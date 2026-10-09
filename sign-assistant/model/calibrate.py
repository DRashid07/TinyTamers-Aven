"""Temperature scaling and abstention thresholds (tau, margin), fitted on the VAL split only.

Owner: A (Data/ML). Run from sign-assistant/ after python -m model.train:
    python -m model.calibrate --target 0.90
Reads model/artifacts/{config.json, model.pt}, data/index.csv, data/vocab.json, data/splits.json
(only the "val" groups; the test split is never read) and data/landmarks/<video_id>.npz.
Writes temperature, tau and margin into config.json and the sweep to val_threshold_sweep.csv.
Coverage and selective accuracy come from model.abstain.decide, so they match what users get.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.abstain import decide
from model.predictor import Predictor
from pose.features import arrays_to_features

ROOT = Path(__file__).resolve().parent.parent
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2]
SWEEP_COLUMNS = ["tau", "margin", "n", "accepted", "correct", "coverage", "selective_accuracy"]


def load_val(index_rows, val_groups, vocab, classes, landmarks_dir, T):
    """VAL clips of the vocabulary -> X (N, T, 192), labels (N,), infos, number without landmarks.

    Words clips are used whole (CONTRACT v2), with the same features as training and live use.
    """
    class_of = {v["dataset_label"]: classes.index(v["id"]) for v in vocab}
    X, labels, infos, missing = [], [], [], 0
    for row in index_rows:
        if row["group"] not in val_groups or row["dataset_label"] not in class_of:
            continue
        path = landmarks_dir / f"{row['video_id']}.npz"
        if not path.exists():
            missing += 1
            continue
        z = np.load(path)
        x, info = arrays_to_features(z["pose"], z["hands"], z["t"], int(z["w"]), int(z["h"]), T=T)
        X.append(x)
        labels.append(class_of[row["dataset_label"]])
        infos.append(info)
    if not X:
        sys.exit(f"No val clips with landmarks ({missing} without landmarks). Nothing written.")
    return np.stack(X), np.array(labels), infos, missing


def val_logits(model, X, batch=256):
    with torch.no_grad():
        return torch.cat([model(torch.as_tensor(X[i:i + batch])) for i in range(0, len(X), batch)]).numpy()


def nll(logits, labels, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    return float(F.cross_entropy(logits / temperature, torch.as_tensor(labels, dtype=torch.long)))


def fit_temperature(logits, labels):
    """Temperature that minimises val NLL (LBFGS on log T, so T stays positive)."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)  # numpy ints are int32 on Windows
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())


def softmax(logits, temperature=1.0):
    """Same as Predictor.probs: softmax(logits / temperature), row-wise."""
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64) / temperature, dim=1).numpy()


def ece(probs, labels, bins=15):
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def sweep(probs, labels, infos, cfg, vocab, taus=TAUS, margins=MARGINS):
    """One row per (tau, margin): coverage = share of val clips decide() accepts (quality gates
    included), selective_accuracy = accuracy on the accepted ones (NaN if none)."""
    rows = []
    for tau in taus:
        for margin in margins:
            trial = {**cfg, "tau": tau, "margin": margin}
            results = [decide(p, info, trial, vocab) for p, info in zip(probs, infos)]
            accepted = sum(r["status"] == "ok" for r in results)
            correct = sum(r["status"] == "ok" and r["id"] == cfg["classes"][y] for r, y in zip(results, labels))
            rows.append({"tau": tau, "margin": margin, "n": len(results), "accepted": accepted,
                         "correct": correct, "coverage": accepted / len(results),
                         "selective_accuracy": correct / accepted if accepted else float("nan")})
    return rows


def choose(rows, target):
    """Return (row, reached). Highest coverage with selective accuracy >= target (ties: the stricter
    pair); if no pair reaches it, the best selective accuracy (ties: higher coverage)."""
    scored = [r for r in rows if r["accepted"]]
    reaching = [r for r in scored if r["selective_accuracy"] >= target]
    if reaching:
        return max(reaching, key=lambda r: (r["coverage"], r["selective_accuracy"], r["tau"], r["margin"])), True
    if not scored:
        return None, False
    return max(scored, key=lambda r: (r["selective_accuracy"], r["coverage"])), False


def update_config(path, temperature, tau, margin):
    """Set temperature, tau and margin in config.json; every other key is kept."""
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(temperature=round(temperature, 4), tau=tau, margin=margin)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_sweep(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "coverage": f"{r['coverage']:.4f}", "selective_accuracy": f"{r['selective_accuracy']:.4f}"})


def format_sweep(rows, chosen):
    lines = [" tau  margin  coverage  sel.acc  accepted/n"]
    for r in rows:
        mark = "  <- chosen" if r is chosen else ""
        lines.append(f"{r['tau']:4.2f}  {r['margin']:6.2f}  {r['coverage']:8.1%}  {r['selective_accuracy']:7.1%}"
                     f"  {r['accepted']:4d}/{r['n']}{mark}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Calibrate temperature, tau and margin on the VAL split.")
    p.add_argument("--target", type=float, default=0.90, help="minimum val selective accuracy (default 0.90)")
    p.add_argument("--model-dir", type=Path, default=ROOT / os.environ.get("MODEL_DIR", "model/artifacts"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    predictor = Predictor(a.model_dir)
    if not predictor.loaded:
        sys.exit(f"No usable model in {a.model_dir} (needs model.pt and config.json from python -m model.train). "
                 "Nothing written.")
    cfg = predictor.config
    vocab = json.loads((a.data_dir / "vocab.json").read_text(encoding="utf-8"))
    if cfg["classes"] != [v["id"] for v in vocab]:
        sys.exit("config.json classes differ from data/vocab.json ids: retrain first. Nothing written.")
    val_groups = set(json.loads((a.data_dir / "splits.json").read_text(encoding="utf-8"))["val"])
    with open(a.data_dir / "index.csv", encoding="utf-8", newline="") as f:
        index_rows = list(csv.DictReader(f))
    X, labels, infos, missing = load_val(index_rows, val_groups, vocab, cfg["classes"],
                                         a.data_dir / "landmarks", cfg["T"])
    logits = val_logits(predictor.model, X)

    temperature = fit_temperature(logits, labels)
    raw, calibrated = softmax(logits), softmax(logits, temperature)
    rows = sweep(calibrated, labels, infos, cfg, vocab)
    chosen, reached = choose(rows, a.target)
    gated = sum(i["valid_ratio"] < cfg["min_valid_ratio"] or i["hand_ratio"] < cfg["min_hand_ratio"] for i in infos)

    print(f"VAL only: {len(labels)} clips in {len(set(labels.tolist()))} classes from {len(val_groups)} groups "
          f"({missing} without landmarks skipped). The test split is not read.")
    print(f"Top-1 accuracy (no abstention): {float((raw.argmax(1) == labels).mean()):.1%}. "
          f"Quality gates (invalid_pose/no_hands) reject {gated} clips at any tau/margin.")
    print(f"Temperature: 1.0 -> {temperature:.4f}")
    print(f"NLL: {nll(logits, labels):.4f} -> {nll(logits, labels, temperature):.4f}   "
          f"ECE: {ece(raw, labels):.4f} -> {ece(calibrated, labels):.4f}")
    print(format_sweep(rows, chosen))
    if chosen is None:
        sys.exit("No (tau, margin) pair accepts any val clip. config.json not changed.")
    summary = (f"tau={chosen['tau']}, margin={chosen['margin']}: val coverage {chosen['coverage']:.1%}, "
               f"selective accuracy {chosen['selective_accuracy']:.1%} ({chosen['accepted']}/{chosen['n']} accepted)")
    if reached:
        print(f"Chosen (highest coverage with val selective accuracy >= {a.target:.0%}): {summary}")
    else:
        print(f"WARNING: no pair reaches val selective accuracy {a.target:.0%}. "
              f"Chosen the best selective accuracy instead: {summary}")

    update_config(a.model_dir / "config.json", temperature, chosen["tau"], chosen["margin"])
    write_sweep(rows, a.model_dir / "val_threshold_sweep.csv")
    print(f"Wrote temperature, tau, margin to {a.model_dir / 'config.json'} and the sweep to "
          f"{a.model_dir / 'val_threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
