"""Evaluate the classifier and its abstain rule on one split -> a section of eval/REPORT.md and a figure.

Owner: D (Direction B/Speech/Eval). Run from sign-assistant/:
    python -m eval.evaluate --split val
    python -m eval.evaluate --split team [--team-dir data/team_recordings]
    python -m eval.evaluate --split test --final [--rerun-reason "..."]   (A, once, after the feature freeze)
The model is loaded through model.predictor and every decision comes from model.abstain, as in the API.
val/test features come from model.dataset.build_features (the training code); team record-mode JSON files go
through the same feature functions after a segment_offline cut. Nothing here writes to the model or its
config: test results never feed back into training or thresholds.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from model.abstain import decide
from model.dataset import build_features
from model.predictor import Predictor
from pose.features import arrays_to_features
from pose.normalize import frames_to_arrays
from pose.segment import load_config, segment_offline

ROOT = Path(__file__).resolve().parent.parent
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ORDER = ["val", "test", "team"]
HEAD = ("# Evaluation report\n\nOwner: D (Direction B/Speech/Eval). Written by "
        "`python -m eval.evaluate --split val|test|team`;\neach split has its own section. The test split is run "
        "once with `--final` after the feature freeze (eval/test_runs.log).")
TITLES = {"val": "Validation (val)", "test": "Test (team recordings, signer-independent)",
          "team": "Team recordings (non-native signers, webcam)"}


def dataset_split(split, vocab, data_dir, T):
    """val/test: index.csv rows of the split's groups -> features via model.dataset.build_features."""
    groups = set(json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))[split])
    labels = {v["dataset_label"] for v in vocab}
    with open(data_dir / "index.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset_label"] in labels and r["group"] in groups]
    data, report = build_features(rows, vocab, data_dir / "landmarks", T)
    no_segment, cfg = 0, load_config()  # diagnostic: AzSLD clips are used whole anyway (CONTRACT v2)
    for vid in data["video_id"]:
        z = np.load(data_dir / "landmarks" / f"{vid}.npz")
        no_segment += segment_offline(z["pose"], z["hands"], z["t"], int(z["w"]), int(z["h"]), cfg) is None
    infos = [{"valid_ratio": float(v), "hand_ratio": float(h)} for v, h in zip(data["valid_ratio"], data["hand_ratio"])]
    return {"X": data["X"], "y": data["y"], "infos": infos, "units": sorted(groups), "unit_name": "groups",
            "missing": sum(report["missing"].values()), "no_segment": no_segment, "n": len(data["y"])}


def team_split(team_dir, vocab, T):
    """Record-mode files {"id", "signer", "w", "h", "frames"} -> segment_offline cut (whole clip if no sign)."""
    index_of = {v["id"]: i for i, v in enumerate(vocab)}
    cfg, X, y, infos, signers, no_segment, skipped = load_config(), [], [], [], set(), 0, 0
    for path in sorted(team_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("id") not in index_of:
            skipped += 1
            continue
        pose, hands, t = frames_to_arrays(rec["frames"])
        span = segment_offline(pose, hands, t, rec["w"], rec["h"], cfg)
        if span is None:
            no_segment += 1
        else:
            pose, hands, t = pose[span[0]:span[1]], hands[span[0]:span[1]], t[span[0]:span[1]]
        x, info = arrays_to_features(pose, hands, t, rec["w"], rec["h"], T=T)
        X.append(x)
        y.append(index_of[rec["id"]])
        infos.append(info)
        signers.add(rec.get("signer", "?"))
    return {"X": np.array(X, dtype=np.float32).reshape(-1, T, 192), "y": np.array(y, dtype=np.int64),
            "infos": infos, "units": sorted(signers), "unit_name": "signers", "missing": skipped,
            "no_segment": no_segment, "n": len(y)}


def metrics(probs, y, infos, cfg, vocab):
    """Everything the report shows, from class probabilities and true class indices."""
    n_classes = len(vocab)
    pred = probs.argmax(1)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (y, pred), 1)
    rows = counts.sum(1, keepdims=True)
    rate = np.divide(counts, rows, out=np.zeros(counts.shape), where=rows > 0)
    present = [c for c in range(n_classes) if rows[c, 0]]
    per_class = {c: rate[c, c] for c in present}
    pairs = sorted(((rate[a, b] + rate[b, a]) / 2, a, b) for a in present for b in present if a < b)[::-1]

    def selective(tau, margin):
        results = [decide(p, i, {**cfg, "tau": tau, "margin": margin}, vocab) for p, i in zip(probs, infos)]
        ok = [r["status"] == "ok" for r in results]
        right = [r["status"] == "ok" and r["id"] == cfg["classes"][t] for r, t in zip(results, y)]
        return sum(ok) / len(y), (sum(right) / sum(ok) if sum(ok) else float("nan")), sum(ok), sum(right)

    return {"top1": float((pred == y).mean()), "macro": float(np.mean(list(per_class.values()))),
            "per_class": per_class, "counts": counts, "rate": rate,
            "pairs": [(r, a, b) for r, a, b in pairs[:5] if r > 0],
            "chosen": selective(cfg["tau"], cfg["margin"]), "taus": [(t, *selective(t, cfg["margin"])) for t in TAUS]}


def save_confusion(rate, vocab, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    glosses = [v["gloss"] for v in vocab]
    fig, ax = plt.subplots(figsize=(12, 11))
    image = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(glosses)), glosses, rotation=90, fontsize=7)
    ax.set_yticks(range(len(glosses)), glosses, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (each row sums to 1)")
    ax.set_title(title)
    fig.colorbar(image, fraction=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def meaning(split, m, cfg, vocab):
    """Plain-language paragraph; says clearly when the numbers are poor."""
    coverage, sel = m["chosen"][0], m["chosen"][1]
    text = [f"Without abstaining, the model names the right sign for {m['top1']:.0%} of these clips; "
            f"counting every sign equally (macro) it is {m['macro']:.0%}."]
    if m["pairs"]:
        _, a, b = m["pairs"][0]
        text.append(f"The worst mix-up is {vocab[a]['gloss']} / {vocab[b]['gloss']}.")
    wrong = m["counts"].sum(1) - np.diag(m["counts"])
    if m["top1"] < m["macro"] - 0.05 and wrong.any():
        c = int(wrong.argmax())
        other = int(np.argmax(np.where(np.arange(len(vocab)) == c, -1, m["counts"][c])))
        text.append(f"Top-1 is lower than macro because most errors come from one big class: {vocab[c]['gloss']} "
                    f"is wrong in {wrong[c]} of {m['counts'][c].sum()} clips, "
                    f"mostly taken for {vocab[other]['gloss']}.")
    text.append(f"With the abstain rule (tau {cfg['tau']}, margin {cfg['margin']}) it answers {coverage:.0%} of the "
                f"clips and says \"Əmin deyiləm\" for the rest; when it answers it is right {sel:.0%} of the time.")
    if m["macro"] < 0.5 or (sel == sel and sel < 0.8) or coverage < 0.2:
        text.append("These numbers are poor: do not rely on the output, and show it only as an unverified hint.")
    if split == "val":
        text.append("Caution: val groups are recording dates of the same AzSLD signers that are in train, so these "
                    "numbers are optimistic for a new signer; only the test and team sections measure that.")
    if split == "team":
        text.append("These are non-native signers on a webcam, the closest we have to real use; with few clips "
                    "per sign each percentage is uncertain.")
    return " ".join(text)


def section(split, m, data, cfg, vocab, info):
    pct = lambda v: "-" if v != v else f"{v:.1%}"  # noqa: E731 - NaN shows as "-"
    cov, sel, accepted, right = m["chosen"]
    lines = [f"<!-- BEGIN {split} -->", f"## {TITLES[split]}", "",
             f"Date {info['date']} UTC · commit {info['commit']} · config sha256 {info['config']} · model "
             f"{cfg['arch']}, {len(cfg['classes'])} classes, temperature {cfg['temperature']}, tau {cfg['tau']}, "
             f"margin {cfg['margin']}", "",
             f"Split: {data['n']} clips from {len(data['units'])} {data['unit_name']}: {', '.join(data['units'])}.",
             f"Clips without landmarks or with an unknown id (skipped): {data['missing']}. "
             f"segment_offline found no sign in {data['no_segment']} clips"
             + (" (team recordings then use the whole clip)." if split == "team"
                else " (diagnostic only: AzSLD clips are used whole, CONTRACT v2)."), "",
             "| metric | value |", "|---|---|",
             f"| top-1 accuracy, no abstention | {pct(m['top1'])} |",
             f"| macro-accuracy (classes are imbalanced) | {pct(m['macro'])} |",
             f"| coverage at tau {cfg['tau']}, margin {cfg['margin']} | {pct(cov)} ({accepted} of {data['n']}) |",
             f"| selective accuracy at tau {cfg['tau']}, margin {cfg['margin']} | {pct(sel)} ({right} of {accepted}) |",
             "", f"![Row-normalised confusion matrix](figures/confusion_{split}.png)", "",
             "Most confused pairs (symmetric rate = mean of the two row-normalised off-diagonal cells):", "",
             "| pair | symmetric rate | counts |", "|---|---|---|"]
    for rate, a, b in m["pairs"]:
        lines.append(f"| {vocab[a]['gloss']} / {vocab[b]['gloss']} | {rate:.1%} | {vocab[a]['gloss']}→"
                     f"{vocab[b]['gloss']} {m['counts'][a, b]} of {m['counts'][a].sum()}, {vocab[b]['gloss']}→"
                     f"{vocab[a]['gloss']} {m['counts'][b, a]} of {m['counts'][b].sum()} |")
    lines += ["", f"Coverage and selective accuracy by tau (margin {cfg['margin']}):", "",
              "| tau | coverage | selective accuracy | answered |", "|---|---|---|---|"]
    lines += [f"| {t} | {pct(c)} | {pct(s)} | {a} |" for t, c, s, a, _ in m["taus"]]
    lines += ["", "Per-class accuracy (no abstention):", "", "| gloss | clips | top-1 |", "|---|---|---|"]
    lines += [f"| {vocab[c]['gloss']} | {m['counts'][c].sum()} | {pct(acc)} |" for c, acc in m["per_class"].items()]
    lines += ["", "**What this means.** " + meaning(split, m, cfg, vocab), f"<!-- END {split} -->"]
    return "\n".join(lines)


def write_section(report, split, text):
    """Replace the split's section in REPORT.md (or add it), keeping the order val, test, team."""
    old = report.read_text(encoding="utf-8") if report.exists() else ""
    found = {s: m.group(0) for s in ORDER
             for m in [re.search(rf"<!-- BEGIN {s} -->.*?<!-- END {s} -->", old, flags=re.S)] if m}
    found[split] = text
    report.write_text(HEAD + "\n\n" + "\n\n".join(found[s] for s in ORDER if s in found) + "\n", encoding="utf-8")


def git_commit():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    return commit + (" (+ uncommitted changes)" if run("status", "--porcelain") else "")


def check_test_log(log, config_hash, reason):
    """Refuse a second test run for the same config unless a reason is given."""
    previous = [line for line in (log.read_text(encoding="utf-8").splitlines() if log.exists() else [])
                if f"config_sha256={config_hash}" in line]
    if previous and not reason:
        sys.exit(f"The test split was already run for this config ({previous[-1]}). "
                 "Pass --rerun-reason \"...\" to run it again; the reason is logged. Nothing run.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the classifier on val, test or team recordings.")
    p.add_argument("--split", choices=ORDER, required=True)
    p.add_argument("--final", action="store_true", help="required for --split test")
    p.add_argument("--rerun-reason", default="", help="needed to run test again for the same config")
    p.add_argument("--team-dir", type=Path, default=ROOT / "data" / "team_recordings")
    p.add_argument("--model-dir", type=Path, default=ROOT / os.environ.get("MODEL_DIR", "model/artifacts"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--eval-dir", type=Path, default=ROOT / "eval", help="REPORT.md, figures/, test_runs.log")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.split == "test" and not a.final:
        sys.exit("--split test needs --final: it is run once, after the feature freeze. Nothing run.")

    predictor = Predictor(a.model_dir)
    if not predictor.loaded:
        sys.exit(f"No usable model in {a.model_dir}. Nothing run.")
    cfg = predictor.config
    vocab = json.loads((a.data_dir / "vocab.json").read_text(encoding="utf-8"))
    if cfg["classes"] != [v["id"] for v in vocab]:
        sys.exit("config.json classes differ from data/vocab.json ids. Nothing run.")
    config_hash = hashlib.sha256((a.model_dir / "config.json").read_bytes()).hexdigest()[:12]
    log = a.eval_dir / "test_runs.log"
    if a.split == "test":
        check_test_log(log, config_hash, a.rerun_reason)

    data = (team_split(a.team_dir, vocab, cfg["T"]) if a.split == "team"
            else dataset_split(a.split, vocab, a.data_dir, cfg["T"]))
    if not data["n"]:
        where = a.team_dir if a.split == "team" else f"the '{a.split}' groups of splits.json"
        sys.exit(f"No clips in {where}. Nothing run.")
    info = {"date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "commit": git_commit(), "config": config_hash}
    if a.split == "test":
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{info['date']} UTC\tcommit={info['commit']}\tconfig_sha256={config_hash}\t"
                    f"clips={data['n']}\treason={a.rerun_reason or 'first run'}\n")

    probs = np.stack([predictor.probs(x) for x in data["X"]])
    m = metrics(probs, data["y"], data["infos"], cfg, vocab)
    title = f"{TITLES[a.split]}: top-1 {m['top1']:.1%}, macro {m['macro']:.1%}"
    save_confusion(m["rate"], vocab, a.eval_dir / "figures" / f"confusion_{a.split}.png", title)
    text = section(a.split, m, data, cfg, vocab, info)
    write_section(a.eval_dir / "REPORT.md", a.split, text)
    print(text)


if __name__ == "__main__":
    main()
