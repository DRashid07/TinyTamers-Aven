"""Train the classifier -> model/artifacts/model.pt and config.json.

Owner: A (Data/ML). Run from sign-assistant/ (AzSLD rows have camera "unknown", so --camera any):
    python -m model.train --arch gru --camera any
    python -m model.train --arch transformer --camera any --out model/artifacts/transformer
Model selection uses val macro-accuracy only; the test split is never loaded.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model.dataset import augment, load_split, load_vocab
from model.net import build_model
from pose.features import FEATURE_VERSION, N_FEATURES

ROOT = Path(__file__).resolve().parent.parent
T = 32


class TrainSet(Dataset):
    def __init__(self, X, y, seed):
        self.X, self.y, self.rng = X, y, np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(augment(self.X[i], self.rng)), int(self.y[i])


def predict(model, X, batch=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)]).argmax(1).numpy()


def scores(pred, y):
    """top-1 accuracy and macro-accuracy (mean per-class recall over the classes present in y)."""
    return float((pred == y).mean()), float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def initialize_from(model, config, directory):
    """Reuse a compatible encoder and classifier rows matched by class id.

    Newly introduced classes keep the model's initial random rows. Validate the
    complete checkpoint before changing the model; calibration belongs to the
    new run and is never copied from the previous config. Return reused classes.
    """
    directory = Path(directory)
    previous = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    if not isinstance(previous, dict):
        raise ValueError("initial checkpoint config must be a JSON object")
    for key in ("arch", "feature_version", "n_features", "T"):
        if key not in previous or key not in config or previous[key] != config[key]:
            raise ValueError(f"initial checkpoint has incompatible {key}")
    for name, settings in (("initial", previous), ("new", config)):
        classes = settings.get("classes")
        if (not isinstance(classes, list) or not classes
                or any(not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9_]+", identifier)
                       for identifier in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError(f"{name} config needs nonempty, unique, valid class ids")

    old_classes, new_classes = previous["classes"], config["classes"]
    old_state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    current = model.state_dict()
    heads = {"head.weight", "head.bias"}
    if not isinstance(old_state, dict) or set(old_state) != set(current) or not heads <= set(current):
        raise ValueError("initial checkpoint parameter names differ from the new model")
    for name, tensor in old_state.items():
        expected = list(current[name].shape)
        if name in heads:
            if not expected or expected[0] != len(new_classes):
                raise ValueError("new model classifier does not match its class ids")
            expected[0] = len(old_classes)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected):
            raise ValueError(f"initial checkpoint has incompatible parameter {name}")

    initialized = {name: tensor.clone() if name in heads else old_state[name]
                   for name, tensor in current.items()}
    old_index = {identifier: index for index, identifier in enumerate(old_classes)}
    copied = 0
    for index, identifier in enumerate(new_classes):
        if identifier in old_index:
            for name in heads:
                initialized[name][index].copy_(old_state[name][old_index[identifier]])
            copied += 1
    model.load_state_dict(initialized, strict=True)
    return copied


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the isolated-sign classifier.")
    p.add_argument("--arch", choices=["gru", "transformer"], default="gru")
    p.add_argument("--camera", default="front", help="front, side, unknown or any (AzSLD: any)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12, help="early stopping on val macro-accuracy")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data",
                   help="folder with vocab.json, index.csv, splits.json and landmarks/")
    p.add_argument("--out", type=Path, default=ROOT / "model" / "artifacts")
    p.add_argument("--init-from", type=Path, help="compatible checkpoint directory; reuse shared class rows by id")
    p.add_argument("--rebuild-cache", action="store_true")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    start = time.time()

    vocab = load_vocab(a.data_dir)
    cache = a.data_dir / "landmarks" / f"features_v{FEATURE_VERSION}.npz"
    X_train, y_train, report = load_split("train", a.camera, data_dir=a.data_dir, cache=cache,
                                         T=T, rebuild=a.rebuild_cache)
    X_val, y_val, _ = load_split("val", a.camera, data_dir=a.data_dir, cache=cache, T=T)
    n_classes = len(vocab)
    train_count, val_count = Counter(y_train.tolist()), Counter(y_val.tolist())
    print(f"features: {'cache' if report['cached'] else 'built'} ({time.time() - start:.0f}s); "
          f"train {len(y_train)} videos, val {len(y_val)} videos, {n_classes} classes; "
          f"missing landmark files: {sum(report['missing'].values())}; team recordings cut by segment_offline: "
          f"{report['cut']}, whole-clip fallbacks: {report['fallback']} (AzSLD clips are used whole, CONTRACT v2)")
    print("videos per class (train/val): " + ", ".join(
        f"{v['id']} {train_count[i]}/{val_count[i]}" for i, v in enumerate(vocab)))
    if not n_classes or not len(y_train) or not len(y_val) or len(train_count) < n_classes:
        sys.exit("Some class has no training video, or val is empty. Nothing written.")

    config = {"feature_version": FEATURE_VERSION, "T": T, "n_features": N_FEATURES, "arch": a.arch,
              "classes": [v["id"] for v in vocab], "temperature": 1.0, "tau": 0.7, "margin": 0.15,
              "min_valid_ratio": 0.6, "min_hand_ratio": 0.5}
    model = build_model(config)
    if a.init_from is not None:
        try:
            copied = initialize_from(model, config, a.init_from)
        except Exception as err:  # a bad checkpoint must fail before writing output artifacts
            sys.exit(f"Cannot initialize from {a.init_from}: {err}. Nothing written.")
        print(f"Initialized encoder from {a.init_from}; reused {copied}/{n_classes} class rows by id, "
              f"{n_classes - copied} new class rows retain random initialization.")
    weights = torch.tensor([1.0 / train_count[c] for c in y_train.tolist()], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(TrainSet(X_train, y_train, a.seed), batch_size=a.batch, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_macro, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total, seen = total + loss.item() * len(yb), seen + len(yb)
        top1, macro = scores(predict(model, X_val), y_val)
        print(f"epoch {epoch:2d}  train loss {total / seen:.4f}  val top-1 {top1:6.1%}  val macro {macro:6.1%}"
              f"  ({time.time() - start:.0f}s)", flush=True)
        if macro > best_macro:
            best_macro, best_epoch, waited = macro, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= a.patience:
                print(f"early stop: no val macro-accuracy gain for {a.patience} epochs")
                break

    model.load_state_dict(best_state)
    pred = predict(model, X_val)
    top1, macro = scores(pred, y_val)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out / "model.pt")
    (a.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nbest epoch {best_epoch}: VAL top-1 {top1:.1%}, VAL macro-accuracy {macro:.1%} "
          f"({len(y_val)} videos). Saved {a.out / 'model.pt'} and config.json ({time.time() - start:.0f}s total).")
    print("tau, margin and temperature are placeholders until python -m model.calibrate.")

    gloss = [v["gloss"] for v in vocab]
    confused = Counter((int(t), int(q)) for t, q in zip(y_val, pred) if t != q)
    print("\n10 most confused pairs on val (true -> predicted: count of the true class's val videos):")
    for (t, q), n in confused.most_common(10):
        print(f"  {gloss[t]} -> {gloss[q]}: {n} of {val_count[t]}")


if __name__ == "__main__":
    main()
