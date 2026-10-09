"""Choose the MVP sign vocabulary and the group split (CONTRACT v2).

Reads data/index.csv; writes data/vocab.json, data/splits.json and
data/reports/vocab_choice.md. Every table in the report is also printed.

Owner: A (Data/ML). Run from sign-assistant/ (AzSLD rows have camera "unknown"):
    python -m data.choose_vocab --min-videos 25 --min-signers 6 --camera any
"""
import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent

# Our domain/demo words in priority order, matched BY HAND to the exact AzSLD_Words_100 label.
# None = the dataset has no class for the word. Never match automatically.
CANDIDATES = [
    ("mən", "MƏN"),
    ("sən", "SƏN"),
    ("həkim", None),
    ("xəstəxana", None),
    ("getmək", "GETMƏK"),
    ("istəmək", "İSTƏMƏK"),
    ("ağrı", None),  # only AĞRIMAQ ("to hurt") exists; unmatched until the team confirms
    ("sabah", "SABAH"),
    ("bu gün", "BU GÜN"),
    ("vaxt", "VAXT"),
    ("kömək", None),
    ("sənəd", "SƏNƏD"),
]

# Lowercase meaning of every AzSLD_Words_100 label, typed by hand. Never use .lower():
# "İ".lower() gives "i" + a combining dot and "I".lower() gives "i" instead of "ı".
AZ = {
    "2": "2", "A": "a", "AD GÜNÜ": "ad günü", "ALMAQ": "almaq", "ANA": "ana", "ATA": "ata",
    "AVTOBUS": "avtobus", "AVTOMOBİL": "avtomobil", "AXŞAM": "axşam", "AZƏRBAYCAN": "azərbaycan",
    "AĞRIMAQ": "ağrımaq", "AİLƏ": "ailə", "BAKI": "bakı", "BALACA": "balaca", "BAZAR": "bazar",
    "BAĞ": "bağ", "BU": "bu", "BU GÜN": "bu gün", "BU HƏFTƏ": "bu həftə", "BURA": "bura",
    "BURDA": "burda", "BİLMİR": "bilmir", "BİLMƏK": "bilmək", "BİLƏR": "bilər", "BİZ": "biz",
    "D": "d", "DONDURMA": "dondurma", "DÜNƏN": "dünən", "EDİR": "edir", "ETİBARNAMƏ": "etibarnamə",
    "EV": "ev", "EŞİTMƏ": "eşitmə", "EŞİTMƏ MƏHDÜDİYYƏTLİ": "eşitmə məhdudiyyətli",
    "FUTBOL": "futbol", "GECİKMƏK": "gecikmək", "GETMƏK": "getmək", "GÖRMƏK": "görmək",
    "GƏLMƏK": "gəlmək", "HAMI": "hamı", "HANSI": "hansı", "HARDA": "harda", "HƏR GÜN": "hər gün",
    "LAZIM": "lazım", "MÜRACİƏT": "müraciət", "MƏKTƏB": "məktəb", "MƏN": "mən", "MƏNİM": "mənim",
    "MƏNƏ": "mənə", "NECƏ": "necə", "NƏDİR": "nədir", "O": "o", "OLAR": "olar", "OLMAQ": "olmaq",
    "ONUN": "onun", "ORA": "ora", "ORDA": "orda", "OXUMAQ": "oxumaq", "OĞUL": "oğul",
    "PENSİYA": "pensiya", "QAPI": "qapı", "QARABAQ": "qarabağ", "QATAR": "qatar", "RƏSİM": "rəsim",
    "SABAH": "sabah", "SALAM": "salam", "SATICI": "satıcı", "SAĞLAM": "sağlam", "SONRA": "sonra",
    "SU": "su", "SİZ": "siz", "SƏN": "sən", "SƏNƏD": "sənəd", "SƏRGİ": "sərgi",
    "TELEFON": "telefon", "UĞUR": "uğur", "UŞAQ": "uşaq", "VAR": "var", "VAXT": "vaxt",
    "VİZA": "viza", "XATIRLADIM": "xatırladım", "XEYİR": "xeyir", "YAXŞI": "yaxşı", "YAŞ": "yaş",
    "YAŞAMAQ": "yaşamaq", "YEMƏK": "yemək", "YENİ": "yeni", "YOX": "yox", "ÇOX": "çox",
    "ÇƏKMƏK": "çəkmək", "ÜNVANLİ": "ünvanlı", "ÜÇÜN": "üçün", "İ": "i", "İNDİ": "indi",
    "İSTƏMƏK": "istəmək", "İSTƏYİRƏM": "istəyirəm", "İŞ": "iş", "İŞARƏ": "işarə",
    "İŞLƏMƏK": "işləmək", "ƏLİL": "əlil", "ƏRZAQ": "ərzaq",
}

# Spelling slips in dataset labels: shown gloss -> correct word (dataset_label stays exact).
GLOSS_FIX = {"QARABAQ": "QARABAĞ", "ÜNVANLİ": "ÜNVANLI",
             "EŞİTMƏ MƏHDÜDİYYƏTLİ": "EŞİTMƏ MƏHDUDİYYƏTLİ"}

_ASCII = str.maketrans("ƏəİıIÖöÜüĞğÇçŞş", "eeiiioouuggccss")


def slug(label):
    """Dataset label -> ascii id [a-z0-9_] (ids only; gloss and az are never derived)."""
    s = unicodedata.normalize("NFC", label).translate(_ASCII).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def load_index(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def class_stats(rows):
    """label -> (videos, distinct groups)."""
    videos, groups = Counter(), defaultdict(set)
    for r in rows:
        videos[r["dataset_label"]] += 1
        if r["group"]:
            groups[r["dataset_label"]].add(r["group"])
    return {lab: (videos[lab], len(groups[lab])) for lab in videos}


def choose(stats, min_videos, min_groups, max_classes):
    """Kept candidates first (priority order), then the classes with most videos, then groups."""
    def ok(lab):
        return lab in stats and stats[lab][0] >= min_videos and stats[lab][1] >= min_groups

    why = {}
    for word, lab in CANDIDATES:
        if lab and ok(lab) and lab not in why:
            why[lab] = f"candidate '{word}'"
    fill = sorted((lab for lab in stats if ok(lab) and lab not in why),
                  key=lambda lab: (-stats[lab][0], -stats[lab][1], lab))
    for lab in fill:
        why[lab] = "fill: most videos/groups"
    labels = list(why)[:max_classes]
    return labels, {lab: why[lab] for lab in labels}


def split_groups(rows, labels, seeds, frac=0.15):
    """CONTRACT v2: the team_* groups are the test split; every other group goes to train or val, with
    about frac of them in val. Keeps the seed whose smallest per-class video count in val is largest
    (ties: val video share closest to frac). Rows with an empty group are train-only and not listed."""
    per = Counter((r["group"], r["dataset_label"]) for r in rows if r["group"])
    groups = sorted({g for g, _ in per})
    test = [g for g in groups if g.startswith("team_")]
    pool = [g for g in groups if not g.startswith("team_")]
    if len(pool) < 2:
        sys.exit(f"Only {len(pool)} dataset groups: cannot make a train/val split. Nothing written.")
    k = max(1, round(frac * len(pool)))
    pool_videos = sum(n for (g, _), n in per.items() if not g.startswith("team_"))
    best = None
    for seed in range(seeds):
        order = pool[:]
        random.Random(seed).shuffle(order)
        parts = {"train": sorted(order[k:]), "val": sorted(order[:k]), "test": test}
        counts = {name: {lab: (sum(per[g, lab] for g in gs), sum(per[g, lab] > 0 for g in gs))
                         for lab in labels} for name, gs in parts.items()}
        worst = min(counts["val"][lab][0] for lab in labels)
        dev = abs(sum(c[0] for c in counts["val"].values()) / pool_videos - frac)
        if best is None or (worst, -dev) > best[0]:
            best = ((worst, -dev), seed, parts, counts)
    return best[1], best[2], best[3], best[0][0]


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def main(argv=None):
    p = argparse.ArgumentParser(description="Choose the vocabulary and the group split.")
    p.add_argument("--min-videos", type=int, default=25, help="min videos per class (after --camera)")
    p.add_argument("--min-signers", type=int, default=6,
                   help="min distinct groups per class (signers; recording dates for AzSLD)")
    p.add_argument("--camera", default="front", help="front, side, unknown or any")
    p.add_argument("--max-classes", type=int, default=40, help="vocabulary size cap (target 30-50)")
    p.add_argument("--seeds", type=int, default=200, help="number of split seeds to try")
    p.add_argument("--data-dir", type=Path, default=DATA, help="folder with index.csv; outputs go here")
    p.add_argument("--group-column", default="group",
                   help="index.csv column with the split unit (CONTRACT v2)")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = [dict(r, group=r.get(a.group_column) or "") for r in load_index(a.data_dir / "index.csv")
            if a.camera in ("any", r["camera"])]
    if not rows:
        sys.exit(f"index.csv has no '{a.camera}' camera rows (is task P1 done?). Nothing written.")
    stats = class_stats(rows)
    labels, why = choose(stats, a.min_videos, a.min_signers, a.max_classes)
    if not labels:
        sys.exit("No class passes the thresholds. Nothing written.")
    missing_az = [lab for lab in labels if lab not in AZ]
    if missing_az:
        sys.exit(f"No hand-written az meaning for {missing_az}: add them to AZ. Nothing written.")
    vocab = [{"id": slug(lab), "gloss": GLOSS_FIX.get(lab, lab), "az": AZ[lab], "dataset_label": lab}
             for lab in labels]
    ids = [v["id"] for v in vocab]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids), ids

    vocab_rows = [r for r in rows if r["dataset_label"] in why]
    no_group = sum(1 for r in vocab_rows if not r["group"])
    seed, parts, counts, worst = split_groups(vocab_rows, labels, a.seeds)
    if worst == 0:
        sys.exit(f"No seed out of {a.seeds} puts every class in val. Nothing written.")

    cand = []
    for i, (word, lab) in enumerate(CANDIDATES, 1):
        v, g = stats.get(lab, (0, 0))
        status = ("not in dataset" if lab is None else f"no {a.camera}-camera videos" if lab not in stats
                  else "too few videos" if v < a.min_videos else "too few groups" if g < a.min_signers
                  else "kept" if lab in why else "cut by --max-classes")
        cand.append((i, word, lab or "-", v, g, status))
    n_videos = {name: sum(c[0] for c in counts[name].values()) for name in parts}
    total = sum(n_videos.values()) or 1
    test_text = (f"{len(parts['test'])} team_* groups" if parts["test"]
                 else "EMPTY, no team recordings (team_* groups) in index.csv yet")
    lines = [
        "# Vocabulary and group split",
        "",
        "Written by `python -m data.choose_vocab` (owner A). No model results were used.",
        "",
        f"Settings: camera={a.camera}, min videos={a.min_videos}, min groups={a.min_signers}, "
        f"max classes={a.max_classes}, seeds tried={a.seeds}, group column={a.group_column}.",
        f"Index: {len(rows)} '{a.camera}' videos in {len(stats)} classes. "
        f"{len(labels)} classes chosen{'' if len(labels) >= 30 else ' (below the 30-50 target)'}.",
        "AzSLD has no signer IDs: a group is the recording date of the Sentences video a clip was cut from "
        "(a proxy, so one signer can be in train and val). The groups columns count those dates.",
        "",
        "## Candidates (priority order)",
        "",
        *md_table(["#", "word", "dataset_label", "videos", "groups", "status"], cand),
        "",
        "Missing from the dataset: " + (", ".join(w for w, lab in CANDIDATES if lab is None) or "none") + ".",
        "",
        "## Vocabulary (class index = row order)",
        "",
        *md_table(["index", "id", "gloss", "az", "dataset_label", "videos", "groups", "why"],
                  [(i, v["id"], v["gloss"], v["az"], v["dataset_label"], *stats[v["dataset_label"]],
                    why[v["dataset_label"]]) for i, v in enumerate(vocab)]),
        "",
        "Gloss differs from dataset_label only for known spelling slips in the dataset: "
        + ", ".join(f"{k} -> {g}" for k, g in GLOSS_FIX.items()) + ".",
        "",
        "## Group split (CONTRACT v2)",
        "",
        f"Dataset groups go to train/val; test = team recordings: {test_text}. "
        f"Seed {seed}: the smallest per-class video count in val is {worst}. "
        f"{no_group} vocab videos have an empty group: train-only, not listed in splits.json.",
        "",
        *md_table(["split", "groups", "videos", "share"],
                  [(n, len(parts[n]), n_videos[n], f"{n_videos[n] / total:.0%}") for n in parts]),
        "",
        "Per class: videos (groups).",
        "",
        *md_table(["id", "gloss", "train", "val", "test"],
                  [(v["id"], v["gloss"], *(f"{counts[n][lab][0]} ({counts[n][lab][1]})" for n in parts))
                   for v, lab in zip(vocab, labels)]),
    ]
    print("\n".join(lines))

    (a.data_dir / "vocab.json").write_text(
        "[\n" + ",\n".join(json.dumps(v, ensure_ascii=False) for v in vocab) + "\n]\n", encoding="utf-8")
    note = (f"CONTRACT v2 split by the '{a.group_column}' column of data/index.csv, camera={a.camera}. "
            "AzSLD groups are recording dates (a proxy, not signers) and go to train/val; "
            f"best of {a.seeds} seeds, every vocab class has >= {worst} videos in val. "
            f"Test = team_* groups ({len(parts['test'])} so far). Empty-group videos are train-only. "
            "Test is read only by python -m eval.evaluate --split test --final.")
    splits = {"seed": seed, **parts, "note": note}
    (a.data_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf-8")
    (a.data_dir / "reports").mkdir(exist_ok=True)
    (a.data_dir / "reports" / "vocab_choice.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
