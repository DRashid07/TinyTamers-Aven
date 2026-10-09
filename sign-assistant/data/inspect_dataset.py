"""Inspect the AzSLD Words_100 subset -> data/index.csv and data/reports/dataset_summary.md.

Owner: A (Data/ML). Run from sign-assistant/: python -m data.inspect_dataset

Needs (gitignored, see data/README_DATA.md):
  data/raw/AzSLD_Words_100/<LABEL>/<video_id>.mp4   unzipped by hand from Zenodo
  data/raw/AzSLD_Sentences_ann/                     annotation JSONs, fetched on the first run

AzSLD has no signer IDs. Words clips are cut from the Sentences recordings: a clip's file name
is a tag "key" in AzSLD_Sentences/<n>/ann/<YYYY-MM-DD HH-MM-SS>.mp4.json, so the recording date
of that file is the clip's split group (CONTRACT.md "data/index.csv").
"""
import csv
import datetime
import io
import json
import re
import statistics
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
WORDS_DIR = ROOT / "data" / "raw" / "AzSLD_Words_100"
ANN_DIR = ROOT / "data" / "raw" / "AzSLD_Sentences_ann"
INDEX_CSV = ROOT / "data" / "index.csv"
SUMMARY_MD = ROOT / "data" / "reports" / "dataset_summary.md"
SENTENCES_URL = "https://zenodo.org/api/records/14222948/files/AzSLD_Sentences.zip/content"
COLUMNS = ["video_id", "video_path", "dataset_label", "signer_id", "camera", "n_frames", "fps", "group"]
ANN_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}-\d{2}-\d{2}\.mp4\.json$")


def http_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), response.headers["Content-Range"]
        except OSError:
            if attempt == 4:
                raise


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(http_range(url, 0, 0)[1].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        data = http_range(self.url, self.pos, self.pos + n - 1)[0]
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_sentence_annotations(url, out_dir):
    """Download only the .json/.csv members of AzSLD_Sentences.zip (~5 MB of 44 GB)."""
    archive = zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=64 * 1024))
    members = [i for i in archive.infolist() if i.filename.endswith((".json", ".csv"))]
    # Read in archive order: one folder's JSONs are adjacent, so the buffer serves most reads.
    for n, info in enumerate(sorted(members, key=lambda i: i.header_offset), 1):
        target = out_dir / info.filename
        if not (target.exists() and target.stat().st_size == info.file_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
        if n % 1000 == 0:
            print(f"  {n}/{len(members)} annotation files")
    (out_dir / ".complete").write_text(f"{len(members)} files from {url}\n", encoding="utf-8")


def load_recording_dates(ann_dir):
    """Map clip key -> date of the sentence recording it was cut from."""
    dates = {}
    for path in ann_dir.glob("AzSLD_Sentences/*/ann/*.json"):
        match = ANN_NAME.match(path.name)
        if match:
            for tag in json.loads(path.read_text(encoding="utf-8"))["tags"]:
                dates[tag["key"]] = match.group(1)
    return dates


def probe(path):
    """Return (n_frames, fps, width, height) read with OpenCV, or None if unreadable."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        return (int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def build_index(words_dir, dates, root=ROOT):
    """Return (rows for index.csv, unreadable paths, Counter of resolutions)."""
    rows, unreadable, resolutions = [], [], Counter()
    for path in sorted(words_dir.glob("*/*.mp4")):
        probed = probe(path)
        if probed is None:
            unreadable.append(path)
            continue
        n_frames, fps, width, height = probed
        resolutions[f"{width}x{height}"] += 1
        date = dates.get(path.stem)
        rows.append({
            "video_id": path.stem,
            "video_path": path.relative_to(root).as_posix(),
            "dataset_label": path.parent.name,
            "signer_id": "",
            "camera": "unknown",
            "n_frames": n_frames,
            "fps": round(fps, 3),
            "group": f"rec_{date}" if date else "",
        })
    duplicates = [v for v, c in Counter(r["video_id"] for r in rows).items() if c > 1]
    if duplicates:
        raise ValueError(f"video_id is not unique: {duplicates[:5]}")
    return rows, unreadable, resolutions


def write_index(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def duration_ms(row):
    return 1000 * row["n_frames"] / row["fps"] if row["fps"] else 0.0


def summarise(rows, unreadable, resolutions):
    """Return the dataset summary as markdown."""
    durations = sorted(duration_ms(r) for r in rows)
    deciles = statistics.quantiles(durations, n=10)
    per_class, per_group = defaultdict(list), defaultdict(list)
    for r in rows:
        per_class[r["dataset_label"]].append(r)
        per_group[r["group"] or "(none)"].append(r)
    class_sizes = sorted(len(v) for v in per_class.values())
    fps_counts = Counter(r["fps"] for r in rows)

    lines = [
        "# AzSLD Words_100: dataset summary",
        "",
        f"Generated by `python -m data.inspect_dataset` on {datetime.date.today()} from "
        "`data/raw/AzSLD_Words_100` (Zenodo, DOI 10.5281/zenodo.14222948).",
        "",
        "## Totals",
        "",
        "| | |",
        "|---|---|",
        f"| classes | {len(per_class)} |",
        f"| videos (readable) | {len(rows)} |",
        f"| unreadable videos | {len(unreadable)} |",
        "| signers | unknown: the dataset has no signer IDs |",
        "| camera | unknown for every video: no camera metadata |",
        f"| recording dates (split groups) | {sum(1 for g in per_group if g != '(none)')} |",
        f"| videos without a recording date | {len(per_group.get('(none)', []))} |",
        f"| fps (OpenCV) | {', '.join(f'{k}: {v}' for k, v in fps_counts.most_common())} |",
        f"| resolution | {', '.join(f'{k}: {v}' for k, v in resolutions.most_common())} |",
        "",
        "## Clip duration (ms)",
        "",
        "| min | p10 | median | p90 | max | mean |",
        "|---|---|---|---|---|---|",
        f"| {durations[0]:.0f} | {deciles[0]:.0f} | {statistics.median(durations):.0f} | {deciles[-1]:.0f} "
        f"| {durations[-1]:.0f} | {statistics.mean(durations):.0f} |",
        "",
        f"Shorter than 300 ms (contract `min_ms`): {sum(d < 300 for d in durations)}. "
        f"Longer than 4000 ms (`max_ms`): {sum(d > 4000 for d in durations)}.",
        "",
        "## Videos per class",
        "",
        f"Videos per class: min {class_sizes[0]}, median {statistics.median(class_sizes):g}, "
        f"max {class_sizes[-1]}. Classes with < 25 videos: {sum(s < 25 for s in class_sizes)}.",
        "Front-camera videos and distinct signers per class cannot be counted: neither is in the data.",
        "Recording dates are the split groups (a proxy, not signers).",
        "",
        "| label | videos | recording dates | median ms |",
        "|---|---|---|---|",
    ]
    for label, items in sorted(per_class.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        dates = {r["group"] for r in items if r["group"]}
        lines.append(f"| {label} | {len(items)} | {len(dates)} | "
                     f"{statistics.median(duration_ms(r) for r in items):.0f} |")
    lines += [
        "",
        "## Videos per recording date",
        "",
        "Stands in for \"videos per signer\", which the data cannot give.",
        "",
        "| group | videos | classes |",
        "|---|---|---|",
    ]
    for group, items in sorted(per_group.items()):
        lines.append(f"| {group} | {len(items)} | {len({r['dataset_label'] for r in items})} |")
    if unreadable:
        lines += ["", "## Unreadable videos", ""] + [f"- {p.as_posix()}" for p in unreadable]
    return "\n".join(lines) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not WORDS_DIR.is_dir():
        sys.exit(f"Missing {WORDS_DIR}: unzip AzSLD_Words_100.zip into data/raw/ (see data/README_DATA.md)")
    if not (ANN_DIR / ".complete").exists():
        print("Fetching the AzSLD_Sentences annotation JSONs (~5 MB of the 44 GB zip) ...")
        fetch_sentence_annotations(SENTENCES_URL, ANN_DIR)
    rows, unreadable, resolutions = build_index(WORDS_DIR, load_recording_dates(ANN_DIR))
    write_index(rows, INDEX_CSV)
    summary = summarise(rows, unreadable, resolutions)
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {INDEX_CSV.relative_to(ROOT).as_posix()} ({len(rows)} rows) and "
          f"{SUMMARY_MD.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
