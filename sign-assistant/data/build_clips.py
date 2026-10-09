"""Cut one reference clip per vocab sign -> data/clips/<id>.mp4 and data/clips_index.json.

Owner: D (Direction B/Speech/Eval). Run from sign-assistant/ (AzSLD rows have camera "unknown"):
    python -m data.build_clips --camera any
For the complete text-to-sign playback vocabulary, keeping existing clips and their provenance:
    python -m data.build_clips --camera any --vocab-file data/playback_vocab.json --skip-existing
For each vocab id it picks ONE video from a TRAIN group of data/splits.json (never val or test). With
landmarks it prefers a found segment (pose.segment.segment_offline), then a high hand_ratio and
valid_ratio, then a duration close to the class median ("checked": true). Without landmarks it takes the
median-duration clip whole ("checked": false). The cut is the segment plus 300 ms on each side,
transcoded by ffmpeg to 480p H.264, no audio, faststart, 25 fps.
"""
import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from pose.features import arrays_to_features
from pose.segment import load_config, segment_offline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAD_MS = 300


def train_candidates(rows, vocab, splits, camera):
    """vocab id -> index rows of that class in TRAIN groups (empty group = train-only, CONTRACT v2)."""
    train = set(splits["train"]) | {""}
    id_of = {v["dataset_label"]: v["id"] for v in vocab}
    out = defaultdict(list)
    for r in rows:
        if r["dataset_label"] in id_of and r["group"] in train and camera in ("any", r["camera"]):
            out[id_of[r["dataset_label"]]].append(r)
    return out


def clip_ms(row):
    return 1000.0 * int(row["n_frames"]) / float(row["fps"])


def landmark_info(row, landmarks_dir, cfg):
    """Quality of one clip from its landmark file, or None if there is no file."""
    path = landmarks_dir / f"{row['video_id']}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    pose, hands, t, w, h = z["pose"], z["hands"], z["t"], int(z["w"]), int(z["h"])
    _, info = arrays_to_features(pose, hands, t, w, h)
    span = segment_offline(pose, hands, t, w, h, cfg)
    start, end = (float(t[span[0]]), float(t[span[1] - 1])) if span else (0.0, clip_ms(row))
    return {"segment": span is not None, "hand_ratio": info["hand_ratio"], "valid_ratio": info["valid_ratio"],
            "start_ms": start, "end_ms": end}


def choose(rows, infos):
    """(row, info or None, checked). infos[i] is landmark_info of rows[i] (None = no landmarks)."""
    median = statistics.median(clip_ms(r) for r in rows)
    with_info = [(r, i) for r, i in zip(rows, infos) if i is not None]
    if not with_info:
        return min(rows, key=lambda r: (abs(clip_ms(r) - median), r["video_id"])), None, False
    row, info = max(with_info, key=lambda ri: (ri[1]["segment"], round(ri[1]["hand_ratio"], 2),
                                               round(ri[1]["valid_ratio"], 2), -abs(clip_ms(ri[0]) - median),
                                               ri[0]["video_id"]))
    return row, info, True


def cut_window(row, info):
    """[start_ms, end_ms] of the segment plus PAD_MS on each side, inside the source clip."""
    start, end = (info["start_ms"], info["end_ms"]) if info else (0.0, clip_ms(row))
    return max(0.0, start - PAD_MS), min(clip_ms(row), end + PAD_MS)


def transcode(ffmpeg, src, out, start_ms, end_ms):
    cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{start_ms / 1000:.3f}", "-i", str(src),
           "-t", f"{(end_ms - start_ms) / 1000:.3f}", "-an", "-vf", "scale=-2:480,fps=25",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out.exists() and out.stat().st_size > 0, result.stderr.strip()[-200:]


def main(argv=None):
    p = argparse.ArgumentParser(description="One reference clip per vocab sign.")
    p.add_argument("--camera", default="front", help="front, side, unknown or any (AzSLD: any)")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--vocab-file", type=Path,
                   help="vocabulary JSON; defaults to <data-dir>/vocab.json (recognition vocabulary)")
    p.add_argument("--skip-existing", action="store_true",
                   help="keep nonempty clips already recorded in clips_index.json; generate missing clips only")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg"), help="path to ffmpeg")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    vocab_path = a.vocab_file or a.data_dir / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    clips_dir = a.data_dir / "clips"
    index_path = a.data_dir / "clips_index.json"
    index = (json.loads(index_path.read_text(encoding="utf-8"))
             if a.skip_existing and index_path.exists() else {})
    existing = {v["id"] for v in vocab
                if isinstance(index.get(v["id"]), dict) and (clips_dir / f"{v['id']}.mp4").is_file()
                and (clips_dir / f"{v['id']}.mp4").stat().st_size > 0}
    if not a.ffmpeg and len(existing) != len(vocab):
        sys.exit("ffmpeg not found: install it or pass --ffmpeg. Nothing written.")

    splits = json.loads((a.data_dir / "splits.json").read_text(encoding="utf-8"))
    with open(a.data_dir / "index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = train_candidates(rows, vocab, splits, a.camera)
    cfg = load_config()
    clips_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for v in vocab:
        if v["id"] in existing:
            print(f"{v['id']:12s} {v['gloss']:12s} kept existing clip and source metadata")
            continue
        index.pop(v["id"], None)
        rows_v = candidates.get(v["id"], [])
        if not rows_v:
            missing.append((v["id"], f"no {a.camera}-camera video in a train group"))
            continue
        row, info, checked = choose(rows_v, [landmark_info(r, a.data_dir / "landmarks", cfg) for r in rows_v])
        start, end = cut_window(row, info)
        ok, err = transcode(a.ffmpeg, ROOT / row["video_path"], clips_dir / f"{v['id']}.mp4", start, end)
        if not ok:
            missing.append((v["id"], f"ffmpeg failed: {err}"))
            continue
        index[v["id"]] = {"source_video_id": row["video_id"], "signer_id": row["signer_id"] or row["group"],
                          "start_ms": round(start), "end_ms": round(end), "checked": checked}
        quality = (f"segment {'yes' if info['segment'] else 'no'}, hand_ratio {info['hand_ratio']:.2f}"
                   if info else "no landmarks")
        print(f"{v['id']:12s} {v['gloss']:12s} {row['video_id']}  {start:5.0f}-{end:5.0f} ms  "
              f"{quality}  checked={checked}  ({len(rows_v)} train candidates)")

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n{sum(v['id'] in index for v in vocab)}/{len(vocab)} clips in {clips_dir}; clips_index.json written.")
    print("ids without a usable clip: " + (", ".join(f"{i} ({why})" for i, why in missing) if missing else "none"))


if __name__ == "__main__":
    main()
