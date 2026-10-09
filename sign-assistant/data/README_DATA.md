# Data

Owner: A (Data/ML). Nothing in `data/raw/`, `data/landmarks/`, `data/clips/` or `data/team_recordings/`
goes into git.

## Source

AzSLD – Azerbaijani Sign Language Dataset, Zenodo DOI 10.5281/zenodo.14222948, CC BY 4.0.
Licence and the attribution we must show: [reports/LICENSE_CHECK.md](reports/LICENSE_CHECK.md).

We use only **AzSLD_Words_100**: 100 word classes, 7,248 short clips.

`vocab.json` contains the 60 trained recognition classes, in model class order. Text-to-sign playback uses
`playback_vocab.json`, which includes all 100 word classes and keeps the original 40 entries intact. The smaller training
set's minimum-video threshold is not a requirement for showing a genuine reference clip.

After building the index and group splits, generate the additional playback clips with:

```bash
python -m data.build_clips --camera any --vocab-file data/playback_vocab.json --skip-existing
```

Pass `--ffmpeg <path>` if ffmpeg is not on PATH. `--skip-existing` preserves nonempty clips with existing source
metadata in `clips_index.json`. New clips use train-group sources only; `checked: false` means no landmark quality
check was available. Ship the full `data/clips/` directory with the app; the videos remain gitignored.

| Zenodo file | size | needed |
|---|---|---|
| AzSLD_Words_100.zip | 1.11 GB | yes |
| AzSLD_Words_200.zip | 1.36 GB | no (superset of Words_100) |
| AzSLD_Fingerspelling.zip | 0.55 GB | no |
| AzSLD_Sentences.zip | 43.8 GB | no: `inspect_dataset.py` fetches only its annotation files (~5 MB) |

## Download by hand

1. Open https://doi.org/10.5281/zenodo.14222948 and download `AzSLD_Words_100.zip`
   (md5 `7ec7550b0a41ad56f130734a718f54e4`). The venue network is slow; pass one copy around on a USB stick.
2. Unzip it into `data/raw/`, so that you get `data/raw/AzSLD_Words_100/<LABEL>/<video_id>.mp4`.
   From `sign-assistant/`:

   ```bash
   python -m zipfile -e path/to/AzSLD_Words_100.zip data/raw/
   ```

   `python -m zipfile` keeps the Azerbaijani folder names (MƏN, SİZ, ...) intact on every OS.

## Build the index

```bash
python -m data.inspect_dataset
```

The first run downloads the AzSLD_Sentences annotation JSONs into `data/raw/AzSLD_Sentences_ann/`
(a few minutes). Then it reads every video with OpenCV and writes:

- `data/index.csv`: one row per video, format in CONTRACT.md.
- `data/reports/dataset_summary.md`: counts, durations and class balance (also printed).

## Curated recognition vocabulary

The active camera model has 60 recognition classes. It retains the original 40 entries
in order and appends these 20 genuine dataset classes:

`SALAM`, `NECƏ`, `YAXŞI`, `SU`, `ATA`, `MƏKTƏB`, `AİLƏ`, `GƏLMƏK`, `AVTOMOBİL`, `OXUMAQ`,
`İŞLƏMƏK`, `VAXT`, `AXŞAM`, `İNDİ`, `ÇOX`, `LAZIM`, `AVTOBUS`, `QAPI`, `BU HƏFTƏ`, `AD GÜNÜ`.

The train/val recording-date groups in `splits.json` remain unchanged. The added signs already exist in
AzSLD_Words_100, so this update requires no new dataset download. `playback_vocab.json` remains the separate
100-class map; recognition class order does not need to match playback list order.
The active GRU was trained on 5,080 prepared training clips and evaluated on 1,280 validation clips.

`python -m data.choose_vocab --min-videos 25 --min-signers 6 --camera any` is the historical 40-class selector.
Its default `--max-classes` is 40, and it rewrites both the vocabulary and the group split. Retain the supplied
curated `vocab.json` and `splits.json` when rebuilding the expanded model:

```bash
python -m pose.extract_dataset --camera any
python -m model.train --arch gru --camera any
python -m model.calibrate --target 0.95
python -m eval.evaluate --split val
```

Existing landmark files are reused; extraction adds missing vocabulary videos. Training must write matching
`config.json` classes and weights, and calibration must run after every training run. Deploy `vocab.json` together
with its matching `model/artifacts/model.pt` and `model/artifacts/config.json`.

The original 40-class metrics are historical and retained in
[../eval/REPORT_40_CLASSES.md](../eval/REPORT_40_CLASSES.md). The active 60-class evaluation is in
[../eval/REPORT.md](../eval/REPORT.md): top-1 69.5%, macro-accuracy 81.5%, coverage 40.0% (512 of 1280),
and selective accuracy 95.9% (491 of 512), at temperature 0.7558, tau 0.9 and margin 0.2.
The added classes have only 2-11 validation clips each. Neither recording-date validation split measures
unseen-signer accuracy; selected saved-landmark `/recognise` checks do not verify fresh webcam signing.

## What the data has, and what it does not

- **Label:** the folder name. Copy it as written (e.g. `QARABAQ`); fix spelling only in `gloss`, never in `dataset_label`.
- **video_id:** the 32-hex file name. It is also a tag `key` in
  `AzSLD_Sentences/<n>/ann/<YYYY-MM-DD HH-MM-SS>.mp4.json`: every Words clip was cut from a sentence recording.
- **No signer ID** in file names, folders or JSON (`labelerLogin` is the annotator, not the signer). `signer_id` is empty.
- **No camera information.** `camera` is `unknown`. The clips we looked at are frontal, 1280x960.
- **30 fps**, although the Zenodo description says 35.
- **group** = `rec_<recording date>`: the split unit (CONTRACT.md v2). Dates stand in for signers. The same
  signer may appear on several dates, so val is not signer-independent. The signer-independent test set is the
  team recordings (`data/team_recordings/`, groups `team_<id>`).
