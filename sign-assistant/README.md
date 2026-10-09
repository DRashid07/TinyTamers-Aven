# AzSL sign assistant

Owner: C (Backend/LLM). Team rules, file ownership and data formats: [CONTRACT.md](CONTRACT.md).

## What it is

A hackathon prototype for Azerbaijani Sign Language (AzSL). Direction A recognises 40 isolated signs from a laptop
webcam and builds an Azerbaijani sentence from them; Direction B shows dataset sign clips for typed or spoken text.

**Safety.** This is an assistive prototype, not a validated translation. It abstains ("Əmin deyiləm, zəhmət olmasa
təkrar edin.") when unsure, always shows the recognised glosses, and must not be used for medical or legal
decisions: use a sign language interpreter. Webcam frames stay in the browser; only landmark coordinates are sent to
the server. If speech is used, it leaves the machine: the sentence text goes to Microsoft Azure for text-to-speech,
and microphone audio goes to Google through Chrome's speech recognition. With an Anthropic key, the glosses and the
typed text are sent to Claude.

![Direction A: status pill, word buffer and composed sentence](docs/direction_a.png)

## Architecture

```mermaid
flowchart LR
  subgraph A["Direction A: sign to text"]
    cam["Webcam"] --> mp["MediaPipe in the browser<br/>pose + hands"]
    mp -- "WebSocket: landmarks only" --> seg["segment<br/>pose/segment.py"]
    seg --> feat["features<br/>pose/features.py"]
    feat --> model["GRU classifier<br/>model/predictor.py"]
    model --> abst{"abstain?<br/>model/abstain.py"}
    abst -- "unsure" --> unsure["Əmin deyiləm"]
    abst -- "ok" --> buf["word buffer<br/>web/app.js"]
    buf -- "Cümlə qur" --> comp["Claude compose<br/>api/compose.py"]
    comp --> tts["Azure TTS<br/>api/speech.py"]
  end
  subgraph B["Direction B: text to sign"]
    txt["typed text or Chrome STT"] --> llm["Claude, vocabulary enum<br/>api/text_to_signs.py"]
    llm --> clips["sign clips<br/>data/clips"]
  end
```

Without an Anthropic key, compose and text-to-signs use deterministic fallbacks (the glosses as words; a prefix
match against the vocabulary). Without an Azure key, `/tts` returns 503 and the "Səsləndir" button hides itself.

## Setup

Python 3.11 and Chrome. Run every command from `sign-assistant/`.

```bash
python3.11 -m venv .venv          # Windows: py -3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # about 500 MB (torch, mediapipe); CPU torch, GPU line in requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
bash scripts/get_models.sh        # MediaPipe .task models -> web/models/ (Windows: run it in Git Bash)
pytest -q                         # all tests should pass
```

Windows: clone into a short path (for example `C:\azsl`). With long paths switched off (the Windows default), a
path over 260 characters breaks the installed `anthropic` package (`ModuleNotFoundError: No module named
'anthropic.types.beta...'`) and the API does not start.

No Python 3.11 installed? With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.11 .venv` and
`uv pip install --python .venv -r requirements.txt` (uv downloads Python 3.11 and installs the same pins).

`.env` (never commit it): `ANTHROPIC_API_KEY` for Claude (optional), `CLAUDE_MODEL` (default `claude-opus-5-5`),
`AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` for text-to-speech (optional), `MODEL_DIR` (default `model/artifacts`).
`data/build_clips.py` also needs `ffmpeg` on the PATH.

## Data and model

Raw videos, landmarks, clips and model weights are not in git. Either:

- **Fast path (demo):** copy from a teammate who ran the pipeline: `model/artifacts/model.pt` and
  `model/artifacts/config.json`, and `data/clips/*.mp4`. Then go to [Run](#run).
- **Full pipeline:** the steps below.

## Data preparation

Details: [data/README_DATA.md](data/README_DATA.md). Runtimes measured on the team laptop (AMD Ryzen AI 7 350,
8 cores, 15 GB RAM, CPU only) on the venue network (about 1 MB/s):

| step | command | writes | runtime |
|---|---|---|---|
| 1. download | by hand from Zenodo: `AzSLD_Words_100.zip`, unzip into `data/raw/` | `data/raw/AzSLD_Words_100/` | 1.11 GB: about 20 min with 8 parallel connections |
| 2. inspect | `python -m data.inspect_dataset` | `data/index.csv`, `data/reports/dataset_summary.md` | 3 min 7 s for 7,248 videos; the first run also fetches ~5 MB of Sentences annotations (about 20 min here) |
| 3. vocabulary and split | `python -m data.choose_vocab --min-videos 25 --min-signers 6 --camera any` | `data/vocab.json`, `data/splits.json`, `data/reports/vocab_choice.md` | 0.1 s |
| 4. extract landmarks | `python -m pose.extract_dataset --camera any` | `data/landmarks/<video_id>.npz` for the 5,863 vocabulary videos | 83 videos/min per process (measured on 24 videos); it runs cores-1 processes, `--shard i/n` splits it over laptops |
| 5. clips | `python -m data.build_clips --camera any` | `data/clips/<id>.mp4` (40), `data/clips_index.json` | 15 s |

`--camera any` is needed because AzSLD has no camera metadata (every clip is "unknown").

## Train, calibrate, evaluate

```bash
python -m model.train --arch gru --camera any   # -> model/artifacts/model.pt, config.json (5 min 44 s on the CPU)
python -m model.calibrate --target 0.90          # temperature, tau, margin on val only (about 5 s)
python -m eval.evaluate --split val              # val section of eval/REPORT.md (10 s)
python -m eval.evaluate --split team             # after the team recordings: data/team_recordings/README.md
python -m eval.evaluate --split test --final     # once, after the feature freeze
```

Re-run `model.calibrate` after every training run: training writes placeholder thresholds.

## Run

```bash
uvicorn api.main:app
```

Open http://localhost:8000 in Chrome and allow the camera (Direction A); Direction B is http://localhost:8000/signs.html.
Without `model/artifacts`, every sign abstains with `model_not_loaded`. `GET /health` and `GET /model-info` show
the state.

## Results

From [eval/REPORT.md](eval/REPORT.md). **There is no unseen-signer number yet:** the test split is empty and the
team-recording section is pending (no recordings). The only numbers are on **val, which is not signer-independent**:
AzSLD has no signer IDs, val is grouped by recording date, and the same signers are in train.

Val: 1166 clips from 9 groups; model gru, 40 classes, temperature 0.8386, tau 0.9, margin 0.2.

| metric | value |
|---|---|
| top-1 accuracy, no abstention | 66.3% |
| macro-accuracy (classes are imbalanced) | 82.2% |
| coverage at tau 0.9, margin 0.2 | 36.8% (429 of 1166) |
| selective accuracy at tau 0.9, margin 0.2 | 96.3% (413 of 429) |

The 5 most confused pairs (symmetric rate): MƏNİM / MƏNƏ 32.2%, O / ONUN 27.3%, ONUN / ORDA 25.7%,
MƏN / MƏNİM 23.0%, O / ORDA 17.6%.

## Limitations

- **Vocabulary:** 40 words (AzSLD_Words_100 classes with at least 25 videos and 6 recording dates).
- **Isolated signs only:** one sign at a time, hands down between signs; no continuous signing.
- **No face or non-manual features:** facial expression, mouth and head movement are ignored.
- **Data vs use:** a studio dataset of native signers (plain wall, 1280x960) against a laptop webcam and new users.
- **Signer split:** no signer IDs, so val is not signer-independent and optimistic; the team recordings
  (non-native signers) are the only signer-independent test and are not recorded yet.
- **Confusable signs:** MƏN / MƏNİM / MƏNƏ and O / ONUN / ORDA; MƏN mostly abstains.
- **Sentences:** Claude can still phrase a sentence wrongly; the glosses are always shown under it.
- **Direction B:** keeps Azerbaijani word order, not AzSL grammar; the fallback prefix match can pick a wrong sign
  (for example "mənzil" -> MƏN) and misses most verb forms.
- **Speech:** STT and TTS quality are not evaluated yet ([docs/SPEECH.md](docs/SPEECH.md)); typed text is the main input.
- **Licence:** CC BY 4.0 per the Zenodo record; a team member still has to confirm it on the Zenodo page
  ([data/reports/LICENSE_CHECK.md](data/reports/LICENSE_CHECK.md)).

## Attribution

- Sign videos and training data: AzSLD – Azerbaijani Sign Language Dataset, N. Alishzade and J. Hasanov, CC BY 4.0,
  DOI [10.5281/zenodo.14222948](https://doi.org/10.5281/zenodo.14222948). Clips cut and re-encoded by the team.
  Paper: Alishzade, N., Hasanov, J. (2025). AzSLD: Azerbaijani sign language dataset for fingerspelling, word, and
  sentence translation with baseline software. Data in Brief 58, 111230. https://doi.org/10.1016/j.dib.2024.111230
- Landmarks: MediaPipe Tasks pose and hand landmarker (Google).
- Sentence composition and text-to-signs: Claude (Anthropic). Text-to-speech: Azure AI Speech. Speech input:
  Chrome Web Speech API.

## Status

- **Works (tested):** Direction A end to end in Chrome with the real model: browser MediaPipe, WebSocket, model,
  abstain, word buffer, compose, `/tts`. On dataset clips used as a fake camera, SABAH, BAKI, GETMƏK and İSTƏMƏK
  were recognised (0.95-0.98) and MƏN abstained; an abstain never added a word. Direction B plays clips in order and
  shows OOV words. Features + inference take 5-13 ms per segment on the CPU.
- **Not working without keys:** Claude composition and Claude text-to-signs (fallbacks used), Azure TTS (503).
- **Measured accuracy:** val only, not signer-independent: top-1 66.3%, macro 82.2%, coverage 36.8% at 96.3%
  selective accuracy. No unseen-signer accuracy.
- **Not verified:** a real webcam with a real person, the team (unseen-signer) recordings, Claude and Azure live,
  Chrome speech recognition, the projector itself, the licence by a person.
- **Next:** record the team set and run `--split team`; merge or drop the confusable pronouns; test with deaf
  native signers; add non-manual features.
