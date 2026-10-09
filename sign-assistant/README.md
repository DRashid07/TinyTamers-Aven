# AzSL sign assistant

Owner: C (Backend/LLM). Team rules, file ownership and data formats: [CONTRACT.md](CONTRACT.md).

## What it is

An assistive prototype for Azerbaijani Sign Language (AzSL), built at the Baku AI hackathon.
It recognises a small vocabulary of isolated signs from webcam landmarks and shows sign clips for text.
It is **not a validated translator**: it abstains when unsure, and its output must not be relied on.

Webcam frames never leave the browser; only landmark coordinates are sent to the server.

## Setup

Python 3.11. Run every command from `sign-assistant/`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in keys; never commit .env
bash scripts/get_models.sh       # MediaPipe models -> web/models/
```

## Data preparation

TODO (A). Raw videos go in `data/raw/` (gitignored). Pipeline:

```bash
python -m data.inspect_dataset   # -> data/index.csv, data/reports/
python -m data.choose_vocab      # -> data/vocab.json
python -m pose.extract_dataset   # -> data/landmarks/<video_id>.npz
python -m model.train            # -> model/artifacts/
python -m model.calibrate
```

## Run

```bash
uvicorn api.main:app
```

Open http://127.0.0.1:8000/. Health check: `GET /health`.

## Evaluation

TODO (D). `python -m eval.evaluate --split val`; the test split only with `--split test --final`.
Results go in `eval/REPORT.md`.

## Limitations

TODO.

## Attribution

Sign data: AzSLD dataset, Alishzade & Hasanov (2025), DOI [10.5281/zenodo.14222948](https://doi.org/10.5281/zenodo.14222948).
TODO: check the dataset licence before sharing anything derived from it.
