# sign-assistant CONTRACT (v2)
Change this file only after telling the whole team.
v2 (task P1): AzSLD has no signer IDs, so data is split by `group`; AzSLD Words clips are used whole (no segment_offline).

## Ownership (one owner per file, so parallel work never conflicts)
| Owner | Files |
|---|---|
| A Data/ML | data/inspect_dataset.py, data/choose_vocab.py, data/index.csv, data/vocab.json, data/splits.json, data/README_DATA.md, data/reports/*, pose/extract_dataset.py, model/dataset.py, model/net.py, model/train.py, model/calibrate.py |
| B Frontend | web/index.html, web/app.js, web/style.css, scripts/get_models.sh |
| C Backend/LLM | pose/normalize.py, pose/features.py, pose/segment.py, pose/segment_config.json, model/abstain.py, model/predictor.py, api/main.py, api/routes_recognise.py, api/llm_client.py, api/compose.py, README.md |
| D Direction B/Speech/Eval | data/build_clips.py, data/clips_index.json, api/text_to_signs.py, api/speech.py, web/signs.html, web/signs.js, web/stt_test.html, eval/*, scripts/speech_smoke.py, docs/* |
Tests: tests/test_<module>.py belongs to the owner of <module>. A new file belongs to whoever creates it (write the owner in its docstring/header).

## Layout
sign-assistant/
  api/      main.py routes_recognise.py llm_client.py compose.py text_to_signs.py speech.py
  pose/     normalize.py features.py segment.py segment_config.json extract_dataset.py
  model/    dataset.py net.py train.py calibrate.py abstain.py predictor.py artifacts/
  eval/     evaluate.py REPORT.md figures/ test_runs.log
  data/     inspect_dataset.py choose_vocab.py build_clips.py vocab.json splits.json index.csv clips_index.json README_DATA.md reports/
            raw/ landmarks/ clips/ team_recordings/        (all gitignored)
  web/      index.html app.js style.css signs.html signs.js stt_test.html models/ (models/ gitignored)
  scripts/  get_models.sh speech_smoke.py
  docs/     SPEECH.md LLM_CHECK.md
  tests/
Python 3.11. Run every command from sign-assistant/ (python -m pose.extract_dataset ...).

## data/index.csv (written by data/inspect_dataset.py)
video_id,video_path,dataset_label,signer_id,camera,n_frames,fps,group
camera is one of front, side, unknown. signer_id is empty if unknown.
group is the split unit: the signer_id when known (team recordings: team_<signer_id>); for AzSLD Words clips
rec_<YYYY-MM-DD>, the date of the Sentences recording the clip was cut from (a proxy, not a signer); empty if unknown.

## data/vocab.json
[{"id": "hekim", "gloss": "HƏKİM", "az": "həkim", "dataset_label": "<exact class name in the dataset>"}]
- id: ascii [a-z0-9_]; used for model classes, file names and URLs. Class index = position in this list.
- Copy gloss/az text as written. Never build it with .upper()/.lower() (Azerbaijani i/İ and ı/I).

## data/splits.json
{"seed": 13, "train": [groups], "val": [groups], "test": [groups], "note": "..."}
Split by group, never by video. AzSLD groups go to train/val; test is the team recordings (team_* groups), the only
signer-independent test. Videos with an empty group are train-only. The test split is read only by `python -m eval.evaluate --split test --final`.

## Frame format (browser -> server; extraction produces the same shape)
Header, once per session: {"type": "start", "w": <video width px>, "h": <video height px>}
Frame: {"type": "frame", "t": <ms float>, "pose": [[x, y, z, vis] x 33] or null, "hands": [[[x, y, z] x 21], ...]}   (0-2 hands, unordered)
- MediaPipe Tasks image-normalised coordinates of the UNMIRRORED frame (the preview may be mirrored with CSS only).
- Models: pose_landmarker_lite.task and hand_landmarker.task (numHands 2). The browser and Python use the same files.

## Arrays (pose/normalize.py)
frames_to_arrays(frames) -> pose (T,33,4) float32, hands (T,2,21,3) float32 (NaN = missing, unordered), t (T,) ms
Landmark files: data/landmarks/<video_id>.npz with keys pose, hands, t, w, h (same shapes).
assign_hands(pose, hands) -> left (T,21,3), right (T,21,3): each detected hand goes to the nearer pose wrist (15 = signer's left, 16 = signer's right). Never use MediaPipe handedness labels.

## Normalisation
1. x *= w / h (aspect fix) for pose and hands.
2. origin = midpoint(pose[11], pose[12]); scale = distance(pose[11], pose[12]) using x, y.
3. A frame is invalid if a shoulder is NaN or its visibility < 0.5.
4. body = (p - origin) / scale. y points down.

## Features, FEATURE_VERSION = 1 (pose/features.py)
arrays_to_features(pose, hands, t, w, h, T=32) -> (X float32 (32, 192), info dict)
192 values per frame:
- pose indices [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]: body x, y -> 22
- left hand then right hand, each: 21 points body x, y (42) + 21 points hand-local x, y = (p - wrist) / distance(wrist, point 9) (42) -> 84 each, 168 total
- left_present, right_present -> 2
A missing hand gives zeros and present = 0. Invalid frames are dropped, then the sequence is resampled to T = 32 by linear interpolation over t.
info = {"n_frames", "valid_ratio", "hand_ratio", "duration_ms"}

## Segmentation (pose/segment.py, parameters in pose/segment_config.json)
A frame is "up" if any present hand has body-coordinate wrist y < raise_y (default 1.0).
Start after start_frames (3) consecutive up frames. End after end_frames (6) consecutive not-up frames.
Valid if min_ms (300) <= duration <= max_ms (4000); otherwise status too_short / too_long.
class Segmenter(w, h, cfg).push(frame_dict) -> None | {"event": "start"} | {"event": "end", "frames": [...], "status": "ok" | "too_short" | "too_long"}
segment_offline(pose, hands, t, w, h, cfg) -> (start_idx, end_idx) or None (first start to last end)
Team recordings are cut with segment_offline so they match live input. AzSLD Words clips are already cut to the
sign (often under 1 s, no rest pose) and are used whole.

## Model artifacts (model/artifacts/)
model.pt (state_dict) and config.json:
{"feature_version": 1, "T": 32, "n_features": 192, "arch": "gru" | "transformer", "classes": [ids], "temperature": 1.0, "tau": 0.7, "margin": 0.15, "min_valid_ratio": 0.6, "min_hand_ratio": 0.5}
model/net.py: build_model(config) -> torch.nn.Module; forward (B, 32, 192) -> logits (B, C)
model/predictor.py: Predictor(dir).probs(X) -> np.ndarray (C,) = softmax(logits / temperature)
model/abstain.py: decide(probs, info, cfg, vocab) -> Result (pure function, no torch)

## Result (API, UI and eval all use this)
{"status": "ok", "id": "hekim", "gloss": "HƏKİM", "confidence": 0.93}
{"status": "abstain", "reason": "low_confidence" | "small_margin" | "too_short" | "too_long" | "no_hands" | "invalid_pose" | "model_not_loaded", "message": "Əmin deyiləm, zəhmət olmasa təkrar edin."}

## API
WS   /ws/recognise      in: header, then frames. out: {"type": "state", "signing": bool} and {"type": "result", ...Result}
POST /recognise         {"w", "h", "frames": [...], "segment": true} -> Result
GET  /vocab             -> vocab.json
GET  /model-info        -> arch, number of classes, tau, margin, temperature
POST /compose-sentence  {"ids": [...]} -> {"sentence": str, "ok": bool, "assumptions": [str], "glosses": [...], "source": "llm" | "fallback"}
POST /text-to-signs     {"text": str} -> {"sequence": [{"kind": "sign", "id", "gloss", "clip": "/clips/<id>.mp4"} | {"kind": "oov", "word"}], "source": "llm" | "fallback"}
POST /tts               {"text": str} -> audio/mpeg; 503 if not configured
Static: / -> web/, /clips -> data/clips/

## LLM
Official `anthropic` Python SDK. Model from env CLAUDE_MODEL (default claude-opus-5-5), effort low, structured outputs (JSON schema), timeout 10 s.
Free alternative (v3, 2026-10-09): if ANTHROPIC_API_KEY is empty and GROQ_API_KEY is set, api/llm_client.structured_call uses Groq's
OpenAI-compatible chat completions with a strict JSON schema, model from GROQ_MODEL (default openai/gpt-oss-120b), reasoning effort low,
timeout 10 s, 1 retry. Callers do not change; "source" stays "llm".
Any error, refusal, timeout or invalid output -> deterministic fallback. Never a guess.
