# Speech: does Azerbaijani TTS and STT work?

Owner: D (Direction B/Speech/Eval). Tools: `python -m scripts.speech_smoke` (TTS) and
`http://127.0.0.1:8000/stt_test.html` in Chrome (STT).

**Status (2026-10-09): the table is NOT complete.** No Azure key was available, the STT page could not
use a microphone on the development machine, and the scores and transcripts need people. The decision
below is therefore provisional; the steps to finish it are at the end.

## Where audio and text go

- **Azure TTS** (`az-AZ-BanuNeural`, `az-AZ-BabekNeural`): the sentence text is sent to Microsoft Azure
  (cloud) and mp3 audio comes back. No user audio is involved.
- **Chrome speech recognition** (`webkitSpeechRecognition`, `az-AZ`): Chrome sends the microphone audio to
  Google's servers (cloud). Users must be told before they speak.
- **Meta MMS-TTS** (`facebook/mms-tts-azj-script_latin`, alternative): runs locally, nothing leaves the
  machine. Licence **CC BY-NC 4.0** (non-commercial only).
- Typed text stays the main input path; it needs no speech service.

## Results

TTS quality: 1 = not understandable, 3 = understandable but clearly robotic or with wrong stress,
5 = natural. It must be judged by an Azerbaijani speaker. STT: write the transcript exactly as shown.

| # | Sentence | Azure Banu (1-5) | Azure Babek (1-5) | MMS-TTS (1-5) | STT speaker 1 | STT speaker 2 |
|---|---|---|---|---|---|---|
| 1 | Sabah həkimə getmək istəyirəm. | not run (no key) | not run (no key) | PENDING | PENDING | PENDING |
| 2 | Başım ağrıyır. | not run (no key) | not run (no key) | PENDING | PENDING | PENDING |
| 3 | Növbəm nə vaxtdır? | not run (no key) | not run (no key) | PENDING | PENDING | PENDING |
| 4 | Mənə kömək edin. | not run (no key) | not run (no key) | PENDING | PENDING | PENDING |
| 5 | Sənədlərimi gətirmişəm. | not run (no key) | not run (no key) | PENDING | PENDING | PENDING |

## What was run on 2026-10-09

- **Azure TTS**: `python -m scripts.speech_smoke --engine azure` prints
  `azure: not configured (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION empty in .env)`. The request (REST
  `https://<region>.tts.speech.microsoft.com/cognitiveservices/v1`, SSML, `audio-24khz-48kbitrate-mono-mp3`)
  has not been sent to Azure yet.
- **MMS-TTS** (the one alternative): its character set covers every letter of the 5 sentences
  (ə, ı, ğ, ş, ç, ö, ü included). It needs `transformers`, which is not in `requirements.txt`; it was
  installed outside the project venv for this test. The 145 MB model download was still running on the slow
  venue network when this file was written.
- **STT page**: `web/stt_test.html` loads, finds `webkitSpeechRecognition`, sets `lang = "az-AZ"` with interim
  results and shows errors. In the development browser the microphone is blocked, so the page showed
  `Xəta: not-allowed`. Whether Chrome actually recognises az-AZ is **not verified yet**.

## Decision (provisional)

- **STT: typed text is the main input.** Chrome `az-AZ` recognition is an optional extra, switched on only
  if both speakers get at least 4 of the 5 sentences right. Reasons: it is unverified, it sends audio to
  Google, and it needs a network.
- **TTS: Azure `az-AZ-BanuNeural`** (Babek as the second voice), if a key is available and the speaker
  gives it 4 or more. It matches the contract (`POST /tts` returns 503 when Azure is not configured), needs
  no extra Python packages and only sends the sentence text. **MMS-TTS is not used in the product**: its
  licence is non-commercial, it needs `transformers` and a 145 MB model. It is kept only as an offline
  backup for the demo if Azure is unavailable.

## How to finish this file

1. Put `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` in `sign-assistant/.env`, run
   `python -m scripts.speech_smoke`, play `scripts/out/*.mp3`, and ask an Azerbaijani speaker to score them.
2. Start `uvicorn api.main:app` on the demo laptop, open `http://127.0.0.1:8000/stt_test.html` in Chrome,
   allow the microphone, and let two different people read the 5 sentences. Copy the transcripts into the
   table.
3. Replace "provisional" above with the final choice.
