# Team recordings: the unseen-signer test set

Owner: D (Direction B/Speech/Eval), task P15. The recordings in this folder are gitignored; only this README is
tracked (added with `git add -f`).

## Why

AzSLD has **no signer IDs** (data/README_DATA.md). Our dataset split groups clips by recording date, so the same
AzSLD signers appear in train and val: **the val numbers are not signer-independent** and are optimistic for a new
person. These team recordings are the **only signer-independent test** we have, and the only check under live
conditions (our webcam, our room, the browser MediaPipe).

The recorders are **not native signers** of AzSL and not deaf. Each person learns a sign from the dataset clip just
before recording it. The numbers therefore say how the system copes with new, inexperienced signers on a webcam;
they do not say how it works for deaf AzSL users. That needs a test with native signers.

## Who and how much

- 3-4 team members, each with their own signer id: lowercase letters, digits and `_` (for example `rashid`).
- Every vocabulary word (data/vocab.json, 40 words), 3-5 times each. About 160 recordings and 25-30 minutes per person.
- Everyone records alone in front of the laptop webcam, upper body and both hands in the picture, plain background,
  good light, at the distance used in the demo.

## Consent and privacy

Record mode saves **landmark coordinates only** (pose and hand points per frame), never video or audio. Each
person agrees before recording, the files stay on the team laptops (this folder is gitignored), and anyone's files
are deleted when they ask.

## Recording one word

1. Start the API (`uvicorn api.main:app`) and open http://127.0.0.1:8000/ in Chrome.
2. Learn the sign first: open signs.html, type the word, press "Göstər" and watch the clip 2-3 times (0.75x helps).
3. Back in index.html, open "Yazma rejimi (komanda üçün)" at the bottom. Type your signer id in "Adınız" and pick the
   word in "İşarə".
4. Hands down. Press "Yaz", wait about 1 second, sign the word once, put your hands down, wait about 1 second,
   press "Dayan". One sign per recording; the evaluation cuts it out with the same segmentation as live use.
5. Repeat 3-5 times, naturally (do not copy your previous attempt). Record again if your hands left the picture.

## Files

The browser downloads `<signer>_<id>_<n>.json`. Move each file to

    data/team_recordings/<signer>/<id>_<n>.json

(drop the leading `<signer>_` from the name). Do not edit the files: each holds `{"id", "signer", "w", "h",
"frames"}` with frames in the CONTRACT.md frame format, and the evaluation reads the id and signer from inside.

## Evaluation

From `sign-assistant/`:

    python -m eval.evaluate --split team

This writes the "Team recordings" section of eval/REPORT.md and eval/figures/confusion_team.png, separate from the
dataset (val/test) numbers. Report them separately and with their limits: few clips per sign, non-native signers.
It reads the `<signer>/` subfolders recursively (P17 fix).
