#!/usr/bin/env bash
# Download pose_landmarker_lite.task and hand_landmarker.task into web/models/.
# Owner: B (Frontend). Run from sign-assistant/: bash scripts/get_models.sh
# Official MediaPipe model URLs, pinned to model version 1. The browser and Python use these same files.
set -euo pipefail

BASE=https://storage.googleapis.com/mediapipe-models
cd "$(dirname "$0")/.."
mkdir -p web/models

for path in pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task \
            hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task; do
  out="web/models/$(basename "$path")"
  if [ -s "$out" ]; then
    echo "exists: $out"
    continue
  fi
  curl -fL --retry 3 -o "$out.part" "$BASE/$path"
  mv "$out.part" "$out"
  echo "downloaded: $out ($(wc -c < "$out") bytes)"
done
