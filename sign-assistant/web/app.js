// Owner: B (Frontend). Webcam -> MediaPipe landmarks -> /ws/recognise, plus record mode.
// Webcam frames never leave the browser; only landmark coordinates are sent.
import {
  DrawingUtils, FilesetResolver, HandLandmarker, PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

// Same version as the Python mediapipe package (requirements.txt), WASM from the same release.
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MAX_SEND_FPS = 30;
const UPPER_BODY = 25; // pose points 0-24: face, shoulders, arms, hands, hips
const POSE_UPPER = PoseLandmarker.POSE_CONNECTIONS.filter((c) => c.start < UPPER_BODY && c.end < UPPER_BODY);

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const drawing = new DrawingUtils(ctx);

let pose, hands, ws;
let lastVideoTime = -1;
let lastSent = -Infinity;
let fpsFrames = 0, fpsMs = 0, fpsStart = performance.now();
let recording = null; // {id, signer, w, h, frames} while "Yaz" is on
const recCounts = {};

function setText(id, text, cls) {
  $(id).textContent = text;
  if (cls !== undefined) $(id).className = cls;
}

// 5 decimals is far below one pixel and keeps messages and files small.
const round = (v) => Math.round(v * 1e5) / 1e5;

async function createLandmarker(Task, vision, options) {
  for (const delegate of ["GPU", "CPU"]) {
    try {
      const task = await Task.createFromOptions(vision, {
        ...options, baseOptions: { ...options.baseOptions, delegate },
      });
      return { task, delegate };
    } catch (err) {
      console.warn(`MediaPipe ${delegate} delegate failed`, err);
    }
  }
  throw new Error("MediaPipe başlamadı (web/models/ faylları varmı? bash scripts/get_models.sh)");
}

async function startCamera() {
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, frameRate: { ideal: 30, max: 30 } },
    audio: false,
  });
  await new Promise((resolve) => { video.onloadedmetadata = resolve; });
  // A background tab may refuse play(); autoplay starts the video once the page is visible.
  video.play().catch((err) => console.warn("video.play()", err));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return; // no new camera frame yet
  lastVideoTime = video.currentTime;

  const t = performance.now();
  const p = pose.detectForVideo(video, t); // the unmirrored video element, never the CSS preview
  const h = hands.detectForVideo(video, t);
  countFps(t, performance.now() - t);
  draw(p, h);

  const frame = {
    type: "frame",
    t,
    pose: p.landmarks.length ? p.landmarks[0].map((l) => [l.x, l.y, l.z, l.visibility ?? 0].map(round)) : null,
    hands: h.landmarks.map((hand) => hand.map((l) => [l.x, l.y, l.z].map(round))),
  };
  if (recording) {
    recording.frames.push(frame);
    if (recording.frames.length % 10 === 0) setText("rec-status", `Yazılır: ${recording.frames.length} kadr`);
  }
  send(frame, t);
}

function countFps(now, ms) {
  fpsFrames += 1;
  fpsMs += ms;
  if (now - fpsStart < 1000) return;
  const fps = Math.round((fpsFrames * 1000) / (now - fpsStart));
  setText("fps", `MediaPipe: ${fps} fps (${Math.round(fpsMs / fpsFrames)} ms/kadr)`, fps < 15 ? "bad" : "ok");
  fpsFrames = 0;
  fpsMs = 0;
  fpsStart = now;
}

function draw(p, h) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const landmarks of p.landmarks) {
    const upper = landmarks.slice(0, UPPER_BODY);
    drawing.drawConnectors(upper, POSE_UPPER, { color: "#4fc3f7", lineWidth: 3 });
    drawing.drawLandmarks(upper, { color: "#0277bd", radius: 3 });
  }
  for (const landmarks of h.landmarks) {
    drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#ffb300", lineWidth: 3 });
    drawing.drawLandmarks(landmarks, { color: "#e65100", radius: 2 });
  }
}

function send(frame, t) {
  // At most 30 per second. 3 ms of slack, because 30 fps camera frames arrive about every 33 ms with jitter.
  if (!ws || ws.readyState !== WebSocket.OPEN || t - lastSent < 1000 / MAX_SEND_FPS - 3) return;
  ws.send(JSON.stringify(frame));
  lastSent = t;
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/recognise`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "start", w: video.videoWidth, h: video.videoHeight }));
    setText("server", "Serverə qoşulub", "ok");
  };
  ws.onmessage = (event) => {
    try {
      showMessage(JSON.parse(event.data));
    } catch (err) {
      console.warn("bad server message", err);
    }
  };
  ws.onclose = () => {
    setText("server", "Server bağlantısı yoxdur", "bad");
    setTimeout(connect, 2000);
  };
}

function showMessage(msg) {
  if (msg.type === "state" && msg.signing) setText("result", "İşarə göstərilir…");
  if (msg.type === "result") {
    setText("result", msg.status === "ok"
      ? `Təxmin: ${msg.gloss} (${Math.round(msg.confidence * 100)}%)`
      : msg.message);
  }
}

async function loadVocab() {
  try {
    const res = await fetch("/vocab");
    const vocab = res.ok ? await res.json() : [];
    if (!vocab.length) throw new Error("empty vocab");
    for (const v of vocab) $("sign").add(new Option(`${v.gloss} (${v.id})`, v.id));
    $("rec-start").disabled = false;
  } catch (err) {
    console.warn("GET /vocab failed", err);
    setText("rec-status", "Lüğət yüklənmədi (GET /vocab)", "bad");
  }
}

// <signer>_<id>_<n>.json; n continues after a page reload when localStorage works.
function nextNumber(key) {
  let stored = 0;
  try { stored = Number(localStorage.getItem(`rec:${key}`)) || 0; } catch { /* storage blocked */ }
  recCounts[key] = Math.max(recCounts[key] || 0, stored) + 1;
  try { localStorage.setItem(`rec:${key}`, recCounts[key]); } catch { /* storage blocked */ }
  return recCounts[key];
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

$("rec-start").onclick = () => {
  const signer = $("signer").value.trim();
  if (!/^[a-z0-9_]+$/.test(signer)) return setText("rec-status", "Ad: yalnız kiçik latın hərfləri, rəqəm və _", "bad");
  if (!pose || !video.videoWidth) return setText("rec-status", "Kamera hazır deyil", "bad");
  recording = { id: $("sign").value, signer, w: video.videoWidth, h: video.videoHeight, frames: [] };
  $("rec-start").disabled = true;
  $("rec-stop").disabled = false;
  setText("rec-status", "Yazılır…", "");
};

$("rec-stop").onclick = () => {
  const rec = recording;
  recording = null;
  $("rec-start").disabled = false;
  $("rec-stop").disabled = true;
  if (!rec || !rec.frames.length) return setText("rec-status", "Heç bir kadr yazılmadı", "bad");
  const name = `${rec.signer}_${rec.id}_${nextNumber(`${rec.signer}_${rec.id}`)}.json`;
  download(name, JSON.stringify(rec));
  setText("rec-status", `Yükləndi: ${name} (${rec.frames.length} kadr)`, "ok");
};

async function main() {
  loadVocab();
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const p = await createLandmarker(PoseLandmarker, vision, {
      baseOptions: { modelAssetPath: "models/pose_landmarker_lite.task" }, runningMode: "VIDEO", numPoses: 1,
    });
    const h = await createLandmarker(HandLandmarker, vision, {
      baseOptions: { modelAssetPath: "models/hand_landmarker.task" }, runningMode: "VIDEO", numHands: 2,
    });
    pose = p.task;
    hands = h.task;
    setText("status", "Kamera açılır…");
    await startCamera();
    setText("status", `Hazır (poza: ${p.delegate}, əllər: ${h.delegate})`, "ok");
  } catch (err) {
    console.error(err);
    setText("status", err.name === "NotAllowedError" ? "Kameraya icazə verilmədi" : `Xəta: ${err.message}`, "bad");
    return;
  }
  connect();
  requestAnimationFrame(loop);
}

main();
