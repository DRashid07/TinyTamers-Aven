// Owner: B (Frontend). Direction A: webcam -> MediaPipe landmarks -> /ws/recognise -> word buffer
// -> /compose-sentence (only on "Cümlə qur") -> /tts, plus record mode.
// Webcam frames never leave the browser; only landmark coordinates are sent.
import {
  DrawingUtils, FilesetResolver, HandLandmarker, PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

// Same version as the Python mediapipe package (requirements.txt), WASM from the same release.
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MAX_SEND_FPS = 30;
const UPPER_BODY = 25; // pose points 0-24: face, shoulders, arms, hands, hips
const POSE_UPPER = PoseLandmarker.POSE_CONNECTIONS.filter((c) => c.start < UPPER_BODY && c.end < UPPER_BODY);
const MAX_WORDS = 12; // POST /compose-sentence accepts 1-12 ids
const HOLD_MS = 2500; // how long a result stays in the status pill
const ABSTAIN_TEXT = "Əmin deyiləm, zəhmət olmasa təkrar edin.";

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
const words = []; // the word buffer: {id, gloss} of "ok" results only, never of an abstain
let pillTimer = null;
let ttsAvailable = true; // false after /tts answers 503 (not configured)

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
    setPill("idle", "Gözləyirəm");
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
    setPill("offline", "Server bağlantısı yoxdur");
    setTimeout(connect, 2000);
  };
}

// Status pill: "Gözləyirəm" -> "İşarə edilir..." -> result, which goes back to idle after holdMs.
function setPill(state, text, detail = "", holdMs = 0) {
  clearTimeout(pillTimer);
  const pill = $("pill");
  pill.className = `pill ${state}`;
  pill.textContent = text;
  if (detail) {
    const small = document.createElement("small");
    small.textContent = detail;
    pill.append(small);
  }
  if (holdMs) pillTimer = setTimeout(() => setPill("idle", "Gözləyirəm"), holdMs);
}

function showMessage(msg) {
  if (msg.type === "state") {
    if (msg.signing) setPill("signing", "İşarə edilir...");
    return;
  }
  if (msg.type !== "result") return;
  if (msg.status === "ok" && msg.id && msg.gloss) {
    addWord(msg.id, msg.gloss);
    setPill("ok", msg.gloss, `əminlik ${Math.round(msg.confidence * 100)}%`, HOLD_MS);
  } else {
    setPill("abstain", ABSTAIN_TEXT, "", HOLD_MS); // an abstain never adds a word
  }
}

function addWord(id, gloss) {
  if (words.length >= MAX_WORDS) {
    setText("buffer-note", `Ən çox ${MAX_WORDS} söz: əvvəl cümlə qurun, sonra Təmizlə.`, "note bad");
    return;
  }
  words.push({ id, gloss });
  renderWords();
}

function renderWords() {
  $("buffer").replaceChildren(...words.map((w) => {
    const li = document.createElement("li");
    li.textContent = w.gloss;
    return li;
  }));
  $("undo").disabled = $("clear").disabled = $("compose").disabled = !words.length;
  setText("buffer-note", words.length ? "" : "Tanınan sözlər burada görünəcək.", "note");
  $("sentence-panel").hidden = true; // a composed sentence belongs to the old buffer
}

function showSentence(result) {
  $("sentence").textContent = result.sentence;
  $("glosses").textContent = `Tanınan işarələr: ${result.glosses.join(" · ")}`;
  $("assumptions").replaceChildren(...result.assumptions.map((a) => {
    const li = document.createElement("li");
    li.textContent = a;
    return li;
  }));
  $("fallback-note").hidden = result.source === "llm" && result.ok;
  $("speak").hidden = !ttsAvailable;
  setText("speak-status", "", "note");
  $("sentence-panel").hidden = false;
}

async function postJson(url, body) {
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

$("undo").onclick = () => { words.pop(); renderWords(); };
$("clear").onclick = () => { words.length = 0; renderWords(); };

$("compose").onclick = async () => {
  const ids = words.map((w) => w.id);
  if (!ids.length) return;
  $("compose").disabled = true;
  setText("buffer-note", "Cümlə qurulur…", "note");
  try {
    const res = await postJson("/compose-sentence", { ids });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const result = await res.json();
    if (ids.join() !== words.map((w) => w.id).join()) return setText("buffer-note", "Sözlər dəyişdi: yenidən Cümlə qur.", "note");
    showSentence(result);
    setText("buffer-note", "", "note");
  } catch (err) {
    setText("buffer-note", `Cümlə qurulmadı (${err.message})`, "note bad");
  } finally {
    $("compose").disabled = !words.length;
  }
};

$("speak").onclick = async () => { // audio only ever starts from this click
  $("speak").disabled = true;
  setText("speak-status", "", "note");
  try {
    const res = await postJson("/tts", { text: $("sentence").textContent });
    if (res.status === 503) {
      ttsAvailable = false;
      $("speak").hidden = true;
      return setText("speak-status", "Səsləndirmə qoşulmayıb.", "note");
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const url = URL.createObjectURL(await res.blob());
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  } catch (err) {
    setText("speak-status", `Səsləndirmə alınmadı (${err.message})`, "note bad");
  } finally {
    $("speak").disabled = false;
  }
};

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
    const text = err.name === "NotAllowedError" ? "Kameraya icazə verilmədi" : `Xəta: ${err.message}`;
    setText("status", text, "bad");
    setPill("offline", text);
    return;
  }
  connect();
  requestAnimationFrame(loop);
}

main();
