// Owner: D (Direction B/Speech/Eval). Text or speech -> POST /text-to-signs -> clips back to back.
// One <video> plays the signs in order with the gloss as subtitle; words without a sign are red chips.
const $ = (id) => document.getElementById(id);
const video = $("video");

let playlist = []; // signs that have a clip, in order: {chip, clip, gloss}
let current = -1;
let playbackVersion = 0;
let playbackActive = false;
let requestVersion = 0;
let requestController = null;

function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = `status ${cls}`;
}

function addChip(text, cls) {
  const li = document.createElement("li");
  li.className = `chip ${cls}`;
  li.textContent = text;
  $("chips").append(li);
  return li;
}

function stopPlayback() {
  playbackVersion += 1;
  playbackActive = false;
  video.pause();
  video.removeAttribute("src");
  video.load();
  playlist[current]?.chip.classList.remove("current");
  current = -1;
  $("subtitle").textContent = "";
}

function render(sequence) {
  stopPlayback();
  $("chips").replaceChildren();
  playlist = [];
  const unknown = [];
  const missing = [];
  for (const item of sequence) {
    if (item.kind === "oov") {
      addChip(`işarə yoxdur: ${item.word}`, "oov");
      unknown.push(item.word);
    } else if (!item.clip) {
      addChip(`${item.gloss} (klip yoxdur)`, "missing");
      missing.push(item.gloss);
    } else playlist.push({ chip: addChip(item.gloss, ""), clip: item.clip, gloss: item.gloss });
  }
  $("coverage").textContent = [
    unknown.length ? `Lüğətdə olmayan sözlər: ${[...new Set(unknown)].join(", ")}.` : "",
    missing.length ? `Videoklipi olmayan işarələr: ${[...new Set(missing)].join(", ")}.` : "",
  ].filter(Boolean).join(" ");
}

function setSpeed() {
  // A new src resets playbackRate to defaultPlaybackRate, so set both.
  video.defaultPlaybackRate = video.playbackRate = Number($("speed").value);
}

function playAt(i) {
  const version = ++playbackVersion;
  playlist[current]?.chip.classList.remove("current");
  current = i;
  if (i >= playlist.length) {
    playbackActive = false;
    $("subtitle").textContent = "";
    setStatus(playlist.length ? "Bitdi" : "Göstəriləcək işarə yoxdur");
    return;
  }
  playbackActive = true;
  const { chip, clip, gloss } = playlist[i];
  chip.classList.add("current");
  $("subtitle").textContent = gloss;
  video.src = clip;
  setSpeed();
  video.play().catch((err) => {
    if (version !== playbackVersion) return;
    if (err.name === "NotAllowedError") setStatus("Oynatmaq üçün Təkrar düyməsini basın", "bad");
  });
}

video.addEventListener("ended", () => {
  if (playbackActive && video.ended) playAt(current + 1);
});
video.addEventListener("error", () => {
  if (!playbackActive || !video.error) return;
  playlist[current]?.chip.classList.add("missing");
  playAt(current + 1); // a clip that does not load is skipped, never replaced
});

async function show() {
  const version = ++requestVersion;
  requestController?.abort();
  requestController = null;
  stopPlayback();
  playlist = [];
  $("chips").replaceChildren();
  $("coverage").textContent = "";
  $("source").textContent = "";
  $("replay").disabled = true;
  const text = $("text").value.trim();
  if (!text) {
    $("show").disabled = false;
    return setStatus("Mətn yazın", "bad");
  }
  const controller = new AbortController();
  requestController = controller;
  $("show").disabled = true;
  setStatus("Hazırlanır…");
  try {
    const res = await fetch("/text-to-signs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { sequence, source } = await res.json();
    if (version !== requestVersion) return;
    render(sequence);
    $("source").textContent = source === "llm"
      ? "Naməlum hissələr dil modeli (LLM) ilə uyğunlaşdırılıb; yoxlanmış tərcümə deyil."
      : "Sadə uyğunlaşdırma: lüğətdəki sözlər, ifadələr və dəstəklənən şəkilçili formalar tanınır.";
    $("replay").disabled = !playlist.length;
    setStatus("");
    playAt(0);
  } catch (err) {
    if (version !== requestVersion || err.name === "AbortError") return;
    setStatus(`Xəta: ${err.message}`, "bad");
  } finally {
    if (version === requestVersion) {
      requestController = null;
      $("show").disabled = false;
    }
  }
}

$("show").onclick = show;
$("replay").onclick = () => playlist.length && playAt(0);
$("speed").onchange = setSpeed;
document.querySelectorAll("[data-example]").forEach((button) => {
  button.onclick = () => {
    $("text").value = button.dataset.example;
    return show();
  };
});

async function loadVocabulary() {
  try {
    const res = await fetch("/sign-vocab");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const entries = await res.json();
    const ready = entries.filter((entry) => entry.clip);
    const unavailable = entries.length - ready.length;
    $("vocab-summary").textContent = `${ready.length} söz və ifadə üçün işarə videosu hazırdır.`
      + (unavailable ? ` ${unavailable} işarənin videoklipi yoxdur.` : "");
    $("vocab-words").textContent = ready.map((entry) => entry.az).sort((a, b) => a.localeCompare(b, "az")).join(", ")
      || "Hazır videoklip yoxdur.";
  } catch {
    $("vocab-summary").textContent = "İşarə lüğəti yüklənmədi; mətn göndərməyi sınaya bilərsiniz.";
    $("vocab-words").textContent = "Səhifəni yeniləyərək lüğəti yenidən yükləyin.";
  }
}

loadVocabulary();

// Speech input (Chrome): hidden where the browser has no speech recognition.
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) {
  $("mic").hidden = false;
  $("mic-note").hidden = false;
  $("mic").onclick = () => {
    const rec = new Recognition();
    rec.lang = "az-AZ";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (event) => {
      $("text").value = event.results[0][0].transcript.slice(0, 300);
      setStatus("Mətni yoxlayın və Göstər düyməsini basın");
    };
    rec.onerror = (event) => setStatus(`Mikrofon xətası: ${event.error}`, "bad");
    rec.onend = () => { $("mic").disabled = false; };
    $("mic").disabled = true;
    setStatus("Dinləyirəm…");
    rec.start();
  };
}
