// Owner: D (Direction B/Speech/Eval). Text or speech -> POST /text-to-signs -> clips back to back.
// One <video> plays the signs in order with the gloss as subtitle; words without a sign are red chips.
const $ = (id) => document.getElementById(id);
const video = $("video");

let playlist = []; // signs that have a clip, in order: {chip, clip, gloss}
let current = -1;

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

function render(sequence) {
  $("chips").replaceChildren();
  playlist = [];
  for (const item of sequence) {
    if (item.kind === "oov") addChip(`işarə yoxdur: ${item.word}`, "oov");
    else if (!item.clip) addChip(`${item.gloss} (klip yoxdur)`, "missing");
    else playlist.push({ chip: addChip(item.gloss, ""), clip: item.clip, gloss: item.gloss });
  }
}

function setSpeed() {
  // A new src resets playbackRate to defaultPlaybackRate, so set both.
  video.defaultPlaybackRate = video.playbackRate = Number($("speed").value);
}

function playAt(i) {
  playlist[current]?.chip.classList.remove("current");
  current = i;
  if (i >= playlist.length) {
    $("subtitle").textContent = "";
    setStatus(playlist.length ? "Bitdi" : "Göstəriləcək işarə yoxdur");
    return;
  }
  const { chip, clip, gloss } = playlist[i];
  chip.classList.add("current");
  $("subtitle").textContent = gloss;
  video.src = clip;
  setSpeed();
  video.play().catch((err) => {
    if (err.name === "NotAllowedError") setStatus("Oynatmaq üçün Təkrar düyməsini basın", "bad");
  });
}

video.addEventListener("ended", () => playAt(current + 1));
video.addEventListener("error", () => {
  playlist[current]?.chip.classList.add("missing");
  playAt(current + 1); // a clip that does not load is skipped, never replaced
});

async function show() {
  const text = $("text").value.trim();
  if (!text) return setStatus("Mətn yazın", "bad");
  $("show").disabled = true;
  setStatus("Hazırlanır…");
  try {
    const res = await fetch("/text-to-signs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { sequence, source } = await res.json();
    render(sequence);
    $("source").textContent = source === "llm"
      ? "Sözlər lüğətə dil modeli (LLM) ilə uyğunlaşdırılıb; yoxlanmış tərcümə deyil."
      : "Sadə uyğunlaşdırma (LLM əlçatan deyil): yalnız sözün əvvəli lüğətlə tutuşdurulur.";
    $("replay").disabled = !playlist.length;
    setStatus("");
    playAt(0);
  } catch (err) {
    setStatus(`Xəta: ${err.message}`, "bad");
  } finally {
    $("show").disabled = false;
  }
}

$("show").onclick = show;
$("replay").onclick = () => playlist.length && playAt(0);
$("speed").onchange = setSpeed;

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
