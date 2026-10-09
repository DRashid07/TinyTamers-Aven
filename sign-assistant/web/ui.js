// Owner: B (Frontend). Page chrome shared by index.html, signs.html and stt_test.html, as in Project V7 and
// University App: light/dark theme, text size, high contrast, AZ/EN language, touch ripple and busy spinner; on the
// camera page also the big screen, copy, the conversation history and the first-visit steps. Page logic stays in
// app.js / signs.js: this file only reads the text they write and, in English, translates their messages.
(() => {
  const root = document.documentElement;
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  const systemDark = matchMedia("(prefers-color-scheme: dark)");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const store = {
    get(key) {
      try { return localStorage.getItem(key); } catch { return null; } // storage blocked
    },
    set(key, value) {
      try { localStorage.setItem(key, value); } catch { /* storage blocked: the choice lasts for this page */ }
    },
  };

  // ---------- English texts. Azerbaijani is the page's own HTML; app.js and signs.js write Azerbaijani. ----------
  const EN = {
    "title.a": "Aven — AzSL assistant",
    "title.b": "Aven — Text to sign",
    "title.c": "Aven — Speech recognition test",
    "skip": "Skip to content",
    "top.disclaimer": "<b>Assistive prototype.</b> Not a validated translation. For medical and legal decisions, ask a sign language interpreter.",
    "stt.disclaimer": "<b>Test page.</b> Speech is recognised on Google's servers (Chrome Web Speech API).",
    "stt.h1": "Speech recognition test (az-AZ)",
    "nav.label": "Direction",
    "nav.a": "Sign to text",
    "nav.b": "Text to sign",
    "theme": "Dark theme",
    "prefs": "Display",
    "prefs.text": "Text size",
    "prefs.t0": "Normal",
    "prefs.t1": "Large",
    "prefs.t2": "Extra large",
    "prefs.contrast": "High contrast",
    "close": "Close",
    "a.camera": "Camera",
    "a.privacy": "The camera image never leaves the browser: only landmark coordinates are sent to the server.",
    "a.words": "Words",
    "a.help": "How it works",
    "a.step1": "Sit so that you are visible from head to waist.",
    "a.step2": "Show one sign, then lower your hands.",
    "a.step3": "When the words are in, press «Build sentence».",
    "a.compose": "Build sentence",
    "a.undo": "Delete last word",
    "a.clear": "Clear",
    "a.sentence": "Sentence",
    "a.fallback": "The sentence could not be built automatically; the words are shown",
    "a.speak": "Read aloud",
    "a.big": "Big screen",
    "a.flip": "Turn the text upside down (for the person opposite)",
    "a.copy": "Copy",
    "a.copied": "Copied",
    "a.selected": "Selected: copy it",
    "a.history": "This conversation",
    "a.copyall": "Copy all",
    "a.sys": "System status",
    "a.credit": "Model trained on the AzSLD dataset: Alishzade & Hasanov (2025), CC BY 4.0,",
    "a.record": "Record mode (team only)",
    "a.recordlegend": "Record mode",
    "a.name": "Your name",
    "a.nameph": "e.g. rashid",
    "a.nametitle": "Only lowercase Latin letters, digits and _",
    "a.sign": "Sign",
    "a.rec": "Record",
    "a.stop": "Stop",
    "ob.1.title": "Sit facing the camera",
    "ob.1.text": "Be visible from head to waist, with light on your face.",
    "ob.2.title": "One sign at a time",
    "ob.2.text": "Show the sign, then lower your hands. The recognised word appears at the top.",
    "ob.3.title": "The image stays with you",
    "ob.3.text": "The camera image never leaves the browser: only landmark coordinates go to the server.",
    "ob.skip": "Skip",
    "ob.next": "Next",
    "ob.start": "Start",
    "b.text": "Text",
    "b.hint": "(up to 300 characters)",
    "b.ph": "Azerbaijani text, e.g. Sabah Bakıya getmək istəyirəm",
    "b.exlabel": "Example sentences",
    "b.examples": "Examples:",
    "b.mic": "Speak",
    "b.show": "Show",
    "b.micnote": "Microphone: Chrome sends the audio to Google's servers to recognise the speech.",
    "b.vocabmore": "See the words that have a sign",
    "b.signs": "Signs",
    "b.speed": "Speed",
    "b.replay": "Replay",
    "b.chips": "Sequence of signs",
    "b.rules": "Signs are shown in the word order of the text; AzSL grammar is not applied. Red: a word with no sign in the vocabulary.",
    "b.credit1": "Videos: AzSLD (Alishzade & Hasanov, 2025), CC BY 4.0,",
    "b.credit2": "Clips cut and re-encoded by the team.",
  };
  // Azerbaijani for the few labels this file writes itself
  const AZ = { "a.copy": "Kopyala", "a.copied": "Kopyalandı", "a.selected": "Seçildi: kopyalayın", "ob.next": "Növbəti", "ob.start": "Başla" };

  // Messages app.js and signs.js write (exact text, then patterns, then parts of a longer text)
  const DYN_EXACT = new Map(Object.entries({
    "Hazırlanır…": "Getting ready…",
    "Kamera açılır…": "Opening the camera…",
    "Gözləyirəm": "Waiting",
    "İşarə edilir...": "Signing...",
    "Əmin deyiləm, zəhmət olmasa təkrar edin.": "Not sure, please sign again.",
    "işarə çox qısa oldu": "the sign was too short",
    "işarə çox uzun oldu: işarələr arasında əlləri aşağı salın": "the sign was too long: lower your hands between signs",
    "əllər görünmədi": "no hands were seen",
    "çiyinlər görünmür və ya kameraya çox yaxınsınız": "shoulders not visible, or you are too close to the camera",
    "tanınmadı və ya bu söz lüğətdə yoxdur": "not recognised, or this word is not in the vocabulary",
    "iki işarə arasında qərar verə bilmədi": "could not decide between two signs",
    "model yüklənməyib": "the model is not loaded",
    "Kamerada insan görünmür": "No one is in the camera",
    "Çiyinləriniz görünmür: bir az geri oturun": "Your shoulders are not visible: sit back a little",
    "Kameraya çox yaxınsınız: geri oturun ki, başdan belə qədər görünəsiniz":
      "You are too close to the camera: sit back so you are visible from head to waist",
    "Kameradan çox uzaqsınız: bir az yaxınlaşın": "You are too far from the camera: come a little closer",
    "Server bağlantısı yoxdur": "No connection to the server",
    "Serverə qoşulub": "Connected to the server",
    "Server: modellər yüklənəndən sonra qoşulur": "Server: connects after the models load",
    "Kameraya icazə verilmədi": "Camera permission was denied",
    "Xəta: MediaPipe başlamadı (web/models/ faylları varmı? bash scripts/get_models.sh)":
      "Error: MediaPipe did not start (are the web/models/ files there? bash scripts/get_models.sh)",
    "Modellər yüklənir…": "Loading models…",
    "Modellər işə salınır…": "Starting the models…",
    "Tanınan sözlər burada görünəcək.": "Recognised words will appear here.",
    "Cümlə qurulur…": "Building the sentence…",
    "Sözlər dəyişdi: yenidən Cümlə qur.": "The words changed: press Build sentence again.",
    "Səsləndirmə qoşulmayıb.": "Reading aloud is not set up.",
    "Lüğət yüklənmədi (GET /vocab)": "The vocabulary did not load (GET /vocab)",
    "Ad: yalnız kiçik latın hərfləri, rəqəm və _": "Name: only lowercase Latin letters, digits and _",
    "Kamera hazır deyil": "The camera is not ready",
    "Yazılır…": "Recording…",
    "Heç bir kadr yazılmadı": "No frames were recorded",
    "Mətn yazın": "Type some text",
    "Bitdi": "Done",
    "Göstəriləcək işarə yoxdur": "No signs to show",
    "Oynatmaq üçün Təkrar düyməsini basın": "Press Replay to play",
    "Naməlum hissələr dil modeli (LLM) ilə uyğunlaşdırılıb; yoxlanmış tərcümə deyil.":
      "Unknown parts were matched by a language model (LLM); not a checked translation.",
    "Sadə uyğunlaşdırma: lüğətdəki sözlər, ifadələr və dəstəklənən şəkilçili formalar tanınır.":
      "Simple matching: vocabulary words, phrases and supported suffixed forms are recognised.",
    "Hazır videoklip yoxdur.": "No video clips are ready.",
    "İşarə lüğəti yüklənir…": "Loading the sign vocabulary…",
    "İşarə lüğəti yüklənmədi; mətn göndərməyi sınaya bilərsiniz.": "The sign vocabulary did not load; you can still try sending text.",
    "Səhifəni yeniləyərək lüğəti yenidən yükləyin.": "Reload the page to load the vocabulary again.",
    "Yüklənir…": "Loading…",
    "Mətni yoxlayın və Göstər düyməsini basın": "Check the text and press Show",
    "Dinləyirəm…": "Listening…",
    "Hazır": "Ready",
    "Dinləyir…": "Listening…",
    "Dayandı": "Stopped",
  }));
  const DYN_PATTERNS = [
    [/^əminlik (\d+)%$/, "confidence $1%"],
    [/^Modellər yüklənir: ([\d.]+) \/ ([\d.]+) MB \(ilk dəfə bir neçə dəqiqə çəkə bilər\)$/,
      "Loading models: $1 / $2 MB (the first time can take a few minutes)"],
    [/^Hazır \(poza: (\w+), əllər: (\w+)\)$/, "Ready (pose: $1, hands: $2)"],
    [/^MediaPipe: (.+) fps \((\d+) ms\/kadr\)$/, "MediaPipe: $1 fps ($2 ms per frame)"],
    [/^Ən çox (\d+) söz: əvvəl cümlə qurun, sonra Təmizlə\.$/, "At most $1 words: build the sentence first, then Clear."],
    [/^Cümlə qurulmadı \((.*)\)$/, "The sentence was not built ($1)"],
    [/^Tanınan işarələr: (.*)$/, "Recognised signs: $1"],
    [/^Səsləndirmə alınmadı \((.*)\)$/, "Reading aloud failed ($1)"],
    [/^Yazılır: (\d+) kadr$/, "Recording: $1 frames"],
    [/^Yükləndi: (.+) \((\d+) kadr\)$/, "Downloaded: $1 ($2 frames)"],
    [/^işarə yoxdur: (.*)$/, "no sign: $1"],
    [/^(.*) \(klip yoxdur\)$/, "$1 (no clip)"],
    [/^(\d+) söz və ifadə üçün işarə videosu hazırdır\.(?: (\d+) işarənin videoklipi yoxdur\.)?$/,
      (match, ready, missing) => `Sign videos are ready for ${ready} words and phrases.${missing ? ` ${missing} signs have no video clip.` : ""}`],
    [/^Mikrofon xətası: (.*)$/, "Microphone error: $1"],
    [/^Xəta: (.*)$/s, "Error: $1"],
  ];
  const DYN_PARTS = [
    ["Lüğətdə olmayan sözlər: ", "Words not in the vocabulary: "],
    ["Videoklipi olmayan işarələr: ", "Signs without a video clip: "],
  ];
  const DYN_IDS = ["pill", "status", "fps", "server", "buffer-note", "glosses", "speak-status", "rec-status",
    "coverage", "source", "vocab-summary", "vocab-words", "chips"];

  let lang = root.lang === "en" ? "en" : "az";
  const onLangChange = [];
  const t = (key) => (lang === "en" ? EN[key] ?? AZ[key] ?? key : AZ[key] ?? key);

  // ---------- theme ----------
  function themeColor() {
    // Phone browsers colour their toolbar with theme-color: keep it equal to the page background.
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = getComputedStyle(root).getPropertyValue("--av-bg").trim() || meta.content;
  }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    for (const button of $$("[data-theme-toggle]")) button.setAttribute("aria-pressed", String(theme === "dark"));
    themeColor();
  }

  applyTheme(root.dataset.theme === "dark" ? "dark" : "light");
  systemDark.addEventListener("change", (event) => {
    if (!store.get("aven-theme")) applyTheme(event.matches ? "dark" : "light");
  });

  for (const button of $$("[data-theme-toggle]")) {
    button.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      if (!reduceMotion.matches) {
        root.classList.add("theme-fade"); // colours change over 0.3 s instead of a hard flash
        setTimeout(() => root.classList.remove("theme-fade"), 400);
      }
      applyTheme(next);
      store.set("aven-theme", next);
    });
  }

  // ---------- text size and contrast ("Aa" panel) ----------
  function applyTextSize(size) {
    if (size === "1" || size === "2") root.dataset.text = size;
    else delete root.dataset.text;
    for (const button of $$("[data-text-size]")) button.setAttribute("aria-pressed", String(button.dataset.textSize === (size || "0")));
  }

  function applyContrast(high) {
    if (high) root.dataset.contrast = "high";
    else delete root.dataset.contrast;
    for (const box of $$("[data-contrast-toggle]")) box.checked = high;
    themeColor();
  }

  applyTextSize(root.dataset.text || "0");
  applyContrast(root.dataset.contrast === "high");
  for (const button of $$("[data-text-size]")) {
    button.addEventListener("click", () => {
      applyTextSize(button.dataset.textSize);
      store.set("aven-text", button.dataset.textSize);
    });
  }
  for (const box of $$("[data-contrast-toggle]")) {
    box.addEventListener("change", () => {
      applyContrast(box.checked);
      store.set("aven-contrast", box.checked ? "high" : "normal");
    });
  }

  // The panel closes on a click outside it and on Esc
  document.addEventListener("click", (event) => {
    for (const panel of $$("details.prefs[open]")) if (!panel.contains(event.target)) panel.open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    for (const panel of $$("details.prefs[open]")) {
      panel.open = false;
      panel.querySelector("summary").focus();
    }
  });

  // ---------- language ----------
  const originals = new Map(); // element -> its Azerbaijani text, HTML and attributes from the page
  function original(element) {
    if (!originals.has(element)) originals.set(element, { text: element.textContent, html: element.innerHTML, attrs: {} });
    return originals.get(element);
  }

  function translateStatic() {
    // original() is read before anything is written, also when the page opens in English
    for (const element of $$("[data-i18n]")) {
      const key = element.dataset.i18n;
      const az = original(element).text;
      element.textContent = lang === "en" && key in EN ? EN[key] : az;
    }
    for (const element of $$("[data-i18n-html]")) {
      const key = element.dataset.i18nHtml;
      const az = original(element).html;
      element.innerHTML = lang === "en" && key in EN ? EN[key] : az; // our own fixed strings only
    }
    for (const element of $$("[data-i18n-attr]")) {
      const saved = original(element).attrs;
      for (const pair of element.dataset.i18nAttr.split(";")) {
        const [attr, key] = pair.split(":");
        if (!(attr in saved)) saved[attr] = element.getAttribute(attr);
        const value = lang === "en" && key in EN ? EN[key] : saved[attr];
        if (value === null) element.removeAttribute(attr);
        else element.setAttribute(attr, value);
      }
    }
  }

  function translateString(text) {
    if (DYN_EXACT.has(text)) return DYN_EXACT.get(text);
    for (const [pattern, replacement] of DYN_PATTERNS) if (pattern.test(text)) return text.replace(pattern, replacement);
    let out = text;
    for (const [az, en] of DYN_PARTS) out = out.split(az).join(en);
    return out;
  }

  const azOf = new WeakMap(); // translated text node -> the Azerbaijani text app.js / signs.js wrote
  function translateTree(element) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.nodeValue;
      const trimmed = text.trim();
      if (!trimmed) continue;
      if (lang === "en") {
        const english = translateString(trimmed);
        if (english !== trimmed) {
          azOf.set(node, text);
          node.nodeValue = text.replace(trimmed, english);
        }
      } else if (azOf.has(node)) {
        node.nodeValue = azOf.get(node);
        azOf.delete(node);
      }
    }
  }

  const dynamicElements = DYN_IDS.map((id) => document.getElementById(id)).filter(Boolean);
  const dynamicObserver = new MutationObserver((records) => {
    if (lang !== "en") return;
    const hosts = new Set();
    for (const { target } of records) {
      const element = target.nodeType === Node.ELEMENT_NODE ? target : target.parentElement;
      const host = dynamicElements.find((candidate) => candidate.contains(element));
      if (host) hosts.add(host);
    }
    for (const host of hosts) translateTree(host); // English text matches nothing, so this settles at once
  });
  for (const element of dynamicElements) {
    dynamicObserver.observe(element, { childList: true, characterData: true, subtree: true });
  }

  function applyLang(next, save) {
    lang = next;
    root.lang = next;
    translateStatic();
    for (const element of dynamicElements) translateTree(element);
    for (const button of $$("[data-lang-toggle]")) {
      button.querySelector("[data-lang-label]").textContent = next === "en" ? "AZ" : "EN";
      const name = next === "en" ? "Azərbaycanca" : "English";
      button.setAttribute("aria-label", name);
      button.title = name;
    }
    for (const update of onLangChange) update();
    if (save) store.set("aven-lang", next);
  }

  for (const button of $$("[data-lang-toggle]")) {
    button.addEventListener("click", () => applyLang(lang === "en" ? "az" : "en", true));
  }

  // ---------- copy ----------
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const area = Object.assign(document.createElement("textarea"), { value: text });
      area.setAttribute("readonly", "");
      area.style.cssText = "position:fixed;opacity:0";
      document.body.append(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      return ok;
    }
  }

  function flashCopied(button, copied = true) {
    if (button.classList.contains("is-copied")) return;
    const label = button.querySelector("[data-i18n]");
    const before = label?.textContent;
    const icon = button.querySelector("use");
    button.classList.add("is-copied");
    if (label) label.textContent = t(copied ? "a.copied" : "a.selected");
    if (copied) icon?.setAttribute("href", "#i-check");
    setTimeout(() => {
      button.classList.remove("is-copied");
      if (label) label.textContent = before;
      icon?.setAttribute("href", "#i-copy");
    }, copied ? 1500 : 2500);
  }

  // When the browser refuses the clipboard, the text is selected instead, so the person can copy it by hand
  function selectText(element) {
    const range = document.createRange();
    range.selectNodeContents(element);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  }

  async function copyFrom(button, text, element) {
    if (!text) return;
    if (await copyText(text)) return flashCopied(button);
    selectText(element);
    flashCopied(button, false);
  }

  for (const button of $$("[data-copy-from]")) {
    button.addEventListener("click", () => {
      const element = document.getElementById(button.dataset.copyFrom);
      if (element) copyFrom(button, element.textContent.trim(), element);
    });
  }

  const glossesOnly = () => {
    const text = document.getElementById("glosses")?.textContent || "";
    return text.includes(": ") ? text.slice(text.indexOf(": ") + 2) : text;
  };

  // ---------- big screen: the sentence over the whole screen, to show it to the other person ----------
  const big = document.querySelector("[data-bigscreen]");
  if (big) {
    const flip = big.querySelector("[data-bigscreen-flip]");
    let wakeLock = null;
    const setFlipped = (flipped) => {
      big.classList.toggle("is-flipped", flipped);
      flip.setAttribute("aria-pressed", String(flipped));
    };
    // Runs on every way out (button, Esc, leaving full screen); it does not rely on the dialog's close event alone
    const endBig = () => {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      wakeLock?.release().catch(() => {});
      wakeLock = null;
    };
    const closeBig = () => {
      if (big.open) big.close();
      endBig();
    };
    const keepAwake = async () => { // the screen must not dim while the other person reads
      try { wakeLock = await navigator.wakeLock?.request("screen"); } catch { wakeLock = null; }
    };
    for (const button of $$("[data-bigscreen-open]")) {
      button.addEventListener("click", () => {
        const sentence = document.getElementById("sentence")?.textContent.trim();
        if (!sentence) return;
        big.querySelector("[data-bigscreen-text]").textContent = sentence;
        big.querySelector("[data-bigscreen-sub]").textContent = glossesOnly();
        setFlipped(false);
        // A <dialog> itself cannot go full screen, so the page does and the dialog covers it. Phones and tablets
        // hide the browser bars this way; iPhone Safari has no page full screen and just shows the dialog.
        root.requestFullscreen?.().catch(() => {});
        big.showModal();
        keepAwake();
      });
    }
    // For a phone lying on the table between the two people
    flip.addEventListener("click", () => setFlipped(!big.classList.contains("is-flipped")));
    big.querySelector("[data-bigscreen-close]").addEventListener("click", closeBig);
    big.addEventListener("cancel", endBig); // Esc
    big.addEventListener("close", endBig);
    document.addEventListener("visibilitychange", () => { // the browser drops the lock when the tab is hidden
      if (big.open && document.visibilityState === "visible") keepAwake();
    });
    document.addEventListener("fullscreenchange", () => {
      if (!document.fullscreenElement && big.open) closeBig(); // Esc leaves full screen and closes in one go
    });
  }

  // ---------- conversation history of this browser tab (sessionStorage), newest first ----------
  const historyCard = document.querySelector("[data-history]");
  const sentence = document.getElementById("sentence");
  if (historyCard && sentence) {
    const KEY = "aven-history";
    const list = historyCard.querySelector("[data-history-list]");
    let entries = [];
    try {
      entries = JSON.parse(sessionStorage.getItem(KEY) || "[]");
    } catch {
      entries = [];
    }
    const save = () => {
      try { sessionStorage.setItem(KEY, JSON.stringify(entries)); } catch { /* storage blocked: kept for this page only */ }
    };
    const timeOf = (at) => new Date(at).toLocaleTimeString(lang === "en" ? "en-GB" : "az-AZ", { hour: "2-digit", minute: "2-digit" });

    const render = () => {
      historyCard.hidden = !entries.length;
      list.replaceChildren(...entries.slice().reverse().map((entry) => {
        const item = document.createElement("li");
        const time = Object.assign(document.createElement("span"), { className: "history-time", textContent: timeOf(entry.at) });
        const text = Object.assign(document.createElement("span"), { className: "history-text", textContent: entry.text });
        const glosses = Object.assign(document.createElement("span"), { className: "history-glosses", textContent: entry.glosses });
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "tool-btn";
        copy.setAttribute("aria-label", t("a.copy"));
        copy.title = t("a.copy");
        copy.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-copy"/></svg>';
        copy.addEventListener("click", () => copyFrom(copy, entry.text, text));
        item.append(time, text, glosses, copy);
        return item;
      }));
    };

    // app.js writes the sentence, then the glosses, in one go; this runs right after both
    new MutationObserver(() => {
      const text = sentence.textContent.trim();
      if (!text) return;
      const last = entries[entries.length - 1];
      if (last && last.text === text && Date.now() - last.at < 3000) return;
      entries.push({ at: Date.now(), text, glosses: glossesOnly() });
      entries = entries.slice(-20);
      save();
      render();
    }).observe(sentence, { childList: true, characterData: true, subtree: true });

    historyCard.querySelector("[data-history-copy]").addEventListener("click", (event) => {
      const all = entries.map((entry) => `${timeOf(entry.at)}  ${entry.text}`).join("\n");
      copyFrom(event.currentTarget, all, list);
    });
    historyCard.querySelector("[data-history-clear]").addEventListener("click", () => {
      entries = [];
      save();
      render();
    });
    onLangChange.push(render);
    render();
  }

  // ---------- first visit: three short steps ----------
  const onboard = document.querySelector("[data-onboard]");
  if (onboard) {
    const steps = $$("[data-step]", onboard);
    const dots = $$(".onboard-dots span", onboard);
    const nextLabel = onboard.querySelector("[data-onboard-next-label]");
    let step = 0;
    const renderStep = () => {
      steps.forEach((section, index) => { section.hidden = index !== step; });
      dots.forEach((dot, index) => dot.classList.toggle("on", index === step));
      onboard.setAttribute("aria-labelledby", steps[step].querySelector("h2").id);
      nextLabel.textContent = t(step === steps.length - 1 ? "ob.start" : "ob.next");
    };
    const finish = () => {
      store.set("aven-onboarded", "1");
      if (onboard.open) onboard.close();
    };
    onboard.querySelector("[data-onboard-next]").addEventListener("click", () => {
      if (step < steps.length - 1) {
        step += 1;
        renderStep();
      } else {
        finish();
      }
    });
    onboard.querySelector("[data-onboard-skip]").addEventListener("click", finish);
    onboard.addEventListener("cancel", () => store.set("aven-onboarded", "1")); // Esc
    for (const button of $$("[data-onboard-open]")) {
      button.addEventListener("click", () => {
        step = 0;
        renderStep();
        onboard.showModal();
      });
    }
    onLangChange.push(renderStep);
    if (!store.get("aven-onboarded")) {
      renderStep();
      onboard.showModal();
    }
  }

  applyLang(lang, false); // the saved language (the head script already set <html lang>)

  // ---------- touch ripple (University App, "Toxunuş"): phones have no hover, so this answers a tap ----------
  document.addEventListener("pointerdown", (event) => {
    const target = event.target.closest(".btn, .chip-btn, .modes a, .theme-toggle, .tool-btn");
    if (!target || target.disabled || reduceMotion.matches) return;
    const box = target.getBoundingClientRect();
    const size = Math.hypot(box.width, box.height) * 2;
    let host = target.querySelector(":scope > .ripple-host");
    if (!host) {
      host = document.createElement("span");
      host.className = "ripple-host";
      host.setAttribute("aria-hidden", "true");
      target.append(host);
    }
    const ripple = document.createElement("span");
    ripple.className = "ripple";
    ripple.style.width = ripple.style.height = `${size}px`;
    ripple.style.left = `${event.clientX - box.left - size / 2}px`;
    ripple.style.top = `${event.clientY - box.top - size / 2}px`;
    ripple.addEventListener("animationend", () => ripple.remove());
    host.append(ripple);
  }, { passive: true });

  // ---------- busy spinner: a [data-busy] button the page disables right after a click waits for the server ----------
  let pressed = null;
  let pressedAt = 0;
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-busy]");
    if (!button) return;
    pressed = button;
    pressedAt = performance.now();
  }, true); // capture: runs before the page's own onclick disables the button

  const busyObserver = new MutationObserver((records) => {
    for (const { target } of records) {
      const busy = target.disabled && target === pressed && performance.now() - pressedAt < 1000;
      target.classList.toggle("is-busy", busy);
      if (busy) target.setAttribute("aria-busy", "true");
      else target.removeAttribute("aria-busy");
    }
  });
  for (const button of $$("button[data-busy]")) {
    busyObserver.observe(button, { attributes: true, attributeFilter: ["disabled"] });
  }
})();
