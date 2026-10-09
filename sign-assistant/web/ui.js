// Owner: B (Frontend). Page chrome shared by index.html, signs.html and stt_test.html: the light/dark theme switch,
// the touch ripple and the busy spinner, as in Project V7 and University App. Page logic stays in app.js / signs.js;
// this file never changes their elements' text or state.
(() => {
  const root = document.documentElement;
  const KEY = "aven-theme";
  const systemDark = matchMedia("(prefers-color-scheme: dark)");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

  function savedTheme() {
    try {
      return localStorage.getItem(KEY);
    } catch {
      return null; // storage blocked: the page follows the system setting
    }
  }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
      button.setAttribute("aria-pressed", String(theme === "dark"));
    }
    // Phone browsers colour their toolbar with theme-color: keep it equal to the page background.
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = getComputedStyle(root).getPropertyValue("--av-bg").trim() || meta.content;
  }

  // The inline script in <head> already set data-theme before the first paint (no flash of the wrong theme).
  applyTheme(root.dataset.theme === "dark" ? "dark" : "light");
  systemDark.addEventListener("change", (event) => {
    if (!savedTheme()) applyTheme(event.matches ? "dark" : "light");
  });

  for (const button of document.querySelectorAll("[data-theme-toggle]")) {
    button.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      if (!reduceMotion.matches) {
        root.classList.add("theme-fade"); // colours change over 0.3 s instead of a hard flash
        setTimeout(() => root.classList.remove("theme-fade"), 400);
      }
      applyTheme(next);
      try {
        localStorage.setItem(KEY, next);
      } catch {
        // storage blocked: the choice lasts until the page is closed
      }
    });
  }

  // Touch feedback (University App, "Toxunuş"): a ripple spreads from the press point. Phones have no hover,
  // so this is the answer to a tap; the button itself never shrinks or moves.
  document.addEventListener("pointerdown", (event) => {
    const target = event.target.closest(".btn, .chip-btn, .modes a, .theme-toggle");
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

  // Busy spinner: when the page script disables a [data-busy] button right after it was clicked (it is waiting for
  // /compose-sentence, /tts or /text-to-signs), the label gives way to a turning ring until the button is enabled
  // again. Disabling for any other reason (no words yet) shows nothing.
  let pressed = null;
  let pressedAt = 0;
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-busy]");
    if (!button) return;
    pressed = button;
    pressedAt = performance.now();
  }, true); // capture: runs before the page's own onclick disables the button

  const observer = new MutationObserver((records) => {
    for (const { target } of records) {
      const busy = target.disabled && target === pressed && performance.now() - pressedAt < 1000;
      target.classList.toggle("is-busy", busy);
      if (busy) target.setAttribute("aria-busy", "true");
      else target.removeAttribute("aria-busy");
    }
  });
  for (const button of document.querySelectorAll("button[data-busy]")) {
    observer.observe(button, { attributes: true, attributeFilter: ["disabled"] });
  }
})();
