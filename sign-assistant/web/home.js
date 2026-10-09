// Owner: B (Frontend). Home page (home.html): the mobile menu and the join window ("Qoşulma sorğusu").
// The form follows Project V7's sign-up window (kursologAuth.js): the Azerbaijani phone mask and check, one error line,
// a busy submit button, a hidden honeypot field and a start time signed by the server (GET /join-token). Aven has no
// accounts and no log-in, so there is no password: the form only sends a request (POST /join-request) and the team
// calls back, as Clopos's join form does. Theme, ripple and spinner styles come from ui.js and style.css.
(() => {
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];

  // ---------- mobile menu (below 1100 px the sections are a panel under the header) ----------
  const menuButton = $("[data-menu-toggle]");
  const nav = $("[data-nav]");
  const setMenu = (open) => {
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Menyunu bağla" : "Menyu");
    nav.toggleAttribute("data-open", open);
  };
  menuButton.addEventListener("click", () => setMenu(!nav.hasAttribute("data-open")));
  nav.addEventListener("click", (event) => {
    if (event.target.closest("a, button")) setMenu(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && nav.hasAttribute("data-open")) {
      setMenu(false);
      menuButton.focus();
    }
  });
  matchMedia("(min-width: 1100px)").addEventListener("change", () => setMenu(false));

  // ---------- join window ----------
  const dialog = $("[data-join]");
  const form = $("#join-form");
  const formView = $("[data-join-form-view]");
  const doneView = $("[data-join-done-view]");
  const errorLine = $("[data-join-error]");
  const submit = $("[data-join-submit]");
  const phone = $("#join-phone");
  const tokenInput = form.elements.form_token;
  const finePointer = matchMedia("(pointer: fine)");

  const PHONE_RE = /^(?:\+994|0)[1-9]\d{8}$/; // same rule as api/join.py: +994 or 0, then 9 digits
  const MESSAGES = {
    network: "Şəbəkə xətası. Yenidən cəhd edin.",
    failed: "Sorğu göndərilmədi. Bir az sonra yenidən cəhd edin.",
  };

  // V7's normalizePhone / formatPhone: "+994 50 123 45 67" or "050 123 45 67" while typing
  const normalizePhone = (value) => (value || "").replace(/[\s\-()]/g, "");

  function formatPhone(value) {
    const raw = normalizePhone(value);
    let prefix = "";
    let digits = raw;
    if (raw.startsWith("+994")) {
      prefix = "+994";
      digits = raw.slice(4);
    } else if (raw.startsWith("+")) {
      prefix = "+";
      digits = raw.slice(1);
    }
    digits = digits.replace(/\D/g, "").slice(0, prefix === "+994" ? 9 : 10);
    const parts = [];
    if (prefix === "+994") {
      if (digits.length > 0) parts.push(digits.slice(0, 2));
      if (digits.length > 2) parts.push(digits.slice(2, 5));
      if (digits.length > 5) parts.push(digits.slice(5, 7));
      if (digits.length > 7) parts.push(digits.slice(7, 9));
      return `${prefix} ${parts.join(" ")}`.trim();
    }
    if (digits.length > 0) parts.push(digits.slice(0, 3));
    if (digits.length > 3) parts.push(digits.slice(3, 6));
    if (digits.length > 6) parts.push(digits.slice(6, 8));
    if (digits.length > 8) parts.push(digits.slice(8, 10));
    return (prefix + parts.join(" ")).trim();
  }

  phone.addEventListener("input", () => {
    const atEnd = phone.selectionStart === phone.value.length;
    phone.value = formatPhone(phone.value);
    if (atEnd) phone.setSelectionRange(phone.value.length, phone.value.length);
  });

  // ---------- errors: V7's one error line, moved under the field to fix (or above the button), field marked ----------
  const fieldOf = {
    name: () => form.elements.name,
    phone: () => phone,
    organization: () => form.elements.organization,
    sector: () => form.elements.sector,
    purpose: () => $("#join-purpose"),
    consent: () => form.elements.consent,
  };

  // The field also names the error line in aria-describedby, next to its own hint
  function describe(input, withError) {
    if (!("describedby" in input.dataset)) input.dataset.describedby = input.getAttribute("aria-describedby") || "";
    const ids = [input.dataset.describedby, withError ? errorLine.id : ""].filter(Boolean).join(" ");
    if (ids) input.setAttribute("aria-describedby", ids);
    else input.removeAttribute("aria-describedby");
  }

  function clearErrors() {
    errorLine.hidden = true;
    errorLine.textContent = "";
    for (const element of $$("[aria-invalid]", form)) element.removeAttribute("aria-invalid");
    for (const input of $$("[data-describedby]", form)) describe(input, false);
  }

  function showError(message, field) {
    const element = field && fieldOf[field]?.();
    const box = element?.closest(".join-field, .check");
    if (box) box.after(errorLine);
    else submit.before(errorLine);
    errorLine.textContent = message;
    errorLine.hidden = false;
    const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!element) {
      errorLine.scrollIntoView({ block: "nearest", behavior: smooth ? "smooth" : "auto" });
      return;
    }
    element.setAttribute("aria-invalid", "true");
    const target = element.matches("fieldset") ? $("input", element) : element;
    describe(target, true);
    target.focus({ preventScroll: true });
    errorLine.scrollIntoView({ block: "center", behavior: smooth ? "smooth" : "auto" }); // the field is right above it
  }

  // A field stops being red, and its message goes, as soon as it is changed
  form.addEventListener("input", (event) => {
    const host = event.target.closest("fieldset[aria-invalid]") || event.target;
    if (!host.hasAttribute("aria-invalid")) return;
    host.removeAttribute("aria-invalid");
    errorLine.hidden = true;
  });

  // Mirrors the server's checks in api/join.py, so a normal mistake never needs a round trip
  function check(data) {
    if (data.name.length < 3 || !/\p{L}/u.test(data.name)) return ["name", "Ad və soyadınızı yazın (ən azı 3 hərf)."];
    if (!PHONE_RE.test(normalizePhone(data.phone))) {
      return ["phone", "Telefon +994 və ya 0 ilə başlayan Azərbaycan nömrəsi olmalıdır, məsələn +994 50 123 45 67."];
    }
    if (!data.sector) return ["sector", "Sahəni seçin."];
    if (!data.purpose) return ["purpose", "Müraciət məqsədini seçin."];
    if (!data.consent) return ["consent", "Davam etmək üçün razılığınızı qeyd edin."];
    return null;
  }

  const collect = () => ({
    name: form.elements.name.value.trim().replace(/\s+/g, " "),
    phone: phone.value,
    organization: form.elements.organization.value.trim().replace(/\s+/g, " "),
    sector: form.elements.sector.value,
    purpose: form.elements.purpose.value,
    consent: form.elements.consent.checked,
    website: form.elements.website.value,
    form_token: tokenInput.value,
  });

  // ---------- the signed start time: fetched when the window opens ----------
  async function fetchToken() {
    try {
      const response = await fetch("/join-token", { cache: "no-store", headers: { Accept: "application/json" } });
      if (!response.ok) return false;
      tokenInput.value = (await response.json()).token || "";
      return Boolean(tokenInput.value);
    } catch {
      return false;
    }
  }

  function setBusy(busy) {
    submit.disabled = busy;
    submit.classList.toggle("is-busy", busy);
    if (busy) submit.setAttribute("aria-busy", "true");
    else submit.removeAttribute("aria-busy");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submit.disabled) return;
    clearErrors();
    const data = collect();
    const problem = check(data);
    if (problem) {
      showError(problem[1], problem[0]);
      return;
    }
    setBusy(true);
    try {
      if (!data.form_token && !(await fetchToken())) {
        showError("Server bağlantısı yoxdur. Bir az sonra yenidən cəhd edin.");
        return;
      }
      data.form_token = tokenInput.value;
      let response;
      try {
        response = await fetch("/join-request", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(data),
        });
      } catch {
        showError(MESSAGES.network);
        return;
      }
      let reply = {};
      try { reply = await response.json(); } catch { /* not JSON: the generic message below */ }
      if (response.ok && reply.ok) {
        showDone(reply.message);
        return;
      }
      if (reply.code === "token") {
        tokenInput.value = ""; // expired or from an older server start: the next try fetches a new one
        fetchToken();
      }
      showError(typeof reply.error === "string" ? reply.error : MESSAGES.failed, reply.field);
    } finally {
      setBusy(false);
    }
  });

  function showDone(message) {
    if (message) $("[data-join-done-text]").textContent = message;
    formView.hidden = true;
    doneView.hidden = false;
    $("[data-join-done-title]").focus();
  }

  function resetForm() {
    form.reset();
    clearErrors();
    tokenInput.value = "";
    formView.hidden = false;
    doneView.hidden = true;
  }

  // ---------- open and close ----------
  let opener = null;
  function openJoin(trigger) {
    if (dialog.open) return;
    opener = trigger || document.activeElement;
    if (!doneView.hidden) resetForm(); // a new request after a sent one starts empty; an unsent one keeps its text
    if (!tokenInput.value) fetchToken();
    document.documentElement.classList.add("join-open"); // V7 locked the page under the window the same way
    dialog.showModal();
    // A phone would pop its keyboard over the sheet, so only a mouse/trackpad user gets the cursor in the name field
    if (finePointer.matches && !formView.hidden) form.elements.name.focus();
  }

  dialog.addEventListener("close", () => {
    document.documentElement.classList.remove("join-open");
    if (location.hash === "#qosulma") history.replaceState(null, "", location.pathname + location.search);
    // A button in the closed menu panel cannot take focus back, so the menu button does
    const back = opener && nav.contains(opener) && !nav.hasAttribute("data-open") ? menuButton : opener;
    if (back && document.contains(back)) back.focus();
    opener = null;
  });

  for (const button of $$("[data-join-open]")) button.addEventListener("click", () => openJoin(button));
  for (const button of $$("[data-join-close]")) button.addEventListener("click", () => dialog.close());

  // A click on the dimmed backdrop closes the window, but not a text selection that ends outside it
  let downOnBackdrop = false;
  dialog.addEventListener("pointerdown", (event) => { downOnBackdrop = event.target === dialog; });
  dialog.addEventListener("click", (event) => {
    if (downOnBackdrop && event.target === dialog) dialog.close();
    downOnBackdrop = false;
  });

  // home.html#qosulma opens the window, so the form can be shared as a link
  const fromHash = () => { if (location.hash === "#qosulma") openJoin(null); };
  window.addEventListener("hashchange", fromHash);
  fromHash();
})();
