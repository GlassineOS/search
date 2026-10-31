// SPDX-License-Identifier: GPL-2.0-or-later
// Semantic search in the site's search box. Loaded by a few inline lines on
// each page at the first focus on (or key in) the box, never before.
//
// The theme's own keyword results show at once. When the worker has loaded,
// its fused (keyword + meaning) list is shown in their place: our own list
// inside the theme's result panel, the theme's list hidden while ours is up.
// Hooks: [data-md-component="search-query"] (the input) and
// [data-md-component="search-result"] (the panel). Nothing else is assumed.
(function () {
  "use strict";
  const script = document.currentScript;
  const base = new URL(script.getAttribute("data-base"), document.baseURI);
  const siteRoot = new URL("../", base);
  const input = document.querySelector('[data-md-component="search-query"]');
  const panel = document.querySelector('[data-md-component="search-result"]');
  if (!input || !panel || !window.Worker) return;

  const box = document.createElement("div");
  box.className = "gds-results";
  box.hidden = true;
  box.innerHTML = '<div class="md-search-result__meta gds-meta"></div>' +
    '<ol class="md-search-result__list gds-list" role="presentation"></ol>';
  panel.appendChild(box);
  const meta = box.firstChild, list = box.lastChild;

  let worker = null, ready = false, seq = 0, active = false;

  const current = () => input.value.trim();

  function setActive(on) {
    active = on;
    panel.classList.toggle("gds-on", on);
    box.hidden = !on;
  }

  function send() {
    const q = current();
    if (!q) { setActive(false); return; }
    if (worker) worker.postMessage({ type: "query", seq: ++seq, q, k: 20 });
  }

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  function snippet(text, q) {
    const words = q.toLowerCase().split(/[^\p{L}\p{N}_-]+/u).filter((w) => w.length > 2);
    const low = text.toLowerCase();
    let at = -1;
    for (const w of words) { const i = low.indexOf(w); if (i >= 0 && (at < 0 || i < at)) at = i; }
    let start = at > 80 ? text.lastIndexOf(" ", at - 60) + 1 : 0;
    let s = text.slice(start, start + 240);
    if (start + 240 < text.length) s = s.replace(/\s+\S*$/, "") + " …";
    if (start > 0) s = "… " + s;
    let html = esc(s);
    for (const w of words) {
      html = html.replace(new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "giu"), "<mark>$1</mark>");
    }
    return html;
  }

  function render(msg) {
    if (msg.q !== current()) return;  // a stale answer
    if (!msg.results.length) { setActive(false); return; }
    meta.textContent = msg.results.length + " results by keywords and meaning";
    list.innerHTML = msg.results.map((r) => {
      const href = new URL(r.url, siteRoot).href;
      const crumb = r.heading !== r.title ? '<p class="gds-crumb">' + esc(r.title) + "</p>" : "";
      return '<li class="md-search-result__item"><a class="md-search-result__link" tabindex="-1" href="' +
        esc(href) + '"><article class="md-search-result__article md-typeset">' + crumb +
        "<h2>" + esc(r.heading) + "</h2><p>" + snippet(r.text, msg.q) + "</p></article></a></li>";
    }).join("");
    setActive(true);
  }

  async function start() {
    const res = await fetch(new URL("manifest.json", base), { cache: "no-cache" });
    if (!res.ok) throw new Error("manifest: HTTP " + res.status);
    const manifest = await res.json();
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = new URL(manifest.web["ui.css"], base).href;
    document.head.appendChild(css);
    worker = new Worker(new URL(manifest.web["worker.js"], base));
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") { ready = true; send(); }
      else if (m.type === "results") render(m);
      else if (m.type === "error") { console.warn("semantic search:", m.message); worker.terminate(); worker = null; }
    };
    worker.postMessage({ type: "init", base: base.href, manifest });
  }

  input.addEventListener("input", () => { if (!current()) setActive(false); else if (ready) send(); });

  // The theme's Enter opens the best of ITS results, its arrows walk its own
  // (now hidden) list: while our list is shown, handle those keys first.
  window.addEventListener("keydown", (e) => {
    if (!active) return;
    const links = Array.from(list.querySelectorAll("a"));
    const el = document.activeElement;
    if (el !== input && !links.includes(el)) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const all = [input, ...links];
      const i = Math.max(0, all.indexOf(el));
      all[(i + (e.key === "ArrowDown" ? 1 : all.length - 1)) % all.length].focus();
    } else if (e.key === "Enter" && el === input) {
      if (links[0]) links[0].click();
    } else {
      return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);

  start().catch((err) => console.warn("semantic search:", err.message || err));
})();
