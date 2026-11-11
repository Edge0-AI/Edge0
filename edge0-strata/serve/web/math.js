// serve/web/math.js - chat math that never changes the source. Engine-independent: a renderer (math-adapter.js) is plugged in.
// Ported from PR #2972 by CC-David-CC. The text of a message stays as the model wrote it (history, copying, the Markdown
// export and the requests all use it); math is only drawn over a placeholder in the page, and a formula that cannot be drawn
// stays as its source text.
"use strict";
(function (root) {
  const escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const escaped = (s, i) => { let n = 0; while (i > 0 && s[--i] === "\\") n++; return n % 2 === 1; };

  // split(text) -> [{raw}, {source, display, raw}, {raw, incomplete: true}, ...]
  // \( \) and $x$ are inline, \[ \] and $$ $$ are display. Code (fences, `spans`), Markdown link targets and prices ($5 and
  // $10) are not math. A formula that is still being written (no closing delimiter yet) stays plain text.
  function split(text) {
    const parts = []; let start = 0, i = 0;
    const push = (end, math) => { if (start < i) parts.push({raw: text.slice(start, i)}); parts.push(math); start = i = end; };
    while (i < text.length) {
      // Code takes precedence, including incomplete fences while streaming.
      const fence = (i === 0 || text[i - 1] === "\n") && text.slice(i).match(/^ {0,3}(`{3,}|~{3,})[^\n]*\n/);
      if (fence) {
        const marker = fence[1][0], count = fence[1].length;
        const end = new RegExp("^ {0,3}" + marker + "{" + count + ",}[ \\t]*(?:\\n|$)", "m").exec(text.slice(i + fence[0].length));
        i = end ? i + fence[0].length + end.index + end[0].length : text.length; continue;
      }
      if (text[i] === "`" && !escaped(text, i)) {
        const ticks = text.slice(i).match(/^`+/)[0]; let end = text.indexOf(ticks, i + ticks.length);
        while (end >= 0 && (text[end - 1] === "`" || text[end + ticks.length] === "`")) end = text.indexOf(ticks, end + ticks.length);
        i = end < 0 ? text.length : end + ticks.length; continue;
      }
      if (escaped(text, i)) { i++; continue; }
      // Never inject a math element into a Markdown link's href attribute.
      if (text.startsWith("](http", i)) {
        const end = text.indexOf(")", i + 2);
        i = end < 0 ? text.length : end + 1; continue;
      }
      let open, close, display;
      if (text.startsWith("\\(", i)) { open = "\\("; close = "\\)"; display = false; }
      else if (text.startsWith("\\[", i)) { open = "\\["; close = "\\]"; display = true; }
      else if (text.startsWith("$$", i)) { open = close = "$$"; display = true; }
      else if (text[i] === "$" && !/\s/.test(text[i + 1] || " ") && text[i - 1] !== "$") { open = close = "$"; display = false; }
      if (!open) { i++; continue; }
      let end = text.indexOf(close, i + open.length);
      while (end >= 0 && (escaped(text, end) || (close === "$" && (/\s/.test(text[end - 1]) || /[\d$]/.test(text[end + 1] || ""))))) end = text.indexOf(close, end + close.length);
      const source = end >= 0 ? text.slice(i + open.length, end) : text.slice(i + open.length);
      // Numeric prices are text, even when separated by a second dollar amount.
      if (open === "$" && (/\n/.test(source) || /\$/.test(source) ||
          /^\d[\d,.]*(?:\s*[A-Za-z]|\s*$)/.test(source))) { i++; continue; }
      if (end < 0) { push(text.length, {raw: text.slice(i), incomplete: true}); break; }
      push(end + close.length, {source, display, raw: text.slice(i, end + close.length)});
    }
    if (start < text.length) parts.push({raw: text.slice(start)});
    return parts;
  }

  // markdown(text, format): format() is the page's Markdown, run on the text with every formula swapped for a placeholder
  // word, then each word becomes <span class="chat-math" data-math="..."> holding the escaped source until it is drawn.
  function markdown(text, format) {
    let prefix = "STRATAMATHPLACEHOLDER"; while (text.includes(prefix)) prefix += "X";
    const slots = [];
    const protectedText = split(text).map((p) => {
      if (p.source === undefined && !p.incomplete) return p.raw;
      const key = prefix + slots.length + "END";
      slots.push(p); return key;
    }).join("");
    let html = format(protectedText);
    slots.forEach((p, i) => {
      const body = p.incomplete ? escape(p.raw) : `<span class="chat-math${p.display ? " chat-math--display" : ""}" data-math="${escape(encodeURIComponent(p.source))}" data-display="${p.display ? "1" : "0"}">${escape(p.raw)}</span>`;
      html = html.replaceAll(prefix + i + "END", () => body);
    });
    return html;
  }

  // drawing: one formula at a time (an engine may keep state between formulas), 256 results cached, a 10 s deadline for the
  // engine to start and 15 s for one formula. A formula over 16 KiB is not drawn. The adapter returns an element that is not
  // in the page; the core clones it into the placeholder if that is still on the screen.
  const cache = new Map(), done = new Map();
  let initialization = null, queue = Promise.resolve(), epoch = 0;
  function reset() { cache.clear(); done.clear(); initialization = null; epoch++; }
  const keyOf = (source, display) => JSON.stringify([source, display]);
  async function result(source, display) {
    if (new TextEncoder().encode(source).length > 16384) return {ok: false, reason: "limit"};
    const key = keyOf(source, display);
    if (cache.has(key)) { const hit = cache.get(key); cache.delete(key); cache.set(key, hit); return hit; }
    if (cache.size >= 256) { const old = cache.keys().next().value; cache.delete(old); done.delete(old); }
    const current = epoch; let work;
    const task = queue.then(async () => {
      if (epoch !== current || cache.get(key) !== task) return {ok: false, reason: "unavailable"};
      try {
        const adapter = root.StrataMathAdapter;
        if (!adapter) return {ok: false, reason: "unavailable"};
        initialization ||= Promise.race([adapter.initialize(), new Promise((_, reject) =>
          setTimeout(() => reject(new Error("Math initialization timed out")), 10000))]);
        await initialization;
        const signal = AbortSignal.timeout(15000);
        let onAbort;
        const deadline = new Promise((resolve) => {
          onAbort = () => resolve({ok: false, reason: "limit"});
          signal.addEventListener("abort", onAbort, {once: true});
        });
        work = Promise.resolve(adapter.render(source, {display, signal}));
        try { return await Promise.race([work, deadline]); }
        finally { signal.removeEventListener("abort", onAbort); }
      } catch (_) { return {ok: false, reason: "unavailable"}; }
    });
    // A browser engine may ignore cancellation. Return source at the deadline,
    // but do not overlap conversions that mutate the engine's macro state.
    queue = task.then(() => work, () => work).then(() => undefined, () => undefined);
    cache.set(key, task);
    task.then((r) => { if (cache.get(key) === task) done.set(key, r); });
    return task;
  }

  function place(slot, r, tr) {
    if (r.ok) slot.replaceChildren(r.element.cloneNode(true));
    else { slot.dataset.mathError = r.reason; slot.title = tr ? tr("math.source", "Math source ({reason})", {reason: r.reason}) : "Math source (" + r.reason + ")"; }
  }
  // hydrate(container): draw the placeholders in a container. A formula drawn before is placed at once, so a streaming
  // answer that is rebuilt on every frame does not flicker back to source text.
  function hydrate(container) {
    const generation = container._mathGeneration = (container._mathGeneration || 0) + 1;
    for (const slot of container.querySelectorAll("[data-math]")) {
      const source = decodeURIComponent(slot.dataset.math), display = slot.dataset.display === "1";
      const known = done.get(keyOf(source, display));
      if (known && known.ok) { place(slot, known, root.tr); continue; }
      result(source, display).then((r) => {
        if (generation !== container._mathGeneration || !container.contains(slot)) return;
        place(slot, r, root.tr);
      });
    }
  }
  function script(url) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script"); s.src = url;
      const timer = setTimeout(() => { s.remove(); reject(new Error("Math script timed out")); }, 10000);
      s.onload = () => { clearTimeout(timer); resolve(); };
      s.onerror = () => { clearTimeout(timer); s.remove(); reject(new Error("Math script unavailable")); };
      document.head.append(s);
    });
  }
  const api = {split, markdown, hydrate, reset, script, result};
  root.StrataMath = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
