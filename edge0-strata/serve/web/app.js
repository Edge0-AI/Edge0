// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", tr("toast.copied", "Copied to clipboard"), "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
  refreshPreviews();                                  // the preview frames carry the page's colours
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  tab = ["chat", "images", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "images", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") $("img-prompt").focus();
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
let apiKey = store.get("apikey", "");
function headers(json = false) {
  const h = {};
  if (apiKey) h.Authorization = "Bearer " + apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = apiKey;
function saveApiKey(notify = false) {
  apiKey = $("api-key").value.trim();                // usable even when browser storage is blocked or full
  const saved = store.set("apikey", apiKey);
  if (notify) {
    if (saved) toast("success", "API key saved", "Kept in this browser only.");
    else toast("warn", "API key used for this page only", "Browser storage is unavailable. Enter it again after reloading.");
  }
}
$("api-key").oninput = () => saveApiKey();
$("api-key").onchange = () => saveApiKey(true);

let health = {model: "strata", images: false, max_context: 0};
// the texts that depend on what this server offers: the empty chat's line and the attach button's hint
function renderEmptyChat() {
  $("attach-btn").title = health.images ? tr("composer.attachPictures", "Attach a text file or a picture (or drop it here)")
                                        : tr("composer.attach", "Attach a text file (or drop it here)");
  $("chat-empty-sub").textContent = tr("chat.emptyModel", "{model} runs on this PC. Nothing leaves it.", {model: health.model});
}
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    renderEmptyChat();
    $("tab-btn-images").hidden = !health.image_generation;
    if (location.hash === "#images" && health.image_generation) showTab("images");
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let metricsEpoch = 0;
// After Load or Unload returns, a poll that started earlier can still say the opposite. Hold the POST's state until a later
// poll agrees, or until the first poll that started after the POST arrives. (PR #3550 by sakuumomo)
let metricsHold = null;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
function metricsAgree(live, state) {
  const s = (live && live.state) || "";
  if (state === "unloaded") return s === "unloaded";
  return s !== "" && s !== "unloaded";
}
async function poll() {
  const epoch = ++metricsEpoch;
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", tr("pill.keyNeeded", "API key needed"));
      if (!keyWarned) { keyWarned = true; toast("warn", tr("pill.keyNeeded", "API key needed"), tr("toast.keyAbout", "This server needs a key: add it under About > Settings."), 6000); }
    } else if (r.ok) {
      const m = await r.json();
      if (metricsHold && epoch <= metricsHold.after && !metricsAgree(m.live, metricsHold.state)) {
        setTimeout(poll, 1000);
        return;
      }
      if (metricsHold && (metricsAgree(m.live, metricsHold.state) || epoch > metricsHold.after)) metricsHold = null;
      lastMetrics = m;
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", tr("pill.unreachable", "Server not reachable"));
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
  // a poll re-sets the sentence every second; refit only when its text changed
  if (text !== lastPillText) { lastPillText = text; fitHeader(); }
}
let lastPillText = null;
// The pill's sentence is the first thing the header row can spare when it runs out of room.
function fitHeader() {
  const header = document.querySelector(".st-header");
  if (!header) return;
  header.classList.remove("st-header--clip");
  if (header.scrollWidth > header.clientWidth) header.classList.add("st-header--clip");
}
window.addEventListener("resize", fitHeader);

// ------------------------------------------------------------------ Load / Unload (POST /load, /unload: the header button)
// Metrics stay "unloaded" until the engine is READY, so a turn or an image job on this page has to paint Loading itself.
// modelOp covers the button's own POST, which returns before the next poll can agree.
let modelOp = null;
let imageBusy = false;
function pageBusy() { return busy != null || imageBusy; }
// "generating an image" is the job a refresh can see. imageBusy is this page's POST before that phase is published. A chat
// phase already on the metrics wins, because Stop still stops that reply.
function imageLive(live) {
  if (live && live.phase === "generating an image") return true;
  const state = live && live.state;
  if (state === "reading" || state === "generating") return false;
  return imageBusy;
}
function paintLive() { if (lastMetrics) render(lastMetrics); }
function turnBlocksUnload(live) {
  const state = (live && live.state) || "";
  return pageBusy() || state === "reading" || state === "generating" || (live && live.queued > 0);
}
function setModelButton(text, title, disabled) {
  const btn = $("unload-btn");
  btn.textContent = text; btn.title = title; btn.disabled = disabled;
  fitHeader();
}
function modelButton(live) {
  if (modelOp === "load") return setModelButton(tr("model.loading", "Loading…"), tr("model.loadingTitle", "The model is loading"), true);
  if (modelOp === "unload") return setModelButton(tr("model.unloading", "Unloading…"), tr("model.unloadingTitle", "The model is unloading"), true);
  const state = (live && live.state) || "";
  const metricsBusy = state === "reading" || state === "generating" || (live && live.queued > 0);
  const imageNow = imageLive(live);
  const unload = tr("model.unload", "Unload");
  if (state === "unloaded") {
    if (imageNow) return setModelButton(tr("model.loading", "Loading…"), tr("model.imageLoading", "An image is already loading the model"), true);
    if (pageBusy()) return setModelButton(tr("model.loading", "Loading…"), tr("model.replyLoading", "A reply is already loading the model"), true);
    return setModelButton(tr("model.load", "Load"), tr("model.loadTitle", "Load the model now. A message also loads it."), false);
  }
  if (imageNow) return setModelButton(unload, tr("model.imageBusy", "An image is generating. It finishes on its own."), true);
  if (!state) return setModelButton("…", tr("model.waiting", "Waiting for the server"), true);
  if (metricsBusy || pageBusy()) return setModelButton(unload, tr("model.stopFirst", "Stop the reply, then unload"), true);
  return setModelButton(unload, tr("model.unloadTitle", "Unload the model and free the GPU"), false);
}
$("unload-btn").onclick = async () => {
  if (modelOp) return;
  const live = (lastMetrics && lastMetrics.live) || {};
  const loading = live.state === "unloaded";
  const busyCopy = (sample, stillLoading) => stillLoading
    ? (imageLive(sample) ? tr("model.imageLoadingDot", "An image is already loading the model.") : tr("model.replyLoadingDot", "A reply is already loading the model."))
    : (imageLive(sample) ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
  if (loading ? pageBusy() : turnBlocksUnload(live)) {
    toast("warn", tr("model.busy", "Model is busy"), busyCopy(live, loading));
    return;
  }
  if (!loading && !confirm(tr("model.confirmUnload", "Unload the model and free the GPU? Load, or the next message, brings it back."))) return;
  // confirm() blocks the poll. Yield so a sample that arrived during the dialog is applied. A confirmed Unload stays an unload.
  if (!loading) {
    modelOp = "unload";
    modelButton(live);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const now = (lastMetrics && lastMetrics.live) || {};
  const loadingNow = loading;
  if (loadingNow ? pageBusy() : turnBlocksUnload(now)) {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton(now);
    toast("warn", tr("model.busy", "Model is busy"), busyCopy(now, loadingNow));
    return;
  }
  modelOp = loadingNow ? "load" : "unload";
  modelButton(now);
  if (loadingNow) toast("info", tr("model.loadingToast", "Loading the model"), tr("model.fewMinutes", "This can take a few minutes."));
  const remember = (state) => {
    if (!lastMetrics) lastMetrics = {};
    lastMetrics.live = Object.assign({}, lastMetrics.live, {state, queued: 0});
    metricsHold = {state, after: metricsEpoch};
  };
  try {
    const r = await fetch(loadingNow ? "load" : "unload", {method: "POST", headers: headers(true), body: "{}"});
    let body = {};
    try { body = await r.json(); } catch (e) { /* the connection dropped */ }
    const status = body.status || "";
    const msg = (body.error && body.error.message) ||
                (r.status === 401 ? tr("toast.keyAbout2", "API key needed: add it under About > Settings.") : tr("toast.httpAnswer", "The server answered HTTP {code}.", {code: r.status}));
    const failTitle = loadingNow ? tr("model.loadFailed", "Load failed") : tr("model.unloadFailed", "Unload failed");
    if (loadingNow) {
      if (r.ok && status === "loaded") {
        remember("idle");
        toast("success", tr("model.loaded", "Model loaded"), tr("model.gpuInUse", "The GPU is in use."));
      } else {
        toast("error", failTitle, msg);
      }
    } else if (status === "unloaded" || status === "not loaded") {
      remember("unloaded");
      toast(status === "unloaded" ? "success" : "info",
            status === "unloaded" ? tr("model.unloaded", "Model unloaded") : tr("model.alreadyUnloaded", "Already unloaded"),
            tr("model.gpuFree", "The GPU is free. Load, or the next message, brings it back."));
    } else if (r.status === 409 || status === "busy") {
      let sample = (lastMetrics && lastMetrics.live) || {};
      try {
        const mr = await fetch("metrics", {headers: headers()});
        if (mr.ok) {
          const m = await mr.json();
          if (m && m.live) { sample = m.live; lastMetrics = m; }
        }
      } catch (e) { /* the 409 body has no phase */ }
      toast("warn", tr("model.busy", "Model is busy"), imageLive(sample)
        ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
    } else if (status === "unsupported") {
      toast("error", tr("model.unsupported", "Unload is not available"), "");
    } else {
      toast("error", failTitle, msg);
    }
  } catch (e) {
    toast("error", loadingNow ? tr("model.loadFailed", "Load failed") : tr("model.unloadFailed", "Unload failed"), tr("model.noAnswer", "The server did not answer."));
  } finally {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton({});
  }
};

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill. modelOp and a local load win over "N queued": with parallel requests, queued stays up through
  // ensure_loaded(), and metrics stay "unloaded" until the engine is READY. An image POST in that window is still a load.
  const localLoad = live.state === "unloaded" && (busy != null || imageBusy);
  if (modelOp === "load") {
    setPill("idle", tr("model.loading", "Loading…"));
  } else if (modelOp === "unload") {
    setPill("idle", tr("model.unloading", "Unloading…"));
  } else if (localLoad) {
    setPill("idle", tr("model.loading", "Loading…"));
  } else if (imageLive(live)) {
    setPill("reading", tr("pill.image", "Generating image…"));
  } else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? tr("pill.readingPct", "Reading prompt · {pct}%", {pct}) : tr("pill.reading", "Reading prompt"));
  } else if (live.state === "generating") {
    setPill("generating", tr("pill.generating", "Generating · {rate} tok/s", {rate: fmt(live.tok_s, 1)}));
  } else if (live.state === "unloaded") {
    setPill("idle", tr("pill.unloaded", "Unloaded"));
  } else {
    setPill("idle", tr("pill.idle", "Idle"));
  }
  modelButton(live);
  if (!modelOp && !localLoad && live.queued > 0) setPill("queued", tr("pill.queued", "{n} queued", {n: live.queued}));
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  // A local turn keeps metrics on "unloaded" for the whole engine start.
  const imageNow = imageLive(live);
  const starting = modelOp === "load" || imageNow || (busy != null && live.state === "unloaded");
  const on = !modelOp && live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) {
    const lit = (b.dataset.s === on || b.dataset.s === live.state) && !(starting && b.dataset.s === "unloaded");
    b.classList.toggle("on", lit);
  }
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (modelOp === "load" || (busy != null && live.state === "unloaded" && !imageNow)) {
    label = "Loading";
    delete prog.dataset.tone;
    detail = "The model is loading.";
  } else if (imageNow && (live.state === "unloaded" || live.state === "reading" || live.state === "generating" || live.state === "idle")) {
    label = live.state === "unloaded" ? "Loading" : "Generating image";
    delete prog.dataset.tone;
    detail = live.state === "unloaded" ? "The model is loading." : "An image is generating. It finishes on its own.";
  } else if (modelOp === "unload") {
    label = "Unloading";
    delete prog.dataset.tone;
    detail = "Freeing the GPU.";
  } else if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (live.state === "unloaded") {
    label = "Unloaded";
    delete prog.dataset.tone;
    detail = "Load brings it back. A message does too.";
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : (st.gpu_note ? "not available on Windows AMD yet" : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"],
                   timeout: ["st-badge--queued", "Timeout"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable (NVML)")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
// An ```svg or ```html block that is complete gets a Preview button (serve/web/preview.js): the block's own code in a sandboxed
// frame, built when it is first opened and again when the theme changes. The state lives on the box (data-preview), so it
// resets if the answer is rebuilt. A page (html) can also be run WITH scripts, after a warning, one block at a time.
function pageTheme() {
  const root = document.documentElement;
  return {dark: root.dataset.theme === "dark", ink: getComputedStyle(root).getPropertyValue("--st-ink").trim()};
}
function fillPreview(fr, scripts, blank) {
  const box = fr.closest(".st-code");
  const doc = blank ? null : StrataPreview.previewDoc(fr.dataset.lang, box.querySelector("pre").textContent, scripts, pageTheme(), window);
  const nf = StrataPreview.makeFrame(doc, {scripts, title: fr.title, lang: fr.dataset.lang});
  fr.replaceWith(nf);
  box.classList.toggle("st-code--run", !!(doc && scripts));
  const rb = box.querySelector("[data-preview-scripts]");
  if (rb) {
    rb.setAttribute("aria-pressed", String(!!(doc && scripts)));
    rb.setAttribute("aria-label", doc && scripts ? tr("preview.stopScripts", "Stop scripts") : tr("preview.runScripts", "Run scripts"));
  }
}
function refreshPreviews() { for (const fr of document.querySelectorAll("iframe[data-opened]")) fillPreview(fr, fr.dataset.scripts === "1"); }
function codeBlock(lang, code, complete) {
  const ok = complete && StrataPreview.canPreview(lang || "", code);
  const kind = ok ? StrataPreview.kindOf(lang) : null;
  const label = kind === "svg" ? tr("preview.showSvg", "Preview the picture") : tr("preview.showPage", "Preview the page");
  const head = `<div class="st-code__head"><span>${esc(lang || tr("code.label", "code"))}</span><span class="st-code__btns">` +
    (kind === "page" ? `<button class="st-btn st-btn--icon" data-preview-scripts aria-label="${esc(tr("preview.runScripts", "Run scripts"))}" aria-pressed="false" title="${esc(tr("preview.runScripts", "Run scripts"))}">${icon("warning")}</button>` : "") +
    (ok ? `<button class="st-btn st-btn--icon" data-preview-toggle data-preview-label="${esc(label)}" aria-label="${esc(label)}" title="${esc(label)}" aria-pressed="false">${icon("image")}</button>` : "") +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="${esc(tr("code.copy", "Copy code"))}">${icon("copy")}</button></span></div>`;
  const prev = ok ? `<div class="st-code__preview"><iframe sandbox="" title="${esc(kind === "svg" ? tr("preview.titleSvg", "Picture preview") : tr("preview.titlePage", "Page preview"))}" data-lang="${esc(lang)}"></iframe></div>` : "";
  return `<div class="st-code">${head}<pre><code>${esc(code)}</code></pre>${prev}</div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest, false); break; }  // still streaming: no preview of a half-written block
    html += codeBlock(m[2].trim(), rest.slice(0, end.index), true);
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
// The sampling shown until the user changes it is the model card's thinking set. Untouched, the page sends NO sampling
// fields: the server picks the card's set for the request's thinking state (thinking 1.0 / 0.95 / 20, not thinking
// 0.7 / 0.8 / 20 with a presence penalty). Before 0.1.45 the page sent 0.6 itself; a stored 0.6 with the other two
// values untouched counts as "not set" too.
const DEFAULTS = {thinking: "high", temperature: 1.0, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true, math: false,
                  chatList: false, chatEdit: false, whileBusy: "wait"};
const OLD_DEFAULT_TEMPERATURE = 0.6;
function samplingUntouched(s) {
  const t = +s.temperature;
  return (t === DEFAULTS.temperature || t === OLD_DEFAULT_TEMPERATURE) && +s.top_p === DEFAULTS.top_p && +s.top_k === DEFAULTS.top_k;
}
let settings = {...DEFAULTS, ...store.get("sampling", {})};
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
let messages = store.get("chat", []);   // the open chat's messages (chats-ui.js: with the saved-chats list on, also that chat's own)
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg, cut, cutAfterTool}
let pending = null;                   // a message parked while an answer is written: {text, images, files, mode}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || tr("chat.image", "image"); wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || tr("chat.image", "image")); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `${tr("chat.you", "You")} · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="${esc(tr("chat.copyAnswer", "Copy the answer"))}" title="${esc(tr("chat.copy", "Copy"))}">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  const actions = msgActions(m, i);                 // edit / regenerate / rewind (chats-ui.js), when switched on
  if (actions) el.appendChild(actions);
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", () => tr("tool.writing", "Writing")], running: ["st-badge--generating", () => tr("tool.running", "Running")],
                    done: ["", () => tr("tool.done", "Done")], error: ["st-badge--error", () => tr("tool.error", "Error")],
                    skipped: ["st-badge--queued", () => tr("tool.skipped", "Not run")]};
function toolHtml(t, k) {
  const [cls, labelOf] = TOOL_STATE[t.state] || ["", () => t.state];
  const label = labelOf();
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">${esc(tr("tool.arguments", "Arguments"))}</div><pre class="tool-call__pre">${esc(args || tr("tool.beingWritten", "(being written)"))}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${esc(t.ok ? tr("tool.result", "Result") : tr("tool.error", "Error"))}${t.chars ? ` · ${esc(tr("tool.chars", "{n} characters", {n: fmt(t.chars)}))}` : ""}` +
              `${t.truncated ? esc(tr("tool.cut", ", cut for the model")) : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// Raster tool results are same-origin cached files; credentials stay in request headers.
const mediaBlobs = new Map();
let mediaEpoch = 0;
function clearToolMedia() {
  mediaEpoch++;
  for (const pending of mediaBlobs.values()) pending.then((url) => URL.revokeObjectURL(url), () => {});
  mediaBlobs.clear();
}
window.addEventListener("pagehide", clearToolMedia);
window.addEventListener("pageshow", (event) => { if (event.persisted) hydrateToolImages(document); });
function safeToolMedia(url) { return typeof url === "string" && /^tool-media\/[0-9a-f]{64}\.(png|jpg|webp)$/.test(url); }
function toolImages(t) {
  return (t.images || []).filter((im) => safeToolMedia(im.url)).map((im) =>
    `<figure class="tool-image"><img data-tool-image="${esc(im.url)}" alt="Tool image" loading="lazy">` +
    `<figcaption><a class="st-btn" data-tool-download="${esc(im.url)}" download="${esc(im.name || "generated-image.png")}">Download image</a></figcaption></figure>`).join("");
}
function hydrateToolImages(root) {
  root.querySelectorAll("[data-tool-image]").forEach((img) => {
    const url = img.dataset.toolImage, epoch = mediaEpoch;
    if (!safeToolMedia(url)) return;
    if (!mediaBlobs.has(url)) mediaBlobs.set(url, fetch(url, {headers: headers()}).then((r) => {
      if (!r.ok) throw new Error("Image unavailable");
      return r.blob();
    }).then((b) => URL.createObjectURL(b)).catch((e) => {if (epoch === mediaEpoch) mediaBlobs.delete(url); throw e;}));
    mediaBlobs.get(url).then((blob) => {
      if (epoch !== mediaEpoch || !img.isConnected) return;
      img.src = blob;
      const a = img.closest("figure").querySelector("[data-tool-download]");
      a.href = blob;
    }).catch(() => { img.alt = "Image unavailable; try refreshing"; });
  });
}
// the answer's text with the tool blocks where the model called them
// math (opt-in, serve/web/math.js): formulas become placeholders before the Markdown runs and are drawn after it
function answerMarkdown(text) {
  return settings.math ? StrataMath.markdown(text, markdown) : markdown(text);
}
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return answerMarkdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += answerMarkdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k) + toolImages(t);
  });
  return html + answerMarkdown(m.text.slice(pos));
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms, images: x.images || [], vision: x.vision === true,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? tr("think.now", "Thinking…") :
      m.thinkSecs != null ? tr("think.for", "Thought for {s} s", {s: fmt(m.thinkSecs, 1)}) : tr("think.done", "Thoughts");
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    hydrateToolImages(bubble);
    if (settings.math) StrataMath.hydrate(bubble);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? tr("chat.stopped", "Stopped") : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  renderSidebar();
  renderParent();
  renderUndoBar();
  scrollDown(true);
}
// Follow the answer while the reader has not left (#4014). Asking "am I within 120 px of the bottom?" on every frame pulled
// the view back after every wheel notch (a notch is about 100 px), so scrolling up during a long answer only worked by
// dragging the scrollbar. Now the reader's own input lets go (wheel up, a finger or a key moving up, the scrollbar), and a
// scroll back to the bottom follows again.
let stickToBottom = true, lastReaderInput = 0;
const FOLLOW_SLACK = 24, FAR_AWAY = 400, INPUT_WINDOW_MS = 400;
function followAfterScroll(distance, wasFollowing, msSinceInput) {
  if (distance <= FOLLOW_SLACK) return true;
  if (msSinceInput <= INPUT_WINDOW_MS || distance > FAR_AWAY) return false;
  return wasFollowing;
}
function scrollDown(force) {
  const s = $("chat-scroll");
  if (force) stickToBottom = true;
  if (stickToBottom) s.scrollTop = s.scrollHeight;
}
(function watchReader() {
  const s = $("chat-scroll");
  const touch = () => { lastReaderInput = Date.now(); };
  s.addEventListener("wheel", (e) => { touch(); if (e.deltaY < 0 && s.scrollTop > 0) stickToBottom = false; }, {passive: true});
  s.addEventListener("touchmove", touch, {passive: true});
  s.addEventListener("pointerdown", touch, {passive: true});
  s.addEventListener("keydown", (e) => {
    touch();
    if (["PageUp", "ArrowUp", "Home"].includes(e.key) && e.target === s && s.scrollTop > 0) stickToBottom = false;
  });
  s.addEventListener("scroll", () => {
    stickToBottom = followAfterScroll(s.scrollHeight - s.scrollTop - s.clientHeight, stickToBottom, Date.now() - lastReaderInput);
  }, {passive: true});
})();

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  // an svg/html block: flip between its code and its sandboxed preview. Closing the preview also turns scripts off.
  const pv = e.target.closest("[data-preview-toggle]");
  if (pv) {
    const box = pv.closest(".st-code");
    const on = box.dataset.preview === "true";
    if (on) delete box.dataset.preview; else box.dataset.preview = "true";
    pv.setAttribute("aria-pressed", String(!on));
    pv.setAttribute("aria-label", on ? pv.dataset.previewLabel : tr("preview.showCode", "Show the code"));
    const fr = box.querySelector("iframe");
    if (on) fillPreview(fr, false, true);
    else if (!fr.dataset.opened) fillPreview(fr, false);
    return;
  }
  const rs = e.target.closest("[data-preview-scripts]");
  if (rs) {
    const fr = rs.closest(".st-code").querySelector("iframe");
    const run = fr.dataset.scripts !== "1";
    if (run && !confirm(tr("preview.confirm", "Run this code with scripts?\n\nIt runs in your browser, cut off from this chat and from the network, but it can still freeze the tab, and it is code the model wrote.\n\nOnly this block, and only until you close the preview."))) return;
    fillPreview(fr, run);
    return;
  }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) {
      const pictures = t.vision ? (t.images || []).filter((im) => safeToolMedia(im.url)) : [];
      const content = pictures.length ? [{type: "text", text: t.result}, ...pictures.map((im) =>
        ({type: "image_url", image_url: {url: "strata-tool-media:" + im.url.split("/").pop()}}))] : t.result;
      out.push({role: "tool", tool_call_id: t.id, content});
    }
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function busyHint() {
  const mode = settings.whileBusy;
  return mode === "queue" ? tr("composer.hintQueue", "Enter sends when the answer is done")
       : mode === "interject" ? tr("composer.hintInterject", "Enter stops the answer here and sends now") : "";
}
// While an answer is written, Send is off ("wait", as always) or parks the message ("queue", "interject": Chat settings)
const parksWhileBusy = () => settings.whileBusy === "queue" || settings.whileBusy === "interject";
function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = (on && !parksWhileBusy()) || !!panel;
  $("input").disabled = !!panel;
  $("composer-hint").textContent = panel ? tr("composer.finishOpen", "Save or cancel what is open above to continue")
                                         : on ? busyHint() : tr("composer.newline", "Shift+Enter: new line");
  if (settings.chatEdit) syncActions();                 // the buttons follow the state: none under the answer being written
  $("s-list").disabled = $("s-edit").disabled = !!busy || !!panel;      // they change the messages on the screen
  renderUndoBar();
  renderPending();
}

// A message sent while an answer is being written (#4004). "queue": it waits and goes when the answer is done. "interject": the
// answer stops where it is (what is written stays, marked interrupted) and the message goes at once; if a tool is running, the
// tool finishes first. Only one message waits. Off by default: with "wait" the page does nothing, as before.
function park(p, own) {
  if (!parksWhileBusy()) return;
  if (pending) { toast("warn", tr("pending.one", "A message is already waiting"), tr("pending.oneText", "Take it back or wait until it is sent.")); return; }
  pending = {...p, mode: settings.whileBusy};
  if (own) { attachments = []; renderAttachments(); $("input").value = ""; autosize(); }
  if (pending.mode === "interject") requestCut();
  setBusy(!!busy);
}
function requestCut() {
  if (!busy || busy.cut) return;
  if ((busy.msg.tools || []).some((t) => t.state === "running")) { busy.cutAfterTool = true; return; }   // let the tool finish
  busy.cut = true;
  busy.controller.abort();
}
function renderPending() {
  let el = $("pending");
  if (!pending) { if (el) el.remove(); return; }
  if (!el) {
    el = node("div", "pending");
    el.id = "pending";
    el.setAttribute("role", "status");
    el.innerHTML = "<span class=\"pending__text\"></span><span class=\"pending__actions\"><button type=\"button\" class=\"st-btn st-btn--secondary\" data-pending-now></button>" +
      "<button type=\"button\" class=\"st-btn st-btn--secondary\" data-pending-back></button></span>";
    el.querySelector("[data-pending-now]").onclick = pendingNow;
    el.querySelector("[data-pending-back]").onclick = releasePending;
    $("composer").before(el);
  }
  const quote = (pending.text || (pending.files[0] || pending.images[0] || {}).name || "").replace(/\s+/g, " ").slice(0, 70);
  const text = !busy ? tr("pending.ready", "Not sent yet: “{text}”", {text: quote})
    : pending.mode === "queue" ? tr("pending.queued", "Sends when this answer is done: “{text}”", {text: quote})
    : busy.cutAfterTool ? tr("pending.afterTool", "Waiting for the tool to finish, then sending: “{text}”", {text: quote})
    : tr("pending.cutting", "Stopping the answer, then sending: “{text}”", {text: quote});
  el.querySelector(".pending__text").textContent = text;
  const now = el.querySelector("[data-pending-now]");
  now.textContent = busy ? tr("pending.sendNow", "Send now") : tr("pending.send", "Send");
  now.hidden = !!busy && pending.mode === "interject";            // already on its way
  el.querySelector("[data-pending-back]").textContent = tr("pending.takeBack", "Take back");
}
function pendingNow() {
  if (!pending) return;
  if (busy) { pending.mode = "interject"; requestCut(); setBusy(true); return; }
  const p = pending;
  pending = null;
  setBusy(false);
  send(p);
}
// the waiting message goes back into the message box (a chat switch or a new chat must not lose it)
function releasePending() {
  if (!pending) return;
  const p = pending;
  pending = null;
  if (!$("input").value.trim()) $("input").value = p.text;
  else $("input").value = `${p.text}\n\n${$("input").value}`;
  attachments = [...p.images, ...p.files, ...attachments];
  renderAttachments(); autosize();
  setBusy(!!busy);
}
// after an answer: an interrupted one sends an interjection, a finished one sends a queued message
function settlePending(cut, finished) {
  if (!pending) return;
  if ((pending.mode === "interject" && cut) || (pending.mode === "queue" && finished)) {
    const p = pending;
    pending = null;
    setBusy(false);
    send(p);
  }
}

// the message box (or a parked message, `payload`) becomes a user turn and a reply is asked for
async function send(payload) {
  const own = !payload;
  const p = payload || {text: $("input").value.trim(), images: attachments.filter((a) => a.kind !== "file"),
                        files: attachments.filter((a) => a.kind === "file")};
  if (!p.text && !p.images.length && !p.files.length) return;
  if (panel) return;                                  // something is open above (an edit): finish it first
  if (busy) { park(p, own); return; }
  clearUndo();
  messages.push({role: "user", text: p.text, images: p.images, files: p.files, time: Date.now()});
  if (own) {
    attachments = [];
    renderAttachments();
    $("input").value = "";
    autosize();
  }
  return generateReply();
}
// an answer for the messages as they are: the last one is a user turn
async function generateReply() {
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m, cut: false, cutAfterTool: false};
  setBusy(true);
  paintLive();

  const body = {model: health.model, messages: apiMessages(), stream: true,
                reasoning_effort: settings.thinking};
  if (!(settings.temperature > 0)) {
    body.temperature = 0;
  } else if (!samplingUntouched(settings)) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = tr("toast.keyNeeded", "This server needs an API key: add it under About > Settings.");
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || tr("toast.engineError", "the engine reported an error"));
        if (j.usage) usage = j.usage;
        if (j.strata_mcp) {
          onTool(m, j.strata_mcp);
          // an interjection waited for a running tool: it has its result now, so the answer can stop
          if (busy.cutAfterTool && !busy.cut && !(m.tools || []).some((t) => t.state === "running")) { busy.cut = true; controller.abort(); }
        }
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", tr("toast.requestFailed", "The request failed"), m.error, 6000); }
  }
  const cut = busy.cut;                                      // an interjection ended it: kept as written, marked interrupted
  if (cut) m.interrupted = true;
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  const ended = cut ? tr("chat.interruptedLower", "interrupted") : tr("chat.stoppedLower", "stopped");
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = tr("chat.tokens", "{n} tokens", {n: fmt(n)}) + (secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : "") + (m.stopped ? ` · ${ended}` : "") +
             (projectionLoaded() ? (settings.esp ? ` · ${tr("chat.projectionOn", "projection on")}` : ` · ${tr("chat.projectionOff", "projection off")}`) : "");
  } else if (m.stopped) {
    m.meta = cut ? tr("chat.interrupted", "Interrupted") : tr("chat.stopped", "Stopped");
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${tr("chat.toolCalls", {one: "{n} tool call", other: "{n} tool calls"}, {n: ran})}`;
  if (m.limit) m.meta = `${m.meta || ""} · ${tr("chat.toolLimit", "stopped at the limit of {n} tool rounds (mcp.max_rounds)", {n: m.limit})}`;
  busy = null;
  setBusy(false);
  paintLive();
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
  settlePending(cut, !m.stopped && !m.error);
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

// one chat, as before: New chat clears it, with an Undo (with the saved-chats list on, its own New chat button does this job)
$("new-btn").onclick = () => {
  if (busy) { stillWriting(); return; }
  if (panel) { closePanel(); }
  clearUndo();
  releasePending();
  if (!messages.length) return;
  const backup = messages;
  clearToolMedia();
  messages = [];
  saveChat();
  renderChat();
  toast("info", tr("chat.newToast", "New chat"), tr("chat.clearedToast", "The last one was cleared."), 6000, {label: tr("undo", "Undo"), run: () => {
    if (busy || messages.length) return;                         // a new question since: the cleared chat is not put back over it
    messages = backup; saveChat(); renderChat();
  }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", tr("export.nothing", "Nothing to save yet")); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", tr("file.picturesOff", "Pictures are off"), tr("file.picturesOffText", "This model was set up for text only.")); continue; }
      if (f.size > 20e6) { toast("warn", tr("file.pictureBig", "Picture too large"), tr("file.pictureBigText", "{name} is over 20 MB.", {name: f.name})); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || tr("file.pasted", "pasted image"), url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", tr("file.notText", "Not a text file"), health.images ? tr("file.notTextPictures", "{name}: attach text files (code, notes, logs, data) or pictures.", {name: f.name})
                                                              : tr("file.notTextOnly", "{name}: attach text files (code, notes, logs, data).", {name: f.name})); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", tr("file.big", "File too large"), tr("file.bigText", "{name} is over 512 KB.", {name: f.name})); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", tr("file.notText", "Not a text file"), tr("file.binaryText", "{name} looks like a binary file.", {name: f.name})); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", tr("file.remove", "Remove"));
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the chat settings drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
const checked = (id) => $(id).getAttribute("aria-checked") === "true";
const segValue = (id) => { const b = [...$(id).children].find((x) => x.getAttribute("aria-checked") === "true"); return b ? b.dataset.v : null; };
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-math").setAttribute("aria-checked", String(!!s.math));
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  $("s-list").setAttribute("aria-checked", String(!!s.chatList));
  $("s-edit").setAttribute("aria-checked", String(!!s.chatEdit));
  for (const b of $("s-busy").children) b.setAttribute("aria-checked", String(b.dataset.v === (s.whileBusy || "wait")));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking};
  if (!(+s.temperature > 0)) d.temperature = 0;
  else if (!samplingUntouched(s)) Object.assign(d, {temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? tr("drawer.greedy", "0 · greedy") : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = segValue("s-thinking");
  $("o-thinking").textContent = sel ? {none: tr("drawer.thinkNone", "answers right away"), low: tr("drawer.thinkLow", "short"),
    medium: tr("drawer.thinkMedium", "medium"), high: tr("drawer.thinkHigh", "thorough (default)")}[sel] : "";
  const busyMode = segValue("s-busy");
  $("o-busy").textContent = {wait: tr("drawer.busyWaitNote", "Send is off until the answer is done, as before."),
    queue: tr("drawer.busyQueueNote", "Your message waits and goes as soon as the answer is done."),
    interject: tr("drawer.busyInterjectNote", "The answer stops where it is (what is written stays) and your message goes at once. A tool that is running finishes first.")}[busyMode] || "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const b of $("s-busy").children) b.onclick = () => { for (const x of $("s-busy").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
for (const id of ["s-math", "s-show", "s-esp", "s-mcp", "s-share", "s-list", "s-edit"]) {
  $(id).onclick = () => $(id).setAttribute("aria-checked", String($(id).getAttribute("aria-checked") !== "true"));
}
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const prev = settings;
  // the list and the edit controls change the messages on the screen: not while an answer is written or something is open
  const locked = !!busy || !!panel;
  settings = {thinking: segValue("s-thinking") || "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              math: checked("s-math"), show: checked("s-show"), esp: checked("s-esp"), mcp: checked("s-mcp"),
              chatList: locked ? !!prev.chatList : checked("s-list"), chatEdit: locked ? !!prev.chatEdit : checked("s-edit"),
              whileBusy: segValue("s-busy") || "wait"};
  store.set("sampling", settings);
  StrataMath.reset();                                  // math on or off: every answer is drawn again
  if (settings.chatList !== !!prev.chatList) await setChatList(settings.chatList, true);
  if (settings.chatEdit !== !!prev.chatEdit) { clearUndo(); syncActions(); }
  if (settings.whileBusy === "wait" && pending) releasePending();    // "wait" parks nothing: a waiting message goes back to the box
  setBusy(!!busy);
  for (const el of $("chat").querySelectorAll(".st-msg--assistant")) {
    const m = messages[+el.dataset.i];
    updateAssistant(el, m, !!busy && busy.msg === m);  // the element send() holds stays the same while an answer streams
  }
  const share = checked("s-share");
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", tr("drawer.saved", "Chat settings saved"), share ? tr("drawer.savedShared", "Other apps (omp, API clients) use these settings from their next request.")
                                                                         : tr("drawer.savedOwn", "Other apps use their own settings again."));
    } catch (e) {
      toast("error", tr("drawer.savedHere", "Saved here, but not for other apps"), e.message, 6000);
    }
    return;
  }
  toast("success", tr("drawer.saved", "Chat settings saved"), settings.temperature === 0 ? tr("drawer.savedGreedy", "Greedy: the same question gives the same answer.") : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ language (serve/web/i18n.js)
// The user's own choice, kept in this browser: English until they pick another. /?lang=de shows one for this visit only.
const urlLang = new URLSearchParams(location.search).get("lang");
$("lang").innerHTML = StrataI18n.LANGUAGES.map(([code, name]) => `<option value="${esc(code)}">${esc(name)}</option>`).join("");
function showLanguage() {
  $("lang").value = StrataI18n.lang;
  renderEmptyChat();
  setSidebar(sidebarOpen, false);
  outputs();
  renderChat();
  setBusy(!!busy);
  if (lastMetrics) render(lastMetrics);
}
$("lang").onchange = async () => {
  store.set("lang", $("lang").value);
  await StrataI18n.setLanguage($("lang").value);
  showLanguage();
};

// ------------------------------------------------------------------ start
async function start() {
  await StrataI18n.setLanguage(urlLang || store.get("lang", "en"));
  $("lang").value = StrataI18n.lang;
  renderEmptyChat();
  initChats();
  if (settings.chatList) await setChatList(true, false);       // the saved-chats list is read before the first draw
  setBusy(false);
  renderChat();
  const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
  if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
  loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
  showTab(location.hash.slice(1) || "chat");
  poll();
  requestAnimationFrame(fitHeader);
}
start();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json"};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  imageBusy = true;
  $("img-go").disabled = true;
  paintLive();
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    imageBusy = false;
    $("img-go").disabled = false;
    paintLive();
  }
};
