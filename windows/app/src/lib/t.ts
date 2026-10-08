// lib/t.ts — single source for UI strings: **English-only** (product decision for the
// overseas release: no language switching). Dot-separated keys with {{var}} interpolation;
// a missing key echoes the key itself (honest surface, never swallowed silently).
type Table = Record<string, unknown>;

const en: Table = {
  app: { name: "edge0", localLabel: "local inference" },
  platform: { note: "Windows · local llama.cpp engine" },
  theme: { light: "Light", dark: "Dark", system: "System" },
  font: { sm: "Small", md: "Default", lg: "Large" },
  nav: {
    models: "Models", service: "Service", settings: "Settings",
    openSidebar: "Open sidebar", closeSidebar: "Close sidebar",
  },
  common: { cancel: "Cancel" },
  sidebar: {
    newChat: "New chat", history: "Chat history", navigation: "Navigation",
    resident: "resident", idle: "idle", noService: "No service",
    scopeShell: "managed in-app",
    bucket: { today: "Today", yesterday: "Yesterday", week: "Previous 7 days", earlier: "Earlier" },
  },
  chat: {
    newThread: "New chat", rename: "Rename", deleteThread: "Delete chat", noThreads: "No chats yet",
    placeholder: "Type a message, Enter to send", send: "Send", stop: "Stop", regenerate: "Regenerate",
    editResend: "Edit & resend", resend: "Resend", copy: "Copy", copyReasoning: "Copy thinking",
    reasoning: "Thinking", aborted: "Response interrupted", saveFailed: "Not saved (conversation store write failed)",
    usage: "tokens in {{p}}/out {{c}}/total {{t}}",
    perf: { decode: "{{v}} tok/s decode", prefill: "{{v}} tok/s prefill", prompt: "prefill {{ms}} ms", mem: "memory {{gi}} GiB" },
    model: "Model", noModel: "(no installed model)",
    light: { resident: "resident", loading: "loading", installed: "idle", absent: "unknown" },
    loadModel: "Load",
    contextUsed: "context {{n}} tok", contextUnknown: "context: unknown",
    greeting: "What would you like to explore?", pickOrCreate: "Select or create a chat",
    serviceShell: "local engine", noInstalledModel: "No models to chat with yet — download one first",
    goDownload: "Go to downloads",
    waitingFirstChunk: "Waiting for the model…", pickModelFirst: "Pick a model first",
    residentForever: "Resident until manually evicted",
    err: {
      notload: "Model not loaded — press Load, or download it",
      busy: "Service busy, retry shortly", cancel: "Generation cancelled",
      worker: "Engine process crashed — restart it from the Service page and retry",
      param: "Request parameters rejected — check the model selection",
      missing: "Model not installed — download it first",
      memload: "Not enough memory to load — use a smaller tier",
      version: "Engine binary missing — set EDGE0_BIN_DIR or start from the Service page",
      pool: "Prefetch pool not active (POOL2 telemetry line missing) — performance claims downgraded",
      protected: "This tier is a seed link (junction) — the shell never deletes through links",
      generic: "Request failed",
    },
  },
  gguf: {
    title: "Local GGUF models (MoE)",
    kicker: "Any GGUF from a supported MoE family runs on the same engine. Expert weights are offloaded so the model fits the GPU the engine reports.",
    folderPlaceholder: "Folder to scan (empty = the local-gguf folder under the edge0 home)",
    scan: "Scan folder", scanning: "Scanning…", noFiles: "No GGUF files found in this folder.",
    pathPlaceholder: "Absolute path to a .gguf file (split models: the first shard)",
    inspect: "Inspect", inspecting: "Reading header…",
    unlisted: "Not in the pinned MoE list",
    arch: "Architecture", shape: "Shape", quant: "Quantisation", size: "File size",
    gpu: "GPU (as the engine reports it)", noGpu: "No GPU reported by the engine",
    ctxLabel: "Context",
    moeShape: "{{layers}} layers · {{experts}} experts ({{used}} active per token)", dense: "dense (no experts to offload)",
    fits: "Fits the GPU", doesNotFit: "Does not fit", unknownFit: "No plan (no GPU budget)",
    mode: "plan: {{mode}}",
    planGpu: "GPU share (est.)", planCpu: "CPU experts (est.)", planKv: "KV cache at this context (est.)", planBudget: "GPU budget after headroom",
    noPlan: "No offload plan: the engine reports no GPU, so the model cannot be planned.",
    load: "Load model", loading: "Loading…",
    loaded: "{{name}} is loaded in the local engine. Chat uses the running engine.",
  },
  settings: {
    title: "Settings", kicker: "Tune edge0 appearance and local service access",
    appearance: "Appearance", appearanceNote: "Theme and font size apply across the app immediately.",
    systemPrompt: "System prompt", systemPromptNote: "Optional instructions sent before each chat. Leave blank to disable.",
    systemPromptEnabled: "System prompt enabled", systemPromptDisabled: "No system prompt will be sent",
    systemPromptReset: "Use default",
    service: "Service", serviceNote: "View a compact summary; full controls, logs, and diagnostics live in the service console.",
    serviceUnavailable: "Service not connected", advancedNote: "Request logs and environment checks live in the service console.",
    openConsole: "Open service console",
  },
  gen: {
    precise: "Precise", balanced: "Balanced", creative: "Creative",
    hint: "Generation style: sampling temperature (Precise 0.2 / Balanced 0.7 / Creative 1.0), applied from the next message.",
  },
  models: {
    title: "Models & Downloads", kicker: "Manage installed tiers, residency, and downloads",
    refresh: "Refresh", loading: "Reading catalog…", retry: "Retry",
    stale: "Catalog manifest expired (mirrors unreachable): browsing OK, downloads paused",
    rev: "Rev", bytes: "Size", files: "Files", source: "Source", singleSource: "single source",
    license: "License", notes: "Model card", pool: "Prefetch pool", repo: "Source repo",
    state: { absent: "Not installed", installed: "Installed", update: "Installed" },
    download: "Download", sourceAuto: "Auto source (speed probe)", sourceMs: "ModelScope", sourceHf: "Hugging Face", sourceHfMirror: "HF mirror",
    load: "Enable", unload: "Evict", loadingModel: "Loading…",
    delete: "Delete", deleteConfirm: "Delete this tier? Both ~/.edge0 directories are removed physically (seed links are refused).",
    loadFailed: "Load failed: {{msg}}",
    pause: "Cancel", cancelTask: "Cancel task", deleteTask: "Delete task",
    sourceSwitched: "source switched", eta: "~{{n}} min left", estimating: "estimating speed…", verifying: "verifying…",
    filesProgress: "{{done}}/{{total}} files",
    phase: {
      idle: "Idle", probe: "Probing sources", download: "Downloading", convert: "On-device converting",
      ready: "Done", error: "Failed", cancelled: "Cancelled",
    },
    err: {
      net: "Mirror or source temporarily unreachable — check network and retry",
      src: "Source rejects resume semantics or unreachable — retry later or switch network",
      disk: "Not enough disk space — free space and retry, or pick a smaller tier",
      hash: "File hash mismatch with manifest — delete the task and retry",
      size: "File size mismatch with manifest — delete the task and retry",
      convert: "On-device conversion failed — check tools/convert_mlx_to_gguf.py, then retry",
      protected: "This tier is a seed link (junction) — the shell unlinks, never deletes targets",
      busy: "Tier is resident — evict it before deleting",
      invalid: "Model files do not match the install record — delete the tier and download again",
      generic: "Request failed",
    },
  },
  welcome: {
    title: "Get started with edge0 on this PC",
    body: "Recommended: {{tier}} ({{bytes}}, stored in {{path}}) — inference runs entirely on-device.",
    bytesUnknown: "size unknown", pathUnknown: "local app directory",
    downloadEnable: "Download & enable", skip: "Skip for now",
    errors: { disk: "Not enough disk space", memory: "Not enough memory to load the recommended tier", engine: "The local engine is unavailable", generic: "First launch did not complete" },
  },
  service: {
    title: "Local API Service", kicker: "Manage engine lifecycle, access, and runtime diagnostics",
    sections: {
      lifecycle: "Lifecycle & ownership", lifecycleNote: "Start or stop the engine and inspect port, version, and uptime.",
      logs: "Request log", logsNote: "Shows the generation-request window since this app started.",
      doctor: "Environment checks", doctorNote: "Shows shell-side measured findings and next actions.",
    },
    panel: {
      title: "Service Panel", start: "Start service", stop: "Stop service",
      busy: "Service operation in progress… (restart ends in-flight requests)", actionFailed: "The service action could not be completed.",
      port: "Listening", bind: "Bind address", scope: "Ownership", uptime: "Uptime",
      version: "Engine version", home: "Storage location", pool: "Prefetch pool", tier: "Resident tier",
      governance: "Process governance", governanceJob: "Job Object armed (OS reclaims the engine when the app dies)",
      governanceSoft: "Normal-exit fallback only (job attach failed — stated honestly)",
      token: "Access token", noCredential: "No credential needed (loopback is exempt)",
      notOwned: "This service was not started by this app; stop it via its owner",
      lan: "Allow LAN", lanOn: "On", lanOff: "Off", lanLater: "Phase 2: LAN access is not implemented yet",
      idle: "Engine not running",
    },
    log: {
      title: "Request Log", count: "{{n}} requests since start",
      note: "Window = generation requests in this app session (in memory; cleared on exit)",
      empty: "No requests since this session started",
      th: { status: "Status", model: "Model", latency: "Latency", tokens: "tokens", source: "Source" },
      status: { done: "done", cancelled: "cancelled", error: "error" },
      source: { loopback: "local" },
    },
    doctor: {
      title: "Environment Check", loading: "Reading check results…", failed: "Cannot fetch environment findings", retry: "Retry",
      next: "Next", goModels: "Go to Models",
      overall: { pass: "All passing", warn: "Warnings" },
      verdict: { pass: "pass", warn: "warn", fail: "fail" },
    },
  },
};

function lookup(key: string): string {
  let cur: unknown = en;
  for (const seg of key.split(".")) {
    if (cur == null || typeof cur !== "object") return key;
    cur = (cur as Table)[seg];
  }
  return typeof cur === "string" ? cur : key;
}

/** t("sidebar.bucket.today") / t("chat.usage", { p, c, t }) — with {{var}} interpolation. */
export function t(key: string, vars?: Record<string, string | number>): string {
  let s = lookup(key);
  if (vars) {
    for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{{${k}}}`, String(v));
  }
  return s;
}

/** Error code → human-readable key (this repo's code set is small: a flat table). */
export function errKey(code: string | null | undefined): string {
  if (!code) return "chat.err.generic";
  if (code.startsWith("E-DL-NET")) return "models.err.net";
  if (code.startsWith("E-DL-SRC")) return "models.err.src";
  if (code.startsWith("E-DL-DISK")) return "models.err.disk";
  if (code.startsWith("E-DL-SHA")) return "models.err.hash";
  if (code.startsWith("E-DL-SIZE")) return "models.err.size";
  if (code.startsWith("E-CONVERT")) return "models.err.convert";
  if (code.startsWith("E-MODEL-PROTECTED")) return "models.err.protected";
  if (code.startsWith("E-MODEL-BUSY")) return "models.err.busy";
  if (code.startsWith("E-MODEL-INVALID")) return "models.err.invalid";
  if (code.startsWith("E-MODEL-MISSING")) return "chat.err.missing";
  if (code.startsWith("E-ENGINE-MISSING")) return "chat.err.version";
  if (code.startsWith("E-ENGINE-DIED")) return "chat.err.worker";
  if (code.startsWith("E-POOL")) return "chat.err.pool";
  if (code.startsWith("E-MEM")) return "chat.err.memload";
  if (code.startsWith("E-DISK")) return "models.err.disk";
  return "chat.err.generic";
}

/** Extract the code from an error string (shell-side format: "E-XXX human text"; take the leading token). */
export function codeOf(err: string | null | undefined): string | null {
  if (!err) return null;
  const m = err.match(/E-[A-Z-]+/);
  return m ? m[0] : null;
}
