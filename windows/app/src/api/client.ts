// api/client.ts — shell IPC wrapper + one-pulse stores. Two-layer architecture:
// data sources are invoke commands + Tauri events only; the frontend never
// invents polling and never fabricates estimates.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type EngStatus = {
  running: boolean; tier?: string; base_url?: string; bound?: string; port?: number;
  pool_mb?: number; uptime_s?: number; log?: string; version?: string | null;
  phys_mem_gb?: number; pool_telemetry?: string | null; pid?: number; job?: boolean;
  model?: string; plan?: GgufPlan | null;
};
export type FileProg = { path: string; done: number; total: number; state: string };
export type TaskView = {
  tier: string; phase: string; bytes_done: number; bytes_total: number;
  source: string; error: string | null; files?: FileProg[];
};
export type CatTier = { repo: string; rev: { modelscope: string; hf: string }; total_bytes: number; pool_mb: number; files: { path: string; size: number; skip?: boolean }[] };

class Sig {
  private ls = new Set<() => void>();
  private v = 0;
  sub = (fn: () => void): (() => void) => {
    this.ls.add(fn);
    return () => { this.ls.delete(fn); };
  };
  version = () => this.v;
  emit() {
    this.v++;
    for (const f of [...this.ls]) f();
  }
}

export const engineStore = {
  status: null as EngStatus | null,
  loadingTier: null as string | null, // in-flight model_load marker (one source of truth for the "loading" light)
  sig: new Sig(),
};

export const downloadStore = {
  tasks: {} as Record<string, TaskView>,
  sig: new Sig(),
  apply(v: TaskView) {
    this.tasks = { ...this.tasks, [v.tier]: v };
    this.sig.emit();
  },
};

export const shellStore = {
  catalog: {} as Record<string, CatTier>,
  paths: { home: "", repo: "", bin_dir: "" },
  installed: {} as Record<string, unknown>,
  sig: new Sig(),
};

/** Ring buffer of request log rows (service-page data = frontend-measured: model / latency / tokens / status; since app start, cleared on exit — labeled honestly). */
export type ReqRow = { ts: number; model: string; status: "done" | "cancelled" | "error"; latencyMs: number; tokens: number | null; code: string | null };
export const reqLog = {
  rows: [] as ReqRow[],
  sig: new Sig(),
  add(r: ReqRow) {
    this.rows = [r, ...this.rows].slice(0, 200);
    this.sig.emit();
  },
};

let bootOnce: Promise<void> | null = null;

/** Idempotent boot: initial fetch + event wiring (shared by main.tsx and first-screen components; repeat calls share one Promise). */
export function boot(): Promise<void> {
  if (!bootOnce) bootOnce = bootNow();
  return bootOnce;
}

async function bootNow(): Promise<void> {
  try {
    shellStore.catalog = await invoke<Record<string, CatTier>>("catalog_get");
    shellStore.paths = await invoke("app_paths");
    await refreshInstalled();
    engineStore.status = await invoke<EngStatus>("engine_status");
    for (const tier of Object.keys(shellStore.catalog)) {
      downloadStore.tasks[tier] = await invoke<TaskView>("download_status", { tier });
    }
  } catch { /* shell not ready yet: stores stay null, UI renders its waiting shape */ }
  void listen<EngStatus>("engine.status", (e) => {
    engineStore.status = e.payload;
    engineStore.sig.emit();
  });
  void listen<TaskView>("download.progress", (e) => downloadStore.apply(e.payload));
  shellStore.sig.emit();
  engineStore.sig.emit();
  downloadStore.sig.emit();
}

export async function refreshInstalled(): Promise<void> {
  shellStore.installed = await invoke<Record<string, unknown>>("models_installed");
  shellStore.sig.emit();
}

/** Models-page refresh button: refresh re-pulls the sources of truth; it is not a poll. */
export async function refreshAll(): Promise<void> {
  try {
    await refreshInstalled();
    engineStore.status = await invoke<EngStatus>("engine_status");
    for (const tier of Object.keys(shellStore.catalog)) {
      downloadStore.tasks[tier] = await invoke<TaskView>("download_status", { tier });
    }
  } finally {
    engineStore.sig.emit();
    downloadStore.sig.emit();
  }
}

export async function loadModel(tier: string): Promise<EngStatus> {
  engineStore.loadingTier = tier;
  engineStore.sig.emit();
  try {
    const s = await invoke<EngStatus>("model_load", { tier });
    engineStore.status = s;
    return s;
  } finally {
    engineStore.loadingTier = null;
    engineStore.sig.emit();
  }
}

export async function unloadModel(): Promise<void> {
  await invoke("model_unload");
  engineStore.status = await invoke<EngStatus>("engine_status");
  engineStore.sig.emit();
}

export const deleteModel = (tier: string): Promise<{ deleted: string }> => invoke("model_delete", { tier });
export const startDownload = (tier: string, source: string | null): Promise<unknown> => invoke("download_start", { tier, source });
export const cancelDownload = (tier: string): Promise<void> => invoke("download_cancel", { tier });
export const doctorRun = (): Promise<{ overall: string; checks: DoctorCheck[] }> => invoke("doctor_run");
export type DoctorCheck = { id: string; verdict: string; detail: string; code?: string | null; next?: string | null };

export function installedTiers(): string[] {
  return Object.keys(shellStore.installed);
}

// —— local GGUF models (MoE families). Facts and the expert-offload plan come from
//    windows/tools/gguf_tool.py; the shell only transports them. ——
export type GgufPlan = {
  fits: boolean | null; mode: string; n_cpu_moe: number; cpu_moe: boolean; args: string[];
  gpu_gib_est: number; cpu_gib_est: number; kv_gib_est: number; budget_gib: number; notes: string[];
};
export type GgufDevice = { id: string; desc: string; total_gb: number; free_gb: number };
export type GgufInfo = {
  ok: boolean; path: string; shards: string[]; name: string; architecture: string; is_moe: boolean;
  layers: number; experts: number; experts_used: number; context_length: number | null;
  embedding_length: number | null; quant: string | null; file_bytes: number; tensor_bytes: number;
  expert_bytes: number; has_chat_template: boolean; kv_bytes_per_token: number; ctx: number;
  family: { id: string; label: string; archs: string[] } | null; warnings: string[];
  plan?: GgufPlan; gpu: GgufDevice | null;
};
export type GgufScanRow = {
  ok: boolean; path: string; name?: string; shards?: string[]; architecture?: string; is_moe?: boolean;
  experts?: number; family?: string | null; family_label?: string | null; quant?: string | null;
  file_bytes?: number; error?: string;
};
export const ggufScan = (dir: string | null): Promise<{ ok: boolean; models: GgufScanRow[] }> =>
  invoke("gguf_scan", { dir });
export const ggufInspect = (path: string, ctx: number): Promise<GgufInfo> => invoke("gguf_inspect", { path, ctx });

/** Load a local GGUF through the shared engine (explicit user action; the shell refuses a plan that does not fit). */
export async function loadGguf(path: string, ctx: number): Promise<EngStatus> {
  engineStore.loadingTier = "local-gguf";
  engineStore.sig.emit();
  try {
    const s = await invoke<EngStatus>("gguf_load", { path, ctx });
    engineStore.status = s;
    return s;
  } finally {
    engineStore.loadingTier = null;
    engineStore.sig.emit();
  }
}
