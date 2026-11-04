import type { DeviceMemory } from './format';
export interface Experts { demand: number; hits: number; loads: number; bytes: number; }
export interface Totals { requests: number; rejected_busy: number; rejected_invalid: number; unauthorized: number; input_tokens: number; reused_tokens: number; computed_tokens: number; generated_tokens: number; checkpoint_restores: number; continued_turns: number; decode_seconds: number; first_token_seconds_mean?: number; prefill_tokens_per_second?: number; decode_tokens_per_second?: number; experts?: Experts; }
export interface RequestRecord { time: string; endpoint: string; source: string; http_status: number; total_seconds: number; input_tokens?: number | null; reused_tokens?: number | null; generated_tokens?: number | null; first_token_seconds?: number | null; prefill_tokens_per_second?: number | null; decode_tokens_per_second?: number | null; decode_seconds?: number | null; stop_reason?: string; ended_at?: string; }
export interface Progress { phase: 'prefill' | 'decode'; input_tokens: number; reused_tokens: number; computed_tokens: number; pending_tokens: number; generated_tokens: number; elapsed_seconds: number; first_token_seconds?: number; prefill_tokens_per_second?: number | null; decode_elapsed_seconds?: number; decode_tokens_per_second?: number | null; }
interface Measurement { values: { current_bytes: number | null; peak_bytes: number | null } | null; source: string; reason: string; }
export interface EngineStatus {
  instance: { id: string; pid: number; started_at: string; uptime_seconds: number };
  listen: { host: string; port: number; auth: boolean };
  model: { alias: string; directory: string; context_length: number; device: string; prefill_device: string };
  engine: { version: string; source_commit: string; ggml_version: string; backends: { name: string; compiled: string }[]; devices: { name: string; backend: string; description: string }[] };
  current: Progress | null; last: RequestRecord | null; recent_requests: RequestRecord[]; totals: Totals;
  memory: { ledger: { limit_bytes: number; current_bytes: number; peak_bytes: number; by_purpose: { resident_weights: number; experts: number; session_state: number; workspace: number; transfer: number; staging: number; metadata: number }; by_device: Record<string, DeviceMemory> }; physical: Measurement; resident: Measurement };
  system: { memory: { total: { bytes: number | null }; available: { bytes: number | null } }; thermal: { state: string | null; source: string; reason: string } };
}
export interface Inspection { path: string; location?: string; ok: boolean; detail: string; model_id: string | null; total_bytes?: number | null; file_count?: number | null; }
export interface Info { version?: string; commit?: string; devices: { backend: string; name: string; description: string }[]; }
export function parseInfo(text: string): Info {
  const read = (key: string) => text.split('\n').find(line => line.startsWith(key + ': '))?.slice(key.length + 2);
  const devices = text.split('\n').filter(line => /^device\.\d+: /.test(line)).flatMap(line => {
    const match = /backend=(\S+) name=(.*?) description="((?:\\.|[^"\\])*)"/.exec(line);
    return match ? [{ backend: match[1], name: match[2], description: JSON.parse('"' + match[3] + '"') as string }] : [];
  }); return { version: read('engine.version'), commit: read('engine.source_commit'), devices };
}
export const sourceLabel = (source: string, platform = globalThis.navigator?.platform ?? '') => /^(127\.|::1$|::ffff:127\.)/.test(source) ? platform.toLowerCase().includes('mac') ? 'this Mac' : 'this PC' : source;

export function deviceLabel(name: string, devices: EngineStatus['engine']['devices'] = []): string {
  const backend = devices.find(device => device.name === name || device.backend.toLowerCase() === name.toLowerCase())?.backend ?? name;
  switch (backend.toLowerCase()) {
    case 'metal': return 'Metal';
    case 'cpu': return 'CPU';
    case 'vulkan': return 'Vulkan';
    default: return backend;
  }
}
export function stopReason(reason?: string): string {
  switch (reason) {
    case 'end_of_generation': return 'stopped at end of answer';
    case 'length': return 'stopped at length limit';
    case 'cancelled': return 'stopped';
    default: return reason ?? '';
  }
}
