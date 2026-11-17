import type { EngineStatus, Totals } from '../status';
import type { Fetcher } from '../chat';
/** A last totals snapshot must not leave Stop or a model switch waiting indefinitely. */
export async function fetchStatus(fetcher: Fetcher, url: string, timeoutMs = 2000): Promise<EngineStatus> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => { timer = setTimeout(() => { const error = new Error('Status request timed out'); controller.abort(error); reject(error); }, timeoutMs); });
  const request = async () => {
    const response = await fetcher(url + '/status', { signal: controller.signal });
    if (!response.ok) { await response.body?.cancel(); throw new Error(`Status failed (${response.status})`); }
    return response.json() as Promise<EngineStatus>;
  };
  try { return await Promise.race([request(), deadline]); }
  finally { clearTimeout(timer!); controller.abort(); }
}
export interface SavedTotals { version: 1; instances: Record<string, Totals>; baselines: Record<string, Totals>; }
export const emptyTotals = (): SavedTotals => ({ version: 1, instances: {}, baselines: {} });
export const observeTotals = (saved: SavedTotals, status: Pick<EngineStatus, 'instance' | 'totals'>): SavedTotals => ({ ...saved, instances: { ...saved.instances, [status.instance.id]: structuredClone(status.totals) } });
export const resetTotals = (saved: SavedTotals): SavedTotals => ({ version: 1, instances: structuredClone(saved.instances), baselines: structuredClone(saved.instances) });
const fields = ['requests', 'rejected_busy', 'rejected_invalid', 'unauthorized', 'input_tokens', 'reused_tokens', 'computed_tokens', 'generated_tokens', 'checkpoint_restores', 'continued_turns', 'decode_seconds'] as const;
export function aggregateTotals(saved: SavedTotals): Totals {
  const result: Totals = { requests: 0, rejected_busy: 0, rejected_invalid: 0, unauthorized: 0, input_tokens: 0, reused_tokens: 0, computed_tokens: 0, generated_tokens: 0, checkpoint_restores: 0, continued_turns: 0, decode_seconds: 0 };
  let decodeForwards = 0;
  for (const [id, total] of Object.entries(saved.instances)) {
    const base = saved.baselines[id];
    for (const field of fields) result[field] += Math.max(0, total[field] - (base?.[field] ?? 0));
    if (total.experts) {
      result.experts ??= { demand: 0, hits: 0, loads: 0, bytes: 0 };
      for (const field of ['demand', 'hits', 'loads', 'bytes'] as const) result.experts[field] += Math.max(0, total.experts[field] - (base?.experts?.[field] ?? 0));
    }
    // The service exposes decode time and a ratio of forwards/time; reconstruct
    // the numerator before adding instances, never average the rates.
    decodeForwards += Math.max(0, (total.decode_tokens_per_second ?? 0) * total.decode_seconds - (base?.decode_tokens_per_second ?? 0) * (base?.decode_seconds ?? 0));
  }
  if (result.decode_seconds > 0) result.decode_tokens_per_second = decodeForwards / result.decode_seconds;
  // The HTTP contract omits first-token sample count and prefill numerator.
  // Cross-instance TTFT and prefill cannot be reconstructed without estimating.
  return result;
}
