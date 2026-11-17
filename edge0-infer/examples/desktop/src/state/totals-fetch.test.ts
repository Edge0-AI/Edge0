import { afterEach, expect, it, vi } from 'vitest';
import { fetchStatus, aggregateTotals, emptyTotals, observeTotals } from './totals';
import sample from '../../../shared/status-sample.json';
import type { EngineStatus } from '../status';

afterEach(() => vi.useRealTimers());
it('records the final service counters without double counting the same instance', async () => {
  const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => Response.json(sample));
  const status = await fetchStatus(fetcher, 'http://127.0.0.1:9000');
  expect(fetcher.mock.calls[0]?.[0]).toBe('http://127.0.0.1:9000/status');
  const once = observeTotals(emptyTotals(), status);
  expect(aggregateTotals(observeTotals(once, status))).toEqual(aggregateTotals(once));
});
it('bounds a stalled request even when the transport does not finish after abort', async () => {
  vi.useFakeTimers(); let signal: AbortSignal | undefined;
  const pending = fetchStatus(async (_url, init) => { signal = init?.signal as AbortSignal; return new Promise<Response>(() => {}); }, 'http://127.0.0.1:9000', 2000);
  const result = expect(pending).rejects.toThrow('Status request timed out');
  await vi.advanceTimersByTimeAsync(2000); await result;
  expect(signal?.aborted).toBe(true);
  expect(vi.getTimerCount()).toBe(0);
});
it('rejects an unavailable service instead of saving fabricated zero counters', async () => {
  await expect(fetchStatus(async () => new Response('unavailable', { status: 503 }), 'http://127.0.0.1:9000')).rejects.toThrow('Status failed (503)');
});
it('keeps counters from a stopped instance when the next model starts', () => {
  const first = { ...sample, instance: { ...sample.instance, id: 'old' }, totals: { ...sample.totals, requests: 3 } } as EngineStatus;
  const second = { ...first, instance: { ...first.instance, id: 'new' }, totals: { ...first.totals, requests: 0 } };
  expect(aggregateTotals(observeTotals(observeTotals(emptyTotals(), first), second)).requests).toBe(3);
});
