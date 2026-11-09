import { expect, it, vi } from 'vitest';
import { healthTimeoutMs, waitForHealth } from './health';

it('aborts a never-returning fetch at the existing total deadline and identifies timeout', async () => {
  vi.useFakeTimers(); const parent = new AbortController(); let requestSignal: AbortSignal | undefined;
  const fetcher = vi.fn((_url: string, init?: RequestInit) => { requestSignal = init?.signal ?? undefined; return new Promise<Response>(() => {}); });
  let settled = false; const pending = waitForHealth(fetcher, 'http://127.0.0.1:1', parent.signal); pending.then(() => { settled = true; }, () => { settled = true; });
  const rejection = expect(pending).rejects.toThrow('Service health check timed out');
  await vi.advanceTimersByTimeAsync(healthTimeoutMs - 1); expect(settled).toBe(false); expect(requestSignal?.aborted).toBe(false);
  await vi.advanceTimersByTimeAsync(1); await rejection; expect(requestSignal?.aborted).toBe(true); expect(parent.signal.aborted).toBe(false); expect(vi.getTimerCount()).toBe(0); vi.useRealTimers();
});
it('keeps user Cancel distinct from timeout and clears all timers', async () => {
  vi.useFakeTimers(); const parent = new AbortController(); let requestSignal: AbortSignal | undefined;
  const pending = waitForHealth(async (_url, init) => { requestSignal = init?.signal ?? undefined; return new Promise<Response>(() => {}); }, 'http://127.0.0.1:1', parent.signal);
  const rejection = expect(pending).rejects.toMatchObject({ name: 'AbortError' }); parent.abort(new DOMException('Stopped', 'AbortError')); await rejection;
  expect(requestSignal?.aborted).toBe(true); expect(vi.getTimerCount()).toBe(0); vi.useRealTimers();
});
it('retries health failure, completes on readiness and removes the overall timer', async () => {
  vi.useFakeTimers(); const fetcher = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(new Response('ready'));
  const parent = new AbortController(), pending = waitForHealth(fetcher, 'http://127.0.0.1:1', parent.signal);
  await vi.advanceTimersByTimeAsync(199); expect(fetcher).toHaveBeenCalledTimes(1); await vi.advanceTimersByTimeAsync(1); await pending;
  expect(fetcher).toHaveBeenCalledTimes(2); expect(fetcher.mock.calls[0][0]).toBe('http://127.0.0.1:1/health'); expect(parent.signal.aborted).toBe(false); expect(vi.getTimerCount()).toBe(0); vi.useRealTimers();
});
it('does not restart the total deadline after unsuccessful responses', async () => {
  vi.useFakeTimers(); const fetcher = vi.fn(async () => new Response('', { status: 503 }));
  const pending = waitForHealth(fetcher, 'http://127.0.0.1:1', new AbortController().signal);
  const rejection = expect(pending).rejects.toThrow('Service health check timed out'); await vi.advanceTimersByTimeAsync(healthTimeoutMs); await rejection;
  expect(fetcher.mock.calls.length).toBeGreaterThan(1); expect(vi.getTimerCount()).toBe(0); vi.useRealTimers();
});
