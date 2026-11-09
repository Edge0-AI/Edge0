import type { Fetcher } from '../chat';

export const healthTimeoutMs = 600_000;
const retryDelay = (signal: AbortSignal) => new Promise<void>((resolve, reject) => {
  signal.throwIfAborted();
  const aborted = () => { clearTimeout(timer); reject(signal.reason); };
  const timer = setTimeout(() => { signal.removeEventListener('abort', aborted); resolve(); }, 200);
  signal.addEventListener('abort', aborted, { once: true });
});

// One deadline covers every probe, including a fetch that never returns.
// Timeout aborts only the probe; the caller's signal still identifies user Cancel.
export async function waitForHealth(fetcher: Fetcher, url: string, parent: AbortSignal): Promise<void> {
  parent.throwIfAborted();
  const request = new AbortController();
  let expired = false, rejectInterrupt!: (error: unknown) => void;
  const interrupted = new Promise<never>((_, reject) => { rejectInterrupt = reject; });
  const cancelled = () => { request.abort(parent.reason); rejectInterrupt(parent.reason); };
  parent.addEventListener('abort', cancelled, { once: true });
  const timer = setTimeout(() => {
    expired = true;
    const error = new Error('Service health check timed out');
    request.abort(error); rejectInterrupt(error);
  }, healthTimeoutMs);
  const probe = async () => {
    while (true) {
      request.signal.throwIfAborted();
      try {
        const response = await fetcher(url + '/health', { signal: request.signal });
        await response.body?.cancel();
        if (response.ok) return;
      } catch (error) { if (request.signal.aborted) throw error; }
      await retryDelay(request.signal);
    }
  };
  try { await Promise.race([probe(), interrupted]); }
  catch (error) { if (expired) throw new Error('Service health check timed out'); throw error; }
  finally { clearTimeout(timer); parent.removeEventListener('abort', cancelled); request.abort(); }
}
