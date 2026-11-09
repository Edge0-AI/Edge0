import { useEffect, useRef } from 'react';
export interface PollFlight { current: Promise<unknown> | null; nextAt?: number; }
export function pollVisible(fn: () => Promise<unknown>, interval: number, visibility: Pick<Document, 'hidden' | 'addEventListener' | 'removeEventListener'> = document, flight: PollFlight = { current: null }) {
  let stopped = false, running = false, timer: ReturnType<typeof setTimeout> | undefined;
  const run = async () => {
    if (stopped || visibility.hidden || running) return;
    const remaining = (flight.nextAt ?? 0) - Date.now();
    if (remaining > 0) { timer = setTimeout(() => void run(), remaining); return; }
    running = true;
    try {
      if (flight.current) await flight.current;
      else { const request = Promise.resolve().then(fn); flight.current = request; try { await request; } finally { flight.nextAt = Date.now() + Math.max(1000, interval); if (flight.current === request) flight.current = null; } }
    } catch { /* The page owns error presentation; retry without an unhandled rejection. */ }
    finally { running = false; if (!stopped && !visibility.hidden) timer = setTimeout(() => void run(), Math.max(1000, interval)); }
  };
  const change = () => { clearTimeout(timer); if (!visibility.hidden && !running) timer = setTimeout(() => void run(), Math.max(1000, interval)); };
  visibility.addEventListener('visibilitychange', change);
  void run();
  return () => { stopped = true; clearTimeout(timer); visibility.removeEventListener('visibilitychange', change); };
}
export function usePolling(fn: () => Promise<unknown>, enabled: boolean, interval = 1000) { const callback = useRef(fn), flight = useRef<Promise<unknown> | null>(null); callback.current = fn; useEffect(() => enabled ? pollVisible(() => callback.current(), interval, document, flight) : undefined, [enabled, interval]); }
