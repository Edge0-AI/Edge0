import { edge0Fetch } from "../bridge/edge0Fetch";
import { daemonStore } from "../bridge/state";
import type { Event, SystemSnapshot } from "../gen";

export const TERMINAL_EVENT_TYPES = [
  "download.completed",
  "download.failed",
  "download.paused",
  "model.load.started",
  "model.load.ready",
  "model.load.failed",
  "model.unloaded",
  "worker.crashed",
  "service.restart",
] as const;

export async function refreshSystem(): Promise<void> {
  try {
    const res = await edge0Fetch("/v1/edge0/system");
    if (!res.ok) return;
    daemonStore.ingest((await res.json()) as SystemSnapshot);
  } catch {
    /* ignore */
  }
}

export interface TaskView {
  id: string;
  tier: string;
  rev: string;
  phase: string;
  source: string | null;
  bytesDone: number;
  bytesTotal: number;
  paused: boolean;
  rateBps: number | null;
  etaS: number | null;
  sourceSwitched: boolean;
  failedCode: string | null;
}

const DL_TYPES = [
  "download.progress",
  "download.paused",
  "download.completed",
  "download.failed",
] as const;

export function latestDownloadEvent(
  store: { latestFolded(type: string, subject: string): Event | undefined },
  taskId: string,
): Event | null {
  let best: Event | null = null;
  for (const t of DL_TYPES) {
    const e = store.latestFolded(t, taskId);
    if (e && (!best || e.seq > best.seq)) best = e;
  }
  return best;
}

export function taskView(
  snap: SystemSnapshot | null,
  store: { latestFolded(type: string, subject: string): Event | undefined },
  tier: string,
): TaskView | null {
  const entry = (snap?.downloads ?? []).find((d) => d.tier === tier);
  if (!entry) return null;
  const v: TaskView = {
    id: entry.id,
    tier: entry.tier,
    rev: entry.rev,
    phase: entry.phase,
    source: entry.source,
    bytesDone: entry.bytes_done,
    bytesTotal: entry.bytes_total,
    paused: entry.paused,
    rateBps: entry.rate_bps ?? null,
    etaS: entry.eta_s ?? null,
    sourceSwitched: entry.source_switched ?? false,
    failedCode: entry.failed?.code ?? null,
  };
  const ev = latestDownloadEvent(store, entry.id);
  if (ev && snap && ev.seq > snap.events_seq && ev.subject === entry.id) {
    const p = ev.payload as Record<string, unknown>;
    const num = (k: string): number | undefined =>
      typeof p[k] === "number" ? (p[k] as number) : undefined;
    if (typeof p.phase === "string") v.phase = p.phase;
    if (typeof p.source === "string") v.source = p.source;
    const bd = num("bytes_done");
    if (bd !== undefined && bd >= v.bytesDone) v.bytesDone = bd;
    if (typeof p.paused === "boolean") v.paused = p.paused;
    const r = num("rate_bps");
    if (r !== undefined) v.rateBps = r;
    const eta = num("eta_s");
    if (eta !== undefined) v.etaS = eta;
    if (p.source_switched === true) v.sourceSwitched = true;
    if (ev.type === "download.failed" && typeof p.code === "string") v.failedCode = p.code;
  }
  return v;
}
