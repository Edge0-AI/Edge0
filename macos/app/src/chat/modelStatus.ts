import type { Event, SystemSnapshot } from "../gen";

export type ModelLight = "resident" | "loading" | "installed" | "absent";

const MODEL_EVENT_TYPES = [
  "model.load.started",
  "model.load.ready",
  "model.load.failed",
  "model.unloaded",
] as const;

export function latestModelEvent(
  store: { latestFolded(type: string, subject: string): Event | undefined },
  modelId: string,
): Event | null {
  let best: Event | null = null;
  for (const t of MODEL_EVENT_TYPES) {
    const e = store.latestFolded(t, modelId);
    if (e && (!best || e.seq > best.seq)) best = e;
  }
  return best;
}

export function deriveLight(
  snapshot: SystemSnapshot | null,
  modelId: string,
  latest: Event | null,
): ModelLight {
  if (!snapshot) return "absent";
  const entry = snapshot.models.find((m) => m.id === modelId);
  if (!entry) return "absent";
  if (latest && latest.subject === modelId && latest.type === "model.load.started") {
    if (entry.state !== "resident") return "loading";
  }
  return entry.state === "resident" ? "resident" : "installed";
}

export function candidateModels(snapshot: SystemSnapshot | null): string[] {
  return (snapshot?.models ?? []).map((m) => m.id);
}
