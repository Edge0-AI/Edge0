import type { Event, ModelEntry, SystemSnapshot } from "../gen";
import type { EventFrame } from "./frames";
import { subscribeEventDriver } from "./ipc";

export type LinkState = "idle" | "live" | "reconnecting" | "resyncing";

type Listener = () => void;

class DaemonStore {
  snapshot: SystemSnapshot | null = null;
  link: LinkState = "idle";
  lastSeq = 0;
  private folded = new Map<string, Event>();
  private listeners = new Set<Listener>();
  private unsubscribe: (() => void) | null = null;

  ingest(snapshot: SystemSnapshot): void {
    this.snapshot = snapshot;
    this.emit();
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    if (!this.unsubscribe) {
      this.unsubscribe = subscribeEventDriver((f) => this.applyFrame(f));
    }
    return () => {
      this.listeners.delete(fn);
    };
  }

  private applyFrame(f: EventFrame): void {
    switch (f.kind) {
      case "snapshot":
        this.snapshot = f.snapshot;
        for (const key of [...this.folded.keys()]) {
          if (key.startsWith("api.request.finished:")) this.folded.delete(key);
        }
        break;
      case "event": {
        this.lastSeq = f.event.seq;
        this.folded.set(`${f.event.type}:${f.event.subject}`, f.event);
        if (this.snapshot && f.event.type === "model.load.ready") {
          for (const m of this.snapshot.models) {
            if (m.id === f.event.subject) m.state = "resident" satisfies ModelEntry["state"];
          }
        }
        break;
      }
      case "status":
        this.link = f.state;
        break;
    }
    this.emit();
  }

  latestFolded(type: string, subject: string): Event | undefined {
    return this.folded.get(`${type}:${subject}`);
  }

  requestsWindow(): Event[] {
    const out: Event[] = [];
    for (const [key, ev] of this.folded) {
      if (key.startsWith("api.request.finished:")) out.push(ev);
    }
    return out.sort((a, b) => b.seq - a.seq);
  }

  clear(): void {
    this.snapshot = null;
    this.folded.clear();
    this.lastSeq = 0;
  }

  maxSeqAmong(types: readonly string[]): number {
    let best = 0;
    for (const [key, ev] of this.folded) {
      const t = key.slice(0, key.lastIndexOf(":"));
      if (types.includes(t) && ev.seq > best) best = ev.seq;
    }
    return best;
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }
}

export const daemonStore = new DaemonStore();
