// state/threads.ts — localStorage-backed thread store (message/row shape mirrors a
// typical chat thread store). A SQLite thread store would be a phase-2 drop-in:
// this layer can be replaced wholesale without touching the UI.
export type MsgUsage = { prompt_tokens: number; completion_tokens: number; total_tokens: number };
export type MsgPerf = { decode_tps?: number; prefill_tps?: number; prompt_ms?: number };
export type MsgMeta = {
  status?: "done" | "aborted" | "error";
  usage?: MsgUsage; perf?: MsgPerf; code?: string | null;
};
export type ThreadMsg = {
  id: string; role: "user" | "assistant"; content: string;
  reasoning?: string; created_at: number; meta?: MsgMeta;
  streaming?: boolean; // in-session transient (stripped before persisting; always false on load)
};
export type ThreadRow = {
  id: string;
  data: { title: string; model: string | null };
  updatedAt: number;
  messages: ThreadMsg[];
};

const STORE_KEY = "edge0.threads.v2";
const ACTIVE_KEY = "edge0.active-thread";

function uid(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

class ThreadStore {
  private rows: ThreadRow[] = [];
  private ver = 0;
  activeId: string | null = null;
  private listeners = new Set<() => void>();
  private persistTimer: ReturnType<typeof setTimeout> | null = null;
  private loaded = false;

  load(): void {
    if (this.loaded) return;
    this.loaded = true;
    try {
      this.rows = (JSON.parse(localStorage.getItem(STORE_KEY) ?? "[]") as ThreadRow[])
        .map((r) => ({
          ...r,
          // converge crash residue: a trailing assistant message without terminal meta means generation was interrupted -> mark aborted
          messages: (() => {
            const ms = r.messages.map((m) => ({ ...m, streaming: false }));
            const last = ms[ms.length - 1];
            if (last && last.role === "assistant" && !last.meta?.status) {
              ms[ms.length - 1] = { ...last, meta: { ...last.meta, status: "aborted" } };
            }
            return ms;
          })(),
        }));
    } catch {
      this.rows = []; // drop a corrupt cache (never crash startup)
    }
    const a = localStorage.getItem(ACTIVE_KEY);
    this.activeId = a && this.rows.some((r) => r.id === a) ? a : null;
  }

  subscribe = (cb: () => void): (() => void) => {
    this.listeners.add(cb);
    return () => { this.listeners.delete(cb); };
  };

  /** snapshot = version number (stable source for useSyncExternalStore; get rows via snapshot()). */
  version(): number {
    return this.ver;
  }

  threads(): ThreadRow[] {
    return this.rows;
  }

  private emit(): void {
    this.ver++;
    for (const l of [...this.listeners]) l();
    if (this.persistTimer) clearTimeout(this.persistTimer);
    this.persistTimer = setTimeout(() => {
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify(
          this.rows.map((r) => ({ ...r, messages: r.messages.map((m) => {
            const { streaming: _s, ...rest } = m;
            return rest;
          }) })),
        ));
      } catch { /* quota exceeded: skip the write (UI must not crash) */ }
    }, 400);
  }

  /** refresh() compat shim kept from the original API shape (this implementation is always in sync). */
  async refresh(): Promise<void> {
    this.load();
    this.emit();
  }

  create(title: string, model: string | null): ThreadRow {
    this.load();
    const row: ThreadRow = { id: uid(), data: { title, model }, updatedAt: Date.now(), messages: [] };
    this.rows = [row, ...this.rows];
    this.activeId = row.id;
    localStorage.setItem(ACTIVE_KEY, row.id);
    this.emit();
    return row;
  }

  rename(id: string, title: string): void {
    this.rows = this.rows.map((r) => (r.id === id ? { ...r, data: { ...r.data, title }, updatedAt: Date.now() } : r));
    this.emit();
  }

  remove(id: string): void {
    this.rows = this.rows.filter((r) => r.id !== id);
    if (this.activeId === id) this.setActive(null);
    this.emit();
  }

  setActive(id: string | null): void {
    this.activeId = id;
    if (id) localStorage.setItem(ACTIVE_KEY, id);
    else localStorage.removeItem(ACTIVE_KEY);
    this.emit();
  }

  setModel(id: string, model: string): void {
    this.rows = this.rows.map((r) => (r.id === id ? { ...r, data: { ...r.data, model } } : r));
    this.emit();
  }

  messages(id: string): ThreadMsg[] {
    return this.rows.find((r) => r.id === id)?.messages ?? [];
  }

  /** Single entry point for message-list mutations (updatedAt written along). */
  patchMessages(id: string, fn: (msgs: ThreadMsg[]) => ThreadMsg[]): void {
    this.rows = this.rows.map((r) =>
      r.id === id ? { ...r, updatedAt: Date.now(), messages: fn(r.messages) } : r,
    );
    this.emit();
  }
}

export const threadStore = new ThreadStore();
threadStore.load();
