import type { ThreadRow } from "../chat/ipc";

type Listener = () => void;

class ThreadStore {
  threads: ThreadRow[] = [];
  activeId: string | null = null;
  private listeners = new Set<Listener>();
  private loaded = false;

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  setActive(id: string | null): void {
    this.activeId = id;
    this.emit();
  }

  async refresh(): Promise<void> {
    const { threadList } = await import("../chat/ipc");
    try {
      this.threads = await threadList();
      this.loaded = true;
    } finally {
      this.emit();
    }
  }

  get isLoaded(): boolean {
    return this.loaded;
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }
}

export const threadStore = new ThreadStore();
