type Phase = "download" | "load";

interface Token {
  tier: string;
  taskId: string;
  phase: Phase;
}

type Listener = () => void;

class AutoLoadStore {
  private token: Token | null = null;
  private listeners = new Set<Listener>();

  get current(): Token | null {
    return this.token;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }

  begin(tier: string, taskId: string): void {
    this.token = { tier, taskId, phase: "download" };
    this.emit();
  }

  claimCompleted(taskId: string): string | null {
    if (!this.token || this.token.phase !== "download" || this.token.taskId !== taskId) return null;
    const tier = this.token.tier;
    this.token = { ...this.token, phase: "load" };
    this.emit();
    return tier;
  }

  consume(tier: string): boolean {
    if (!this.token || this.token.phase !== "load" || this.token.tier !== tier) return false;
    this.token = null;
    this.emit();
    return true;
  }

  cancel(): void {
    if (this.token) {
      this.token = null;
      this.emit();
    }
  }
}

export const autoLoad = new AutoLoadStore();
