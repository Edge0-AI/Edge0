// One-shot intent token for "Download & enable" (in memory; consumed on use).
// This module is the ONLY place that may auto-chain a finished download into a load —
// every other load path requires an explicit user click.
type Phase = "download" | "load";

interface Token {
  tier: string;
  taskId: string; // download tasks are keyed by tier (shell-side pull.rs)
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

  /** User pressed "Download & enable" (the only legal entry point). */
  begin(tier: string, taskId: string): void {
    this.token = { tier, taskId, phase: "download" };
    this.emit();
  }

  /** download ready for the same task: move to the load phase (returns tier). One-shot: never revives. */
  claimCompleted(tier: string): string | null {
    if (!this.token || this.token.phase !== "download" || this.token.taskId !== tier) return null;
    const t = this.token.tier;
    this.token = { ...this.token, phase: "load" };
    this.emit();
    return t;
  }

  /** load succeeded: consume the token (returns tier for thread creation); later duplicate ready events are no-ops. */
  consume(tier: string): boolean {
    if (!this.token || this.token.phase !== "load" || this.token.tier !== tier) return false;
    this.token = null;
    this.emit();
    return true;
  }

  /** failure / abandon / close: invalidate immediately (never persisted, so a restart naturally drops it). */
  cancel(): void {
    if (this.token) {
      this.token = null;
      this.emit();
    }
  }
}

export const autoLoad = new AutoLoadStore();
