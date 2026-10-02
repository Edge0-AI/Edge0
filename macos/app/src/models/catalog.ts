import type { CatalogResponse } from "../gen";
import { edge0Fetch } from "../bridge/edge0Fetch";
import { ApiError } from "../bridge/frames";

export type CatalogStatus = "idle" | "loading" | "ready" | "error";

interface Listener {
  (): void;
}

class CatalogStore {
  status: CatalogStatus = "idle";
  data: CatalogResponse | null = null;
  error: { code: string | null; message: string } | null = null;
  lastError: Error | null = null;
  private listeners = new Set<Listener>();
  private inflight: Promise<void> | null = null;

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }

  async refresh(): Promise<void> {
    if (this.inflight) return this.inflight;
    this.status = "loading";
    this.emit();
    const p = (async () => {
      try {
        const res = await edge0Fetch("/v1/edge0/catalog");
        if (!res.ok) throw await apiErrFrom(res);
        const j = (await res.json()) as CatalogResponse;
        this.data = j;
        this.error = null;
        this.status = "ready";
      } catch (e) {
        const ae = e instanceof ApiError ? e : null;
        this.error = { code: ae?.code ?? null, message: ae?.message ?? String(e) };
        this.lastError = e instanceof Error ? e : new Error(String(e));
        this.status = "error";
      } finally {
        this.inflight = null;
        this.emit();
      }
    })();
    this.inflight = p;
    return p;
  }
}

export const catalogStore = new CatalogStore();

async function apiErrFrom(res: Response): Promise<ApiError> {
  try {
    const j = (await res.json()) as { error?: { code?: string; message?: string } };
    return new ApiError(res.status, j?.error?.code ?? null, j?.error?.message ?? `HTTP ${res.status}`);
  } catch {
    return new ApiError(res.status, null, `HTTP ${res.status}`);
  }
}
