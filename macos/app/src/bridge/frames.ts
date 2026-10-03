import type { Event } from "../gen/Event";
import type { SystemSnapshot } from "../gen/SystemSnapshot";

export type StreamFrame =
  | { kind: "chunk"; data: string }
  | { kind: "done" }
  | { kind: "error"; code: string | null; message: string };

export type EventFrame =
  | { kind: "snapshot"; snapshot: SystemSnapshot }
  | { kind: "event"; event: Event }
  | { kind: "status"; state: "live" | "reconnecting" | "resyncing" };

export interface ApiErrorShape {
  status: number | null;
  code: string | null;
  message: string;
}

export class ApiError extends Error implements ApiErrorShape {
  status: number | null;
  code: string | null;
  constructor(status: number | null, code: string | null, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function toApiError(raw: string): ApiError {
  try {
    const o = JSON.parse(raw) as { status?: number | null; code?: string | null; message?: string };
    if (o && typeof o === "object" && "message" in o) {
      return new ApiError(o.status ?? null, o.code ?? null, o.message ?? raw);
    }
  } catch {
    /* non-JSON errors are passed through as-is */
  }
  return new ApiError(null, null, raw);
}
