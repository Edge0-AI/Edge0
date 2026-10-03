import type { Usage } from "../gen";
import { ApiError } from "../bridge/frames";

export interface SseEvent {
  data: string;
}

export function splitSseEvents(buf: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = [];
  let rest = buf;
  for (;;) {
    const m = rest.match(/\r?\n\r?\n/);
    if (!m || m.index === undefined) break;
    const block = rest.slice(0, m.index);
    rest = rest.slice(m.index + m[0].length);
    const dataLines: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (dataLines.length > 0) events.push({ data: dataLines.join("\n") });
  }
  return { events, rest };
}

/** Keep `delta.content` and `delta.reasoning_content` on separate sinks so reasoning never enters the visible body. */
export interface ChatDeltaSink {
  onText(text: string): void;
  onReasoning(text: string): void;
  onUsage(usage: Usage, perf?: PerfEdge0): void;
}

export interface PerfEdge0 {
  ttft_ms?: number;
  prefill_tps?: number;
  decode_tps?: number;
  footprint_mb?: number;
}

export async function consumeChatSse(
  res: Response,
  sink: ChatDeltaSink,
  signal?: AbortSignal,
): Promise<void> {
  const reader = res.body?.getReader();
  if (!reader) throw new ApiError(res.status, null, "streaming response has no body");
  let aborted = false;
  const onAbort = () => {
    aborted = true;
    void reader.cancel();
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  const decoder = new TextDecoder();
  let buf = "";
  let doneFlagged = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const { events, rest } = splitSseEvents(buf);
    buf = rest;
    for (const ev of events) {
      if (ev.data === "[DONE]") {
        doneFlagged = true;
        continue;
      }
      let obj: {
        choices?: { delta?: { content?: string; reasoning_content?: string }; finish_reason?: string | null }[];
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
          total_tokens?: number;
          edge0?: Record<string, unknown> | null;
        } | null;
        error?: { code?: string; message?: string };
      };
      try {
        obj = JSON.parse(ev.data);
      } catch {
        continue;
      }
      if (obj.error) {
        throw new ApiError(res.status, obj.error.code ?? "E-SRV-WORKER", obj.error.message ?? "in-stream error");
      }
      const delta = obj.choices?.[0]?.delta;
      if (delta?.content) sink.onText(delta.content);
      if (delta?.reasoning_content) sink.onReasoning(delta.reasoning_content);
      if (obj.usage && typeof obj.usage.total_tokens === "number") {
        const e = obj.usage.edge0 ?? {};
        const num = (k: string): number | undefined =>
          typeof e[k] === "number" ? (e[k] as number) : undefined;
        const perf: PerfEdge0 = {
          ttft_ms: num("ttft_ms"),
          prefill_tps: num("prefill_tps"),
          decode_tps: num("decode_tps"),
          footprint_mb: num("footprint_mb"),
        };
        sink.onUsage(
          {
            prompt_tokens: obj.usage.prompt_tokens ?? 0,
            completion_tokens: obj.usage.completion_tokens ?? 0,
            total_tokens: obj.usage.total_tokens,
          },
          perf,
        );
      }
    }
  }
  signal?.removeEventListener("abort", onAbort);
  if (aborted) return;
  if (!doneFlagged && buf.trim().length > 0) {
    throw new ApiError(res.status, "E-SRV-WORKER", "stream ended without [DONE] (connection truncated)");
  }
}

export async function parseErrorEnvelope(res: Response): Promise<ApiError> {
  let code: string | null = null;
  let message = `HTTP ${res.status}`;
  try {
    const j = (await res.json()) as { error?: { code?: string; message?: string } };
    code = j?.error?.code ?? null;
    message = j?.error?.message ?? message;
  } catch {
  }
  return new ApiError(res.status, code, message);
}
