import type { ChatTransport, UIMessage, UIMessageChunk } from "ai";

import { ApiError } from "../bridge/frames";
import type { Usage } from "../gen";
import { edge0Fetch, FAKE_ORIGIN } from "../bridge/edge0Fetch";
import { currentTemperature } from "../lib/generation";
import { consumeChatSse, parseErrorEnvelope, type PerfEdge0 } from "./sse";

export interface AssistantPersistence {
  beginAssistant(): Promise<string>;
  append(id: string, contentDelta: string | null, reasoningDelta: string | null): Promise<void>;
  finish(
    id: string,
    status: "done" | "aborted" | "error",
    code: string | null,
    usage: Usage | null,
  ): Promise<void>;
  dropAssistant(fromId: string): Promise<void>;
}

export interface TransportModelCtx {
  model: string;
  params?: Record<string, unknown>;
  systemPrompt?: string | null;
}

export interface TransportDeps {
  getModel: () => TransportModelCtx | null;
  getPersistence: () => AssistantPersistence;
  flushMs?: number;
  flushEvery?: number;
}

type Fetcher = typeof edge0Fetch;

const TEXT_ID = "text-main";
const REASONING_ID = "reasoning-main";

export function toWireMessages(
  messages: UIMessage[],
  systemPrompt?: string | null,
): { role: "system" | "user" | "assistant"; content: string }[] {
  const out: { role: "system" | "user" | "assistant"; content: string }[] = [];
  if (systemPrompt?.trim()) {
    out.push({ role: "system", content: systemPrompt });
  }
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    let text = "";
    for (const part of m.parts) {
      if (part.type === "text") text += part.text;
    }
    if (text.length > 0 || m.role === "user") out.push({ role: m.role, content: text });
  }
  return out;
}

class DeltaSink {
  pendingContent = "";
  pendingReasoning = "";
  sinceFlush = 0;
  constructor(
    private readonly onFlush: () => void,
    private readonly flushEvery: number,
  ) {}
  collect(content: string | null, reasoning: string | null) {
    if (content) this.pendingContent += content;
    if (reasoning) this.pendingReasoning += reasoning;
    this.sinceFlush += content ? 1 : 0;
    if (this.sinceFlush >= this.flushEvery) this.onFlush();
  }
  take(): [string | null, string | null] {
    const pair: [string | null, string | null] = [
      this.pendingContent || null,
      this.pendingReasoning || null,
    ];
    this.pendingContent = "";
    this.pendingReasoning = "";
    this.sinceFlush = 0;
    return pair;
  }
}

export function createEdge0Transport(
  deps: TransportDeps,
  fetcher: Fetcher = edge0Fetch,
): ChatTransport<UIMessage> {
  const flushMs = deps.flushMs ?? 250;
  const flushEvery = deps.flushEvery ?? 32;

  return {
    async sendMessages(options): Promise<ReadableStream<UIMessageChunk>> {
      const modelCtx = deps.getModel();
      const persist = deps.getPersistence();

      const stream = new ReadableStream<UIMessageChunk>({
        start(controller) {
          void run(controller);
        },
      });

      async function run(controller: ReadableStreamDefaultController<UIMessageChunk>) {
        const push = (c: UIMessageChunk) => controller.enqueue(c);

        if (options.trigger === "regenerate-message" && options.messageId) {
          try {
            await persist.dropAssistant(options.messageId);
          } catch {
          }
        }

        let assistantId: string;
        try {
          assistantId = await persist.beginAssistant();
        } catch (e) {
          push({ type: "start" });
          push({ type: "error", errorText: JSON.stringify({ local: true, message: String(e) }) });
          push({ type: "finish" });
          controller.close();
          return;
        }
        push({ type: "start", messageId: assistantId });

        if (!modelCtx?.model) {
          await persist
            .finish(assistantId, "error", "E-MODEL-MISSING", null)
            .catch(() => undefined);
          push({ type: "error", errorText: JSON.stringify({ code: "E-MODEL-MISSING", message: "No model selected" }) });
          push({ type: "finish" });
          controller.close();
          return;
        }

        let saveFailed = false;
        const sink = new DeltaSink(() => void flush(), flushEvery);
        let openChannel: "reasoning" | "text" | null = null;
        let usage: Usage | null = null;
        let perf: PerfEdge0 | null = null;

        async function flush() {
          const [c, r] = sink.take();
          if (!c && !r) return;
          try {
            await persist.append(assistantId, c, r);
          } catch {
            saveFailed = true;
          }
        }
        const timer = setInterval(() => void flush(), flushMs);

        const switchTo = (ch: "text" | "reasoning") => {
          if (openChannel === ch) return;
          if (openChannel === "text") push({ type: "text-end", id: TEXT_ID });
          if (openChannel === "reasoning") push({ type: "reasoning-end", id: REASONING_ID });
          if (ch === "text") push({ type: "text-start", id: TEXT_ID });
          else push({ type: "reasoning-start", id: REASONING_ID });
          openChannel = ch;
        };

        let terminal: { status: "done" | "aborted" | "error"; code: string | null };
        try {
          const res = await fetcher(`${FAKE_ORIGIN}/v1/chat/completions`, {
            method: "POST",
            stream: true,
            body: {
              model: modelCtx.model,
              messages: toWireMessages(options.messages, modelCtx.systemPrompt),
              stream: true,
              temperature: currentTemperature(),
              // Chat needs a full reply; daemon default 64 is a fixture cap, not a chat cap.
              max_tokens: 1024,
              ...(modelCtx.params ?? {}),
            },
            signal: options.abortSignal ?? undefined,
          });
          if (!res.ok) {
            throw await parseErrorEnvelope(res); // 404 NOTLOAD / 429 BUSY / 507 …
          }
          await consumeChatSse(res, {
            onText: (s) => {
              switchTo("text");
              push({ type: "text-delta", id: TEXT_ID, delta: s });
              sink.collect(s, null);
            },
            onReasoning: (s) => {
              switchTo("reasoning");
              push({ type: "reasoning-delta", id: REASONING_ID, delta: s });
              sink.collect(null, s);
            },
            onUsage: (u, p) => {
              usage = u;
              perf = p ?? null;
            },
          }, options.abortSignal ?? undefined);
          terminal = options.abortSignal?.aborted ? { status: "aborted", code: null } : { status: "done", code: null };
        } catch (e) {
          const apiErr = e instanceof ApiError ? e : null;
          if (options.abortSignal?.aborted) {
            terminal = { status: "aborted", code: null };
          } else {
            const code = apiErr?.code ?? "E-SRV-WORKER";
            const message = apiErr?.message ?? String(e);
            push({ type: "error", errorText: JSON.stringify({ code, message, status: apiErr?.status ?? null }) });
            terminal = { status: "error", code };
          }
        } finally {
          clearInterval(timer);
        }

        if (openChannel === "text") push({ type: "text-end", id: TEXT_ID });
        if (openChannel === "reasoning") push({ type: "reasoning-end", id: REASONING_ID });

        await flush();
        try {
          await persist.finish(assistantId, terminal.status, terminal.code, usage);
        } catch {
          saveFailed = true;
        }
        push({ type: "finish", messageMetadata: { usage, perf, saveFailed, status: terminal.status } });
        controller.close();
      }

      return stream;
    },

    async reconnectToStream() {
      return null;
    },
  };
}

export function decodeErrorChunk(errorText: string): { code: string | null; message: string; status: number | null } {
  try {
    const o = JSON.parse(errorText) as { code?: string | null; message?: string; status?: number | null };
    return { code: o.code ?? null, message: o.message ?? errorText, status: o.status ?? null };
  } catch {
    return { code: null, message: errorText, status: null };
  }
}
