// chat/useThreadChat.ts — per-thread generation hook. No AI SDK dependency: talks
// straight to the loopback llama-server OpenAI SSE endpoint. The final chunk's
// usage+timings are recorded as measured (missing keys render nothing; zero estimates).
import { useCallback, useRef, useState } from "react";
import { engineStore, reqLog } from "../api/client";
import { threadStore, type MsgUsage, type ThreadMsg, type ThreadRow } from "../state/threads";
import { currentTemperature } from "../lib/generation";
import { storedSystemPrompt } from "../lib/systemPrompt";
import { codeOf } from "../lib/t";

export type ThreadChatApi = {
  messages: ThreadMsg[];
  busy: boolean;
  send: (text: string) => Promise<void>;
  stop: () => void;
  regenerate: () => Promise<void>;
  editResend: (msgId: string, text: string) => Promise<void>;
  wireJson: () => { role: string; content: string }[]; // the exact wire form used for token metering
};

export function toWire(history: ThreadMsg[]): { role: string; content: string }[] {
  const out: { role: string; content: string }[] = [];
  const sp = storedSystemPrompt();
  if (sp) out.push({ role: "system", content: sp });
  for (const m of history) {
    if (m.role === "user") out.push({ role: "user", content: m.content });
    else if (m.content.trim()) out.push({ role: "assistant", content: m.content });
  }
  return out;
}

const uid = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export function useThreadChat(thread: ThreadRow, model: string | null): ThreadChatApi {
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchLast = useCallback(
    (patch: Partial<ThreadMsg>) => {
      threadStore.patchMessages(thread.id, (ms) =>
        ms.length ? ms.map((m, i) => (i === ms.length - 1 ? { ...m, ...patch } : m)) : ms,
      );
    },
    [thread.id],
  );

  const run = useCallback(
    async (history: ThreadMsg[]) => {
      const base = engineStore.status?.running ? engineStore.status.base_url : undefined;
      if (!base) {
        patchLast({ meta: { status: "error", code: "E-NOT-LOADED" } });
        return;
      }
      setBusy(true);
      const t0 = Date.now();
      const ac = new AbortController();
      abortRef.current = ac;
      let acc = "";
      let reason = "";
      let usage: MsgUsage | undefined;
      let perf: { decode_tps?: number; prefill_tps?: number; prompt_ms?: number } | undefined;
      try {
        const resp = await fetch(`${base}/v1/chat/completions`, {
          method: "POST",
          signal: ac.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: model ?? "edge0",
            messages: toWire(history),
            max_tokens: 1024,
            temperature: currentTemperature(),
            stream: true,
            stream_options: { include_usage: true }, // final-chunk usage comes straight from the engine
            // enable_thinking:false => answer text streams directly (server-side normalization point)
            chat_template_kwargs: { enable_thinking: false },
          }),
        });
        if (!resp.ok || !resp.body) throw new Error(`E-ENGINE HTTP ${resp.status}`);
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            const tr = line.trim();
            if (!tr.startsWith("data:")) continue;
            const payload = tr.slice(5).trim();
            if (payload === "[DONE]") continue;
            try {
              const d = JSON.parse(payload);
              const delta = d.choices?.[0]?.delta ?? {};
              acc += delta.content ?? "";
              reason += delta.reasoning_content ?? "";
              if (d.usage) {
                usage = {
                  prompt_tokens: d.usage.prompt_tokens ?? 0,
                  completion_tokens: d.usage.completion_tokens ?? 0,
                  total_tokens: d.usage.total_tokens ?? 0,
                };
              }
              if (d.timings && typeof d.timings === "object") {
                const nums: Record<string, number> = {};
                for (const [k, v] of Object.entries(d.timings)) if (typeof v === "number") nums[k] = v;
                perf = {
                  decode_tps: nums.predicted_per_second,
                  prefill_tps: nums.prompt_per_second,
                  prompt_ms: nums.prompt_ms,
                };
              }
              patchLast({ content: acc, reasoning: reason, streaming: true });
            } catch {
              /* tolerate a split (partial) JSON line; next chunk completes it */
            }
          }
        }
        patchLast({ content: acc, reasoning: reason, streaming: false, meta: { status: "done", usage, perf } });
        reqLog.add({ ts: Date.now(), model: model ?? "edge0", status: "done", latencyMs: Date.now() - t0, tokens: usage?.completion_tokens ?? null, code: null });
      } catch (e) {
        const aborted = (e as Error).name === "AbortError";
        patchLast({
          content: acc, reasoning: reason, streaming: false,
          meta: aborted
            ? { status: "aborted", usage, perf }
            : { status: "error", code: codeOf(String(e)) ?? "E-ENGINE" },
        });
        reqLog.add({
          ts: Date.now(), model: model ?? "edge0",
          status: aborted ? "cancelled" : "error",
          latencyMs: Date.now() - t0, tokens: null, code: aborted ? null : codeOf(String(e)),
        });
      } finally {
        setBusy(false);
        abortRef.current = null;
      }
    },
    [model, patchLast, thread.id],
  );

  const send = useCallback(
    async (text: string) => {
      const history: ThreadMsg[] = [
        ...threadStore.messages(thread.id),
        { id: uid(), role: "user", content: text, created_at: Date.now() },
      ];
      threadStore.patchMessages(thread.id, (ms) => [
        ...ms,
        history[history.length - 1],
        { id: uid(), role: "assistant", content: "", created_at: Date.now(), streaming: true },
      ]);
      await run(history);
    },
    [run, thread.id],
  );

  const regenerate = useCallback(async () => {
    const msgs = threadStore.messages(thread.id);
    const lastA = msgs.map((m) => m.role).lastIndexOf("assistant");
    if (lastA < 0) return;
    const history = msgs.slice(0, lastA); // drop the previous answer
    threadStore.patchMessages(thread.id, () => [
      ...history,
      { id: uid(), role: "assistant", content: "", created_at: Date.now(), streaming: true },
    ]);
    await run(history);
  }, [run, thread.id]);

  const editResend = useCallback(
    async (msgId: string, text: string) => {
      const msgs = threadStore.messages(thread.id);
      const idx = msgs.findIndex((m) => m.id === msgId);
      if (idx < 0) return;
      const edited: ThreadMsg = { id: uid(), role: "user", content: text, created_at: Date.now() };
      const history = [...msgs.slice(0, idx), edited];
      threadStore.patchMessages(thread.id, () => [
        ...history,
        { id: uid(), role: "assistant", content: "", created_at: Date.now(), streaming: true },
      ]);
      await run(history);
    },
    [run, thread.id],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const wireJson = useCallback(() => toWire(threadStore.messages(thread.id)), [thread.id]);

  return { messages: threadStore.messages(thread.id), busy, send, stop, regenerate, editResend, wireJson };
}
