// MessageItem — one rendered message: user bubble (right, editable + resend), assistant
// full-width with a drawn avatar, action bar on hover, collapsible reasoning, and a
// measured usage/perf line that renders nothing when keys are missing.
import { Streamdown } from "streamdown";
import { Brain, Check, Copy, Pencil, RotateCcw } from "lucide-react";
import { useState } from "react";

import type { MsgPerf, ThreadMsg } from "../../state/threads";
import { codeOf, errKey, t } from "../../lib/t";

export function MessageItem({
  message,
  streaming,
  isLastAssistant,
  busy,
  onRegenerate,
  onEditResend,
}: {
  message: ThreadMsg;
  streaming: boolean;
  isLastAssistant: boolean;
  busy: boolean;
  onRegenerate: () => void;
  onEditResend: (newText: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [copied, setCopied] = useState<"text" | "reasoning" | null>(null);
  const isUser = message.role === "user";
  const text = message.content;
  const reasoning = message.reasoning ?? "";
  const meta = message.meta ?? {};
  const live = streaming && isLastAssistant && !isUser;
  const isAborted = meta.status === "aborted";
  const isError = meta.status === "error";

  async function copy(value: string, which: "text" | "reasoning") {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(which);
      setTimeout(() => setCopied(null), 1200);
    } catch {
      /* clipboard denied in webview: fail silently, never block the UI */
    }
  }

  if (isUser) {
    return (
      <article data-msg-role="user" className="group/msg flex flex-col items-end py-2.5">
        {editing ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setEditing(false);
              if (draft.trim()) onEditResend(draft);
            }}
            className="w-full max-w-[88%]"
          >
            <textarea
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={3}
              className="w-full resize-y rounded-2xl border border-accent/50 bg-surface2 p-3 text-sm outline-none"
            />
            <div className="flex justify-end gap-2 pt-1.5">
              <button type="submit" className="e0-btn e0-btn-primary">{t("chat.resend")}</button>
              <button type="button" onClick={() => setEditing(false)} className="e0-btn e0-btn-ghost">{t("common.cancel")}</button>
            </div>
          </form>
        ) : (
          <>
            <div className="max-w-[88%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-accent px-4 py-3 text-sm leading-relaxed text-accentink shadow-sm">
              {text}
            </div>
            <div className="mt-0.5 flex h-6 gap-1 opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100">
              {text && (
                <IconAction label={t("chat.copy")} onClick={() => void copy(text, "text")}>
                  {copied === "text" ? <Check size={13} /> : <Copy size={13} />}
                </IconAction>
              )}
              {!busy && (
                <IconAction label={t("chat.editResend")} onClick={() => { setEditing(true); setDraft(text); }}>
                  <Pencil size={13} />
                </IconAction>
              )}
            </div>
          </>
        )}
      </article>
    );
  }

  return (
    <article data-msg-role="assistant" className="group/msg flex gap-3 border-t border-line/40 py-5 first:border-t-0">
      <Avatar />
      <div className="min-w-0 flex-1">
        {reasoning && (
          <details className="mb-3 max-w-full rounded-xl border border-line bg-surface2/70 px-3 py-2" data-testid="reasoning-block">
            <summary className="flex cursor-pointer select-none items-center gap-1.5 text-xs font-medium text-muted">
              <Brain size={13} aria-hidden />
              {t("chat.reasoning")}
            </summary>
            <div className="mt-1.5 grid gap-1.5">
              <div className="whitespace-pre-wrap text-xs leading-relaxed text-muted">{reasoning}</div>
              <button type="button" onClick={() => void copy(reasoning, "reasoning")} className="flex w-fit items-center gap-1 text-xs text-muted hover:text-fg">
                {copied === "reasoning" ? <Check size={12} /> : <Copy size={12} />}
                {t("chat.copyReasoning")}
              </button>
            </div>
          </details>
        )}

        <div
          className="e0-prose prose prose-sm max-w-none break-words text-sm [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
          data-testid="message-body"
        >
          <Streamdown mode={live ? "streaming" : "static"}>{text}</Streamdown>
          {live && !text && <span className="inline-block h-4 w-1 animate-pulse bg-fg align-middle" />}
        </div>

        {isError && (
          <p className="mt-1.5 flex items-center gap-2 rounded-lg bg-err/10 px-2.5 py-1.5 text-xs text-err" role="alert">
            <span>{t(errKey(meta.code))}</span>
            {meta.code && <code className="rounded bg-err/10 px-1">{meta.code}</code>}
          </p>
        )}
        {isAborted && <p className="mt-1 text-xs text-warn">{t("chat.aborted")}</p>}

        {/* per-turn performance metrics: shown as soon as complete, stays; renders nothing when all fields are absent (no-fake-values rule) */}
        {!live && <PerfLine perf={meta.perf} />}

        <div className="mt-0.5 flex h-6 items-center gap-1 opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100">
          {text && (
            <IconAction label={t("chat.copy")} onClick={() => void copy(text, "text")}>
              {copied === "text" ? <Check size={13} /> : <Copy size={13} />}
            </IconAction>
          )}
          {isLastAssistant && !busy && (
            <IconAction label={t("chat.regenerate")} onClick={onRegenerate}>
              <RotateCcw size={13} />
            </IconAction>
          )}
          {meta.usage && (
            <span className="ml-auto text-[11px] tabular-nums text-muted" data-testid="usage-line">
              {t("chat.usage", {
                p: meta.usage.prompt_tokens,
                c: meta.usage.completion_tokens,
                t: meta.usage.total_tokens,
              })}
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

function IconAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick} className="e0-icon-btn !size-7 !rounded-md">
      {children}
    </button>
  );
}

/** Per-turn performance line (timings measured in the final stream chunk): renders nothing when no value is present. */
function PerfLine({ perf }: { perf: MsgPerf | undefined }) {
  const p = (perf ?? {}) as Record<string, number | undefined>;
  const parts: string[] = [];
  if (typeof p.decode_tps === "number") parts.push(t("chat.perf.decode", { v: p.decode_tps.toFixed(1) }));
  if (typeof p.prefill_tps === "number") parts.push(t("chat.perf.prefill", { v: p.prefill_tps.toFixed(1) }));
  if (typeof p.prompt_ms === "number") parts.push(t("chat.perf.prompt", { ms: Math.round(p.prompt_ms) }));
  if (parts.length === 0) return null;
  return (
    <p className="mt-1 flex flex-wrap gap-x-2 text-[11px] tabular-nums text-muted" data-testid="perf-line">
      {parts.map((s, i) => (
        <span key={i}>{s}</span>
      ))}
    </p>
  );
}

/** Drawn monochrome avatar (brand mark). */
function Avatar() {
  return (
    <span aria-hidden className="mt-0.5 grid h-7 w-7 shrink-0 place-content-center rounded-full border border-line bg-surface2 text-fg">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M7 1l1.6 4.4L13 7l-4.4 1.6L7 13 5.4 8.6 1 7l4.4-1.6L7 1z" fill="currentColor" />
      </svg>
    </span>
  );
}
