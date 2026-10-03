import { Streamdown } from "streamdown";
import { createMathPlugin } from "@streamdown/math";
// Models commonly write `$d$`; leaving single-dollar math off skips those. Currency like "$5 and $10" can false-positive.
const mathPlugin = createMathPlugin({ singleDollarTextMath: true });
import { Brain, Check, Copy, Pencil, RotateCcw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useState } from "react";

import {
  messageReasoning,
  messageText,
  type ChatMessageMeta,
  type ChatUIMessage,
} from "../../chat/records";
import { errorNextKey } from "../../chat/errorMap";

export function MessageItem({
  message,
  streaming,
  isLastAssistant,
  busy,
  onRegenerate,
  onEditResend,
}: {
  message: ChatUIMessage;
  streaming: boolean;
  isLastAssistant: boolean;
  busy: boolean;
  onRegenerate: () => void;
  onEditResend: (newText: string) => void;
}) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [copied, setCopied] = useState<"text" | "reasoning" | null>(null);
  const isUser = message.role === "user";
  const text = messageText(message);
  const reasoning = messageReasoning(message);
  const meta = message.metadata ?? {};
  const live = streaming && isLastAssistant && !isUser;
  const isAborted = meta.status === "aborted" || (live && false);
  const isError = meta.status === "error";

  async function copy(value: string, which: "text" | "reasoning") {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(which);
      setTimeout(() => setCopied(null), 1200);
    } catch {
      /* clipboard may be unavailable */
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
              <button type="submit" className="e0-btn e0-btn-primary">
                {t("chat.resend")}
              </button>
              <button type="button" onClick={() => setEditing(false)} className="e0-btn e0-btn-ghost">
                {t("common.cancel")}
              </button>
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
                <IconAction
                  label={t("chat.editResend")}
                  onClick={() => {
                    setEditing(true);
                    setDraft(text);
                  }}
                >
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
          className="prose prose-sm dark:prose-invert max-w-none break-words text-sm [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
          data-testid="message-body"
        >
          <Streamdown mode={live ? "streaming" : "static"} plugins={{ math: mathPlugin }}>{text}</Streamdown>
          {live && !text && <span className="inline-block h-4 w-1 animate-pulse bg-fg align-middle" />}
        </div>

        {isError && meta.code && (
          <p className="mt-1.5 flex items-center gap-2 rounded-lg bg-err/10 px-2.5 py-1.5 text-xs text-err" role="alert">
            <span>{t(errorNextKey(meta.code))}</span>
            <code className="rounded bg-err/10 px-1">{meta.code}</code>
          </p>
        )}
        {isAborted && <p className="mt-1 text-xs text-warn">{t("chat.aborted")}</p>}
        {meta.saveFailed && <p className="mt-1 text-xs text-warn">{t("chat.saveFailed")}</p>}

        
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
              {t("chat.usage", { p: meta.usage.prompt_tokens, c: meta.usage.completion_tokens, t: meta.usage.total_tokens })}
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

function IconAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="e0-icon-btn !size-7 !rounded-md"
    >
      {children}
    </button>
  );
}

function PerfLine({ perf }: { perf: ChatMessageMeta["perf"] }) {
  const { t } = useTranslation();
  if (!perf || !Object.values(perf).some((v) => typeof v === "number")) return null;
  return (
    <p className="mt-1 flex flex-wrap gap-x-2 text-[11px] tabular-nums text-muted" data-testid="perf-line">
      {typeof perf.decode_tps === "number" && <span>{t("chat.perf.decode", { v: perf.decode_tps })}</span>}
      {typeof perf.prefill_tps === "number" && <span>{t("chat.perf.prefill", { v: perf.prefill_tps })}</span>}
      {typeof perf.ttft_ms === "number" && <span>{t("chat.perf.ttft", { ms: perf.ttft_ms })}</span>}
      {typeof perf.footprint_mb === "number" && (
        <span>{t("chat.perf.mem", { gi: (perf.footprint_mb / 1024).toFixed(2) })}</span>
      )}
    </p>
  );
}

function Avatar() {
  return (
    <span
      aria-hidden
      className="mt-0.5 grid h-7 w-7 shrink-0 place-content-center rounded-full border border-line bg-surface2 text-fg"
    >
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M7 1l1.6 4.4L13 7l-4.4 1.6L7 13 5.4 8.6 1 7l4.4-1.6L7 1z" fill="currentColor" />
      </svg>
    </span>
  );
}
