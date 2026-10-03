// ComposerBar — floating rounded composer: Enter sends / Shift+Enter newlines,
// the button becomes Stop while generating, ModelPill + GenPill on the left.
import { ArrowUp, Square } from "lucide-react";
import { useState } from "react";

import type { ThreadChatApi } from "../../chat/useThreadChat";
import { t } from "../../lib/t";
import { GenPill } from "./GenPill";
import { ModelPill } from "./ModelPill";

export function ComposerBar({
  chat,
  model,
  onModelChange,
}: {
  chat: ThreadChatApi;
  model: string;
  onModelChange: (id: string) => void;
}) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);

  async function submit() {
    const trimmed = text.trim();
    if (!trimmed || sending) return;
    setSending(true);
    setText("");
    try {
      await chat.send(trimmed);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="e0-composer" data-testid="composer">
      <div className="e0-composer-shell">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void submit();
            }
          }}
          rows={2}
          placeholder={t("chat.placeholder")}
          className="e0-composer-textarea w-full resize-none bg-transparent text-sm outline-none placeholder:text-muted"
        />
        <div className="e0-composer-toolbar">
          <ModelPill model={model} onModelChange={onModelChange} wireJson={chat.wireJson} />
          <GenPill />
          <span className="min-w-0 flex-1 truncate text-xs text-muted">{!model ? t("chat.pickModelFirst") : ""}</span>
          {chat.busy ? (
            <button
              type="button"
              aria-label={t("chat.stop")}
              title={t("chat.stop")}
              onClick={chat.stop}
              className="e0-send-btn bg-fg text-surface"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              type="button"
              aria-label={t("chat.send")}
              title={t("chat.send")}
              onClick={() => void submit()}
              disabled={!model || !text.trim()}
              className="e0-send-btn disabled:opacity-40"
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
