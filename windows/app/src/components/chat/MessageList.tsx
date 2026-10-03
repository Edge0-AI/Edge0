// MessageList — full-width list + "waiting for first chunk" placeholder (all threads live locally; no pagination).
import type { ThreadChatApi } from "../../chat/useThreadChat";
import { t } from "../../lib/t";
import { MessageItem } from "./MessageItem";

export function MessageList({ chat }: { chat: ThreadChatApi }) {
  const lastAssistantIdx = chat.messages.map((m) => m.role).lastIndexOf("assistant");
  return (
    <div data-testid="message-list">
      {chat.messages.map((m, i) => (
        <MessageItem
          key={m.id ?? i}
          message={m}
          streaming={chat.busy}
          isLastAssistant={i === lastAssistantIdx}
          busy={chat.busy}
          onRegenerate={() => void chat.regenerate()}
          onEditResend={(next) => void chat.editResend(m.id, next)}
        />
      ))}
      {chat.busy && chat.messages[chat.messages.length - 1]?.role !== "assistant" && (
        <p className="px-3 text-xs text-muted" data-testid="waiting-first-chunk">
          {t("chat.waitingFirstChunk")}
        </p>
      )}
    </div>
  );
}
