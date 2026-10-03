import { useChat } from "@ai-sdk/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  messageFinish,
  messageAppend,
  messagesDeleteFrom,
  messagesPage,
  messageStart,
  newMessageId,
  threadRename,
  type ThreadRow,
} from "./ipc";
import { rowToUiMessage, type ChatUIMessage } from "./records";
import { createEdge0Transport } from "./transport";

export interface ThreadChatApi {
  messages: ChatUIMessage[];
  status: "submitted" | "streaming" | "ready" | "error";
  send: (text: string) => Promise<void>;
  stop: () => void;
  regenerate: () => Promise<void>;
  editResend: (messageId: string, newText: string) => Promise<void>;
  loadEarlier: () => Promise<boolean>;
  hasMoreEarlier: boolean;
  historyLoaded: boolean;
  busy: boolean;
}

export function useThreadChat(
  thread: ThreadRow,
  model: string,
  onThreadChanged: () => void,
  onRoundSettled?: (assistantText: string) => void,
  systemPrompt?: string | null,
): ThreadChatApi {
  const [hasMoreEarlier, setHasMoreEarlier] = useState(false);
  const oldestRef = useRef<{ createdAt: number; id: string } | null>(null);
  const modelRef = useRef(model);
  modelRef.current = model;

  const transport = useMemo(
    () =>
      createEdge0Transport({
        getModel: () =>
          modelRef.current
            ? { model: modelRef.current, params: {}, systemPrompt }
            : null,
        getPersistence: () => ({
          beginAssistant: async () => {
            const row = await messageStart({
              threadId: thread.id,
              role: "assistant",
              content: "",
              model: modelRef.current,
            });
            return row.id;
          },
          append: (id, c, r) => messageAppend(id, c, r),
          finish: (id, s, code, usage) => messageFinish(id, s, code, usage).then(() => undefined),
          dropAssistant: (fromId) => messagesDeleteFrom(thread.id, fromId).then(() => undefined),
        }),
      }),
    [thread.id, systemPrompt],
  );

  const { messages, sendMessage, regenerate: chatRegenerate, stop, setMessages, status } = useChat<ChatUIMessage>({
    id: thread.id,
    messages: [],
    transport,
    onFinish: () => {
      onThreadChanged();
      onRoundSettledRef.current?.("");
    },
  });

  const [historyLoaded, setHistoryLoaded] = useState(false);
  useEffect(() => {
    setHistoryLoaded(false);
    void (async () => {
      const rows = await messagesPage({ threadId: thread.id, limit: 50 });
      setHasMoreEarlier(rows.length === 50);
      oldestRef.current = rows.length > 0 ? { createdAt: rows[0].createdAt, id: rows[0].id } : null;
      const list = rows.map(rowToUiMessage);
      if (list.length > 0) {
        setMessages((cur) => (cur.length === 0 ? list : cur));
      }
      setHistoryLoaded(true);
    })();
  }, [thread.id, setMessages]);

  const onRoundSettledRef = useRef(onRoundSettled);
  onRoundSettledRef.current = onRoundSettled;

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const id = newMessageId();
      await messageStart({
        threadId: thread.id,
        role: "user",
        content: trimmed,
        model: modelRef.current,
        id,
      });
      await messageFinish(id, "done", null, null);
      if (messages.length === 0) {
        const t = trimmed.slice(0, 30);
        await threadRename(thread.id, trimmed.length > 30 ? t + "…" : t).catch(() => undefined);
      }
      await sendMessage({ id, role: "user", parts: [{ type: "text", text: trimmed }] });
      onThreadChanged();
    },
    [sendMessage, thread.id, onThreadChanged, messages.length],
  );

  const regenerate = useCallback(async () => {
    await chatRegenerate();
    onThreadChanged();
  }, [chatRegenerate, onThreadChanged]);

  const editResend = useCallback(
    async (messageId: string, newText: string) => {
      const idx = messages.findIndex((m) => m.id === messageId);
      if (idx < 0) return;
      await messagesDeleteFrom(thread.id, messageId);
      setMessages(messages.slice(0, idx));
      await send(newText);
    },
    [messages, setMessages, send, thread.id],
  );

  const loadEarlier = useCallback(async (): Promise<boolean> => {
    const before = oldestRef.current;
    if (!before) return false;
    const rows = await messagesPage({ threadId: thread.id, beforeCreatedAt: before.createdAt, beforeId: before.id, limit: 50 });
    if (rows.length === 0) {
      setHasMoreEarlier(false);
      return false;
    }
    oldestRef.current = { createdAt: rows[0].createdAt, id: rows[0].id };
    const older = rows.map(rowToUiMessage);
    setMessages((cur) => [...older, ...cur]);
    setHasMoreEarlier(rows.length === 50);
    return true;
  }, [setMessages, thread.id]);

  return {
    messages,
    status,
    send,
    stop,
    regenerate,
    editResend,
    loadEarlier,
    hasMoreEarlier,
    historyLoaded,
    busy: status === "streaming" || status === "submitted",
  };
}
