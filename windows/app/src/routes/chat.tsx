// routes/chat.tsx — chat main area: centered max-width message column + floating bottom
// composer; no selected thread = big-title empty state; nothing installed = WelcomeCard;
// the boot snapshot decides the waiting shape.
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

import { engineStore, installedTiers, shellStore } from "../api/client";
import { useThreadChat } from "../chat/useThreadChat";
import { isAtBottom } from "../chat/scroll";
import { t } from "../lib/t";
import { storedSystemPrompt, subscribeSystemPrompt } from "../lib/systemPrompt";
import { threadStore, type ThreadRow } from "../state/threads";
import { ComposerBar } from "../components/chat/ComposerBar";
import { MessageList } from "../components/chat/MessageList";
import { WelcomeCard } from "./welcome";

export function ChatPage() {
  useSyncExternalStore(threadStore.subscribe, () => threadStore.version());
  useSyncExternalStore(engineStore.sig.sub, engineStore.sig.version);
  useSyncExternalStore(shellStore.sig.sub, shellStore.sig.version);

  const st = engineStore.status;
  const active = threadStore.threads().find((x) => x.id === threadStore.activeId) ?? null;
  const hasInstalled = installedTiers().length > 0;

  if (!st) {
    return (
      <div className="e0-chat-empty text-sm text-muted">
        <span className="mx-auto grid size-10 place-items-center rounded-2xl bg-surface2 text-fg">
          <span className="size-2 rounded-full bg-warn animate-pulse" aria-hidden />
        </span>
        <p className="font-medium text-fg">{t("service.panel.idle")}</p>
        <p className="text-xs">{t("platform.note")}</p>
      </div>
    );
  }
  if (!hasInstalled) return <WelcomeCard />;
  if (!active) {
    return (
      <div className="e0-chat-empty">
        <span className="mx-auto grid size-12 place-items-center rounded-2xl bg-fg text-surface shadow-sm">
          <span className="text-lg font-semibold" aria-hidden>+</span>
        </span>
        <h2 className="mt-1 text-3xl font-semibold tracking-tight">{t("chat.greeting")}</h2>
        <p className="text-sm text-muted">{t("chat.pickOrCreate")}</p>
      </div>
    );
  }
  return <ChatPane key={active.id} thread={active} />;
}

function ChatPane({ thread }: { thread: ThreadRow }) {
  const st = engineStore.status;
  const [model, setModel] = useState(thread.data.model ?? (st?.running ? st.tier ?? "" : ""));
  useSyncExternalStore(subscribeSystemPrompt, () => storedSystemPrompt(), () => storedSystemPrompt());
  const scrollerRef = useRef<HTMLDivElement>(null);
  // Follow-bottom: pin to the bottom on streaming increments only while the user is already near the bottom.
  const stickBottom = useRef(true);

  const chat = useThreadChat(thread, model || null);

  const onModelChange = useCallback(
    (id: string) => {
      setModel(id);
      threadStore.setModel(thread.id, id);
    },
    [thread.id],
  );

  useEffect(() => {
    const last = chat.messages[chat.messages.length - 1];
    // right after the user sends: always jump to bottom (clear intent)
    if (last?.role === "user") stickBottom.current = true;
    const el = scrollerRef.current;
    if (el && stickBottom.current) el.scrollTo({ top: el.scrollHeight });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.messages]);

  const onScrollerScroll = () => {
    const el = scrollerRef.current;
    if (el) stickBottom.current = isAtBottom(el);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="e0-chat-header">
        <div className="min-w-0">
          <div className="e0-chat-header-title">{thread.data.title}</div>
          <div className="e0-chat-header-meta">
            <span className={"size-2 rounded-full " + (st?.running ? "bg-ok" : "bg-muted")} aria-hidden />
            {t("chat.serviceShell")}
          </div>
        </div>
        <span className="hidden text-xs text-muted sm:inline">{model || t("chat.noModel")}</span>
      </header>
      <div ref={scrollerRef} onScroll={onScrollerScroll} className="min-h-0 flex-1 overflow-y-auto">
        <div className="e0-chat-column space-y-1 px-5 pb-4 pt-7 sm:px-6">
          <MessageList chat={chat} />
        </div>
      </div>
      <ComposerBar chat={chat} model={model} onModelChange={onModelChange} />
    </div>
  );
}
