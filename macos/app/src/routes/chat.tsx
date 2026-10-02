import { useSyncExternalStore } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../bridge/state";
import type { SystemSnapshot } from "../gen";
import { useThreadChat } from "../chat/useThreadChat";
import { threadCreate, threadSetModel, type ThreadRow } from "../chat/ipc";
import { type ChatUIMessage } from "../chat/records";
import { toWireMessages } from "../chat/transport";
import { threadStore } from "../state/threads";
import { isAtBottom } from "../chat/scroll";
import { MessageList } from "../components/chat/MessageList";
import { ComposerBar } from "../components/chat/ComposerBar";
import { WelcomeCard } from "./welcome";
import { storedSystemPrompt, subscribeSystemPrompt } from "../lib/systemPrompt";

export function ChatPage() {
  const { t } = useTranslation();
  const threads = useSyncExternalStore((cb) => threadStore.subscribe(cb), () => threadStore.threads);
  const activeId = useSyncExternalStore((cb) => threadStore.subscribe(cb), () => threadStore.activeId);
  const snap = useSyncExternalStore((cb) => daemonStore.subscribe(cb), () => daemonStore.snapshot);
  const [seed, setSeed] = useState<{ threadId: string; text: string } | null>(null);

  const active = threads.find((x) => x.id === activeId) ?? null;
  const hasInstalled = (snap?.models ?? []).length > 0;

  if (!snap) {
    return (
      <div className="e0-chat-empty text-sm text-muted">
        <span className="mx-auto grid size-10 place-items-center rounded-2xl bg-surface2 text-fg">
          <span className="size-2 rounded-full bg-warn animate-pulse" aria-hidden />
        </span>
        <p className="font-medium text-fg">{t("chat.waitingService")}</p>
        <p className="text-xs">{t("platform.note")}</p>
      </div>
    );
  }
  if (!hasInstalled) {
    return <WelcomeCard />;
  }
  if (!active) {
    const promptSuggestions = [
      t("chat.suggestion.water"),
      t("chat.suggestion.story"),
      t("chat.suggestion.shirts"),
      t("chat.suggestion.averageSpeed"),
    ];
    const sendSuggestion = async (text: string) => {
      try {
        const row = await threadCreate(t("chat.newThread"), defaultModel(snap));
        await threadStore.refresh();
        setSeed({ threadId: row.id, text });
        threadStore.setActive(row.id);
      } catch {
        setSeed(null);
      }
    };
    return (
      <div className="e0-chat-empty">
        <span className="mx-auto grid size-12 place-items-center rounded-2xl bg-fg text-surface shadow-sm">
          <span className="text-lg font-semibold" aria-hidden>+</span>
        </span>
        <h2 className="mt-1 text-3xl font-semibold tracking-tight">{t("chat.greeting")}</h2>
        <p className="text-sm text-muted">{t("chat.pickOrCreate")}</p>
        <div className="mx-auto mt-4 grid w-full max-w-[760px] gap-2 sm:grid-cols-2" data-testid="chat-suggestions">
          {promptSuggestions.map((text) => (
            <button
              key={text}
              type="button"
              onClick={() => void sendSuggestion(text)}
              className="rounded-xl border border-line bg-surface2/60 px-4 py-3 text-left text-sm leading-snug text-fg/80 transition-colors hover:bg-surface2 hover:text-fg"
            >
              {text}
            </button>
          ))}
        </div>
      </div>
    );
  }
  return (
    <ChatPane
      key={active.id}
      thread={active}
      snapshot={snap}
      initialPrompt={seed?.threadId === active.id ? seed.text : null}
      onInitialConsumed={() => setSeed((s) => (s?.threadId === active.id ? null : s))}
      onChanged={() => void threadStore.refresh()}
    />
  );
}

function ChatPane({
  thread,
  snapshot,
  initialPrompt,
  onInitialConsumed,
  onChanged,
}: {
  thread: ThreadRow;
  snapshot: SystemSnapshot;
  initialPrompt: string | null;
  onInitialConsumed: () => void;
  onChanged: () => void;
}) {
  const { t } = useTranslation();
  const [model, setModel] = useState(thread.data.model ?? defaultModel(snapshot) ?? "");
  const systemPrompt = useSyncExternalStore(
    subscribeSystemPrompt,
    storedSystemPrompt,
    storedSystemPrompt,
  );
  const scrollerRef = useRef<HTMLDivElement>(null);
  const stickBottom = useRef(true);
  const [meterNonce, setMeterNonce] = useState(0);

  const chat = useThreadChat(thread, model, () => {
    onChanged();
    setMeterNonce((n) => n + 1);
  }, undefined, systemPrompt);

  const onModelChange = useCallback(
    (id: string) => {
      setModel(id);
      void threadSetModel(thread.id, id);
    },
    [thread.id],
  );

  const seedSentRef = useRef(false);
  useEffect(() => {
    // History must finish filling first, or the refill treats the in-flight seed as empty UI and duplicates it.
    if (!initialPrompt || !chat.historyLoaded || seedSentRef.current) return;
    seedSentRef.current = true;
    onInitialConsumed();
    void chat.send(initialPrompt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPrompt, chat.historyLoaded]);

  const wireJson = useCallback(
    () => toWireMessages(chat.messages as ChatUIMessage[] as never, systemPrompt),
    [chat.messages, systemPrompt],
  );

  useEffect(() => {
    // Stick to the bottom only when the user was already near it (or just sent).
    const last = chat.messages[chat.messages.length - 1];
    if (last?.role === "user") stickBottom.current = true;
    const el = scrollerRef.current;
    if (el && stickBottom.current) el.scrollTo({ top: el.scrollHeight });
  }, [chat.messages]);

  const onScrollerScroll = () => {
    const el = scrollerRef.current;
    if (el) stickBottom.current = isAtBottom(el);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="e0-chat-header">
        <div className="min-w-0 overflow-hidden">{/* min-w-0 is required for title ellipsis */}
          <div className="e0-chat-header-title">{thread.data.title}</div>
          <div className="e0-chat-header-meta">
            <span className={"size-2 rounded-full " + (snapshot ? "bg-ok" : "bg-muted")} aria-hidden />
            {snapshot.daemon.scope === "launchagent" ? t("chat.serviceAgent") : t("chat.serviceMenu")}
          </div>
        </div>
        <span className="hidden text-xs text-muted sm:inline">{model || t("chat.noModel")}</span>
      </header>
      <div ref={scrollerRef} onScroll={onScrollerScroll} className="min-h-0 flex-1 overflow-y-auto">
        <div className="e0-chat-column space-y-1 px-5 pb-4 pt-7 sm:px-6">
          {chat.hasMoreEarlier && (
            <button
              type="button"
              onClick={() => void chat.loadEarlier()}
              className="mx-auto block rounded-full px-3 py-1 text-xs text-muted hover:bg-surface2 hover:text-fg"
            >
              {t("chat.loadEarlier")}
            </button>
          )}
          <MessageList chat={chat} />
        </div>
      </div>
      <ComposerBar
        chat={chat}
        model={model}
        onModelChange={onModelChange}
        wireJson={wireJson}
        meterNonce={meterNonce}
      />
    </div>
  );
}

function defaultModel(snap: SystemSnapshot | null): string | null {
  const resident = snap?.models.find((m) => m.state === "resident");
  return (resident ?? snap?.models[0])?.id ?? null;
}
