// AppShell — sidebar + full-width main area (Atomic-Chat style three-segment layout).
// Hash routing; state comes from the single-pulse stores in api/client.
// autoLoad intent chain: download ready -> issue one load -> on success create a thread and enter chat.
import { BookOpen, Menu, Plus, Server, Settings, Sparkles, X } from "lucide-react";
import { useEffect, useState, useSyncExternalStore, type ReactNode } from "react";

import {
  downloadStore,
  engineStore,
  loadModel,
  refreshInstalled,
  shellStore,
} from "../api/client";
import { autoLoad } from "../lib/autoLoad";
import { t } from "../lib/t";
import { bucketThreads } from "../chat/buckets";
import { threadStore } from "../state/threads";
import { SidebarThreadItem } from "./SidebarThreadItem";

const go = (path: string) => {
  window.location.hash = `#${path}`;
};

export function AppShell({ view, children }: { view: string; children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const threadsVer = useSyncExternalStore(threadStore.subscribe, () => threadStore.version());
  const engVer = useSyncExternalStore(engineStore.sig.sub, engineStore.sig.version);
  const dlVer = useSyncExternalStore(downloadStore.sig.sub, downloadStore.sig.version);
  const shellVer = useSyncExternalStore(shellStore.sig.sub, shellStore.sig.version);
  void threadsVer; void engVer; void dlVer; void shellVer; // pulse-driven rerenders; data read straight from stores

  // autoLoad one-shot intent chain: ready -> explicitly issue one load -> on success create thread and enter chat.
  useEffect(() => {
    const check = () => {
      const cur = autoLoad.current;
      if (!cur) return;
      if (cur.phase === "download") {
        const tv = downloadStore.tasks[cur.tier];
        if (tv && (tv.phase === "error" || tv.phase === "cancelled")) autoLoad.cancel();
        else if (tv?.phase === "ready") {
          const tier = autoLoad.claimCompleted(cur.tier);
          if (tier) {
            void (async () => {
              await refreshInstalled();
              try {
                await loadModel(tier);
              } catch {
                autoLoad.cancel();
              }
            })();
          }
        }
      } else {
        const st = engineStore.status;
        if (st?.running && st.tier === cur.tier && autoLoad.consume(cur.tier)) {
          const row = threadStore.create(t("chat.newThread"), cur.tier);
          threadStore.setActive(row.id);
          go("/"); // entering a session means returning to the chat page
        } else if (!st?.running && engineStore.loadingTier === null) {
          autoLoad.cancel(); // load failed (engine not running and not loading)
        }
      }
    };
    const off1 = downloadStore.sig.sub(check);
    const off2 = engineStore.sig.sub(check);
    const off3 = autoLoad.subscribe(check);
    check();
    return () => {
      off1();
      off2();
      off3();
    };
  }, []);

  const threads = threadStore.threads();
  const activeId = threadStore.activeId;
  const buckets = bucketThreads(threads, Date.now());
  const st = engineStore.status;
  const running = !!st?.running;
  const serviceLine = running
    ? `:${st?.port} · ${t("sidebar.resident")} ${st?.tier}`
    : st
      ? t("sidebar.idle")
      : t("sidebar.noService");

  function newChat() {
    const row = threadStore.create(t("chat.newThread"), running ? (st?.tier ?? null) : null);
    threadStore.setActive(row.id);
    go("/"); // "New chat" clicked from a non-chat page must land back on the chat page
    setSidebarOpen(false);
  }

  const navCls = (active: boolean) =>
    "flex items-center gap-2 rounded-lg px-3 py-2 text-sm " +
    (active ? "bg-surface2 text-fg" : "text-muted hover:bg-surface2 hover:text-fg");

  return (
    <div className="e0-app-shell text-fg">
      {sidebarOpen && (
        <button type="button" aria-label={t("nav.closeSidebar")} className="e0-sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}
      <aside data-testid="app-sidebar" className={"e0-sidebar " + (sidebarOpen ? "e0-sidebar-open" : "")} data-open={sidebarOpen ? "true" : "false"}>
        <div className="flex items-center justify-between px-4 pb-2 pt-4">
          <div className="flex items-center gap-2">
            <span className="grid size-8 place-items-center rounded-xl bg-fg text-surface shadow-sm">
              <Sparkles size={16} aria-hidden />
            </span>
            <div>
              <span className="block text-sm font-bold tracking-tight">{t("app.name")}</span>
              <span className="block text-[10px] font-medium uppercase tracking-[0.16em] text-muted">{t("app.localLabel")}</span>
            </div>
          </div>
          <button type="button" aria-label={t("nav.closeSidebar")} className="e0-icon-btn md:hidden" onClick={() => setSidebarOpen(false)}>
            <X size={16} aria-hidden />
          </button>
        </div>
        <div className="px-3 pb-3">
          <button type="button" onClick={newChat} className="e0-btn e0-btn-primary w-full">
            <Plus size={16} aria-hidden />
            {t("sidebar.newChat")}
          </button>
        </div>
        <nav aria-label={t("sidebar.navigation")} className="space-y-1 px-2 pb-2">
          <a href="#/models" aria-label={t("nav.models")} data-testid="nav-models" className={navCls(view === "/models")} onClick={() => setSidebarOpen(false)}>
            <BookOpen size={16} aria-hidden />
            {t("nav.models")}
          </a>
          <a href="#/service" aria-label={t("nav.service")} data-testid="nav-service" className={navCls(view === "/service")} onClick={() => setSidebarOpen(false)}>
            <Server size={16} aria-hidden />
            {t("nav.service")}
          </a>
        </nav>
        <nav aria-label={t("sidebar.history")} className="min-h-0 flex-1 overflow-y-auto px-2">
          <h2 className="px-3 pb-2 pt-2 text-[10px] font-bold uppercase tracking-[0.16em] text-muted">{t("sidebar.history")}</h2>
          {buckets.map((b) => (
            <section key={b.key} className="mb-2">
              <h2 className="px-2 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wide text-muted">
                {t(`sidebar.bucket.${b.key}`)}
              </h2>
              <ul className="space-y-0.5">
                {b.items.map((th) => (
                  <SidebarThreadItem
                    key={th.id}
                    thread={th}
                    active={th.id === activeId}
                    onSelect={() => {
                      threadStore.setActive(th.id);
                      setSidebarOpen(false);
                      go("/");
                    }}
                    onRename={(title) => threadStore.rename(th.id, title)}
                    onDelete={() => threadStore.remove(th.id)}
                  />
                ))}
              </ul>
            </section>
          ))}
          {threads.length === 0 && <p className="px-3 py-3 text-xs leading-relaxed text-muted">{t("chat.noThreads")}</p>}
        </nav>
        <div className="e0-divider mx-3" />
        <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted">
          <span aria-label={serviceLine} className={"inline-block size-2 shrink-0 rounded-full " + (running ? "bg-ok" : st ? "bg-warn" : "bg-muted")} />
          <span className="min-w-0 flex-1 truncate" data-testid="service-summary">{serviceLine}</span>
          <a href="#/settings" aria-label={t("nav.settings")} className="e0-icon-btn ml-auto" onClick={() => setSidebarOpen(false)}>
            <Settings size={16} aria-hidden />
          </a>
        </div>
      </aside>
      <main data-testid="app-main" className="e0-main-frame">
        <div className="e0-mobile-toggle absolute left-4 top-4 z-10">
          <button type="button" aria-label={t("nav.openSidebar")} title={t("nav.openSidebar")}
            className="e0-icon-btn border border-line bg-surface shadow-sm" onClick={() => setSidebarOpen(true)}>
            <Menu size={17} aria-hidden />
          </button>
        </div>
        <div className="h-full min-h-0 overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}
