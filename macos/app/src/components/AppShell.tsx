import { Link, Outlet, useRouter } from "@tanstack/react-router";
import { BookOpen, Home, Menu, Plus, Server, Settings, Sparkles, X } from "lucide-react";
import { useEffect, useState, useSyncExternalStore } from "react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../bridge/state";
import { edge0Fetch } from "../bridge/edge0Fetch";
import { threadCreate, threadDelete, threadRename } from "../chat/ipc";
import { bucketThreads } from "../chat/buckets";
import { threadStore } from "../state/threads";
import { autoLoad } from "../models/autoLoad";
import { refreshSystem } from "../models/system";
import { SidebarThreadItem } from "./SidebarThreadItem";

export function AppShell() {
  const { t } = useTranslation();
  const router = useRouter();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const threadsState = useSyncExternalStore(
    (cb) => threadStore.subscribe(cb),
    () => threadStore.threads,
  );
  const activeId = useSyncExternalStore(
    (cb) => threadStore.subscribe(cb),
    () => threadStore.activeId,
  );
  const snapTick = useSyncExternalStore(
    (cb) => daemonStore.subscribe(cb),
    () => daemonStore.snapshot,
  );

  useEffect(() => {
    void threadStore.refresh();
  }, []);

  useEffect(() => {
    const check = () => {
      const cur = autoLoad.current;
      if (!cur) return;
      if (cur.phase === "download") {
        const comp = daemonStore.latestFolded("download.completed", cur.taskId);
        const fail = daemonStore.latestFolded("download.failed", cur.taskId);
        if (fail && (!comp || fail.seq >= comp.seq)) {
          autoLoad.cancel();
          return;
        }
        if (comp) {
          const tier = autoLoad.claimCompleted(comp.subject);
          if (tier) {
            void (async () => {
              await refreshSystem();
              try {
                const res = await edge0Fetch(`/v1/edge0/models/${tier}/load`, {
                  method: "POST",
                  body: {},
                });
                if (!res.ok) autoLoad.cancel();
              } catch {
                autoLoad.cancel();
              }
            })();
          }
        }
      } else {
        const ready = daemonStore.latestFolded("model.load.ready", cur.tier);
        const fail = daemonStore.latestFolded("model.load.failed", cur.tier);
        if (fail) {
          autoLoad.cancel();
          return;
        }
        if (ready && autoLoad.consume(ready.subject)) {
          void (async () => {
            await refreshSystem();
            const row = await threadCreate(t("chat.newThread"), cur.tier);
            await threadStore.refresh();
            threadStore.setActive(row.id);
            void router.navigate({ to: "/" });
          })();
        }
      }
    };
    const off1 = daemonStore.subscribe(check);
    const off2 = autoLoad.subscribe(check);
    check();
    return () => {
      off1();
      off2();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const snap = snapTick;
  const buckets = bucketThreads(threadsState, Date.now());
  const resident = snap?.models.find((m) => m.state === "resident");
  const serviceLine = snap
    ? `:${snap.daemon.port} - ${resident ? `${t("sidebar.resident")} ${resident.id}` : t("sidebar.idle")}`
    : t("sidebar.noService");

  async function newChat() {
    const defaultModel = (resident ?? snap?.models[0])?.id ?? null;
    const row = await threadCreate(t("chat.newThread"), defaultModel);
    await threadStore.refresh();
    threadStore.setActive(row.id);
    void router.navigate({ to: "/" });
  }

  function goHome() {
    threadStore.setActive(null);
    setSidebarOpen(false);
    void router.navigate({ to: "/" });
  }

  return (
    <div className="e0-app-shell text-fg">
      {sidebarOpen && (
        <button
          type="button"
          aria-label={t("nav.closeSidebar")}
          className="e0-sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        data-testid="app-sidebar"
        className={"e0-sidebar " + (sidebarOpen ? "e0-sidebar-open" : "")}
        data-open={sidebarOpen ? "true" : "false"}
      >
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
          <button
            type="button"
            aria-label={t("nav.closeSidebar")}
            className="e0-icon-btn md:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={16} aria-hidden />
          </button>
        </div>
        <div className="px-3 pb-3">
          <button
            type="button"
            onClick={() => void newChat()}
            className="e0-btn e0-btn-primary w-full"
          >
            <Plus size={16} aria-hidden />
            {t("sidebar.newChat")}
          </button>
        </div>
        <nav aria-label={t("sidebar.navigation")} className="space-y-1 px-2 pb-2">
          <button
            type="button"
            aria-label={t("nav.home")}
            data-testid="nav-home"
            onClick={goHome}
            className={
              "flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-surface2 hover:text-fg " +
              (activeId === null && router.state.location.pathname === "/" ? "bg-surface2 text-fg" : "text-muted")
            }
          >
            <Home size={16} aria-hidden />
            {t("nav.home")}
          </button>
          <Link
            to="/models"
            aria-label={t("nav.models")}
            data-testid="nav-models"
            activeProps={{ className: "bg-surface2 text-fg" }}
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted hover:bg-surface2 hover:text-fg"
            onClick={() => setSidebarOpen(false)}
          >
            <BookOpen size={16} aria-hidden />
            {t("nav.models")}
          </Link>
          <Link
            to="/service"
            aria-label={t("nav.service")}
            data-testid="nav-service"
            activeProps={{ className: "bg-surface2 text-fg" }}
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted hover:bg-surface2 hover:text-fg"
            onClick={() => setSidebarOpen(false)}
          >
            <Server size={16} aria-hidden />
            {t("nav.service")}
          </Link>
        </nav>
        <nav aria-label={t("sidebar.history")} className="min-h-0 flex-1 overflow-y-auto px-2">
          <h2 className="px-3 pb-2 pt-2 text-[10px] font-bold uppercase tracking-[0.16em] text-muted">
            {t("sidebar.history")}
          </h2>
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
                      void router.navigate({ to: "/" });
                    }}
                    onRename={(title) => void threadRename(th.id, title).then(() => threadStore.refresh())}
                    onDelete={() =>
                      void threadDelete(th.id).then(() => {
                        if (activeId === th.id) threadStore.setActive(null);
                        void threadStore.refresh();
                      })
                    }
                  />
                ))}
              </ul>
            </section>
          ))}
          {threadsState.length === 0 && <p className="px-3 py-3 text-xs leading-relaxed text-muted">{t("chat.noThreads")}</p>}
        </nav>
        <div className="e0-divider mx-3" />
        <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted">
          <span
            aria-label={serviceLine}
            className={
              "inline-block size-2 shrink-0 rounded-full " +
              (snap ? "bg-ok" : "bg-muted")
            }
          />
          <span className="min-w-0 flex-1 truncate" data-testid="service-summary">
            {serviceLine}
          </span>
          <Link
            to="/settings"
            aria-label={t("nav.settings")}
            className="e0-icon-btn ml-auto"
            onClick={() => setSidebarOpen(false)}
          >
            <Settings size={16} aria-hidden />
          </Link>
        </div>
      </aside>
      <main data-testid="app-main" className="e0-main-frame">
        <div className="e0-mobile-toggle absolute left-4 top-4 z-10">
          <button
            type="button"
            aria-label={t("nav.openSidebar")}
            title={t("nav.openSidebar")}
            className="e0-icon-btn border border-line bg-surface shadow-sm"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={17} aria-hidden />
          </button>
        </div>
        <div className="h-full min-h-0 overflow-y-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
