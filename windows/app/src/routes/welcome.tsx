// routes/welcome.tsx — first-run guide card (shown in the chat area when nothing is
// installed; completes the loop within three clicks). "Download & enable" = one explicit
// intent: create the task + arm the one-shot token (auto-load on completion lives in the
// AppShell chain).
import { Sparkles, Download } from "lucide-react";
import { useState, useSyncExternalStore } from "react";

import { downloadStore, shellStore, startDownload } from "../api/client";
import { autoLoad } from "../lib/autoLoad";
import { humanBytes } from "../lib/byteFormat";
import { codeOf, errKey, t } from "../lib/t";
import { DownloadCardBody } from "../models/cards";

export function WelcomeCard() {
  useSyncExternalStore(downloadStore.sig.sub, downloadStore.sig.version);
  useSyncExternalStore(shellStore.sig.sub, shellStore.sig.version);

  const [dismissed, setDismissed] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const cat = shellStore.catalog;
  const tier = cat["8b"] ? "8b" : Object.keys(cat)[0] ?? null;
  const rec = tier ? cat[tier] : null;
  const task = tier ? downloadStore.tasks[tier] : undefined;
  const path = shellStore.paths.home || null;

  if (dismissed) {
    return (
      <div className="e0-chat-empty">
        <p className="text-lg font-medium">{t("chat.noInstalledModel")}</p>
        <a
          href="#/models"
          className="justify-self-center rounded-full border border-line px-4 py-1.5 text-sm text-muted hover:bg-surface2 hover:text-fg"
        >
          {t("chat.goDownload")}
        </a>
      </div>
    );
  }

  async function downloadAndEnable() {
    if (!tier) return;
    setErr(null);
    setBusy(true);
    try {
      await startDownload(tier, null);
      autoLoad.begin(tier, tier); // task key = tier (shell pull.rs)
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="e0-chat-empty px-6" data-testid="welcome-card">
      <div className="e0-chat-column mx-auto space-y-5 text-center">
        <span className="mx-auto grid size-12 place-items-center rounded-2xl bg-fg text-surface shadow-sm">
          <Sparkles size={20} aria-hidden />
        </span>
        <div>
          <h2 className="text-3xl font-semibold tracking-tight">{t("welcome.title")}</h2>
          <p className="mx-auto mt-2 max-w-lg text-sm leading-relaxed text-muted">
            {t("welcome.body", {
              tier: tier ?? "8b",
              bytes: rec ? humanBytes(rec.total_bytes) : t("welcome.bytesUnknown"),
              path: path ?? t("welcome.pathUnknown"),
            })}
          </p>
        </div>
        {task && task.phase !== "idle" ? (
          <div className="mx-auto w-full max-w-lg text-left">
            <DownloadCardBody task={task} />
          </div>
        ) : (
          <button
            type="button"
            data-testid="welcome-go"
            disabled={busy || !rec}
            onClick={() => void downloadAndEnable()}
            className="e0-btn e0-btn-primary mx-auto min-h-10 px-5 text-sm"
          >
            <Download size={15} aria-hidden />
            {t("welcome.downloadEnable")}
          </button>
        )}
        {err && (
          <p data-testid="welcome-err" role="alert" className="e0-status e0-status-err mx-auto max-w-lg justify-start rounded-xl px-3 py-2 text-left">
            <span>
              <span className="block font-semibold">{t("welcome.errors.generic")}</span>
              <span className="mt-0.5 block font-normal">
                {t(errKey(codeOf(err)))} · {codeOf(err) ?? ""}
              </span>
            </span>
          </p>
        )}
        <div>
          <button
            type="button"
            data-testid="welcome-skip"
            onClick={() => setDismissed(true)}
            className="text-xs text-muted underline hover:text-fg"
          >
            {t("welcome.skip")}
          </button>
        </div>
      </div>
    </div>
  );
}
