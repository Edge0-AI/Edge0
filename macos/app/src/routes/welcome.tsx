import { useEffect, useState } from "react";
import { Link } from "@tanstack/react-router";
import { invoke } from "@tauri-apps/api/core";
import { Download, Sparkles } from "lucide-react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../bridge/state";
import { edge0Fetch } from "../bridge/edge0Fetch";
import { ApiError } from "../bridge/frames";
import { catalogStore } from "../models/catalog";
import { autoLoad } from "../models/autoLoad";
import { refreshSystem, taskView } from "../models/system";
import { downloadErrKey } from "../models/errMap";
import { humanBytes } from "../models/byteFormat";
import { DownloadCardForTier } from "../models/cards";

export function WelcomeCard() {
  const { t } = useTranslation();
  const [, force] = useState(0);
  useEffect(() => {
    const off1 = daemonStore.subscribe(() => force((n) => n + 1));
    const off2 = catalogStore.subscribe(() => force((n) => n + 1));
    const off3 = autoLoad.subscribe(() => force((n) => n + 1));
    return () => {
      off1();
      off2();
      off3();
    };
  }, []);
  useEffect(() => {
    void catalogStore.refresh();
  }, []);
  const [dismissed, setDismissed] = useState(false);
  const [err, setErr] = useState<{ code: string | null; message: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [headless, setHeadless] = useState(false);

  const snap = daemonStore.snapshot;
  const rec = catalogStore.data?.tiers.find((x) => x.tier === "edge0-8b") ?? null;
  const task = taskView(snap, daemonStore, "edge0-8b");

  if (dismissed) {
    return (
      <div className="e0-chat-empty">
        <p className="text-lg font-medium">{t("chat.noInstalledModel")}</p>
        <Link
          to="/models"
          className="justify-self-center rounded-full border border-line px-4 py-1.5 text-sm text-muted hover:bg-surface2 hover:text-fg"
        >
          {t("chat.goDownload")}
        </Link>
      </div>
    );
  }

  async function downloadAndEnable() {
    setErr(null);
    setBusy(true);
    try {
      if (headless) {
        await invoke("agent_set", { enable: true });
      }
      const res = await edge0Fetch("/v1/edge0/downloads", {
        method: "POST",
        body: { tier: "edge0-8b", source: null },
      });
      if (!res.ok) throw await errFrom(res);
      const j = (await res.json()) as { id: string };
      autoLoad.begin("edge0-8b", j.id);
      void refreshSystem();
    } catch (e) {
      const ae = e instanceof ApiError ? e : null;
      setErr({ code: ae?.code ?? null, message: ae?.message ?? String(e) });
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
            model: "edge0-8b",
            bytes: rec ? humanBytes(rec.bytes_total) : t("welcome.bytesUnknown"),
            path: snap?.daemon.home ?? t("welcome.pathUnknown"),
          })}
          </p>
        </div>
        <fieldset className="mx-auto w-full max-w-lg rounded-2xl border border-line bg-surface2/60 p-4 text-left">
          <legend className="px-1 text-xs font-semibold text-fg">{t("welcome.startupTitle")}</legend>
          <label className="mt-1 flex cursor-pointer items-start gap-3 text-sm">
            <input
              type="checkbox"
              checked={headless}
              onChange={(e) => setHeadless(e.target.checked)}
              className="mt-0.5 size-4 accent-accent"
            />
            <span>
              <span className="block font-medium">{t("welcome.headlessLabel")}</span>
              <span className="mt-0.5 block text-xs leading-relaxed text-muted">{t("welcome.headlessNote")}</span>
            </span>
          </label>
        </fieldset>
        {task ? (
          <DownloadCardForTier tier="edge0-8b" />
        ) : (
          <button
            type="button"
            data-testid="welcome-go"
            disabled={busy || !catalogStore.data || catalogStore.data.stale}
            title={catalogStore.data?.stale ? t("models.stale") : undefined}
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
              <span className="block font-semibold">{t(welcomeErrorTitle(err.code))}</span>
              <span className="mt-0.5 block font-normal">{t(downloadErrKey(err.code))} - {err.code ?? ""}</span>
            </span>
          </p>
        )}
        {catalogStore.status === "error" && !catalogStore.data && (
          <p data-testid="welcome-catalog-err" className="text-xs text-muted">
            {t(downloadErrKey(catalogStore.error?.code ?? null))}
            <button
              type="button"
              data-testid="welcome-retry"
              className="ml-2 underline"
              onClick={() => void catalogStore.refresh()}
            >
              {t("models.retry")}
            </button>
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

async function errFrom(res: Response): Promise<ApiError> {
  try {
    const j = (await res.json()) as { error?: { code?: string; message?: string } };
    return new ApiError(res.status, j?.error?.code ?? null, j?.error?.message ?? `HTTP ${res.status}`);
  } catch {
    return new ApiError(res.status, null, `HTTP ${res.status}`);
  }
}

function welcomeErrorTitle(code: string | null): string {
  if (code === "E-DL-DISK") return "welcome.errors.disk";
  if (code === "E-MEM-LOAD") return "welcome.errors.memory";
  if (code === "E-GPU-DEVICE" || code === "E-SRV-WORKER") return "welcome.errors.engine";
  return "welcome.errors.generic";
}
