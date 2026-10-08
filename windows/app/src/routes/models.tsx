// routes/models.tsx — model catalog + download management. Tiers come from the embedded
// catalog table; local facts = models_installed + engine status + folded download tasks.
// The UI never estimates.
import { BookOpen, Download, Power, RefreshCw, Trash2 } from "lucide-react";
import { useState, useSyncExternalStore } from "react";

import {
  deleteModel,
  downloadStore,
  engineStore,
  loadModel,
  refreshAll,
  shellStore,
  startDownload,
  unloadModel,
} from "../api/client";
import { deriveLight } from "../chat/modelStatus";
import { humanBytes } from "../lib/byteFormat";
import { codeOf, errKey, t } from "../lib/t";
import type { CatTier } from "../api/client";
import { DownloadCardBody } from "../models/cards";
import { LocalGgufPanel } from "../models/gguf";

export function ModelsPage() {
  const [, force] = useState(0);
  useSyncExternalStore(shellStore.sig.sub, shellStore.sig.version);
  useSyncExternalStore(downloadStore.sig.sub, downloadStore.sig.version);
  useSyncExternalStore(engineStore.sig.sub, engineStore.sig.version);
  void force;

  const tiers = Object.entries(shellStore.catalog);
  const ready = tiers.length > 0;

  return (
    <section className="e0-page space-y-4" data-testid="models-page">
      <div className="e0-page-header">
        <div>
          <h2 className="e0-page-title flex items-center gap-2">
            <BookOpen size={18} aria-hidden />
            {t("models.title")}
          </h2>
          <p className="e0-page-kicker">{t("models.kicker")}</p>
        </div>
        <button type="button" data-testid="models-refresh" onClick={() => void refreshAll()} className="e0-btn e0-btn-secondary">
          <RefreshCw size={12} aria-hidden />
          {t("models.refresh")}
        </button>
      </div>

      {!ready && <p className="text-sm text-muted">{t("models.loading")}</p>}

      {tiers.map(([tier, rec]) => (
        <TierCard key={tier} tier={tier} rec={rec} />
      ))}

      <LocalGgufPanel />
    </section>
  );
}

function TierCard({ tier, rec }: { tier: string; rec: CatTier }) {
  const [source, setSource] = useState("");
  const [opError, setOpError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const st = engineStore.status;
  const installed = shellStore.installed[tier] !== undefined;
  const task = downloadStore.tasks[tier];
  const inFlight = task && (task.phase === "probe" || task.phase === "download" || task.phase === "convert");
  const light = deriveLight(st, engineStore.loadingTier, Object.keys(shellStore.installed), tier);
  const stateLabel = installed ? "installed" : "absent";
  const resident = light === "resident";
  const filesCount = rec.files.filter((f) => !f.skip).length;

  async function run(fn: () => Promise<unknown>) {
    setOpError(null);
    setConfirmDelete(false);
    try {
      await fn();
      await refreshAll();
    } catch (e) {
      const s = String(e);
      setOpError(`${t(errKey(codeOf(s)))}${codeOf(s) ? ` · ${codeOf(s)}` : ""}`);
      await refreshAll().catch(() => undefined);
    }
  }

  return (
    <div data-testid={`tier-${tier}`} className="e0-section flex flex-col gap-3 p-5">
      <div className="flex items-baseline justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold">{tier}</h3>
          <p className="text-xs text-muted" data-testid={`tier-state-${tier}`}>{t(`models.state.${stateLabel}`)}</p>
        </div>
        <span data-testid={`light-${tier}`} data-light={light} className={"e0-status " + (light === "resident" ? "e0-status-ok" : light === "loading" ? "e0-status-warn" : "")}>
          <span className={"size-1.5 rounded-full " + (light === "resident" ? "bg-ok" : light === "loading" ? "bg-warn" : "bg-muted")} aria-hidden />
          {t(`chat.light.${light}`)}
        </span>
      </div>

      <dl className="order-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-line/60 pt-3 text-xs tabular-nums">
        <Field k={t("models.rev")} v={rec.rev.hf.slice(0, 12)} />
        <Field k={t("models.bytes")} v={humanBytes(rec.total_bytes)} />
        <Field k={t("models.files")} v={String(filesCount)} />
        <Field k={t("models.pool")} v={`${rec.pool_mb} MB`} />
        <Field k={t("models.source")} v="Hugging Face (primary) + mirrors" />
        <Field k={t("models.repo")} v={rec.repo} />
      </dl>

      {inFlight && task && (
        <div className="order-2">
          <DownloadCardBody task={task} />
        </div>
      )}
      {!inFlight && task && (task.phase === "error" || task.phase === "cancelled") && (
        <div className="order-2">
          <DownloadCardBody task={task} />
        </div>
      )}

      {opError && (
        <p data-testid={`tier-operr-${tier}`} role="alert" className="e0-status e0-status-err order-2 w-full justify-start rounded-lg">
          {opError}
        </p>
      )}

      {!installed && !inFlight && (
        <div className="order-1 flex flex-wrap items-center gap-2">
          <select
            data-testid={`source-select-${tier}`}
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs"
          >
            <option value="">{t("models.sourceAuto")}</option>
            <option value="hf">{t("models.sourceHf")}</option>
            <option value="hf-mirror">{t("models.sourceHfMirror")}</option>
            <option value="modelscope">{t("models.sourceMs")}</option>
          </select>
          <button
            type="button"
            data-testid={`download-${tier}`}
            onClick={() => void run(() => startDownload(tier, source || null))}
            className="e0-btn e0-btn-primary"
          >
            <Download size={12} aria-hidden />
            {t("models.download")}
          </button>
        </div>
      )}

      {installed && (
        <div className="order-1 flex flex-wrap items-center gap-2">
          {!inFlight && light !== "resident" && (
            <button
              type="button"
              data-testid={`load-${tier}`}
              disabled={light === "loading"}
              onClick={() => void run(() => loadModel(tier))}
              className="e0-btn e0-btn-secondary"
            >
              <Power size={12} aria-hidden />
              {light === "loading" ? t("models.loadingModel") : t("models.load")}
            </button>
          )}
          {resident && (
            <button type="button" data-testid={`unload-${tier}`} onClick={() => void run(() => unloadModel())} className="e0-btn e0-btn-secondary">
              {t("models.unload")}
            </button>
          )}
          {resident && <span className="text-xs text-muted">{t("chat.residentForever")}</span>}
          {confirmDelete ? (
            <span className="flex items-center gap-1 text-xs">
              <span>{t("models.deleteConfirm")}</span>
              <button type="button" data-testid={`delete-yes-${tier}`} onClick={() => void run(() => deleteModel(tier))} className="e0-btn bg-err text-white">
                {t("models.delete")}
              </button>
              <button type="button" data-testid={`delete-no-${tier}`} onClick={() => setConfirmDelete(false)} className="e0-btn e0-btn-secondary">
                {t("common.cancel")}
              </button>
            </span>
          ) : (
            <button
              type="button"
              data-testid={`delete-${tier}`}
              disabled={!!inFlight}
              onClick={() => setConfirmDelete(true)}
              className="e0-btn e0-btn-danger"
            >
              <Trash2 size={12} aria-hidden />
              {t("models.delete")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-muted">{k}</dt>
      <dd className="text-fg truncate" title={v}>{v}</dd>
    </>
  );
}
