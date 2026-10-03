import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { BookOpen, Download, RefreshCw, Trash2 } from "lucide-react";

import { daemonStore } from "../bridge/state";
import { edge0Fetch } from "../bridge/edge0Fetch";
import { ApiError } from "../bridge/frames";
import type { CatalogTier } from "../gen";
import { catalogStore } from "../models/catalog";
import {
  refreshSystem,
  taskView,
  TERMINAL_EVENT_TYPES,
} from "../models/system";
import { DownloadCardBody, ResidentCountdown } from "../models/cards";
import { downloadErrKey } from "../models/errMap";
import { humanBytes } from "../models/byteFormat";
import { deriveLight, latestModelEvent, type ModelLight } from "../chat/modelStatus";

export function ModelsPage() {
  const { t } = useTranslation();
  const [, force] = useState(0);
  useEffect(() => {
    const off1 = daemonStore.subscribe(() => force((n) => n + 1));
    const off2 = catalogStore.subscribe(() => force((n) => n + 1));
    return () => {
      off1();
      off2();
    };
  }, []);

  useEffect(() => {
    let lastTerm = 0;
    return daemonStore.subscribe(() => {
      const m = daemonStore.maxSeqAmong(TERMINAL_EVENT_TYPES);
      if (m > lastTerm) {
        lastTerm = m;
        void refreshSystem().then(() => void catalogStore.refresh());
      }
    });
  }, []);

  useEffect(() => {
    void catalogStore.refresh();
    void refreshSystem();
  }, []);

  const catalog = catalogStore;
  const stale = catalog.data?.stale ?? false;

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
        <button
          type="button"
          data-testid="models-refresh"
          onClick={() => {
            void catalogStore.refresh();
            void refreshSystem();
          }}
          className="e0-btn e0-btn-secondary"
        >
          <RefreshCw
            size={12}
            aria-hidden
            className={catalog.status === "loading" ? "animate-spin" : ""}
          />
          {t("models.refresh")}
        </button>
      </div>

      {stale && (
        <p data-testid="models-stale" className="e0-status e0-status-warn w-full justify-start rounded-xl px-3 py-2">
          {t("models.stale")}
        </p>
      )}

      {catalog.status === "error" && !catalog.data && (
        <p
          data-testid="models-error"
          className="e0-status e0-status-err w-full justify-start rounded-xl px-3 py-2"
        >
          {t(downloadErrKey(catalog.error?.code ?? null))}
          {catalog.error?.code ? ` - ${catalog.error.code}` : ""}
          <button
            type="button"
            data-testid="models-retry"
            onClick={() => void catalogStore.refresh()}
            className="ml-2 underline"
          >
            {t("models.retry")}
          </button>
        </p>
      )}

      {!catalog.data && catalog.status !== "error" && (
        <p className="text-sm text-muted">{t("models.loading")}</p>
      )}

      {catalog.data?.tiers.map((tier) => (
        <TierCard
          key={tier.tier}
          tier={tier}
          stale={stale}
          direct={catalog.data?.signer_key_id === "direct"}
        />
      ))}
    </section>
  );
}

function TierCard({ tier, stale, direct }: { tier: CatalogTier; stale: boolean; direct?: boolean }) {
  const { t } = useTranslation();
  const [, force] = useState(0);
  useEffect(() => daemonStore.subscribe(() => force((n) => n + 1)), []);
  const [source, setSource] = useState<string>("");
  const [keepAlive, setKeepAlive] = useState<string>("");
  const [opError, setOpError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const snap = daemonStore.snapshot;
  const task = taskView(snap, daemonStore, tier.tier);
  const installed = snap?.models.find((m) => m.id === tier.tier);
  const light: ModelLight = deriveLight(snap, tier.tier, latestModelEvent(daemonStore, tier.tier));
  const stateLabel: "absent" | "installed" | "update" = tier.update_available
    ? "update"
    : installed || tier.installed
      ? "installed"
      : "absent";

  async function run(fn: () => Promise<void>) {
    setOpError(null);
    setConfirmDelete(false);
    try {
      await fn();
      void refreshSystem();
    } catch (e) {
      const ae = e instanceof ApiError ? e : null;
      setOpError(ae ? `${t(downloadErrKey(ae.code))}${ae.code ? ` - ${ae.code}` : ""}` : t("models.err.generic"));
    }
  }

  return (
    <div
      data-testid={`tier-${tier.tier}`}
      className="e0-section flex flex-col gap-3 p-5"
    >
      <div className="flex items-baseline justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold">{tier.tier}</h3>
          <p className="text-xs text-muted" data-testid={`tier-state-${tier.tier}`}>
            {t(`models.state.${stateLabel}`)}
            {installed?.bundled ? ` - ${t("models.bundled")}` : ""}
            {tier.update_available && tier.installed
              ? ` - ${t("models.updateFrom", { rev: tier.installed })}`
              : ""}
          </p>
        </div>
        <span
          data-testid={`light-${tier.tier}`}
          data-light={light}
          className={
            "e0-status " +
            (light === "resident" ? "e0-status-ok" : light === "loading" ? "e0-status-warn" : "")
          }
        >
          <span className={"size-1.5 rounded-full " + (light === "resident" ? "bg-ok" : light === "loading" ? "bg-warn" : "bg-muted")} aria-hidden />
          {t(`chat.light.${light}`)}
        </span>
      </div>

      <dl className="order-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-line/60 pt-3 text-xs">
        <Field k={t("models.rev")} v={tier.rev} />
        <Field k={t("models.bytes")} v={humanBytes(tier.bytes_total)} />
        <Field k={t("models.files")} v={String(tier.files_count)} />
        <Field
          k={t("models.source")}
          v={
            tier.primary_source +
            (tier.has_modelscope && tier.has_huggingface ? "" : ` - ${t("models.singleSource")}`)
          }
        />
        {tier.license && <Field k={t("models.license")} v={tier.license} />}
        {tier.notes_url && (
          <div className="col-span-2">
            <a
              href={tier.notes_url}
              target="_blank"
              rel="noreferrer"
              data-testid={`tier-notes-${tier.tier}`}
              className="text-accent underline"
            >
              {t("models.notes")}
            </a>
          </div>
        )}
      </dl>

      {task && <div className="order-2"><DownloadCardBody task={task} /></div>}

      {opError && (
        <p data-testid={`tier-operr-${tier.tier}`} role="alert" className="e0-status e0-status-err order-2 w-full justify-start rounded-lg">
          {opError}
        </p>
      )}

      {!task && (stateLabel === "absent" || stateLabel === "update") && (
        <div className="order-1 flex flex-wrap items-center gap-2">
          {direct ? (
            <span data-testid={`source-direct-${tier.tier}`} className="rounded-full border border-line bg-surface2 px-3 py-1.5 text-xs text-muted">
              {t("models.sourceDirect")}
            </span>
          ) : (
            <select
              data-testid={`source-select-${tier.tier}`}
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs"
            >
              <option value="">{t("models.sourceAuto")}</option>
              <option value="modelscope">{t("models.sourceMs")}</option>
              <option value="huggingface">{t("models.sourceHf")}</option>
            </select>
          )}
          <button
            type="button"
            data-testid={`download-${tier.tier}`}
            disabled={stale}
            title={stale ? t("models.stale") : undefined}
            onClick={() =>
              void run(async () => {
                const res = await edge0Fetch("/v1/edge0/downloads", {
                  method: "POST",
                  body: direct ? { tier: tier.tier } : { tier: tier.tier, source: source || null },
                });
                if (!res.ok) throw await apiErr(res);
              })
            }
            className="e0-btn e0-btn-primary"
          >
            <Download size={12} aria-hidden />
            {stateLabel === "update" ? t("models.updateDownload") : t("models.download")}
          </button>
        </div>
      )}

      {installed && !task && (
        <div className="order-1 flex flex-wrap items-center gap-2">
          <select
            data-testid={`keepalive-${tier.tier}`}
            value={keepAlive}
            onChange={(e) => setKeepAlive(e.target.value)}
            className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs"
          >
            <option value="">{t("models.keepDefault")}</option>
            <option value="30m">{t("models.keep30")}</option>
            <option value="inf">{t("models.keepInf")}</option>
          </select>
          {light !== "resident" && light !== "loading" && (
            <button
              type="button"
              data-testid={`load-${tier.tier}`}
              onClick={() =>
                void run(async () => {
                  const res = await edge0Fetch(`/v1/edge0/models/${tier.tier}/load`, {
                    method: "POST",
                    body: keepAlive ? { keep_alive: keepAlive } : {},
                  });
                  if (!res.ok) throw await apiErr(res);
                })
              }
              className="e0-btn e0-btn-secondary"
            >
              {t("models.load")}
            </button>
          )}
          {light === "loading" && <span className="text-xs text-muted">{t("models.loadingModel")}</span>}
          {light === "resident" && (
            <button
              type="button"
              data-testid={`unload-${tier.tier}`}
              onClick={() =>
                void run(async () => {
                  const res = await edge0Fetch(`/v1/edge0/models/${tier.tier}/unload`, {
                    method: "POST",
                    body: {},
                  });
                  if (!res.ok) throw await apiErr(res);
                })
              }
              className="e0-btn e0-btn-secondary"
            >
              {t("models.unload")}
            </button>
          )}
          <ResidentCountdown unloadAt={installed.unload_at ?? null} resident={light === "resident"} />
          {installed.bundled ? (
            <span className="text-xs text-muted">{t("models.bundledHint")}</span>
          ) : confirmDelete ? (
            <span className="flex items-center gap-1 text-xs">
              <span>{t("models.deleteConfirm")}</span>
              <button
                type="button"
                data-testid={`delete-yes-${tier.tier}`}
                onClick={() =>
                  void run(async () => {
                    const res = await edge0Fetch(`/v1/edge0/models/${tier.tier}`, {
                      method: "DELETE",
                    });
                    if (!res.ok) throw await apiErr(res);
                    void catalogStore.refresh();
                  })
                }
                className="e0-btn bg-err text-white"
              >
                {t("models.delete")}
              </button>
              <button
                type="button"
                data-testid={`delete-no-${tier.tier}`}
                onClick={() => setConfirmDelete(false)}
                className="e0-btn e0-btn-secondary"
              >
                {t("common.cancel")}
              </button>
            </span>
          ) : (
            <button
              type="button"
              data-testid={`delete-${tier.tier}`}
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
      <dd className="text-fg">{v}</dd>
    </>
  );
}

async function apiErr(res: Response): Promise<ApiError> {
  try {
    const j = (await res.json()) as { error?: { code?: string; message?: string } };
    return new ApiError(res.status, j?.error?.code ?? null, j?.error?.message ?? `HTTP ${res.status}`);
  } catch {
    return new ApiError(res.status, null, `HTTP ${res.status}`);
  }
}
