// models/gguf.tsx — local GGUF models (MoE families) on the shared engine. The shell asks
// tools/gguf_tool.py for each file's facts and expert-offload plan; every number shown is
// the tool's, computed for the GPU the engine reports. Loading is an explicit click, and
// the Load button stays disabled unless the plan says the model fits.
import { FileSearch, FolderOpen, Power } from "lucide-react";
import { useState } from "react";

import { ggufInspect, ggufScan, loadGguf } from "../api/client";
import type { GgufInfo, GgufPlan, GgufScanRow } from "../api/client";
import { humanBytes } from "../lib/byteFormat";
import { t } from "../lib/t";

const CTX_CHOICES = [4096, 8192, 16384, 32768];
const FIELD = "min-w-0 flex-1 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs";

/** GGUF errors already carry their E-GGUF code and an English sentence; show them unchanged. */
function errorText(e: unknown): string {
  return String(e);
}

export function LocalGgufPanel() {
  const [dir, setDir] = useState("");
  const [rows, setRows] = useState<GgufScanRow[] | null>(null);
  const [path, setPath] = useState("");
  const [ctx, setCtx] = useState(8192);
  const [info, setInfo] = useState<GgufInfo | null>(null);
  const [busy, setBusy] = useState<"scan" | "inspect" | "load" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<string | null>(null);

  async function run(kind: "scan" | "inspect" | "load", fn: () => Promise<void>) {
    setBusy(kind);
    setErr(null);
    try {
      await fn();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  const scan = () =>
    run("scan", async () => {
      const r = await ggufScan(dir.trim() || null);
      setRows(r.models);
    });

  const inspect = (p: string) =>
    run("inspect", async () => {
      const v = await ggufInspect(p, ctx);
      setPath(p);
      setInfo(v);
      setLoaded(null);
    });

  const load = () => {
    if (!info) return;
    const target = info.path;
    const name = info.name;
    void run("load", async () => {
      const s = await loadGguf(target, ctx);
      setLoaded(s.model ?? name);
    });
  };

  const canLoad = !!info && info.plan?.fits === true && busy === null;

  return (
    <div data-testid="gguf-panel" className="e0-section flex flex-col gap-3 p-5">
      <div>
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <FileSearch size={16} aria-hidden />
          {t("gguf.title")}
        </h3>
        <p className="text-xs text-muted">{t("gguf.kicker")}</p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input
          data-testid="gguf-dir"
          value={dir}
          onChange={(e) => setDir(e.target.value)}
          placeholder={t("gguf.folderPlaceholder")}
          className={FIELD}
        />
        <button type="button" data-testid="gguf-scan" disabled={busy !== null} onClick={() => void scan()} className="e0-btn e0-btn-secondary">
          <FolderOpen size={12} aria-hidden />
          {busy === "scan" ? t("gguf.scanning") : t("gguf.scan")}
        </button>
      </div>

      {rows !== null && rows.length === 0 && <p className="text-xs text-muted">{t("gguf.noFiles")}</p>}
      {rows !== null && rows.length > 0 && (
        <ul data-testid="gguf-rows" className="flex flex-col divide-y divide-line/60 text-xs">
          {rows.map((r, i) => (
            <li key={r.path} className="flex flex-wrap items-center justify-between gap-2 py-2">
              <div className="min-w-0">
                <div className="truncate font-medium">{r.name ?? r.path}</div>
                <div className="truncate text-muted" title={r.path}>
                  {r.ok
                    ? [
                        r.family_label ?? t("gguf.unlisted"),
                        r.quant ?? null,
                        r.file_bytes !== undefined ? humanBytes(r.file_bytes) : null,
                      ]
                        .filter((x) => x !== null)
                        .join(" · ")
                    : r.error}
                </div>
              </div>
              {r.ok && (
                <button
                  type="button"
                  data-testid={`gguf-inspect-${i}`}
                  disabled={busy !== null}
                  onClick={() => void inspect(r.path)}
                  className="e0-btn e0-btn-secondary"
                >
                  {t("gguf.inspect")}
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap items-center gap-2 border-t border-line/60 pt-3">
        <input
          data-testid="gguf-path"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder={t("gguf.pathPlaceholder")}
          className={FIELD}
        />
        <select
          data-testid="gguf-ctx"
          value={ctx}
          onChange={(e) => setCtx(Number(e.target.value))}
          aria-label={t("gguf.ctxLabel")}
          className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs"
        >
          {CTX_CHOICES.map((c) => (
            <option key={c} value={c}>
              {`${t("gguf.ctxLabel")} ${c}`}
            </option>
          ))}
        </select>
        <button
          type="button"
          data-testid="gguf-inspect-path"
          disabled={busy !== null || !path.trim()}
          onClick={() => void inspect(path.trim())}
          className="e0-btn e0-btn-secondary"
        >
          {busy === "inspect" ? t("gguf.inspecting") : t("gguf.inspect")}
        </button>
      </div>

      {err && (
        <p data-testid="gguf-error" role="alert" className="e0-status e0-status-err w-full justify-start rounded-lg">
          {err}
        </p>
      )}

      {info && (
        <div data-testid="gguf-info" className="flex flex-col gap-3 border-t border-line/60 pt-3">
          <div className="flex items-baseline justify-between gap-2">
            <div className="min-w-0">
              <h4 className="truncate text-sm font-semibold">{info.name}</h4>
              <p className="truncate text-xs text-muted" title={info.path}>
                {info.path}
              </p>
            </div>
            <span className={"e0-status " + (info.family ? "e0-status-ok" : "e0-status-warn")}>
              {info.family ? info.family.label : t("gguf.unlisted")}
            </span>
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs tabular-nums">
            <Kv k={t("gguf.arch")} v={info.architecture} />
            <Kv
              k={t("gguf.shape")}
              v={
                info.is_moe
                  ? t("gguf.moeShape", { layers: info.layers, experts: info.experts, used: info.experts_used })
                  : t("gguf.dense")
              }
            />
            <Kv k={t("gguf.quant")} v={info.quant ?? "—"} />
            <Kv k={t("gguf.size")} v={humanBytes(info.file_bytes)} />
            <Kv
              k={t("gguf.gpu")}
              v={info.gpu ? `${info.gpu.desc} · ${info.gpu.free_gb.toFixed(1)} GiB free` : t("gguf.noGpu")}
            />
            <Kv k={t("gguf.ctxLabel")} v={String(info.ctx)} />
          </dl>

          {info.plan ? <PlanBlock plan={info.plan} /> : <p className="text-xs text-muted">{t("gguf.noPlan")}</p>}

          {info.warnings.length > 0 && (
            <ul className="flex flex-col gap-1 text-xs text-muted">
              {info.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <button type="button" data-testid="gguf-load" disabled={!canLoad} onClick={() => void load()} className="e0-btn e0-btn-primary">
              <Power size={12} aria-hidden />
              {busy === "load" ? t("gguf.loading") : t("gguf.load")}
            </button>
            {loaded && (
              <span data-testid="gguf-loaded" className="text-xs text-muted">
                {t("gguf.loaded", { name: loaded })}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function PlanBlock({ plan }: { plan: GgufPlan }) {
  const tone = plan.fits === true ? "e0-status-ok" : plan.fits === false ? "e0-status-err" : "e0-status-warn";
  const verdict = plan.fits === true ? t("gguf.fits") : plan.fits === false ? t("gguf.doesNotFit") : t("gguf.unknownFit");
  return (
    <div data-testid="gguf-plan" className="flex flex-col gap-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className={"e0-status " + tone}>{verdict}</span>
        <code className="rounded bg-surface px-1.5 py-0.5 font-mono">{plan.args.join(" ")}</code>
        <span className="text-muted">{t("gguf.mode", { mode: plan.mode })}</span>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 tabular-nums">
        <Kv k={t("gguf.planGpu")} v={`${plan.gpu_gib_est.toFixed(2)} GiB`} />
        <Kv k={t("gguf.planCpu")} v={`${plan.cpu_gib_est.toFixed(2)} GiB`} />
        <Kv k={t("gguf.planKv")} v={`${plan.kv_gib_est.toFixed(2)} GiB`} />
        <Kv k={t("gguf.planBudget")} v={`${plan.budget_gib.toFixed(2)} GiB`} />
      </dl>
      {plan.notes.map((n, i) => (
        <p key={i} className="text-muted">
          {n}
        </p>
      ))}
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-muted">{k}</dt>
      <dd className="truncate text-fg" title={v}>
        {v}
      </dd>
    </>
  );
}
