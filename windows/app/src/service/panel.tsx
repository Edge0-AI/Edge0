// service/panel.tsx — lifecycle panel. Skin boundaries: forwards existing commands only,
// reads measured fields only. This shell has no daemon token: loopback is access-exempt
// and stated as such. LAN = phase-2 placeholder row (kept visible, disabled + honest
// title).
import { useState, useSyncExternalStore } from "react";

import {
  engineStore,
  installedTiers,
  loadModel,
  shellStore,
  unloadModel,
} from "../api/client";
import { codeOf, errKey, t } from "../lib/t";

export function ServicePanel() {
  useSyncExternalStore(engineStore.sig.sub, engineStore.sig.version);
  useSyncExternalStore(shellStore.sig.sub, shellStore.sig.version);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const st = engineStore.status;
  const running = !!st?.running;
  const loading = engineStore.loadingTier !== null;
  const tiers = installedTiers();
  const pick = tiers.includes("8b") ? "8b" : tiers[0] ?? null;

  async function act(name: string, fn: () => Promise<unknown>) {
    setBusy(name);
    setError(null);
    try {
      await fn();
    } catch (e) {
      const s = String(e);
      setError(`${t(errKey(codeOf(s)))}${codeOf(s) ? ` · ${codeOf(s)}` : ""}`);
    } finally {
      setBusy(null);
    }
  }

  const inFlight = busy !== null || loading;

  return (
    <section className="e0-section overflow-hidden p-5" data-testid="svc-panel">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">{t("service.panel.title")}</h3>
        <div className="flex items-center gap-2">
          {running ? (
            <button type="button" data-testid="svc-stop" disabled={inFlight} onClick={() => void act("stop", () => unloadModel())} className="e0-btn e0-btn-secondary">
              {t("service.panel.stop")}
            </button>
          ) : (
            <button
              type="button"
              data-testid="svc-start"
              disabled={inFlight || !pick}
              title={pick ? undefined : t("chat.noInstalledModel")}
              onClick={() => pick && void act("start", () => loadModel(pick))}
              className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accentink hover:opacity-90 disabled:opacity-50"
            >
              {loading ? t("models.loadingModel") : t("service.panel.start")}
            </button>
          )}
        </div>
      </div>

      {inFlight && (
        <p data-testid="svc-busy" className="mb-2 text-xs text-muted">{t("service.panel.busy")}</p>
      )}
      {error && (
        <p data-testid="svc-error" role="alert" className="mb-2 rounded-lg bg-err/10 px-3 py-2 text-xs text-err">{error}</p>
      )}

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs" data-testid="svc-fields">
        <dt className="text-muted">{t("service.panel.port")}</dt>
        <dd data-testid="svc-port" className="tabular-nums">{st?.port ? `${st.bound}:${st.port}` : "—"}</dd>
        <dt className="text-muted">{t("service.panel.bind")}</dt>
        <dd data-testid="svc-bind">{st?.bound ?? "—"}</dd>
        <dt className="text-muted">{t("service.panel.scope")}</dt>
        <dd data-testid="svc-scope">{running ? t("sidebar.scopeShell") : "—"}</dd>
        <dt className="text-muted">{t("service.panel.tier")}</dt>
        <dd data-testid="svc-tier">{running ? st?.tier ?? "—" : "—"}</dd>
        <dt className="text-muted">{t("service.panel.pool")}</dt>
        <dd data-testid="svc-pool" className="tabular-nums">{running && st?.pool_mb ? `${st.pool_mb} MB` : "—"}</dd>
        <dt className="text-muted">{t("service.panel.uptime")}</dt>
        <dd data-testid="svc-uptime" className="tabular-nums">{running ? `${st?.uptime_s ?? 0}s` : "—"}</dd>
        <dt className="text-muted">{t("service.panel.version")}</dt>
        <dd data-testid="svc-version">{st?.version ?? "—"}</dd>
        <dt className="text-muted">{t("service.panel.governance")}</dt>
        <dd data-testid="svc-governance" className={running && !st?.job ? "text-warn" : ""}>
          {running ? (st?.job ? t("service.panel.governanceJob") : t("service.panel.governanceSoft")) : "—"}
        </dd>
        <dt className="text-muted">{t("service.panel.home")}</dt>
        <dd data-testid="svc-home" className="break-all">{shellStore.paths.home || "—"}</dd>
      </dl>

      {/* token row: loopback direct access needs no credential — the fact, stated plainly (no mask, no copy button) */}
      <div className="mt-3 flex items-center gap-2 text-xs" data-testid="svc-token-row">
        <span className="text-muted">{t("service.panel.token")}</span>
        <span data-testid="svc-token-none">{t("service.panel.noCredential")}</span>
      </div>

      {/* LAN toggle: phase-2 placeholder — row kept visible, disabled + honest title */}
      <div className="mt-3 flex items-center justify-between border-t border-line pt-3 text-xs" data-testid="svc-lan-row">
        <span className="text-muted">{t("service.panel.lan")}</span>
        <button type="button" role="switch" aria-checked={false} data-testid="svc-lan-switch" disabled title={t("service.panel.lanLater")} className="e0-btn e0-btn-secondary disabled:opacity-50">
          {t("service.panel.lanOff")}
        </button>
      </div>
    </section>
  );
}
