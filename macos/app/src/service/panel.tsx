import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../bridge/state";
import type { SystemSnapshot } from "../gen";
import { isLoopback, useServiceState } from "./state";

export function ServicePanel() {
  const { t } = useTranslation();
  const [svc] = useServiceState();
  const [, force] = useState(0);
  const snap = daemonStore.snapshot;
  const phase = svc?.phase.phase ?? "probing";
  const running = phase === "spawned" || phase === "adopted";

  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copyState, setCopyState] = useState<"masked" | "copied" | "no-credential">("masked");

  async function act(name: string, fn: () => Promise<unknown>) {
    setBusy(name);
    setError(null);
    try {
      await fn();
    } catch (e) {
      const raw = typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
      const msg = raw === "no-credential" ? t("service.panel.noCredential") : t("service.panel.actionFailed");
      setError(msg);
      if (raw.includes("no-credential")) setCopyState("no-credential");
    } finally {
      setBusy(null);
      force((n) => n + 1);
    }
  }

  const inFlight = busy !== null;
  const versionLine = snap
    ? svc?.versionMismatch
      ? t("service.panel.versionMismatch", {
          daemon: snap.daemon.host_app_version ?? "?",
          app: svc?.appVersion ?? "?",
        })
      : `v${snap.daemon.version}`
    : "—";

  return (
    <section className="e0-section overflow-hidden p-5" data-testid="svc-panel">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">{t("service.panel.title")}</h3>
        <div className="flex items-center gap-2">
          {running ? (
            <button
              type="button"
              data-testid="svc-stop"
              disabled={inFlight || phase === "adopted"}
              title={phase === "adopted" ? t("service.panel.notOwned") : undefined}
              onClick={() => void act("stop", () => invoke("service_stop"))}
              className="e0-btn e0-btn-secondary"
            >
              {t("service.panel.stop")}
            </button>
          ) : (
            (phase === "stopped" || phase === "exited") && (
              <button
                type="button"
                data-testid="svc-start"
                disabled={inFlight}
                onClick={() => void act("start", () => invoke("service_start"))}
                className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accentink hover:opacity-90 disabled:opacity-50"
              >
                {t("service.panel.start")}
              </button>
            )
          )}
        </div>
      </div>

      {inFlight && (
        <p data-testid="svc-busy" className="mb-2 text-xs text-muted">
          {t("service.panel.busy")}
        </p>
      )}
      {error && (
        <p data-testid="svc-error" role="alert" className="mb-2 rounded-lg bg-err/10 px-3 py-2 text-xs text-err">
          {error}
        </p>
      )}

      <dl
        className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs"
        data-testid="svc-fields"
      >
        <dt className="text-muted">{t("service.panel.port")}</dt>
        <dd data-testid="svc-port">{snap ? `${snap.daemon.bound}:${snap.daemon.port}` : "—"}</dd>
        <dt className="text-muted">{t("service.panel.bind")}</dt>
        <dd data-testid="svc-bind">{snap?.daemon.bound ?? "—"}</dd>
        <dt className="text-muted">{t("service.panel.scope")}</dt>
        <dd data-testid="svc-scope">
          {snap ? `${snap.daemon.scope} - ${snap.daemon.started_by}` : "—"}
        </dd>
        <dt className="text-muted">{t("service.panel.uptime")}</dt>
        <dd data-testid="svc-uptime">{snap ? `${snap.daemon.uptime_s}s` : "—"}</dd>
        <dt className="text-muted">{t("service.panel.version")}</dt>
        <dd data-testid="svc-version" className={svc?.versionMismatch ? "text-warn" : ""}>
          {versionLine}
        </dd>
        <dt className="text-muted">{t("service.panel.home")}</dt>
        <dd data-testid="svc-home" className="break-all">
          {snap?.daemon.home ?? "—"}
        </dd>
      </dl>

      
      <div className="mt-3 flex items-center gap-2 text-xs" data-testid="svc-token-row">
        <span className="text-muted">{t("service.panel.token")}</span>
        {copyState === "no-credential" ? (
          <span data-testid="svc-token-none">{t("service.panel.noCredential")}</span>
        ) : (
          <>
            <span data-testid="svc-token-mask" className="tracking-widest text-muted">
              {t("service.panel.mask")}
            </span>
            <button
              type="button"
              data-testid="svc-token-copy"
              disabled={inFlight}
              onClick={() =>
                void act("copy", async () => {
                  await invoke("token_copy");
                  setCopyState("copied");
                })
              }
              className="e0-btn e0-btn-secondary disabled:opacity-50"
            >
              {t("service.panel.copy")}
            </button>
            {copyState === "copied" && (
              <span data-testid="svc-token-copied" className="text-warn">
                {t("service.panel.copiedWarn")}
              </span>
            )}
          </>
        )}
      </div>

      <LanSwitch snap={snap} busy={inFlight} onAct={act} />
      <AgentSwitch snap={snap} phase={phase} busy={inFlight} onAct={act} />
    </section>
  );
}

function LanSwitch({
  snap,
  busy,
  onAct,
}: {
  snap: SystemSnapshot | null;
  busy: boolean;
  onAct: (name: string, fn: () => Promise<unknown>) => Promise<void>;
}) {
  const { t } = useTranslation();
  const [confirming, setConfirming] = useState(false);
  const lanOn = snap ? !isLoopback(snap.daemon.bound) : false;
  const target = !lanOn;

  return (
    <div className="mt-3 space-y-1 border-t border-line pt-3 text-xs" data-testid="svc-lan-row">
      <div className="flex items-center justify-between">
        <span className="text-muted">{t("service.panel.lan")}</span>
        {confirming ? (
          <span className="flex items-center gap-2">
            <span data-testid="svc-lan-confirm-text">{t("service.panel.lanConfirm")}</span>
            <button
              type="button"
              data-testid="svc-lan-yes"
              disabled={busy}
              onClick={() => {
                setConfirming(false);
                void onAct("lan", () => invoke("lan_set", { on: target }));
              }}
              className="rounded bg-accent px-2 py-0.5 text-accentink disabled:opacity-50"
            >
              {t("service.panel.lanGoOn")}
            </button>
            <button
              type="button"
              data-testid="svc-lan-no"
              onClick={() => setConfirming(false)}
              className="rounded border border-line px-2 py-0.5"
            >
              {t("common.cancel")}
            </button>
          </span>
        ) : (
          <button
            type="button"
            role="switch"
            aria-checked={lanOn}
            data-testid="svc-lan-switch"
            data-on={lanOn}
            disabled={busy || !snap}
            onClick={() => setConfirming(true)}
            className="e0-btn e0-btn-secondary disabled:opacity-50"
          >
            {lanOn ? t("service.panel.lanOn") : t("service.panel.lanOff")}
          </button>
        )}
      </div>
      {lanOn && (
        <p data-testid="svc-lan-guide" className="text-muted">
          {t("service.panel.lanGuide")}
        </p>
      )}
    </div>
  );
}

function AgentSwitch({
  snap,
  phase,
  busy,
  onAct,
}: {
  snap: SystemSnapshot | null;
  phase: string;
  busy: boolean;
  onAct: (name: string, fn: () => Promise<unknown>) => Promise<void>;
}) {
  const { t } = useTranslation();
  if (!snap || (phase !== "spawned" && phase !== "adopted")) return null;
  const onAgent = snap.daemon.scope === "launchagent";
  return (
    <div className="mt-3 flex items-center justify-between border-t border-line pt-3 text-xs" data-testid="svc-agent-row">
      <span className="text-muted">{t("service.panel.agent")}</span>
      <button
        type="button"
        data-testid="svc-agent-toggle"
        data-to={onAgent ? "menu" : "agent"}
        disabled={busy}
        onClick={() => void onAct("agent", () => invoke("agent_set", { enable: !onAgent }))}
        className="e0-btn e0-btn-secondary disabled:opacity-50"
      >
        {onAgent ? t("service.panel.agentToMenu") : t("service.panel.agentToLaunchd")}
      </button>
    </div>
  );
}
