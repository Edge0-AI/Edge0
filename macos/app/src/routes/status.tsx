import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { edge0Fetch } from "../bridge/edge0Fetch";
import { daemonStore } from "../bridge/state";
import type { SystemSnapshot } from "../gen";

interface ServiceState {
  phase: {
    phase: string;
    health?: {
      daemon_version: string;
      host_app_version: string | null;
      scope: string;
      started_by: string;
      uptime_s: number;
      pid: number;
    };
    code?: number | null;
    detail?: string;
    base?: string;
    spawnedByUs?: boolean;
  };
  appVersion: string;
  versionMismatch: boolean;
  spawnWasDev: boolean;
  pinned: boolean;
}

export function StatusPage() {
  const { t } = useTranslation();
  const [, force] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [svc, setSvc] = useState<ServiceState | null>(null);

  useEffect(() => {
    const off = daemonStore.subscribe(() => force((n) => n + 1));
    void refresh();
    const poll = setInterval(() => {
      invoke<ServiceState>("service_state").then(setSvc).catch(() => undefined);
    }, 2000);
    return () => {
      off();
      clearInterval(poll);
    };
  }, []);

  const act = useCallback(async (cmd: "service_start" | "service_restart") => {
    try {
      await invoke(cmd);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  async function refresh() {
    try {
      const res = await edge0Fetch("/v1/edge0/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      daemonStore.ingest((await res.json()) as SystemSnapshot);
      setError(null);
    } catch (e) {
      setError(t("status.unavailable"));
    }
  }

  const snap = daemonStore.snapshot;
  const resident = snap?.models.filter((m) => m.state === "resident") ?? [];
  return (
    <section className="space-y-3" data-testid="status-page">
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">{t("status.title")}</h2>
        <span
          data-testid="stream-link"
          className={
            "rounded px-2 py-0.5 text-xs " +
            (daemonStore.link === "live" || daemonStore.lastSeq > 0
              ? "bg-accent text-white"
              : "border border-line text-muted")
          }
        >
          {t(`status.${linkKey(daemonStore.link, daemonStore.lastSeq)}`)}
        </span>
      </div>

      {svc && <LifecycleBanner svc={svc} onStart={() => void act("service_start")} onRestart={() => void act("service_restart")} />}

      {error && !snap && (
        <p data-testid="status-error" className="rounded border border-line p-3 text-sm text-muted">
          {error}
        </p>
      )}

      {snap ? (
        <dl
          className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm"
          data-testid="snapshot-fields"
        >
          <Field k={t("status.daemon")} v={`v${snap.daemon.version}`} testid="daemon-version" />
          <Field
            k={t("status.hostApp")}
            v={snap.daemon.host_app_version ?? "—"}
            testid="host-app"
          />
          <Field
            k={t("status.scope")}
            v={snap.daemon.scope}
            extra={snap.daemon.started_by}
            testid="scope"
          />
          <Field
            k={t("status.port")}
            v={`${snap.daemon.bound}:${snap.daemon.port}`}
            testid="port"
          />
          <Field k={t("status.uptime")} v={fmtUptime(snap.daemon.uptime_s, t)} testid="uptime" />
          <Field
            k={t("status.diskFree")}
            v={`${snap.hardware.disk_free_gib.toFixed(1)} GiB`}
            testid="disk"
          />
          <Field
            k={t("status.throttled")}
            v={snap.daemon.throttled ? t("status.yes") : t("status.no")}
            testid="throttled"
          />
          <Field k={t("status.seq")} v={String(snap.events_seq)} testid="events-seq" />
        </dl>
      ) : (
        !error && <p className="text-sm text-muted">{t("status.waiting")}</p>
      )}

      <div data-testid="resident-models" className="space-y-1">
        <h3 className="text-sm font-medium">{t("status.models")}</h3>
        {resident.length === 0 && (
          <p className="text-sm text-muted" data-testid="no-resident">
            {t("status.noResident")}
          </p>
        )}
        {resident.map((m) => (
          <p key={m.id} className="text-sm">
            <span data-testid={`resident-${m.id}`}>{m.id}</span>
            {m.unload_at && (
              <span className="text-muted"> - {t("status.residentUntil")} {fmtHHMM(m.unload_at)}</span>
            )}
          </p>
        ))}
      </div>

      
      <p className="text-xs text-muted">{t("platform.note")}</p>
    </section>
  );
}

function LifecycleBanner({
  svc,
  onStart,
  onRestart,
}: {
  svc: ServiceState;
  onStart: () => void;
  onRestart: () => void;
}) {
  const { t } = useTranslation();
  const p = svc.phase;
  if (svc.versionMismatch) {
    return (
      <div
        data-testid="mismatch-banner"
        className="flex items-center justify-between gap-3 rounded border border-yellow-500/50 p-3 text-sm"
      >
        <span>
          {t("status.mismatch", {
            daemon: p.health?.host_app_version ?? "?",
            app: svc.appVersion,
          })}
        </span>
        <button
          data-testid="restart-service"
          className="rounded bg-accent px-3 py-1 text-white"
          onClick={onRestart}
        >
          {t("status.restartService")}
        </button>
      </div>
    );
  }
  if (p.phase === "exited") {
    return (
      <div
        data-testid="exited-banner"
        className="flex items-center justify-between gap-3 rounded border border-line p-3 text-sm"
      >
        <span>{t("status.exited", { code: p.code ?? t("status.unknown") })}</span>
        <button data-testid="start-service" className="rounded bg-accent px-3 py-1 text-white" onClick={onStart}>
          {t("status.startService")}
        </button>
      </div>
    );
  }
  if (p.phase === "stopped") {
    return (
      <div data-testid="stopped-banner" className="flex items-center justify-between gap-3 rounded border border-line p-3 text-sm">
        <span className="text-muted">{p.detail}</span>
        <button data-testid="start-service" className="rounded bg-accent px-3 py-1 text-white" onClick={onStart}>
          {t("status.startService")}
        </button>
      </div>
    );
  }
  if (p.phase === "remote") {
    return (
      <p data-testid="remote-note" className="text-sm text-muted">
        {t("status.remote", { base: p.base ?? "?" })}
      </p>
    );
  }
  if (svc.spawnWasDev && (p.phase === "spawned" || p.spawnedByUs)) {
    return (
      <p data-testid="dev-badge" className="text-xs text-muted">
        {t("status.devDaemon")}
      </p>
    );
  }
  return null;
}

function Field({ k, v, extra, testid }: { k: string; v: string; extra?: string; testid: string }) {
  return (
    <>
      <dt className="text-muted">{k}</dt>
      <dd data-testid={testid}>
        {v}
        {extra ? <span className="text-muted"> - {extra}</span> : null}
      </dd>
    </>
  );
}

function linkKey(link: string, lastSeq: number): string {
  if (link !== "live") return `stream_${link}`;
  return lastSeq > 0 ? "stream_live" : "stream_idle";
}

function fmtUptime(s: number, t: (k: string) => string): string {
  if (s < 60) return `${s}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}${t("status.hours")}${m}${t("status.minutes")}` : `${m}${t("status.minutes")}`;
}

function fmtHHMM(iso: string): string {
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
