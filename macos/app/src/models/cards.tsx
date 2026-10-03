import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../bridge/state";
import { edge0Fetch } from "../bridge/edge0Fetch";
import { refreshSystem, taskView, type TaskView } from "./system";
import { downloadErrKey } from "./errMap";
import { etaMinutes, formatClock, humanBytes, humanRate, remainingSeconds } from "./byteFormat";

export function DownloadCardForTier({ tier }: { tier: string }) {
  const [, force] = useState(0);
  useEffect(() => daemonStore.subscribe(() => force((n) => n + 1)), []);
  const task = taskView(daemonStore.snapshot, daemonStore, tier);
  if (!task) return null;
  return <DownloadCardBody task={task} />;
}

export function DownloadCardBody({ task }: { task: TaskView }) {
  const { t } = useTranslation();
  const pct = task.bytesTotal > 0 ? Math.min(100, (task.bytesDone / task.bytesTotal) * 100) : 0;
  const showEta = task.phase === "fetch" && task.etaS !== null;
  return (
    <div data-testid={`dl-${task.id}`} className="space-y-1 rounded-lg bg-surface2/50 p-3 text-left text-xs">
      <div className="flex items-center justify-between">
        <span data-testid={`dl-phase-${task.id}`}>{t(`models.phase.${task.phase}`)}</span>
        <span className="text-muted">
          {humanBytes(task.bytesDone)} / {humanBytes(task.bytesTotal)}
          
          {task.phase === "fetch" && task.rateBps !== null && ` - ${humanRate(task.rateBps)}`}
        </span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded bg-line"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        data-testid={`dl-bar-${task.id}`}
      >
        <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
      </div>
      <div className="flex items-center justify-between gap-3">
        <span
          data-testid={`dl-eta-${task.id}`}
          className="min-w-0 flex-1 truncate text-muted"
        >
          {task.failedCode
            ? `${t(downloadErrKey(task.failedCode))} - ${task.failedCode}`
            : showEta
              ? t("models.eta", { n: etaMinutes(task.etaS as number) })
              : task.phase === "fetch"
                ? t("models.estimating")
                : t("models.verifying")}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {task.sourceSwitched && (
            <span data-testid={`dl-switched-${task.id}`} className="text-warn">
              {t("models.sourceSwitched")}
            </span>
          )}
          {task.failedCode ? (
            <button
              type="button"
              data-testid={`dl-retry-${task.id}`}
              onClick={() => void post(`/v1/edge0/downloads/${task.id}/resume`)}
              className="e0-btn e0-btn-secondary"
            >
              {t("models.retry")}
            </button>
          ) : task.paused ? (
            <button
              type="button"
              data-testid={`dl-resume-${task.id}`}
              onClick={() => void post(`/v1/edge0/downloads/${task.id}/resume`)}
              className="e0-btn e0-btn-secondary"
            >
              {t("models.resume")}
            </button>
          ) : task.phase === "fetch" ? (
            <button
              type="button"
              data-testid={`dl-pause-${task.id}`}
              onClick={() => void post(`/v1/edge0/downloads/${task.id}/pause`)}
              className="e0-btn e0-btn-secondary"
            >
              {t("models.pause")}
            </button>
          ) : null}
          <button
            type="button"
            data-testid={`dl-cancel-${task.id}`}
            onClick={() =>
              void edge0Fetch(`/v1/edge0/downloads/${task.id}`, { method: "DELETE" })
                .then(refreshSystem)
                .catch(() => undefined)
            }
            className="rounded-full border border-line px-2 py-0.5 text-muted hover:text-err"
          >
            {t("models.deleteTask")}
          </button>
        </div>
      </div>
    </div>
  );
}

export function ResidentCountdown({ unloadAt, resident }: { unloadAt: string | null; resident: boolean }) {
  const { t } = useTranslation();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!resident || !unloadAt) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [resident, unloadAt]);
  if (!resident) return null;
  const clock = formatClock(unloadAt);
  const rem = remainingSeconds(unloadAt, now);
  return (
    <span data-testid="resident-countdown" className="text-xs text-muted">
      {t("models.residentUntil")} {clock ?? t("models.unknown")}
      {rem !== null && ` - ${t("models.remaining", { n: Math.ceil(rem / 60) })}`}
    </span>
  );
}

async function post(path: string): Promise<void> {
  try {
    await edge0Fetch(path, { method: "POST", body: {} });
  } catch {
    /* ignore */
  }
  await refreshSystem();
}
