// models/cards.tsx — in-flight/failed download task cards (one implementation shared by
// the catalog page and the welcome card). Speed is frontend-measured: sliding-window
// deltas over the shell's progress byte stream (the shell sends no rate field).
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import { cancelDownload, downloadStore, startDownload, type TaskView } from "../api/client";
import { etaMinutes, humanBytes, humanRate } from "../lib/byteFormat";
import { codeOf, errKey, t } from "../lib/t";

export function DownloadCardForTier({ tier }: { tier: string }) {
  useSyncExternalStore(downloadStore.sig.sub, downloadStore.sig.version);
  const task = downloadStore.tasks[tier];
  if (!task || task.phase === "idle") return null;
  return <DownloadCardBody task={task} />;
}

export function DownloadCardBody({ task }: { task: TaskView }) {
  const pct = task.bytes_total > 0 ? Math.min(100, (task.bytes_done / task.bytes_total) * 100) : 0;
  const downloading = task.phase === "download";
  const rateRef = useRef<{ b: number; t: number } | null>(null);
  const [rate, setRate] = useState<number | null>(null);

  useEffect(() => {
    if (!downloading) {
      setRate(null);
      rateRef.current = null;
      return;
    }
    const now = Date.now();
    const prev = rateRef.current;
    rateRef.current = { b: task.bytes_done, t: now };
    if (prev && now - prev.t > 500) {
      const db = task.bytes_done - prev.b;
      if (db >= 0) setRate(db / ((now - prev.t) / 1000));
    }
  }, [task.bytes_done, downloading]);

  const failed = task.phase === "error" && task.error;
  const showEta = downloading && rate !== null && rate > 0;
  const filesDone = task.files ? task.files.filter((f) => f.state === "done").length : 0;

  return (
    <div data-testid={`dl-${task.tier}`} className="space-y-1 rounded-lg bg-surface2/50 p-3 text-left text-xs">
      <div className="flex items-center justify-between">
        <span data-testid={`dl-phase-${task.tier}`}>{t(`models.phase.${task.phase}`)}</span>
        <span className="text-muted tabular-nums">
          {humanBytes(task.bytes_done)} / {humanBytes(task.bytes_total)}
          {downloading && rate !== null && ` · ${humanRate(rate)}`}
        </span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded bg-line"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        data-testid={`dl-bar-${task.tier}`}
      >
        <div className={"h-full " + (failed ? "bg-err" : "bg-accent")} style={{ width: `${pct}%` }} />
      </div>
      <div className="flex items-center justify-between gap-3">
        <span data-testid={`dl-eta-${task.tier}`} className="min-w-0 flex-1 truncate text-muted">
          {failed
            ? `${t(errKey(codeOf(task.error)))} · ${codeOf(task.error) ?? "E-DL"}`
            : showEta
              ? t("models.eta", { n: etaMinutes((task.bytes_total - task.bytes_done) / (rate as number)) })
              : downloading
                ? t("models.estimating")
                : task.files?.length
                  ? t("models.filesProgress", { done: filesDone, total: task.files.length })
                  : ""}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {failed ? (
            <button
              type="button"
              data-testid={`dl-retry-${task.tier}`}
              onClick={() => void startDownload(task.tier, null).catch(() => undefined)}
              className="e0-btn e0-btn-secondary"
            >
              {t("models.retry")}
            </button>
          ) : downloading ? (
            <button
              type="button"
              data-testid={`dl-cancel-${task.tier}`}
              onClick={() => void cancelDownload(task.tier)}
              className="rounded-full border border-line px-2 py-0.5 text-muted hover:text-err"
            >
              {t("models.cancelTask")}
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
