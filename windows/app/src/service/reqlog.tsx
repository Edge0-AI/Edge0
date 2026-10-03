// service/reqlog.tsx — request log table. Data source = frontend-measured generation
// requests (useThreadChat writes into the reqLog ring): since app start, in memory,
// cleared on exit — labeled honestly in the section header.
import { useSyncExternalStore } from "react";

import { reqLog, type ReqRow } from "../api/client";
import { t } from "../lib/t";

export function statusTone(status: ReqRow["status"]): "ok" | "warn" | "err" {
  if (status === "done") return "ok";
  if (status === "error") return "err";
  return "warn";
}

export function RequestLogSection() {
  useSyncExternalStore(reqLog.sig.sub, reqLog.sig.version);
  const rows = reqLog.rows;
  return (
    <section className="e0-section overflow-hidden p-5" data-testid="svc-log">
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold">{t("service.log.title")}</h3>
        <span data-testid="svc-log-count" className="text-xs text-muted tabular-nums">
          {t("service.log.count", { n: rows.length })}
        </span>
      </div>
      <p data-testid="svc-log-note" className="mb-2 text-xs text-muted">{t("service.log.note")}</p>
      {rows.length === 0 ? (
        <p data-testid="svc-log-empty" className="text-xs text-muted">{t("service.log.empty")}</p>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="py-1 pr-2 font-normal">{t("service.log.th.status")}</th>
              <th className="py-1 pr-2 font-normal">{t("service.log.th.model")}</th>
              <th className="py-1 pr-2 font-normal">{t("service.log.th.latency")}</th>
              <th className="py-1 pr-2 font-normal">{t("service.log.th.tokens")}</th>
              <th className="py-1 font-normal">{t("service.log.th.source")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const tone = statusTone(r.status);
              return (
                <tr key={`${r.ts}-${i}`} data-testid={`svc-log-row-${i}`} className="border-t border-line/60">
                  <td className="py-1 pr-2">
                    <span className="flex items-center gap-1">
                      <span
                        className={"inline-block size-2 rounded-full " + (tone === "ok" ? "bg-accent" : tone === "err" ? "bg-err" : "bg-warn")}
                        aria-hidden
                      />
                      {t(`service.log.status.${r.status}`)}
                      {r.code && <span className="text-muted">{r.code}</span>}
                    </span>
                    <span className="ml-1 text-[10px] text-muted tabular-nums">
                      {new Date(r.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </span>
                  </td>
                  <td className="py-1 pr-2">{r.model || "—"}</td>
                  <td className="py-1 pr-2 tabular-nums">{r.latencyMs}ms</td>
                  <td className="py-1 pr-2 tabular-nums">{r.tokens ?? "—"}</td>
                  <td className="py-1">{t("service.log.source.loopback")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
