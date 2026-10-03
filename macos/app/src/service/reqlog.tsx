import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../bridge/state";
import type { Event } from "../gen";
import { requestCodeKey } from "./codeMap";

export interface RequestRow {
  seq: number;
  subject: string;
  ts: string;
  status: string;
  code: string | null;
  model: string;
  latencyMs: number | null;
  ttftMs: number | null;
  tokens: number | null;
  source: string;
}

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
const str = (v: unknown): string => (typeof v === "string" ? v : "");

export function rowFromEvent(e: Event): RequestRow {
  const p = (e.payload ?? {}) as Record<string, unknown>;
  return {
    seq: e.seq,
    subject: e.subject,
    ts: e.ts,
    status: str(p.status),
    code: p.code == null ? null : str(p.code),
    model: str(p.model),
    latencyMs: num(p.latency_ms),
    ttftMs: num(p.ttft_ms),
    tokens: num(p.tokens),
    source: str(p.source),
  };
}

export function statusTone(status: string): "ok" | "warn" | "err" {
  if (status === "done") return "ok";
  if (status === "error") return "err";
  return "warn";
}

export function RequestLogSection() {
  const { t } = useTranslation();
  const [, force] = useState(0);
  useEffect(() => daemonStore.subscribe(() => force((n) => n + 1)), []);

  const rows = daemonStore.requestsWindow().map(rowFromEvent);
  return (
    <section className="e0-section overflow-hidden p-5" data-testid="svc-log">
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold">{t("service.log.title")}</h3>
        <span data-testid="svc-log-count" className="text-xs text-muted">
          {t("service.log.count", { n: rows.length })}
        </span>
      </div>
      <p data-testid="svc-log-note" className="mb-2 text-xs text-muted">
        {t("service.log.note")}
      </p>
      {rows.length === 0 ? (
        <p data-testid="svc-log-empty" className="text-xs text-muted">
          {t("service.log.empty")}
        </p>
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
            {rows.map((r) => {
              const tone = statusTone(r.status);
              return (
                <tr key={r.seq} data-testid={`svc-log-row-${r.subject}`} className="border-t border-line/60">
                  <td className="py-1 pr-2">
                    <span className="flex items-center gap-1">
                      <span
                        data-testid={`svc-log-dot-${r.subject}`}
                        className={
                          "inline-block size-2 rounded-full " +
                          (tone === "ok" ? "bg-accent" : tone === "err" ? "bg-err" : "bg-warn")
                        }
                        aria-hidden
                      />
                      {t(`service.log.status.${r.status}`)}
                      {r.code ? (
                        <span data-testid={`svc-log-code-${r.subject}`} className="text-muted">
                          {r.code} - {t(requestCodeKey(r.code))}
                        </span>
                      ) : null}
                    </span>
                  </td>
                  <td className="py-1 pr-2">{r.model || "—"}</td>
                  <td className="py-1 pr-2">
                    {r.latencyMs ?? "—"}ms
                    {r.ttftMs !== null && (
                      <span className="text-muted"> - TTFT {r.ttftMs}ms</span>
                    )}
                  </td>
                  <td className="py-1 pr-2">{r.tokens ?? "—"}</td>
                  <td className="py-1">{t(`service.log.source.${r.source}`)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
