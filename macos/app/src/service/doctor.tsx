import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "@tanstack/react-router";
import { Stethoscope } from "lucide-react";

import { edge0Fetch } from "../bridge/edge0Fetch";
import { requestCodeKey } from "./codeMap";

interface DoctorCheck {
  id: string;
  verdict: string;
  detail: string;
  code?: string | null;
  next?: string | null;
}
interface DoctorReport {
  overall: string;
  checks: DoctorCheck[];
}

const ACTION_ROUTES: Record<string, { to: "/models"; labelKey: string }> = {
  "E-DL-DISK": { to: "/models", labelKey: "service.doctor.goModels" },
  "E-MODEL-HASH": { to: "/models", labelKey: "service.doctor.goModels" },
};

export function DoctorSection() {
  const { t } = useTranslation();
  const [report, setReport] = useState<DoctorReport | null>(null);
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchDoctor = useCallback(async () => {
    setLoading(true);
    setFailed(false);
    try {
      const res = await edge0Fetch("/v1/edge0/doctor");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setReport((await res.json()) as DoctorReport);
    } catch {
      setReport(null);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDoctor();
  }, [fetchDoctor]);

  const verdictTone = (v: string) => (v === "pass" ? "ok" : v === "warn" ? "warn" : "err");
  const cls = (tone: string) =>
    tone === "ok" ? "bg-accent/15 text-accent" : tone === "warn" ? "bg-warn/15 text-warn" : "bg-err/15 text-err";

  return (
    <section className="e0-section overflow-hidden p-5" data-testid="svc-doctor">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          <Stethoscope size={14} aria-hidden />
          {t("service.doctor.title")}
        </h3>
        {report && (
          <span data-testid="doctor-overall" className={"rounded-full px-2 py-0.5 text-xs " + cls(verdictTone(report.overall))}>
            {t(`service.doctor.overall.${report.overall}`)}
          </span>
        )}
      </div>

      {loading && !report && <p className="text-xs text-muted">{t("service.doctor.loading")}</p>}

      {failed && (
        <p data-testid="doctor-failed" className="text-xs text-muted">
          {t("service.doctor.failed")}
          <button
            type="button"
            data-testid="doctor-retry"
            onClick={() => void fetchDoctor()}
            className="ml-2 underline"
          >
            {t("service.doctor.retry")}
          </button>
        </p>
      )}

      {report && (
        <ul className="space-y-1">
          {report.checks.map((c) => {
            const action = c.code ? ACTION_ROUTES[c.code] : undefined;
            return (
              <li
                key={c.id}
                data-testid={`doctor-item-${c.id}`}
                className="flex items-baseline gap-2 border-t border-line/60 py-1.5 text-xs first:border-t-0"
              >
                <span className={"w-14 shrink-0 rounded-full px-1.5 py-0.5 text-center " + cls(verdictTone(c.verdict))}>
                  {t(`service.doctor.verdict.${c.verdict}`)}
                </span>
                <span data-testid={`doctor-detail-${c.id}`} className="min-w-0 flex-1">
                  <span className="text-muted">{c.id}</span> - {c.detail}
                  {c.code && (
                    <span data-testid={`doctor-code-${c.id}`} className="text-muted">
                      {" "}
                      - {c.code} {t(requestCodeKey(c.code))}
                    </span>
                  )}
                  {c.next && (
                    <span data-testid={`doctor-next-${c.id}`} className="block text-muted">
                      {t("service.doctor.next")}: {c.next}
                    </span>
                  )}
                </span>
                {action && (
                  <Link
                    to={action.to}
                    data-testid={`doctor-action-${c.id}`}
                    className="e0-btn e0-btn-secondary shrink-0"
                  >
                    {t(action.labelKey)}
                  </Link>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
