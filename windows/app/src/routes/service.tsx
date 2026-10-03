// routes/service.tsx — local API service screen with three landmarks: lifecycle /
// request log / environment checks. Adds no new control paths or endpoints — a skin
// over existing shell capabilities.
import { Server } from "lucide-react";

import { DoctorSection } from "../service/doctor";
import { ServicePanel } from "../service/panel";
import { RequestLogSection } from "../service/reqlog";
import { t } from "../lib/t";

export function ServicePage() {
  return (
    <div className="e0-page space-y-5" data-testid="service-page">
      <header className="e0-page-header">
        <div>
          <h2 className="e0-page-title flex items-center gap-2">
            <Server size={18} aria-hidden />
            {t("service.title")}
          </h2>
          <p className="e0-page-kicker">{t("service.kicker")}</p>
        </div>
      </header>
      <section aria-labelledby="service-lifecycle-title" data-testid="service-lifecycle-section">
        <div className="mb-2 px-1">
          <h3 id="service-lifecycle-title" className="e0-section-title">{t("service.sections.lifecycle")}</h3>
          <p className="e0-section-note">{t("service.sections.lifecycleNote")}</p>
        </div>
        <ServicePanel />
      </section>
      <section aria-labelledby="service-log-title" data-testid="service-log-section">
        <div className="mb-2 px-1">
          <h3 id="service-log-title" className="e0-section-title">{t("service.sections.logs")}</h3>
          <p className="e0-section-note">{t("service.sections.logsNote")}</p>
        </div>
        <RequestLogSection />
      </section>
      <section aria-labelledby="service-doctor-title" data-testid="service-doctor-section">
        <div className="mb-2 px-1">
          <h3 id="service-doctor-title" className="e0-section-title">{t("service.sections.doctor")}</h3>
          <p className="e0-section-note">{t("service.sections.doctorNote")}</p>
        </div>
        <DoctorSection />
      </section>
    </div>
  );
}
