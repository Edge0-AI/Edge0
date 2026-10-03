import { Link } from "@tanstack/react-router";
import { useState, useSyncExternalStore } from "react";
import { Settings2, SlidersHorizontal, Server, RotateCcw } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ShellControls } from "../components/ShellControls";
import { daemonStore } from "../bridge/state";
import { DEFAULT_SYSTEM_PROMPT, setSystemPrompt, storedSystemPrompt } from "../lib/systemPrompt";

export function SettingsPage() {
  const { t } = useTranslation();
  const [systemPrompt, setPrompt] = useState(() => storedSystemPrompt() ?? "");
  const snap = useSyncExternalStore(
    (cb) => daemonStore.subscribe(cb),
    () => daemonStore.snapshot,
  );
  return (
    <div className="e0-page space-y-5">
      <header className="e0-page-header">
        <div>
          <h2 className="e0-page-title flex items-center gap-2">
            <Settings2 size={19} aria-hidden />
            {t("settings.title")}
          </h2>
          <p className="e0-page-kicker">{t("settings.kicker")}</p>
        </div>
      </header>
      <section className="e0-section overflow-hidden" aria-labelledby="settings-appearance-title">
        <div className="e0-section-header">
          <div>
            <h3 id="settings-appearance-title" className="e0-section-title">{t("settings.appearance")}</h3>
            <p className="e0-section-note">{t("settings.appearanceNote")}</p>
          </div>
          <SlidersHorizontal size={17} className="text-muted" aria-hidden />
        </div>
        <div className="px-5 pb-5">
          <ShellControls />
        </div>
      </section>
      <section className="e0-section overflow-hidden" aria-labelledby="settings-prompt-title">
        <div className="e0-section-header">
          <div>
            <h3 id="settings-prompt-title" className="e0-section-title">{t("settings.systemPrompt")}</h3>
            <p className="e0-section-note">{t("settings.systemPromptNote")}</p>
          </div>
        </div>
        <div className="border-t border-line/60 px-5 pb-5 pt-4">
          <textarea
            value={systemPrompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              setSystemPrompt(e.target.value);
            }}
            placeholder={DEFAULT_SYSTEM_PROMPT}
            rows={4}
            aria-label={t("settings.systemPrompt")}
            className="w-full resize-y rounded-xl border border-line bg-surface2 p-3 text-sm leading-relaxed outline-none focus:border-accent"
          />
          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-xs text-muted">
              {systemPrompt.trim() ? t("settings.systemPromptEnabled") : t("settings.systemPromptDisabled")}
            </span>
            <button
              type="button"
              className="e0-btn e0-btn-secondary"
              onClick={() => {
                setPrompt(DEFAULT_SYSTEM_PROMPT);
                setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
              }}
            >
              <RotateCcw size={13} aria-hidden />
              {t("settings.systemPromptReset")}
            </button>
          </div>
        </div>
      </section>
      <section className="e0-section overflow-hidden" aria-labelledby="settings-service-title">
        <div className="e0-section-header">
          <div>
            <h3 id="settings-service-title" className="e0-section-title">{t("settings.service")}</h3>
            <p className="e0-section-note">{t("settings.serviceNote")}</p>
          </div>
          <Server size={17} className="text-muted" aria-hidden />
        </div>
        <div className="flex items-center gap-3 border-t border-line/60 px-5 py-4">
          <span className={"size-2 rounded-full " + (snap ? "bg-ok" : "bg-muted")} aria-hidden />
          <span className="min-w-0 flex-1 truncate text-sm text-muted">
            {snap ? `${snap.daemon.bound}:${snap.daemon.port} - v${snap.daemon.version}` : t("settings.serviceUnavailable")}
          </span>
          <Link
            to="/service"
            data-testid="service-console-link"
            className="e0-btn e0-btn-secondary shrink-0"
          >
            {t("settings.openConsole")}
          </Link>
        </div>
      </section>
      <p className="px-1 text-xs text-muted">{t("settings.advancedNote")}</p>
    </div>
  );
}
