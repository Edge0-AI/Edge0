// ShellControls — theme + font-size control groups (no radix: native data-state keeps the same shape).
import { useEffect, useState } from "react";
import { t } from "../lib/t";
import {
  applyFont,
  applyTheme,
  storedFont,
  storedTheme,
  type FontScale,
  type Theme,
} from "../lib/theme";

export function ShellControls() {
  const [theme, setTheme] = useState<Theme>(storedTheme());
  const [font, setFont] = useState<FontScale>(storedFont());

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);
  useEffect(() => {
    applyFont(font);
  }, [font]);
  // When following the system preference, re-resolve on system changes.
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => theme === "system" && applyTheme("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-line pb-3" data-testid="shell-controls">
      <div className="e0-toggle-group" role="group" aria-label={t("theme.system")}>
        {(["light", "dark", "system"] as const).map((v) => (
          <button key={v} type="button" data-state={v === theme ? "on" : "off"} className="e0-toggle"
            aria-pressed={v === theme} onClick={() => setTheme(v)}>
            {t(`theme.${v}`)}
          </button>
        ))}
      </div>
      <div className="e0-toggle-group" role="group" aria-label={t("font.md")}>
        {(["sm", "md", "lg"] as const).map((v) => (
          <button key={v} type="button" data-state={v === font ? "on" : "off"} className="e0-toggle"
            aria-pressed={v === font} onClick={() => setFont(v)}>
            {t(`font.${v}`)}
          </button>
        ))}
      </div>
    </div>
  );
}
