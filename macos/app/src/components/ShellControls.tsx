import * as RadixToggleGroup from "@radix-ui/react-toggle-group";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  applyFont,
  applyTheme,
  storedFont,
  storedTheme,
  type FontScale,
  type Theme,
} from "../lib/theme";

const ToggleGroup = RadixToggleGroup.Root;
const Toggle = RadixToggleGroup.Item;

export function ShellControls() {
  const { t } = useTranslation();
  const [theme, setTheme] = useState<Theme>(storedTheme());
  const [font, setFont] = useState<FontScale>(storedFont());

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);
  useEffect(() => {
    applyFont(font);
  }, [font]);
  useEffect(() => {
    // Re-resolve when the OS theme changes while the app is set to "system".
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => theme === "system" && applyTheme("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  const seg =
    "e0-toggle";
  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-line pb-3" data-testid="shell-controls">
      <ToggleGroup
        type="single"
        value={theme}
        onValueChange={(v) => v && setTheme(v as Theme)}
        className="e0-toggle-group"
        aria-label={t("status.title")}
      >
        {(["light", "dark", "system"] as const).map((v) => (
          <Toggle key={v} value={v} className={seg}>
            {t(`theme.${v}`)}
          </Toggle>
        ))}
      </ToggleGroup>
      <ToggleGroup
        type="single"
        value={font}
        onValueChange={(v) => v && setFont(v as FontScale)}
        className="e0-toggle-group"
      >
        {(["sm", "md", "lg"] as const).map((v) => (
          <Toggle key={v} value={v} className={seg}>
            {t(`font.${v}`)}
          </Toggle>
        ))}
      </ToggleGroup>
    </div>
  );
}
