// GenPill — three generation styles, switched in place on the composer toolbar; the next message reads it live.
import { useState } from "react";
import { t } from "../../lib/t";
import { GEN_TEMPERATURE, setGen, storedGen, type GenPreset } from "../../lib/generation";

const PRESETS: GenPreset[] = ["precise", "balanced", "creative"];

export function GenPill() {
  const [gen, setGenState] = useState<GenPreset>(() => storedGen());
  return (
    <div
      className="flex shrink-0 items-center gap-0.5 rounded-full border border-line bg-surface2 px-1 py-0.5"
      data-testid="gen-pill"
      title={t("gen.hint")}
    >
      {PRESETS.map((p) => (
        <button
          key={p}
          type="button"
          data-testid={`gen-${p}`}
          data-state={gen === p ? "on" : "off"}
          aria-pressed={gen === p}
          onClick={() => {
            setGen(p);
            setGenState(p);
          }}
          className="rounded-full px-2 py-0.5 text-[11px] text-muted transition-colors hover:text-fg data-[state=on]:bg-accent data-[state=on]:text-accentink"
          title={`${t(`gen.${p}`)} · ${GEN_TEMPERATURE[p]}`}
        >
          {t(`gen.${p}`)}
        </button>
      ))}
    </div>
  );
}
