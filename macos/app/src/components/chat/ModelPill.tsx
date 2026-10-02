import { useEffect, useRef, useState } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { daemonStore } from "../../bridge/state";
import { edge0Fetch } from "../../bridge/edge0Fetch";
import { ApiError } from "../../bridge/frames";
import { candidateModels, deriveLight, latestModelEvent, type ModelLight } from "../../chat/modelStatus";
import { errorNextKey } from "../../chat/errorMap";

export function ModelPill({
  model,
  onModelChange,
  wireJson,
  meterNonce,
}: {
  model: string;
  onModelChange: (id: string) => void;
  wireJson: () => { role: string; content: string }[];
  meterNonce: number;
}) {
  const { t } = useTranslation();
  const [, force] = useState(0);
  const [open, setOpen] = useState(false);
  const [loadError, setLoadError] = useState<{ code: string | null; message: string } | null>(null);
  const [contextTokens, setContextTokens] = useState<number | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wireRef = useRef(wireJson);
  wireRef.current = wireJson;
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => daemonStore.subscribe(() => force((n) => n + 1)), []);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const snap = daemonStore.snapshot;
  const candidates = candidateModels(snap);
  const light = model ? deriveLight(snap, model, latestModelEvent(daemonStore, model)) : "absent";

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => void measure(), 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meterNonce, model, snap?.daemon.port]);

  async function measure() {
    if (!model || snap === null) {
      setContextTokens(null);
      return;
    }
    try {
      const res = await edge0Fetch("/v1/edge0/apply-template", {
        method: "POST",
        body: { model, messages: wireRef.current(), add_generation_prompt: true },
      });
      if (!res.ok) throw new ApiError(res.status, null, `HTTP ${res.status}`);
      const j = (await res.json()) as { token_count?: number };
      setContextTokens(typeof j.token_count === "number" ? j.token_count : null);
    } catch {
      setContextTokens(null);
    }
  }

  async function load(id: string) {
    setLoadError(null);
    try {
      const res = await edge0Fetch(`/v1/edge0/models/${id}/load`, { method: "POST", body: {} });
      if (!res.ok) throw await apiErrFrom(res);
    } catch (e) {
      const ae = e instanceof ApiError ? e : null;
      setLoadError({ code: ae?.code ?? null, message: ae?.message ?? String(e) });
    }
  }

  return (
    <div ref={rootRef} className="relative" data-testid="model-pill">
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex min-w-0 max-w-[18rem] items-center gap-1.5 rounded-full border border-line bg-surface2 px-2.5 py-1.5 text-xs font-medium text-fg hover:border-accent/50 hover:bg-surface"
      >
        <LightDot light={light} />
        <span className="max-w-[10rem] truncate">{model || t("chat.noModel")}</span>
        <span className="text-muted" data-testid="context-meter">
          {contextTokens === null ? t("chat.contextUnknown") : t("chat.contextUsed", { n: contextTokens })}
        </span>
        <ChevronDown size={12} aria-hidden />
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute bottom-full left-0 z-20 mb-2 w-72 rounded-2xl border border-line bg-surface p-2 shadow-lg"
        >
          {candidates.length === 0 && <p className="px-2 py-1.5 text-sm text-muted">{t("chat.noModel")}</p>}
          {candidates.map((id) => {
            const l = deriveLight(snap, id, latestModelEvent(daemonStore, id));
            return (
              <div key={id} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-surface2">
                <button
                  type="button"
                  role="option"
                  aria-selected={id === model}
                  onClick={() => {
                    onModelChange(id);
                    setOpen(false);
                  }}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                >
                  <LightDot light={l} />
                  <span className="min-w-0 flex-1 truncate">{id}</span>
                  <span className="text-xs text-muted">{t(`chat.light.${l}`)}</span>
                </button>
                {(l === "installed" || l === "loading") && id === model && (
                  <button
                    type="button"
                    onClick={() => void load(id)}
                    disabled={l === "loading"}
                    className="flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-xs text-muted hover:text-fg disabled:opacity-60"
                  >
                    {l === "loading" && <Loader2 size={11} className="animate-spin" aria-hidden />}
                    {t("chat.loadModel")}
                  </button>
                )}
              </div>
            );
          })}
          {loadError && (
            <p className="mt-1 rounded-lg bg-err/10 px-2 py-1.5 text-xs text-err" role="alert">
              {t(errorNextKey(loadError.code))} {loadError.code ? ` -  ${loadError.code}` : ""}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function LightDot({ light }: { light: ModelLight }) {
  const cls =
    light === "resident" ? "bg-ok" : light === "loading" ? "bg-warn animate-pulse" : light === "installed" ? "bg-muted" : "bg-line";
  return <span data-testid="model-light" data-light={light} className={"inline-block h-2 w-2 shrink-0 rounded-full " + cls} aria-hidden />;
}

async function apiErrFrom(res: Response): Promise<ApiError> {
  try {
    const j = (await res.json()) as { error?: { code?: string; message?: string } };
    return new ApiError(res.status, j?.error?.code ?? null, j?.error?.message ?? `HTTP ${res.status}`);
  } catch {
    return new ApiError(res.status, null, `HTTP ${res.status}`);
  }
}
