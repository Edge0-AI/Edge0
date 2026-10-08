// ModelPill — model pill at the composer's left end. Data comes from the shell:
// candidates = installed list, light = pure modelStatus function, token metering =
// measured via /apply-template -> /tokenize (unavailable = unknown, never estimated).
import { ChevronDown, Loader2 } from "lucide-react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import { engineStore, installedTiers, loadModel, shellStore } from "../../api/client";
import { threadStore } from "../../state/threads";
import { candidateModels, deriveLight, type ModelLight } from "../../chat/modelStatus";
import { codeOf, errKey, t } from "../../lib/t";

function LightDot({ light }: { light: ModelLight }) {
  const cls =
    light === "resident" ? "bg-ok" : light === "loading" ? "bg-warn animate-pulse" : light === "installed" ? "bg-muted" : "bg-line";
  return <span data-testid="model-light" data-light={light} className={"inline-block h-2 w-2 shrink-0 rounded-full " + cls} aria-hidden />;
}

export function ModelPill({
  model,
  onModelChange,
  wireJson,
}: {
  model: string;
  onModelChange: (id: string) => void;
  wireJson: () => { role: string; content: string }[];
}) {
  const [, force] = useState(0);
  const [open, setOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [contextTokens, setContextTokens] = useState<number | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wireRef = useRef(wireJson);
  wireRef.current = wireJson;
  const rootRef = useRef<HTMLDivElement>(null);

  useSyncExternalStore(engineStore.sig.sub, engineStore.sig.version);
  useSyncExternalStore(shellStore.sig.sub, shellStore.sig.version);
  useEffect(() => engineStore.sig.sub(() => force((n) => n + 1)), []);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const st = engineStore.status;
  const installed = installedTiers();
  const candidates = candidateModels(installed);
  // A local GGUF file served by the running engine is the resident model, whatever tier the thread names.
  const ggufResident = st?.running && st.tier === "local-gguf" ? (st.model ?? "local GGUF") : null;
  const light = ggufResident ? "resident" : model ? deriveLight(st, engineStore.loadingTier, installed, model) : "absent";

  // metering: on message/model/port change, debounce 400 ms then measure via the two-hop call.
  const msgVer = useSyncExternalStore(
    threadStore.subscribe,
    () => threadStore.version(),
  );
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => void measure(), 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [msgVer, model, st?.port]);

  async function measure() {
    const base = engineStore.status?.running ? engineStore.status.base_url : undefined;
    if (!base || !model) {
      setContextTokens(null);
      return;
    }
    try {
      const r1 = await fetch(`${base}/apply-template`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: wireRef.current(), add_generation_prompt: true }),
      });
      if (!r1.ok) throw new Error(`HTTP ${r1.status}`);
      const j1 = (await r1.json()) as { prompt?: string };
      if (typeof j1.prompt !== "string") throw new Error("no prompt");
      const r2 = await fetch(`${base}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: j1.prompt }),
      });
      if (!r2.ok) throw new Error(`HTTP ${r2.status}`);
      const j2 = (await r2.json()) as { tokens?: unknown[] };
      setContextTokens(Array.isArray(j2.tokens) ? j2.tokens.length : null);
    } catch {
      setContextTokens(null); // unavailable = unknown: no estimate, no fake zero
    }
  }

  async function load(id: string) {
    setLoadError(null);
    try {
      await loadModel(id);
    } catch (e) {
      setLoadError(t(errKey(codeOf(String(e))))) + (codeOf(String(e)) ? ` · ${codeOf(String(e))}` : "");
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
        <span className="max-w-[10rem] truncate">{ggufResident ?? (model || t("chat.noModel"))}</span>
        <span className="text-muted" data-testid="context-meter">
          {contextTokens === null ? t("chat.contextUnknown") : t("chat.contextUsed", { n: contextTokens })}
        </span>
        <ChevronDown size={12} aria-hidden />
      </button>

      {open && (
        <div role="listbox" className="absolute bottom-full left-0 z-20 mb-2 w-72 rounded-2xl border border-line bg-surface p-2 shadow-lg">
          {candidates.length === 0 && <p className="px-2 py-1.5 text-sm text-muted">{t("chat.noModel")}</p>}
          {candidates.map((id) => {
            const l = deriveLight(st, engineStore.loadingTier, installed, id);
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
              {loadError}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
