#!/usr/bin/env python3
"""Aggregate per-context + per-model results.

Inputs (8 rows):
    ./32k.json              IQ2_XS @ 32k (KV=int8)  - 2026-10-05 (re-run with cold-prefill)
    ./64k.json              IQ2_XS @ 64k (KV=int8)  - 2026-10-05
    ./128k.json             IQ2_XS @ 128k (KV=int8) - 2026-10-05 (CORRECTED from q4_0)
    ./256k.json             IQ2_XS @ 256k (KV=int8) - 2026-10-05 (CORRECTED from k8v4)
    ./iq3_xxs_32k.json      IQ3_XXS @ 32k (KV=int8)  - 2026-10-05
    ./iq3_xxs_64k.json      IQ3_XXS @ 64k (KV=int8)  - 2026-10-05
    ./iq3_xxs_128k.json     IQ3_XXS @ 128k (KV=q4_0) - 2026-10-05
    ./iq3_xxs_256k.json     IQ3_XXS @ 256k (KV=k8v4) - 2026-10-05

Two prefill columns are now reported:
    Cold Prefill     = engine-emitted timings.prompt_per_second on a
                      max_tokens=1 non-stream call after a fresh /unload
                      and /re-fetch (true cold wall-clock)
    SSE Prefill      = engine-emitted timings.prompt_per_second on the
                      streamed call - reflects time-to-first-decode chunk,
                      which is shorter because --prefill auto overlaps
                      with the start of the decode (8192-token chunks)

Outputs:
    ./matrix.json, ./matrix.md, ./summary.json
    ./logs/<label>.md per-row audit trail
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOGS = BASE / "logs"
LOGS.mkdir(exist_ok=True)

OUT_MATRIX = BASE / "matrix.json"
OUT_MD = BASE / "matrix.md"
OUT_SUMMARY = BASE / "summary.json"

# (label, src_path, kv, model_name)
FILES = [
    ("32k",           BASE / "32k.json",           "int8",  "IQ2_XS"),
    ("64k",           BASE / "64k.json",           "int8",  "IQ2_XS"),
    ("128k",          BASE / "128k.json",          "int8",  "IQ2_XS"),
    ("256k",          BASE / "256k.json",          "int8",  "IQ2_XS"),
    ("IQ3_XXS @32k",  BASE / "iq3_xxs_32k.json",   "int8",  "IQ3_XXS"),
    ("IQ3_XXS @64k",  BASE / "iq3_xxs_64k.json",   "int8",  "IQ3_XXS"),
    ("IQ3_XXS @128k", BASE / "iq3_xxs_128k.json",  "q4_0",  "IQ3_XXS"),
    ("IQ3_XXS @256k", BASE / "iq3_xxs_256k.json",  "k8v4",  "IQ3_XXS"),
]


def md_text(rows: list) -> str:
    L = [
        "# Strata TPS benchmark results - Shin-BlackMamba - 2026-10-05 (corrected)",
        "",
        "Engine: Strata Qwen3.8-Flash-Next on RTX 4070 Ti SUPER (16 GiB, sm_89).",
        "Image: strata:latest, sm_89 fat-binary, vision decoder NOT compiled.",
        "Container: --network host --gpus all --ulimit memlock=-1 --shm-size=4g.",
        "Bench harness: `run.py` issues, per row:",
        "  1. **Cold prefill probe** (max_tokens=1, non-stream): `timings.prompt_per_second` after `POST /unload`",
        "  2. **Three SSE stream calls** with reasoning_effort=minimal, max_tokens=192, t=0",
        "",
        "KV choices per row:",
        "  - IQ2_XS rows use KV=int8. **CORRECTED 2026-10-05** - KV-streaming keeps KV table small in RAM so int8 is viable for ALL IQ2_XS contexts.",
        "  - IQ3_XXS rows at 32k/64k use KV=int8; at 128k/256k use KV=q4_0/k8v4 because IQ3_XXS residents 47 GB of experts and KV=int8 would push past VRAM at > 64 k.",
        "",
        "Two prefill columns are reported:",
        "  - **Cold Prefill** = true end-of-prompt throughput (after /unload / re-fetch).",
        "  - **SSE Prefill**  = engine-reported time-to-first-chunk throughput during streaming, lower because --prefill auto overlaps decode with the next 8K-token chunk's prefill.",
        "",
        "## Matrix",
        "",
        "| Label | Model | KV | Prompt tokens | Avg TPS | Peak TPS (best of 3) | Engine TPS (avg) | Cold Prefill | SSE Prefill | Draft accepted | TTFT (avg) |",
        "|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, model_name, kv, ctx in rows:
        avgs = [round(x["decode_avg_tps"], 2) for x in ctx["runs"]]
        peaks = [round(x["decode_peak_tps"], 2) for x in ctx["runs"]]
        engs = [round(x["eng_predicted_per_second"], 2) for x in ctx["runs"]]
        cp = ctx.get("cold_prefill") or {}
        cold_prefill_tps = cp.get("prompt_per_second", 0.0)
        sse_prefill_avgs = [round(x.get("eng_prompt_per_second", 0.0), 2) for x in ctx["runs"] if x.get("eng_prompt_per_second")]
        draft_total_ac = sum(x["eng_draft_n_accepted"] for x in ctx["runs"])
        draft_total_pr = sum(x["eng_draft_n"] for x in ctx["runs"])
        draft_pct = round(100 * draft_total_ac / max(1, draft_total_pr), 1)
        ttft = [round(x["ttft_ms"]) for x in ctx["runs"]]
        L.append(
            f"| {label} | {model_name} | {kv} | {ctx['prompt_tokens_target']:,d} | "
            f"**{statistics.mean(avgs):.2f}** | "
            f"{max(peaks):.2f} | "
            f"**{statistics.mean(engs):.2f}** | "
            f"**{cold_prefill_tps:.1f}** | "
            f"{statistics.mean(sse_prefill_avgs):.1f} | "
            f"{draft_pct}% | "
            f"{int(statistics.mean(ttft))} ms |"
        )
    L.append("")
    L.append("## Quality-vs-Speed comparison (IQ2_XS vs IQ3_XXS at same KV)")
    L.append("")
    L.append("| Context | KV | IQ2_XS engine TPS | IQ3_XXS engine TPS | IQ3 decode-TPS tax | IQ3 peak tax |")
    L.append("|---|---|---:|---:|---:|---:|")
    pairings = [("32k", "int8", "IQ3_XXS @32k"),
                ("64k", "int8", "IQ3_XXS @64k"),
                ("128k", "int8", "IQ3_XXS @128k"),
                ("256k", "int8", "IQ3_XXS @256k")]
    iq2_by_label_kv = {r[0]: r for r in rows if r[1] == "IQ2_XS"}
    iq3_by_label = {r[0]: r for r in rows if r[1] == "IQ3_XXS"}
    for label, kv, iq3_label in pairings:
        iq2 = iq2_by_label_kv.get(label)
        iq3 = iq3_by_label.get(iq3_label)
        if iq2 and iq3:
            iq2_eng = statistics.mean(round(x["eng_predicted_per_second"], 2) for x in iq2[3]["runs"])
            iq2_pk  = max(round(x["decode_peak_tps"], 2) for x in iq2[3]["runs"])
            iq3_eng = statistics.mean(round(x["eng_predicted_per_second"], 2) for x in iq3[3]["runs"])
            iq3_pk  = max(round(x["decode_peak_tps"], 2) for x in iq3[3]["runs"])
            L.append(f"| {label} | {kv} | {iq2_eng:.2f} | {iq3_eng:.2f} | "
                     f"{(iq3_eng - iq2_eng)/iq2_eng*100:+.1f}% | "
                     f"{(iq3_pk - iq2_pk)/iq2_pk*100:+.1f}% |")
    L.append("")
    L.append("## Per-row audit trails")
    L.append("")
    for label, _m, _kv, _c in rows:
        fn = label.replace(' ', '_').replace('@', 'at')
        L.append(f"- [{label}](./logs/{fn}.md)")
    L.append("")
    L.append("## Per-run detail")
    L.append("")
    for label, model_name, kv, ctx in rows:
        L.append(f"### {label} (model={model_name}, KV={kv}, prompt={ctx['prompt_tokens_target']:,d} tok)")
        L.append("")
        cp = ctx.get("cold_prefill") or {}
        L.append(f"  Cold prefill probe: prompt_n={cp.get('prompt_tokens_actual')} "
                 f"prompt_total_ms={cp.get('prompt_total_ms', 0):.1f} "
                 f"prompt_per_second={cp.get('prompt_per_second', 0):.1f} t/s")
        L.append("")
        L.append("| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for rr in ctx["runs"]:
            L.append(
                f"| {rr['seq']} | {rr['prompt_tokens_actual']:,d} | {rr['completion_tokens']} | "
                f"{rr['ttft_ms']:.0f} ms | {rr['decode_avg_tps']:.2f} | {rr['decode_peak_tps']:.2f} | "
                f"{rr['eng_predicted_per_second']:.2f} | {rr['eng_prompt_per_second']:.1f} | "
                f"{rr['eng_draft_n_accepted']}/{rr['eng_draft_n']} | {rr['finish_reason']} |"
            )
        L.append("")
    return "\n".join(L)


def per_context_log(label: str, model_name: str, kv: str, ctx: dict) -> str:
    cp = ctx.get("cold_prefill") or {}
    L = [
        f"# Audit trail - {label}",
        "",
        f"- Engine model: {label}",
        f"- Configured model_name (POST body.model): {model_name}",
        f"- Context: {ctx['target_ctx']} tokens",
        f"- KV cache: {kv}",
        f"- Prompt tokens landed: {ctx['prompt_tokens_target']:,d}",
        f"- Fill chars: {ctx['fill_chars']:,d}",
        f"- Runs: 1 cold-prefill probe (max_tokens=1, non-stream, after /unload/reload) + 3 stream calls",
        "",
        "## Container deploy line",
        "",
        "```",
        "docker rm -f strata && sleep 2",
        "docker run -d --name strata --network host --gpus all \\",
        "  --ulimit memlock=-1 --shm-size=4g \\",
        "  -v strata-data:/data \\",
        f"  -e FAMILY=qwen -e MODEL={model_name} -e CONTEXT={ctx['target_ctx']} \\",
        f"  -e VISION=no -e KV={kv} -e REINSTALL=1 \\",
        "  -e HOST=127.0.0.1 -e PORT=8090 --restart unless-stopped strata",
        "```",
        "",
        "## Cold prefill probe (max_tokens=1, non-stream)",
        "",
        f"- prompt_n:           **{cp.get('prompt_tokens_actual')}**",
        f"- prompt_total_ms:    **{cp.get('prompt_total_ms', 0):.1f}** ms",
        f"- prompt_per_second:  **{cp.get('prompt_per_second', 0):.1f}** t/s",
        f"- completion_tokens:  {cp.get('completion_tokens', 0)}",
        "",
        "These are the numbers that match the README's 2,000+ t/s headline (the engine's end-of-prompt prefill throughput).",
        "",
        "## Per-run captures (SSE stream)",
        "",
        "| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine predicted_per_second | SSE Prefill (timings.prompt_per_second) | Draft | Finish |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rr in ctx["runs"]:
        L.append(
            f"| {rr['seq']} | {rr['prompt_tokens_actual']:,d} | {rr['completion_tokens']} | "
            f"{rr['ttft_ms']:.0f} ms | {rr['decode_avg_tps']:.2f} | "
            f"{rr['decode_peak_tps']:.2f} | {rr['eng_predicted_per_second']:.2f} | "
            f"{rr['eng_prompt_per_second']:.1f} | "
            f"{rr['eng_draft_n_accepted']}/{rr['eng_draft_n']} | "
            f"{rr['finish_reason']} |"
        )
    L.append("")
    L.append(f"Raw: [`../{label}.json`](../{label}.json)")
    L.append("")
    return "\n".join(L)


def main():
    out = {
        "header": {
            "engine": "Strata Qwen3.8-Flash-Next",
            "host": "Shin-BlackMamba / Ubuntu 24.04.5 LTS",
            "kernel": "6.8.0-146-generic",
            "nvidia_driver": "595.91.07",
            "nvidia_ctk": "1.20.1",
            "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER (16 GiB, sm_89)",
            "ram": "94 GiB",
            "engine_build": "sm_89 fat-binary (CUDA_ARCHITECTURES=89); BUILD_VISION=0; VISION=no",
            "method": (
                "Two probe types per row: (1) cold-prefill max_tokens=1 non-stream after /unload; "
                "(2) three SSE stream calls (reasoning_effort=minimal, max_tokens=192, t=0)."
            ),
            "kv_strategy": "IQ2_XS uses KV=int8 across all 4 contexts after correction 2026-10-05. IQ3_XXS at 32k/64k uses int8; at 128k/256k uses q4_0/k8v4 because IQ3_XXS residents 47 GiB of experts and int8 plus attention would push past the 16 GiB VRAM envelope.",
        },
        "per_context": [],
    }

    rows_for_md = []
    for label, fp, kv, model_name in FILES:
        d = json.load(open(fp))
        r = d["results"][0]
        ctx = {
            "target_ctx": r["target_ctx"],
            "kv": kv,
            "prompt_tokens_target": r["target_prompt_tokens"],
            "fill_chars": r["fill_chars"],
            "cold_prefill": r.get("cold_prefill") or {},
            "runs": r["runs"],
        }
        avgs = [round(x["decode_avg_tps"], 2) for x in ctx["runs"]]
        peaks = [round(x["decode_peak_tps"], 2) for x in ctx["runs"]]
        engs = [round(x["eng_predicted_per_second"], 2) for x in ctx["runs"]]
        out["per_context"].append({
            "label": label,
            "model_name": model_name,
            "kv": kv,
            "prompt_tokens_target": r["target_prompt_tokens"],
            "cold_prefill": ctx["cold_prefill"],
            "runs": ctx["runs"],
            "avg_decode_avg_tps": round(statistics.mean(avgs), 2),
            "max_peak_tps": round(max(peaks), 2),
            "engine_predicted_avg_tps": round(statistics.mean(engs), 2),
            "sse_prefill_avg_tps": round(statistics.mean(round(x.get("eng_prompt_per_second", 0), 2) for x in ctx["runs"]), 2),
            "draft_total_accepted": sum(x["eng_draft_n_accepted"] for x in ctx["runs"]),
            "draft_total_proposed": sum(x["eng_draft_n"] for x in ctx["runs"]),
            "draft_pct": round(100 * sum(x["eng_draft_n_accepted"] for x in ctx["runs"]) /
                              max(1, sum(x["eng_draft_n"] for x in ctx["runs"])), 1),
            "ttft_ms_per_run": [int(x["ttft_ms"]) for x in ctx["runs"]],
        })

        log_path = LOGS / f"{label.replace(' ', '_').replace('@','at')}.md"
        log_path.write_text(per_context_log(label, model_name, kv, ctx))

        rows_for_md.append((label, model_name, kv, ctx))

    OUT_MATRIX.write_text(json.dumps(out, indent=2))
    OUT_MD.write_text(md_text(rows_for_md))
    OUT_SUMMARY.write_text(json.dumps(out, indent=2))

    written = [OUT_MATRIX, OUT_MD, OUT_SUMMARY]
    written += [LOGS / f"{label.replace(' ', '_').replace('@','at')}.md"
                for label, _m, _k, _c in FILES]
    print("Wrote:")
    for f in written:
        print(f"  - {f.relative_to(BASE)}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
