#!/usr/bin/env python3
"""Collect data/*.json (the measured runs; files named *-discarded-* are left out) into runs.csv and summary.json,
with per-run memory peaks from data/telemetry.jsonl.

Units: prompt tok/s = freshly read prompt tokens / engine prompt_ms; decode tok/s = engine-generated tokens /
engine decode_ms (GET /metrics, 0.1 ms resolution); TTFT and total = client wall time, seconds, from just before the
HTTP request to the first non-empty text delta / the end of the stream (loopback, includes the server's tokenizing).
GPU memory = nvidia-smi memory.used (MiB, whole card); host RAM used = MemTotal - MemAvailable (GiB, whole system).
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def telemetry():
    rows = []
    path = DATA / "telemetry.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def peaks(tel, t0, t1):
    sel = [r for r in tel if t0 - 1 <= r["epoch_s"] <= t1 + 1]
    if not sel:
        return {}
    gpu = [int(r["gpu"].split(",")[0]) for r in sel]
    host = [(r["memory"]["MemTotal_KiB"] - r["memory"]["MemAvailable_KiB"]) / 2**20 for r in sel]
    swap = [(r["memory"]["SwapTotal_KiB"] - r["memory"]["SwapFree_KiB"]) / 2**20 for r in sel]
    rss = [r.get("rss_KiB", {}).get("strata", 0) / 2**20 for r in sel]
    return {"samples": len(sel), "gpu_mem_used_mib_max": max(gpu), "host_used_gib_max": round(max(host), 2),
            "swap_used_gib_max": round(max(swap), 2), "engine_rss_gib_max": round(max(rss), 2)}


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": round(statistics.median(values), 2), "min": round(min(values), 2), "max": round(max(values), 2),
            "n": len(values)}


def main():
    tel = telemetry()
    rows = []
    for f in sorted(DATA.glob("*.json")):
        if "discarded" in f.name or not f.name.startswith(("fresh-", "followup-", "image-")):
            continue
        r = json.loads(f.read_text())
        e, m = r.get("engine") or {}, r.get("metrics_record") or {}
        read = e.get("read")
        prompt_ms = m.get("prompt_ms") if m.get("prompt_tokens") == e.get("prompt_tokens") else e.get("prompt_ms_log")
        decode_ms = m.get("decode_ms") if m.get("prompt_tokens") == e.get("prompt_tokens") else e.get("decode_ms_log")
        gen = m.get("engine_generated", e.get("generated"))
        row = {"label": r["label"], "kind": r.get("kind", "image"), "target": r.get("target"),
               "prompt_tokens": e.get("prompt_tokens"), "reused": e.get("reused"), "read": read,
               "prompt_ms": prompt_ms, "prompt_tok_s": round(read / (prompt_ms / 1000), 1) if read and prompt_ms else None,
               "generated": gen, "decode_ms": decode_ms,
               "decode_tok_s": round(gen / (decode_ms / 1000), 1) if gen and decode_ms else None,
               "drafts": f"{e.get('drafts_accepted')}/{e.get('drafts_proposed')}",
               "draft_accept_pct": round(100 * e["drafts_accepted"] / e["drafts_proposed"], 1) if e.get("drafts_proposed") else None,
               "decode_hit_pct": e.get("hit_rate_pct"), "kv_vram_pct": e.get("kv_vram_pct"),
               "kv_ram_mib": e.get("kv_ram_mib"), "ttft_s": round(r["client_ttft_s"], 2) if r.get("client_ttft_s") else None,
               "total_s": round(r["client_total_s"], 2), "finish": r.get("finish_reason"),
               "needle": r.get("needle"), "needle_found": r.get("needle_found"), "correct": r.get("correct"),
               "foreign_overlap": r.get("foreign_overlap"), "attempt": r.get("attempt")}
        row.update(peaks(tel, r["epoch_start"], r.get("epoch_end", r["epoch_start"] + r["client_total_s"])))
        rows.append(row)
    with (HERE / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        key = f"{r['kind']}-{r['target']}" if r["kind"] != "image" else "image"
        groups.setdefault(key, []).append(r)
    summary = {k: {f: stats([r.get(f) for r in g]) for f in
                   ("prompt_tokens", "reused", "read", "prompt_tok_s", "generated", "decode_tok_s", "draft_accept_pct",
                    "decode_hit_pct", "kv_vram_pct", "ttft_s", "total_s", "gpu_mem_used_mib_max", "host_used_gib_max",
                    "swap_used_gib_max", "engine_rss_gib_max")} for k, g in groups.items()}
    if tel:
        summary["telemetry_whole_session"] = peaks(tel, tel[0]["epoch_s"], tel[-1]["epoch_s"])
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for k, v in summary.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
