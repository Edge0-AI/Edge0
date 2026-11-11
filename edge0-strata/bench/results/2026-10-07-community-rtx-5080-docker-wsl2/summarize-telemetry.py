#!/usr/bin/env python3
"""Turn a telemetry.jsonl from monitor.py into memory-summary.json.

Reports the observed peaks and ranges of the engine's own memory counters, in the
units it reports them in (bytes), so the report can state where each number came from.
"""
import argparse
import json
import statistics
from pathlib import Path

FIELDS = ("gpu_mem_used", "gpu_mem_total", "gpu_util", "gpu_temp_c", "gpu_power_w",
          "gpu_pcie_gen", "gpu_pcie_width", "ram_used", "ram_total", "disk_read_mb",
          "vram_free_mib", "expert_cache_mib")

GIB = 1024 ** 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    samples, broken = [], 0
    for line in args.telemetry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            broken += 1                     # one line can be lost where a sampler restarts
            continue
        if "error" not in row:
            samples.append(row)

    times = [row["t"] for row in samples]
    summary = {
        "source": str(args.telemetry.name),
        "samples": len(samples),
        "unparseable_lines": broken,
        "sampled_seconds": round(max(times) - min(times), 1) if times else 0,
        "note": ("Each sample is one GET /metrics, so the values are the engine's own view. "
                 "Sampled peaks can miss shorter spikes between samples."),
        "fields": {},
    }
    for field in FIELDS:
        values = [row[field] for row in samples if row.get(field) is not None]
        if not values:
            continue
        entry = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if field in ("gpu_mem_used", "gpu_mem_total", "ram_used", "ram_total"):
            entry["max_gib"] = round(max(values) / GIB, 2)
            entry["median_gib"] = round(statistics.median(values) / GIB, 2)
        summary["fields"][field] = entry

    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["fields"], indent=2))


if __name__ == "__main__":
    main()
