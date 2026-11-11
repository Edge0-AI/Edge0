"""Derive a compact, auditable summary from saved benchmark artifacts."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results/2026-10-09-rtx3060-0141")
out = parser.parse_args().out.resolve()


def rows(name):
    path = out / name
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


requests = rows("requests.jsonl")
telemetry = rows("telemetry.jsonl")
thermal = rows("thermal-flags.jsonl")
events = rows("events.jsonl")
starts = {}
for event in events:
    if event.get("event") == "request_start":
        starts.setdefault((event.get("model"), event["label"]), []).append(dt.datetime.fromisoformat(event["utc"]).timestamp())


def condition(row):
    mode = row.get("weights_mode", "native")
    env = row.get("engine_environment", {})
    config = out / "configs" / f"{row['server_number']:03d}-{row['model']}-{row['context']}.json"
    args = json.loads(config.read_text(encoding="utf-8"))["args"]
    budget = float(args[args.index("--resident-budget-gib")+1]) if "--resident-budget-gib" in args else None
    return mode, budget, env.get("STRATA_RESIDENT_HEADROOM_GIB"), env.get("STRATA_UNBUFFERED_LOAD", "auto")
result = []
groups = {}
for row in requests:
    if row["group"] not in ("speed", "language", "bridge", "recall"):
        continue
    workload = row["label"].split("-")[1] if row["group"] == "language" else (
        "depth-" + row["label"].split("-d")[-1] if row["group"] == "recall" else "code")
    key = (row["model"], *condition(row), row["context"], row["group"], workload, row.get("target"))
    groups.setdefault(key, []).append(row)
for (model, mode, budget, headroom, file_io, context, group, workload, target), trials in sorted(groups.items(), key=str):
    valid = [r for r in trials if r.get("ok") and r.get("token_count_matches") and r.get("engine", {}).get("reused") == 0]
    windows = []
    for trial in trials:
        previous = [s for s in starts.get((model, trial["label"]), []) if s <= trial["epoch"]]
        if previous:
            windows.append((trial["label"], max(previous), trial["epoch"]))
    sensors = [r for r in telemetry if r.get("model") == model and any(
        r.get("case") == label and start <= r["epoch"] <= end for label, start, end in windows)]
    item = {"model": model, "weights_mode": mode, "context": context, "group": group, "workload": workload, "target": target,
            "resident_budget_gib": budget, "resident_headroom_gib": headroom, "file_io": file_io,
            "attempts": len(trials), "valid_fresh": len(valid),
            "actual_prompt_tokens": sorted({r["local_prompt_tokens"] for r in valid}),
            "output_tokens": [r["engine"].get("output_tokens") for r in valid],
            "recall_pass": sum(bool(r.get("correct")) for r in valid if "expected" in r),
            "recall_trials": sum("expected" in r for r in valid)}
    metrics = {
        "prefill_tok_s": [r["engine"].get("prompt_read", 0)*1000/r["engine"]["prompt_ms"] for r in valid if r["engine"].get("prompt_ms", 0) > 0],
        "decode_tok_s": [r["engine"]["decode_tok_s"] for r in valid if r["engine"].get("decode_tok_s")],
        "client_ttft_s": [r["client_ttft_s"] for r in valid if r.get("client_ttft_s") is not None],
        "client_elapsed_s": [r["client_elapsed_s"] for r in valid if r.get("client_elapsed_s") is not None]}
    for name, key in (("decode_file_mb", "file_mb"), ("decode_ram_blobs", "ram_blobs"),
                      ("decode_file_blobs", "file_blobs"), ("expert_hit_rate", "hit_rate"), ("pcie_share", "pcie_share")):
        metrics[name] = [r["engine"][key] for r in valid if r.get("engine", {}).get(key) is not None]
    for name, values in metrics.items():
        item[name] = {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
    for name, operation in (("ram_available_bytes", min), ("commit_fraction", max), ("vram_mib", max),
                            ("temperature_c", max), ("graphics_mhz", min), ("gpu_power_w", max)):
        values = [r[name] for r in sensors if r.get(name) is not None]
        item[name + ("_min" if operation is min else "_max")] = operation(values) if values else None
    result.append(item)

grades = rows("grades.jsonl")
quality = {}
for row in grades:
    quality.setdefault(row["model"] + "/" + row["group"], []).append(row["pass"])
overview = {"request_count": len(requests), "ok_requests": sum(bool(r.get("ok")) for r in requests),
            "failed_requests": [{k: r.get(k) for k in ("model", "label", "context", "weights_mode", "error_type")} for r in requests if not r.get("ok")],
            "thermal_flag_samples": len(thermal),
            "sw_thermal_active_samples": sum(r.get("clocks_event_reasons.sw_thermal_slowdown") == "Active" for r in thermal),
            "quality": {k: {"pass": sum(v), "attempted": len(v)} for k, v in quality.items()},
            "measurements": result}
(out / "analysis.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with (out / "measurements.csv").open("w", encoding="utf-8-sig", newline="") as f:
    keys = list(result[0]) if result else []
    writer = csv.DictWriter(f, fieldnames=keys)
    if keys:
        writer.writeheader()
        writer.writerows([{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in result])
print(json.dumps({"requests": len(requests), "measurement_conditions": len(result), "failed_requests": len(overview["failed_requests"])}))
