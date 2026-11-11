#!/usr/bin/env python3
"""Interleaved independent-engine A/B pairs with explicitly declared configuration differences.

One complete shape warmup and one measurement per engine. Reporting stays descriptive;
the quality gates, all workload regressions and resource observations must be reviewed separately.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from bench_engine import save
from single_gpu import ROOT, validate_config


def summarize(directory, manifest):
    groups = {}
    for item in manifest["sessions"]:
        data = json.loads((directory / item["name"] / "results.json").read_text())
        if item["status"] != "passed" or data["binary_sha256"] != manifest["configs"][item["arm"]]["binary_sha256"]:
            raise ValueError("failed session or changed engine")
        if data["fixtures_sha256"] != manifest["fixtures_sha256"]:
            raise ValueError("changed fixtures")
        records = [r for r in data["records"] if r["phase"] == "measurement"]
        expected = {(n, inc) for n in manifest["lengths"] for inc in manifest["increments"]} if manifest["mode"] == "incremental" else {(n, 0) for n in manifest["lengths"]}
        actual = set()
        for r in records:
            key = (r["base_tokens"], r["increment"]) if manifest["mode"] == "incremental" else (r["input_tokens"], 0)
            if key in actual:
                raise ValueError("duplicate measurement")
            actual.add(key)
            if not r["output_ids"] or r["finish"] not in {"length", "stop"}:
                raise ValueError("incomplete generation")
            if (manifest["mode"] == "fresh" and r["reused"] != 0) or (manifest["mode"] == "incremental" and not key[0] - 1 <= r["reused"] <= key[0]):
                raise ValueError("incorrect reuse path")
            groups.setdefault(key, {}).setdefault(item["pair"], {})[item["arm"]] = r
        if actual != expected:
            raise ValueError("missing measurement")
    rows = []
    for key, pairs in sorted(groups.items()):
        row = {"history_or_input_tokens": key[0], "increment": key[1], "pairs": []}
        for index, pair in sorted(pairs.items()):
            a, b = pair["baseline"], pair["candidate"]
            equal_keys = ("input_ids_sha256", "output_ids", "generated", "finish", "reused", "prompt_read", "drafts_accepted", "drafts_offered")
            changed = [k for k in equal_keys if a[k] != b[k]]
            values = {k: {"baseline": a[k], "candidate": b[k], "time_reduction_pct": 100 * (1 - b[k] / a[k])}
                      for k in ("prompt_ms", "decode_ms", "ttft_s", "wall_s")}
            row["pairs"].append({"index": index, "differing_work_fields": changed, **values})
        row["identical_work"] = all(not p["differing_work_fields"] for p in row["pairs"])
        for metric in ("prompt_ms", "decode_ms", "ttft_s", "wall_s"):
            reductions = [p[metric]["time_reduction_pct"] for p in row["pairs"]]
            row[metric] = {"median_paired_time_reduction_pct": statistics.median(reductions),
                           "min_paired_time_reduction_pct": min(reductions), "max_paired_time_reduction_pct": max(reductions),
                           "faster_pairs": sum(v > 0 for v in reductions),
                           "regressions_over_2pct": sum(v < -2 for v in reductions),
                           **{f"{arm}_median": statistics.median(p[metric][arm] for p in row["pairs"]) for arm in ("baseline", "candidate")}}
        rows.append(row)
    if any(len(r["pairs"]) != manifest["pairs"] for r in rows):
        raise ValueError("incomplete pair count")
    return {"scope": "descriptive paired results; no automatic acceptance or quality claim", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--mode", choices=["fresh", "incremental"], default="fresh")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 4096, 32768])
    parser.add_argument("--increments", type=int, nargs="+", default=[256, 512, 900, 2048, 4096])
    parser.add_argument("--vary-env", action="append", default=[],
                        help="explicit environment difference for a configuration experiment; default allows none")
    args = parser.parse_args()
    if args.pairs < 1 or min(args.lengths + args.increments) < 1:
        parser.error("positive pair count and lengths required")
    configs = {}
    for arm, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        cfg = json.loads(path.read_text())
        validate_config(cfg)
        configs[arm] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "binary_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(), "config": cfg}
        if any(k in cfg["env"] for k in ("STRATA_STATE_HASH", "STRATA_STATE_HASH_GDN", "STRATA_DUMP_FIRST_LOGITS",
                                        "STRATA_TRACE", "STRATA_PREFILL_LAYER_HASH_T", "STRATA_PREFILL_GEMM_CAPTURE",
                                        "STRATA_PREFILL_TIMING", "STRATA_DECODE_TIMING", "STRATA_IO_STATS",
                                        "STRATA_HIPBLASLT_VERBOSE", "ROCBLAS_LAYER", "HIPBLASLT_LOG_LEVEL")):
            raise ValueError("performance config contains diagnostic instrumentation")
    for key in ("args", "tokenizer", "cwd", "lib_dirs", "r9700_numa"):
        if configs["baseline"]["config"].get(key) != configs["candidate"]["config"].get(key):
            raise ValueError("paired comparison requires identical controls: " + key)
    a_env, b_env = (configs[arm]["config"]["env"] for arm in ("baseline", "candidate"))
    changed_env = {key for key in a_env.keys() | b_env.keys() if a_env.get(key) != b_env.get(key)}
    if changed_env != set(args.vary_env):
        raise ValueError("actual environment differences do not match --vary-env: " + str(sorted(changed_env)))
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "interleaved A/B; independent engine per arm; one warmup and measurement per shape",
                "vary_env": {key: {"baseline": a_env.get(key), "candidate": b_env.get(key)} for key in sorted(changed_env)},
                "mode": args.mode, "pairs": args.pairs, "configs": configs, "lengths": args.lengths,
                "increments": args.increments, "max_new": 256,
                "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(), "sessions": []}
    save(args.out / "matrix.json", manifest)
    for pair in range(args.pairs):
        for arm in (("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")):
            c = configs[arm]; cfg = c["config"]
            name = f"{pair:02d}-{arm}"
            numa = cfg.get("r9700_numa", {"cpu_nodes": "0,1", "preferred_node": "1"})
            runner = "bench_engine.py" if args.mode == "fresh" else "bench_incremental.py"
            extra = ["--shape-warmups", "1"] if args.mode == "fresh" else ["--increments", *map(str, args.increments)]
            cmd = [sys.executable, str(ROOT / "tools/hip/r9700/single_gpu.py"),
                   "--device-binary", str(Path(cfg["exe"]).with_name("strata-device")),
                   "--config", c["path"], "--record", str((args.out / f"{name}-launch.json").resolve()),
                   "--cpu-nodes", numa["cpu_nodes"], "--preferred-node", str(numa["preferred_node"]), "run", "--",
                   sys.executable, str(ROOT / "tools/hip/r9700" / runner), "--config", c["path"],
                   "--fixtures", str(args.fixtures.resolve()), "--out", str((args.out / name).resolve()),
                   "--lengths", *map(str, args.lengths), "--repeats", "1", "--max-new", "256", *extra]
            entry = {"pair": pair, "arm": arm, "name": name, "command": cmd, "status": "running",
                     "start_utc": datetime.now(timezone.utc).isoformat()}
            manifest["sessions"].append(entry); save(args.out / "matrix.json", manifest)
            print(entry["start_utc"], "start", name, flush=True)
            with (args.out / f"{name}-console.log").open("x") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode,
                         end_utc=datetime.now(timezone.utc).isoformat())
            save(args.out / "matrix.json", manifest)
            print(entry["end_utc"], entry["status"], name, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
    summary = summarize(args.out, manifest)
    save(args.out / "summary.json", summary)
    manifest["complete"] = True
    save(args.out / "matrix.json", manifest)


if __name__ == "__main__":
    main()
