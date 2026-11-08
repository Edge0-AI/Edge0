"""35B conversion, streamed benchmarks and a measured 4 GiB RSS check, in Docker.

Given the downloaded 35B, conversion checks all tensors and adapter shapes.
Given a budgeted Linux server, chat answers and positive counters prove activation.
Given 1/4/8 concurrent requests, record decode, TTFT, load and process memory.
The 4 GiB check reports a failure if sampled RSS or kernel high-water RSS exceeds it.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.request import Request, urlopen

from build_baseline import cache_options, check_pin
from compare_server import decode_rate
from example_cases import CASES
from example_server import collect_stream, metrics, payload
from resource_watch import host_snapshot, watched_process, within_budget
from server_test_utils import ROOT, digest, free_port, require, save_json, server_args, wait_ready


def environment(features=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E0_", "LLAMA_", "GGML_"))}
    return dict(env, OMP_WAIT_POLICY="PASSIVE", **(features or {}))


def convert(args):
    command = [sys.executable, str(ROOT / "windows/tools/convert_mlx_to_gguf.py"),
               "--dir", str(args.model_dir), "--out", str(args.converted_dir)]
    before, started = host_snapshot(), time.monotonic()
    with watched_process(command, environment(), args.output_dir / "conversion.log",
                         args.output_dir / "conversion-memory.jsonl") as (process, watch):
        require(process.wait() == 0, "35B conversion failed; see conversion.log")
    result = dict(wall_s=time.monotonic() - started, before=before, after=host_snapshot(), memory=watch.summary)
    save_json(args.output_dir / "conversion.json", result)
    print(f"PASS conversion: {result['wall_s']:.1f}s", flush=True)


def artifacts(args):
    manifest = json.loads((args.converted_dir / "manifest.json").read_text())
    for name, expected in manifest["inputs_sha256"].items():
        require(digest(args.model_dir / name) == expected, f"source SHA mismatch: {name}")
    for kind, name in (("gguf", "edge0-35b.gguf"), ("lora", "lora_edge0_35b-gguf.gguf")):
        require(digest(args.converted_dir / name) == manifest["artifacts"][kind], f"artifact SHA mismatch: {kind}")
    return manifest


def features(args, mode):
    if mode == "vanilla":
        return {}, None, 0
    budget = 4096 if mode == "lowram" else 12288
    env = dict(E0_PREROUTER=str(args.model_dir / "prerouter_edge0_35b.safetensors"), E0_PREFETCH="1")
    return env, args.converted_dir / "lora_edge0_35b-gguf.gguf", budget


def command(args, mode, port):
    env, lora, budget = features(args, mode)
    binary = (args.vanilla_build if mode == "vanilla" else args.linux_build) / "bin/llama-server"
    result = server_args(binary, args.converted_dir / "edge0-35b.gguf", port, 8, lora, 512, 8)
    if budget:
        result += ["--mem-budget-mb", str(budget)]
    # Avoid permanent copies of MoE tensors; both builds use identical settings.
    result += ["--no-repack"]
    return result, environment(env), budget


def benchmark_payload():
    query = payload(dict(prompt="Continue this ordered list with numbers only: " + ", ".join(map(str, range(1, 81)))))
    return dict(query, max_tokens=128, ignore_eos=True)


def streamed_request(port, query, output, barrier=None):
    if barrier:
        barrier.wait(timeout=30)
    before, started = host_snapshot(), time.monotonic()
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(query).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=1800) as response:
        require(response.status == 200, "chat completion failed")
        row = collect_stream(response, time.monotonic, started)
    finished = time.monotonic()
    row.update(request=query, before=before, after=host_snapshot(), started_s=started, finished_s=finished,
               client_wall_s=finished - started, **metrics(row))
    save_json(output, row)
    return row


def measure(args, port, mode, streams):
    before, started = host_snapshot(), time.monotonic()
    barrier = threading.Barrier(streams)
    with ThreadPoolExecutor(max_workers=streams) as pool:
        jobs = [pool.submit(streamed_request, port, benchmark_payload(),
                            args.output_dir / f"{mode}-p{streams}-{i}.json", barrier) for i in range(streams)]
        rows = [job.result() for job in jobs]
    wall = time.monotonic() - started
    require(all(row["generated_tokens"] == 128 for row in rows), "benchmark decode was truncated")
    require(max(row["started_s"] for row in rows) < min(row["finished_s"] for row in rows),
            "benchmark requests did not overlap")
    result = dict(streams=streams, rows=rows, wall_s=wall, before=before, after=host_snapshot(),
                  decode_tok_s=decode_rate([row["timings"] for row in rows]),
                  aggregate_wall_tok_s=sum(row["generated_tokens"] for row in rows) / wall)
    print(f"{mode} p{streams}: decode {result['decode_tok_s']:.2f} tok/s, wall {wall:.2f}s", flush=True)
    return result


def check_trace(log, budget=None):
    text = Path(log).read_text()
    if budget:
        require(f"[mem-budget] enabled: {budget} MiB" in text, "memory budget not activated")
    require(re.search(r"loaded \d+ tensors", text), "LoRA load not confirmed")
    require("INIT ok: H=2048 E=256 K=4" in text, "35B prerouter not initialized with GGUF dimensions")
    matches = re.findall(r"steps=(\d+) heads=(\d+) experts=(\d+).*?advice=(\d+) failures=(\d+)", text)
    require(matches, "no prerouter accounting")
    steps, heads, experts, advice, failures = map(int, matches[-1])
    require(min(steps, heads, experts, advice) > 0 and failures == 0, "prefetch not active or failed")
    return dict(steps=steps, heads=heads, experts=experts, advice=advice, failures=failures,
                budget_failures=len(re.findall(r"\[mem-budget\] enforcement failed", text)))


def public_command(argv):
    """The command line with the home folder shortened to ~, so saved results name no local user."""
    home = str(Path.home())
    return [str(arg).replace(home, "~") for arg in argv]


def run_server(args, mode):
    port, log = free_port(), args.output_dir / f"server-{mode}.log"
    argv, env, budget = command(args, mode, port)
    result = dict(mode=mode, before=host_snapshot(), budget_mib=budget, command=public_command(argv), samples=[])
    started = time.monotonic()
    with watched_process(argv, env, log, args.output_dir / f"memory-{mode}.jsonl") as (process, watch):
        wait_ready(process, port, log)
        result["load_s"] = time.monotonic() - started
        watch.check()
        run_workloads(args, mode, port, result, watch)
    require(process.returncode == 0, f"server exited with {process.returncode}")
    result.update(after=host_snapshot(), memory=watch.summary)
    result["trace"] = check_trace(log, budget) if budget else None
    result["rss_within_budget"] = within_budget(watch.summary, budget) if budget else None
    save_json(args.output_dir / f"{mode}.json", result)
    print(f"PASS {mode} e2e; RSS within budget: {result['rss_within_budget']}", flush=True)
    return result


def run_workloads(args, mode, port, result, watch):
    query = payload(CASES[0])
    if mode == "lowram":
        query["max_tokens"] = 64
    result["book"] = streamed_request(port, query, args.output_dir / f"{mode}-book.json")
    require(result["book"]["finish_reason"] in ("stop", "length"), "unexpected book finish reason")
    for streams in ((1,) if mode == "lowram" else (1, 4, 8)):
        watch.check()
        result["samples"].append(measure(args, port, mode, streams))
        save_json(args.output_dir / f"{mode}-partial.json", result)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vanilla-source", type=Path, required=True)
    parser.add_argument("--vanilla-build", type=Path, required=True)
    parser.add_argument("--linux-build", type=Path, default=ROOT / "wt/linux/build-cpu")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("vanilla", "active", "lowram"), default=["vanilla", "active", "lowram"])
    args = parser.parse_args()
    args.converted_dir = args.converted_dir or Path(str(args.model_dir) + "-gguf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def preflight(args):
    point = host_snapshot()
    require(point["memory_bytes"]["MemAvailable"] >= 4 << 30, "less than 4 GiB RAM available")
    require(cache_options(args.linux_build) == cache_options(args.vanilla_build), "CMake settings differ")
    result = dict(before=point, pin=check_pin(args.vanilla_source), cmake=cache_options(args.linux_build))
    save_json(args.output_dir / "preflight.json", result)


def main():
    args = arguments()
    preflight(args)
    if args.convert:
        convert(args)
    manifest = artifacts(args)
    for mode in args.modes:
        run_server(args, mode)
    require(artifacts(args) == manifest, "inference changed converted artifacts or inputs")
    print("PASS 35B artifact SHA verification; see each mode's measured RSS verdict", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: 35B run: {error}", file=sys.stderr, flush=True)
        raise
