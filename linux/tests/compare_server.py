"""Alternate real-weight vanilla/Linux decode runs with matching server settings.

Reports observations under load; every load snapshot must be <4 for the 95% gate.
Build vanilla first with build_baseline.py. Run this script inside Docker.
Use --require-quiet to stop on contention and --diagnostic-controls to separate
LoRA, prerouter and advice costs without changing the acceptance criterion.
"""

import argparse
from pathlib import Path
import statistics

from build_baseline import cache_options, check_pin
from e2e_server import check_trace
from server_test_utils import (ROOT, completion, digest, request, require,
                               running_server, save_json, snapshot)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vanilla-source", type=Path, required=True)
    parser.add_argument("--vanilla-build", type=Path, required=True)
    parser.add_argument("--linux-build", type=Path, default=ROOT / "wt/linux/build-cpu")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--require-quiet", action="store_true",
                        help="save UNDER_LOAD and exit 2 before model work, or if contention appears")
    parser.add_argument("--diagnostic-controls", action="store_true",
                        help="also measure Linux LoRA-only and prerouter accounting-only modes")
    args = parser.parse_args()
    return prepare_arguments(args)


def prepare_arguments(args):
    require(args.rounds >= 3, "at least three rounds are required")
    args.converted_dir = args.converted_dir or Path(str(args.model_dir) + "-gguf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def quiet_check(args, result, samples=()):
    if not args.require_quiet:
        return True
    point = snapshot()
    result.setdefault("quiet_checks", []).append(point)
    loads = [point["load"][0]] + [sample[when]["load"][0]
                                for sample in samples for when in ("before", "after")]
    if max(loads) < 4:
        return True
    result["summary"] = {"status": "UNDER_LOAD", "reason": "1-minute load must be below 4",
                         "max_checked_load": max(loads)}
    save_json(args.output_dir / "compare.json", result)
    print(f"UNDER_LOAD: load={max(loads):.2f}; quiet comparison stopped (exit 2)", flush=True)
    return False


def prompt_tokens(port):
    response = request(port, "/tokenize", {"content": "The natural numbers in ascending order are " * 100})
    tokens = response["tokens"][:256]
    require(len(tokens) == 256, "benchmark prompt must contain exactly 256 tokens")
    return tokens


def decode_rate(timings):
    # The server samples token one during prefill; predicted_ms covers N-1
    # decode steps. Use its timing convention for single and parallel streams.
    seconds = max(row["predicted_ms"] for row in timings) / 1000
    require(seconds > 0, "invalid decode duration")
    return sum(row["predicted_n"] - 1 for row in timings) / seconds


def measure(port, prompt, streams):
    before = snapshot()
    response = completion(port, prompt if streams == 1 else [prompt] * streams, 128)
    after = snapshot()
    rows = [response] if streams == 1 else response
    require(isinstance(rows, list) and len(rows) == streams, "missing completion streams")
    timings = [row["timings"] for row in rows]
    require(all(len(row["tokens"]) == 128 for row in rows), "benchmark decode was truncated")
    require(all(row["prompt_n"] == 256 and row["predicted_n"] == 128 for row in timings),
            "benchmark token counts do not match")
    return {"streams": streams, "tok_s": decode_rate(timings),
            "timings": timings, "before": before, "after": after}


def server_features(args, mode):
    features = {"E0_PREROUTER": str(args.model_dir / "prerouter_edge0_8b.safetensors"),
                "E0_PREFETCH": "1" if mode == "active" else "0"} if mode in ("active", "accounting") else {}
    lora = args.converted_dir / "lora_edge0_8b-gguf.gguf" if mode in ("active", "accounting", "lora") else None
    return lora, features


def run_server(args, mode, threads, round_index):
    build = args.vanilla_build if mode == "vanilla" else args.linux_build
    log = args.output_dir / f"bench-{round_index}-{mode}-t{threads}.log"
    lora, features = server_features(args, mode)
    with running_server(build / "bin/llama-server", args.converted_dir / "edge0-8b.gguf",
                        log, threads=threads, lora=lora, features=features, batch=2048) as port:
        tokens = prompt_tokens(port)
        completion(port, tokens, 16)
        samples = [measure(port, tokens, n) for n in ((1, 4, 8) if threads == 8 else (1,))]
    trace = check_trace(log, "advice" if mode == "active" else "accounting") if features else None
    require(features or "[pref-trace]" not in log.read_text(), "dormant hook activated")
    if mode == "lora":
        check_trace(log, "dormant")
    return dict(mode=mode, threads=threads, round=round_index, samples=samples, trace=trace)


def grouped_rates(runs):
    groups = {}
    quiet = True
    for run in runs:
        for sample in run["samples"]:
            key = (run["mode"], run["threads"], sample["streams"])
            groups.setdefault(key, []).append(sample["tok_s"])
            quiet &= max(sample[when]["load"][0] for when in ("before", "after")) < 4
    return groups, quiet


def workload_summary(groups, threads, streams):
    rates = {mode: statistics.median(values) for (mode, t, p), values in groups.items()
             if (t, p) == (threads, streams)}
    ratios = {mode: rates[mode] / rates["vanilla"] for mode in ("dormant", "active")}
    row = dict(threads=threads, streams=streams, median_tok_s=rates, ratios=ratios)
    if all(mode in rates for mode in ("lora", "accounting")):
        pairs = {"lora_vs_dormant": ("lora", "dormant"), "accounting_vs_lora": ("accounting", "lora"),
                 "active_vs_accounting": ("active", "accounting"), "active_vs_lora": ("active", "lora")}
        row["feature_ratios"] = {key: rates[a] / rates[b] for key, (a, b) in pairs.items()}
    return row


def summary(runs):
    groups, quiet = grouped_rates(runs)
    rows = [workload_summary(groups, threads, streams) for threads, streams in ((8, 1), (8, 4), (8, 8), (16, 1))]
    passed = all(min(row["ratios"].values()) >= 0.95 for row in rows)
    return {"status": ("PASS" if passed else "FAIL") if quiet else "UNDER_LOAD",
            "observed_within_5_percent": passed, "rows": rows}


def run_round(args, index, result):
    modes = ("vanilla", "dormant", "lora", "accounting", "active") if args.diagnostic_controls else (
        "vanilla", "dormant", "active")
    modes = modes if index % 2 else tuple(reversed(modes))
    for threads in (8, 16):
        for mode in modes:
            if not quiet_check(args, result):
                return False
            run = run_server(args, mode, threads, index)
            result["runs"].append(run)
            save_json(args.output_dir / "compare.json", result)
            rates = ", ".join(f"p{s['streams']}={s['tok_s']:.2f}" for s in run["samples"])
            print(f"round={index} {mode} t{threads}: {rates}", flush=True)
            if not quiet_check(args, result, run["samples"]):
                return False
    return True


def main():
    args = arguments()
    result = {"runs": []}
    if not quiet_check(args, result):
        return 2
    options = cache_options(args.linux_build)
    require(cache_options(args.vanilla_build) == options, "baseline and Linux CMake options differ")
    result.update(pin=check_pin(args.vanilla_source), cmake=options,
                  model_sha256=digest(args.converted_dir / "edge0-8b.gguf"))
    result["rate_formula"] = "sum(predicted_n - 1) / max(predicted_ms / 1000)"
    for index in range(1, args.rounds + 1):
        if not run_round(args, index, result):
            return 2
    result["summary"] = summary(result["runs"])
    save_json(args.output_dir / "compare.json", result)
    print(result["summary"], flush=True)
    require(result["summary"]["status"] != "FAIL", "quiet-machine speed gate failed")


if __name__ == "__main__":
    raise SystemExit(main())
