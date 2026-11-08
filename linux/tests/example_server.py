"""Measure four serial and concurrent 8B chat examples; run inside Docker.

Given the converted 8B with LoRA/prerouter/advice, requests return complete
answers and timing evidence. Known synthetic labels measure tagging accuracy.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen

from e2e_server import check_trace
from example_cases import CASES, score_tags
from server_test_utils import ROOT, digest, require, running_server, save_json, snapshot


def payload(case):
    return {"messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0, "seed": 4242, "max_tokens": 400, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}


def ingest(result, event, clock, started):
    if "error" in event:
        raise RuntimeError(f"server error event: {event['error']}")
    result["events"].append(event)
    for choice in event.get("choices", []):
        content = choice.get("delta", {}).get("content") or ""
        if content and result["ttft_s"] is None:
            result["ttft_s"] = clock() - started
        result["answer"] += content
        if choice.get("finish_reason"):
            result["finish_reason"] = choice["finish_reason"]
    for key in ("usage", "timings"):
        if event.get(key):
            result[key] = event[key]


def collect_stream(source, clock, started):
    result = dict(events=[], answer="", ttft_s=None)
    done = False
    for line in source:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            done = True
            break
        ingest(result, json.loads(data), clock, started)
    require(done and result["answer"].strip(), "truncated stream or empty answer")
    require(all(key in result for key in ("timings", "usage", "finish_reason")), "missing final statistics")
    return result


def metrics(result):
    timing, usage = result["timings"], result["usage"]
    count, milliseconds = timing["predicted_n"], timing["predicted_ms"]
    require(1 < count <= 400 and milliseconds > 0, "invalid decode timing/count")
    require(count == usage["completion_tokens"], "usage and timing token counts differ")
    require(usage["prompt_tokens"] == timing["prompt_n"] + timing.get("cache_n", 0), "prompt count mismatch")
    return dict(prompt_tokens=usage["prompt_tokens"], generated_tokens=count,
                decode_steps=count - 1, decode_tok_s=(count - 1) * 1000 / milliseconds,
                server_wall_s=(timing["prompt_ms"] + milliseconds) / 1000,
                server_prefill_s=timing["prompt_ms"] / 1000)


def run_request(port, case, mode, output, barrier=None):
    before, query = snapshot(), payload(case)
    if barrier:
        barrier.wait(timeout=30)
    started = time.monotonic()
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                      data=json.dumps(query).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        require(response.status == 200, "chat request failed")
        row = collect_stream(response, time.monotonic, started)
    row.update(case=case["id"], mode=mode, request=query, client_wall_s=time.monotonic() - started,
               before=before, after=snapshot(), **metrics(row))
    row["score"] = score_tags(case, row["answer"]) if "labels" in case else None
    save_json(output / f"{mode}-{case['id']}.json", row)
    print(f"{mode} {case['id']}: {row['generated_tokens']} tokens, "
          f"TTFT {row['ttft_s']:.2f}s, decode {row['decode_tok_s']:.2f} tok/s", flush=True)
    return row


def run_group(args, mode):
    parallel = 1 if mode == "serial" else 4
    log = args.output_dir / f"server-{mode}.log"
    features = {"E0_PREROUTER": str(args.model_dir / "prerouter_edge0_8b.safetensors"), "E0_PREFETCH": "1"}
    with running_server(args.binary, args.converted_dir / "edge0-8b.gguf", log, threads=8,
                        lora=args.converted_dir / "lora_edge0_8b-gguf.gguf", features=features, parallel=parallel) as port:
        before, started = snapshot(), time.monotonic()
        if parallel == 1:
            rows = [run_request(port, case, mode, args.output_dir) for case in CASES]
        else:
            barrier = threading.Barrier(4)
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = [pool.submit(run_request, port, case, mode, args.output_dir, barrier) for case in CASES]
                rows = [job.result() for job in jobs]
        wall = time.monotonic() - started
    group = dict(before=before, after=snapshot(), wall_s=wall, trace=check_trace(log, "advice"),
                 aggregate_tok_s=sum(row["generated_tokens"] for row in rows) / wall)
    return rows, group


def timing_table(rows):
    text = ["| Run | Prompt | Prompt tokens | Generated | TTFT s | Decode tok/s | Server total s | Client wall s | Load before/after |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        load = f"{row['before']['load'][0]:.2f}/{row['after']['load'][0]:.2f}"
        text.append(f"| {row['mode']} | {row['case']} | {row['prompt_tokens']} | {row['generated_tokens']} | "
                    f"{row['ttft_s']:.3f} | {row['decode_tok_s']:.2f} | {row['server_wall_s']:.3f} | "
                    f"{row['client_wall_s']:.3f} | {load} |")
    return text


def score_text(score):
    fields = ", ".join(f"{key} {value}/8" for key, value in score["fields"].items())
    text = f"JSON valid: {score['valid']}; exact labels {score['exact']}/8; {fields}."
    if score.get("fence_removed"):
        text += " Markdown code fence removed for scoring; the raw answer violates the JSON-only instruction."
    if score.get("error"):
        text += f" Parse/shape error: {score['error']}."
    if score["mismatches"]:
        text += " Mismatches: " + json.dumps(score["mismatches"])
    return text


def case_report(case, rows):
    text = [f"## {case['title']}", "", "Prompt:", "", "```text", case["prompt"], "```", ""]
    if "labels" in case:
        text.extend(["Ground truth (in ID order):", "", "```json", json.dumps(case["labels"]), "```", ""])
    for row in rows:
        if row["case"] == case["id"]:
            text.extend([f"### {row['mode'].capitalize()} answer", "", "````text", row["answer"], "````", "",
                         f"Finish reason: `{row['finish_reason']}`.", ""])
            if row["score"]:
                text.extend([score_text(row["score"]), ""])
    return text


def report_intro():
    return ["# Linux 8B example exchanges — 2026-11-06", "",
            "CPU, 8 threads; converted Edge0 8B, LoRA + prerouter + POSIX page advice enabled. "
            "`OMP_WAIT_POLICY=PASSIVE`, loopback HTTP, disposable Docker container. "
            "Four separate requests with `--parallel 1`, then the same four concurrently with `--parallel 4`. "
            "Context 16384, batch/microbatch 512/512, seed 4242, temperature 0, max_tokens 400, "
            "thinking disabled and prompt caching disabled; no warmup request.", "",
            "Measurements are under load and provisional; they do not establish the quiet speed gate. "
            "TTFT is client time to first nonempty streamed content. Server total is "
            "(prompt_ms + predicted_ms)/1000; client wall also includes HTTP and scheduling. "
            "Decode tok/s = (predicted_n - 1)/(predicted_ms/1000), excluding the first token sampled in prefill. "
            "Prompt/generated counts come from server usage and are checked against timings. "
            "Load is the host 1-minute average.", ""]


def write_report(path, result):
    text = report_intro() + timing_table(result["rows"]) + [""]
    parallel = result["groups"]["parallel"]
    text.extend([f"Parallel aggregate: **{parallel['aggregate_tok_s']:.2f} generated tok/s**, "
                 f"including prefill, across **{parallel['wall_s']:.3f} s** for all four requests. "
                 "This is a single mixed-workload observation, not a decode-only benchmark.", "",
                 "Prerouter/page-advice activation was verified in both server logs; counters:", "",
                 "```json", json.dumps({key: value["trace"] for key, value in result["groups"].items()}, indent=2), "```", ""])
    text.extend(result.get("content_notes", []) + [""])
    for case in CASES:
        text.extend(case_report(case, result["rows"]))
    text.extend(["Raw requests, streamed JSON events, timings, host snapshots, checksums, and server logs are "
                 "saved in the run's output directory. All inputs above are synthetic; "
                 "these eight-item scores are demonstrations, not a general accuracy benchmark.", ""])
    Path(path).write_text("\n".join(text))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "wt/linux/build-cpu/bin/llama-server")
    args = parser.parse_args()
    args.converted_dir = args.converted_dir or Path(str(args.model_dir) + "-gguf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def check_artifacts(args):
    manifest = json.loads((args.converted_dir / "manifest.json").read_text())
    files = {"gguf": args.converted_dir / "edge0-8b.gguf",
             "lora": args.converted_dir / "lora_edge0_8b-gguf.gguf",
             "prerouter": args.model_dir / "prerouter_edge0_8b.safetensors"}
    expected = dict(manifest["artifacts"], prerouter=manifest["inputs_sha256"][files["prerouter"].name])
    hashes = {key: digest(path) for key, path in files.items()}
    require(hashes == expected, "model/adapter/sidecar SHA mismatch")
    return hashes


def main():
    args = arguments()
    result = dict(before=snapshot(), artifacts=check_artifacts(args), rows=[], groups={})
    for mode in ("serial", "parallel"):
        rows, group = run_group(args, mode)
        result["rows"].extend(rows)
        result["groups"][mode] = group
        save_json(args.output_dir / "examples.json", result)
    require(check_artifacts(args) == result["artifacts"], "inference changed model artifacts")
    result["after"] = snapshot()
    save_json(args.output_dir / "examples.json", result)
    write_report(args.output_dir / "examples.md", result)
    print("PASS: eight measured answers, active LoRA/prerouter/advice, unchanged artifacts", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: examples failed: {error}", file=sys.stderr, flush=True)
        raise
