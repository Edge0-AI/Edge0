#!/usr/bin/env python3
"""Serial fresh-prompt benchmark, adapted from the community MI50/RTX 5090 harness.

Start a dedicated Strata server separately. This script never starts or stops it.
Expert adaptation and the OS file cache stay warm between requests; prefix reuse
is rejected. Engine file-tier counters describe decode, not physical disk reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def executable_record(config):
    """Resolve the configured binary exactly as serve/server.py does, without executing it."""
    executable = Path(config["exe"])
    if not executable.is_absolute():
        executable = Path(config.get("cwd") or ".") / executable
    executable = executable.resolve()
    checksum = hashlib.sha256()
    with executable.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return {"path": str(executable), "sha256": checksum.hexdigest(), "size_bytes": executable.stat().st_size}


def engine_provenance_record(path, executable):
    """Keep compact build provenance and verify its executable artifact, when recorded."""
    path = path.resolve()
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8-sig"))
    artifacts = source.get("artifacts", {})
    if isinstance(artifacts, list):
        entries = [(entry["file"], entry) for entry in artifacts]
    elif isinstance(artifacts, dict):
        entries = list(artifacts.items())
    else:
        raise ValueError("Provenance artifacts must be a list or dictionary")
    matches = [entry for name, entry in entries
               if Path(name).name.casefold() == Path(executable["path"]).name.casefold()]
    if len(matches) > 1:
        raise ValueError("Provenance has multiple artifacts matching the configured executable")
    expected = matches[0]["sha256"].lower() if matches else None
    if expected is not None and expected != executable["sha256"]:
        raise ValueError("Provenance executable SHA-256 does not match the configured executable")
    toolchain_keys = ("cmake", "ninja", "msvc", "windows_sdk", "nvcc", "cuda", "cuda_architectures",
                      "cxx_standard", "cuda_standard", "build_type", "portable_avx2", "msvc_runtime",
                      "parallel_jobs", "tests_built")
    toolchain = {key: value for key, value in source.get("toolchain", {}).items() if key in toolchain_keys}
    # Release BUILD.json records CUDA and target architectures at the top level.
    toolchain.update({key: source[key] for key in ("cuda", "archs", "ptx", "portable") if key in source})
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "version": source.get("strata_version", source.get("version")), "commit": source.get("commit"),
            "fingerprint": source.get("engine_source_fingerprint", source.get("source_fingerprint")),
            "toolchain": toolchain, "repository": source.get("fork_repository"),
            "llama_cpp_commit": source.get("llama_cpp_pinned_commit"),
            "executable_hash_verified": expected is not None, "recorded_executable_sha256": expected}


def validate_completion(before, after, chunks, expected, finish, done, text):
    if not done:
        raise RuntimeError("The stream ended without [DONE]")
    if not text:
        raise RuntimeError("The response contained no text")
    if finish not in ("length", "stop"):
        raise RuntimeError(f"Unsuccessful finish reason: {finish!r}")
    if after["totals"]["requests"] != before["totals"]["requests"] + 1:
        raise RuntimeError("Request counter mismatch; another client or engine restart invalidated this trial")
    if after["live"]["state"] != "idle" or after["live"].get("queued", 0):
        raise RuntimeError("The dedicated server was not idle after the request")
    engine = after["requests"][0]
    usage_chunk = next((chunk for chunk in reversed(chunks) if "usage" in chunk), None)
    if not usage_chunk or not usage_chunk.get("timings"):
        raise RuntimeError("The final stream chunk lacks usage or engine timings")
    usage, timings = usage_chunk["usage"], usage_chunk["timings"]
    if engine.get("reused") != 0 or usage["prompt_tokens_details"]["cached_tokens"] != 0:
        raise RuntimeError("Fresh prompt reused cached tokens; restart this arm before resuming")
    if engine["prompt_tokens"] != expected or usage["prompt_tokens"] != expected:
        raise RuntimeError(f"Prompt token count mismatch; expected {expected}")
    if engine.get("prompt_total") != expected or engine.get("prompt_read") != expected:
        raise RuntimeError("The engine did not report reading the complete prompt")
    if engine["finish"] != finish or engine["output_tokens"] != usage["completion_tokens"]:
        raise RuntimeError("Stream usage or finish reason does not match /metrics")
    if timings["prompt_n"] != expected or timings["predicted_n"] != usage["completion_tokens"]:
        raise RuntimeError("Final stream timings do not match the completed request")
    for name, timing_name in (("prompt_ms", "prompt_ms"), ("decode_ms", "predicted_ms")):
        if engine[name] <= 0 or abs(engine[name] - timings[timing_name]) > 0.11:
            raise RuntimeError(f"Invalid or mismatched engine timing: {name}")
    if engine["engine_generated"] <= 0:
        raise RuntimeError("The engine reported no generated tokens")
    return engine, usage, timings


def perform(url, out, session, label, attempt, req, expected, interval=1.0, timeout=1800):
    stem = f"{label}-attempt-{attempt}"
    body = json.dumps(req, ensure_ascii=False).encode("utf-8")
    (out / f"{stem}-request.json").write_bytes(body + b"\n")
    row = {"label": label, "attempt": attempt, "session": session, "status": "failed",
           "expected_prompt_tokens": expected, "request_sha256": hashlib.sha256(body).hexdigest(),
           "started_epoch_s": time.time(), "raw_stream": f"{stem}-sse.jsonl",
           "telemetry": f"{stem}-telemetry.jsonl", "text": "", "reasoning": ""}
    chunks, texts, reasoning, telemetry_errors = [], [], [], []
    stop = threading.Event()
    monitor = None
    started = None

    def sample(output):
        metrics = get(url + "/metrics")
        hardware = dict(metrics["hardware"])
        if hardware.get("ram_total") is not None and hardware.get("ram_used") is not None:
            hardware["ram_available"] = hardware["ram_total"] - hardware["ram_used"]
        sample_row = {"epoch_s": time.time(), "metrics_time": metrics.get("time"),
                      "live": metrics["live"], "hardware": hardware,
                      "completed_requests": metrics["totals"]["requests"]}
        output.write(json.dumps(sample_row) + "\n")
        output.flush()

    def monitor_loop(output):
        while not stop.wait(interval):
            try:
                sample(output)
            except Exception as error:
                telemetry_errors.append(f"{type(error).__name__}: {error}")
                output.write(json.dumps({"epoch_s": time.time(), "error": telemetry_errors[-1]}) + "\n")
                output.flush()
                break

    try:
        before = get(url + "/metrics")
        row["metrics_before"] = before
        if before["live"]["state"] != "idle" or before["live"].get("queued", 0):
            raise RuntimeError("The dedicated server must already be loaded and idle")
        if expected + req["max_tokens"] > before["engine"]["max_context"]:
            raise RuntimeError("Prompt plus output cap exceeds the configured context")
        with (out / row["telemetry"]).open("w", encoding="utf-8") as telemetry:
            sample(telemetry)
            monitor = threading.Thread(target=monitor_loop, args=(telemetry,), daemon=True)
            monitor.start()
            try:
                with (out / row["raw_stream"]).open("w", encoding="utf-8") as raw:
                    started = time.perf_counter()
                    first, finish, done = None, None, False
                    wire = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                                  headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(wire, timeout=timeout) as response:
                        for line in response:
                            raw.write(json.dumps({"elapsed_s": time.perf_counter() - started,
                                                  "line": line.decode("utf-8")}, ensure_ascii=False) + "\n")
                            raw.flush()
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == b"[DONE]":
                                done = True
                                break
                            chunk = json.loads(payload)
                            chunks.append(chunk)
                            if chunk.get("error"):
                                raise RuntimeError(f"Stream error: {chunk['error']}")
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content") or ""
                                thought = delta.get("reasoning_content") or ""
                                if (content or thought) and first is None:
                                    first = time.perf_counter() - started
                                texts.append(content)
                                reasoning.append(thought)
                                finish = choice.get("finish_reason") or finish
                    row.update(client_ttft_s=first, client_elapsed_s=time.perf_counter() - started,
                               finish_reason=finish, stream_done=done)
                    after = get(url + "/metrics")
                    row["metrics_after"] = after
                    engine, usage, timings = validate_completion(before, after, chunks, expected,
                                                                  finish, done, "".join(texts + reasoning))
                    row.update(engine=engine, usage=usage, timings=timings,
                               prefill_tok_s=expected / (engine["prompt_ms"] / 1000),
                               decode_tok_s=engine["engine_generated"] / (engine["decode_ms"] / 1000))
            finally:
                stop.set()
                monitor.join()
                try:
                    sample(telemetry)
                except Exception as error:
                    telemetry_errors.append(f"{type(error).__name__}: {error}")
            if telemetry_errors:
                raise RuntimeError("Telemetry failed: " + "; ".join(telemetry_errors))
        row["status"] = "ok"
    except urllib.error.HTTPError as error:
        row["error"] = {"type": type(error).__name__, "message": str(error), "http_status": error.code,
                        "body": error.read().decode("utf-8", errors="replace")}
    except Exception as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        stop.set()
        if monitor is not None and monitor.is_alive():
            monitor.join()
        row.update(text="".join(texts), reasoning="".join(reasoning), finished_epoch_s=time.time())
        if started is not None and "client_elapsed_s" not in row:
            row["client_elapsed_s"] = time.perf_counter() - started
        write_json(out / f"{stem}-record.json", row)
        write_json(out / f"{stem}-chunks.json", chunks)
    return row


def summary(rows, targets):
    groups = {}
    for target in targets:
        selected = [r for r in rows if r["label"].startswith(f"tokens-{target}-")]
        good = [r for r in selected if r["status"] == "ok"]
        fields = ("client_ttft_s", "client_elapsed_s", "prefill_tok_s", "decode_tok_s")
        groups[str(target)] = {"successful_runs": len(good), "failed_attempts": len(selected) - len(good),
                               **{field: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                                  for field in fields if (v := [r[field] for r in good])}}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Server configuration, saved with relevant fields only")
    parser.add_argument("--engine-provenance", type=Path,
                        help="Build provenance JSON or release BUILD.json; requires --config")
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="strata")
    parser.add_argument("--label", required=True, help="Configuration label, such as iq2-xs-3090-resident")
    parser.add_argument("--out", type=Path, required=True, help="A separate directory for each configuration")
    parser.add_argument("--targets", default="1024,4096,16384")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="Keep completed trials and preserve failed attempts")
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",")]
    if not targets or min(targets) <= 0 or len(set(targets)) != len(targets):
        parser.error("Targets must be unique positive token counts")
    if args.runs < 1 or args.max_tokens < 1 or args.timeout <= 0 or args.telemetry_interval <= 0:
        parser.error("Runs, output cap, timeout, and telemetry interval must be positive")
    if args.engine_provenance and not args.config:
        parser.error("--engine-provenance requires --config to identify the executable")
    args.root, args.pack, args.out = args.root.resolve(), args.pack.resolve(), args.out.resolve()
    url = args.url.rstrip("/")
    sys.path[:0] = [str(args.root), str(args.root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages

    directory = args.pack / "tokenizer"
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").splitlines(),
                          json.loads((directory / "token_type.json").read_text(encoding="utf-8")))
    template = ChatTemplate(directory / "chat_template.jinja")

    def request(content, maximum=args.max_tokens):
        return {"model": args.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "reasoning_effort": "none", "max_tokens": maximum,
                "stream": True, "stream_options": {"include_usage": True}}

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    initial = get(url + "/metrics")
    if initial["live"]["state"] != "idle" or initial["live"].get("queued", 0):
        raise RuntimeError("Start a dedicated, loaded and idle server before this benchmark")
    config = {}
    executable = None
    if args.config:
        source = json.loads(args.config.read_text(encoding="utf-8-sig"))
        executable = executable_record(source)
        keys = ("exe", "cwd", "args", "gpu", "layer_split", "split_skip_if_fits", "backend", "tokenizer", "log",
                "model_name", "parallel", "expert_profile_save", "expert_profile_save_every")
        config = {key: source[key] for key in keys if key in source}
        compute_variables = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "STRATA_STAGE_TRIM", "STRATA_SPLIT_OWN",
                             "STRATA_ARENA_PIN_GIB", "STRATA_RESIDENT_PIN", "STRATA_RESIDENT_HEADROOM_GIB",
                             "STRATA_ARENA_MMAP", "STRATA_PREFILL_HELP", "STRATA_FETCH_THREADS", "STRATA_LOOKAHEAD",
                             "STRATA_DECODE_TIMING", "STRATA_SPLIT_TIMING", "STRATA_VERIFY_PROFILE",
                             "STRATA_SPLIT_MISS_MS", "STRATA_PF_FUSED", "STRATA_FILE_RELEASE"}
        config["env"] = {key: value for key, value in source.get("env", {}).items()
                         if key in compute_variables}
    revision = subprocess.run(["git", "-c", f"safe.directory={args.root.as_posix()}", "rev-parse", "HEAD"],
                              cwd=args.root, capture_output=True, text=True, check=True).stdout.strip()
    provenance = engine_provenance_record(args.engine_provenance, executable) if args.engine_provenance else None
    manifest = {"label": args.label, "frontend_source_commit": revision, "pack": str(args.pack),
                "config": config, "engine_executable": executable,
                "engine_provenance": provenance, "engine_reported_version": initial["engine"].get("version"),
                "targets": targets, "runs": args.runs, "max_tokens": args.max_tokens,
                "model": args.model, "server_model": initial["engine"]["model"],
                "max_context": initial["engine"]["max_context"],
                "hardware_static": initial.get("hardware_static"),
                "tokenizer_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                     for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "method": "Serial warm-engine trials; default adaptive expert state retained; zero prompt reuse required",
                "disk_counter_note": "file_mb is the decode file tier, possibly OS-cached; disk_read_mb is system-wide physical I/O"}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError("Output already has a manifest; use a new output directory or --resume")
        if digest(json.loads(manifest_path.read_text(encoding="utf-8"))) != digest(manifest):
            raise RuntimeError("Resume manifest differs; use a new output directory")
    elif args.resume:
        raise RuntimeError("Cannot resume: output has no manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    rows_path = args.out / "results.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if args.resume and rows_path.exists() else []
    session = uuid.uuid4().hex[:12]
    write_json(args.out / f"session-{session}.json", {"session": session, "epoch_s": time.time(),
               "python": sys.version, "platform": platform.platform(), "initial_metrics": initial,
               "initial_status": get(url + "/v1/status"), "resumed": args.resume})

    def run(label, req):
        if any(row["label"] == label and row["status"] == "ok" for row in rows):
            print("skip completed", label, flush=True)
            return
        attempt = sum(row["label"] == label for row in rows) + 1
        print("start", args.label, label, "tokens", count(req), flush=True)
        row = perform(url, args.out, session, label, attempt, req, count(req),
                      args.telemetry_interval, args.timeout)
        rows.append(row)
        write_json(rows_path, rows)
        write_json(args.out / "summary.json", summary(rows, targets))
        print(row["status"], label, json.dumps({key: row.get(key) for key in
              ("client_ttft_s", "prefill_tok_s", "decode_tok_s", "error")}), flush=True)
        if row["status"] != "ok":
            raise RuntimeError(f"Benchmark stopped after failed trial {label}; evidence was retained")

    run(f"warmup-{session}", request(f"Warmup {session}. Reply with exactly the word READY.", 16))
    filler = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                       for i in range(12000))
    ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
              "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")
    for target in targets:
        for trial in range(1, args.runs + 1):
            # Checkpoints cover a complete turn or >=16K prompt tokens by default;
            # this early differing nonce prevents a whole checkpoint from matching.
            prefix = f"Benchmark nonce: series-{target}-trial-{trial}.\nReview this synthetic Python module:\n"
            low, high = 0, len(filler)
            while low < high:
                middle = (low + high + 1) // 2
                if count(request(prefix + filler[:middle] + ending)) <= target:
                    low = middle
                else:
                    high = middle - 1
            req = request(prefix + filler[:low] + ending)
            if not target - 20 <= count(req) <= target:
                raise RuntimeError(f"Could not construct target {target}: {count(req)}")
            run(f"tokens-{target}-run-{trial}", req)
    print("Complete:", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
