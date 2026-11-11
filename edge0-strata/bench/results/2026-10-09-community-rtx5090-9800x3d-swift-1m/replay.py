#!/usr/bin/env python3
"""Replay frozen requests against an already configured, running Strata server.

No launch, restart, model selection, or configuration changes are performed.
Set STRATA_BENCHMARK_API_KEY if the server needs a bearer token.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
SECRET = os.environ.get("STRATA_BENCHMARK_API_KEY", "")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def error_text(error):
    text = str(error)
    return text.replace(SECRET, "[redacted]") if SECRET else text


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Client:
    def __init__(self, url, timeout):
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("--url must be an http(s) origin without credentials, path, query, or fragment")
        self.origin, self.timeout = url.rstrip("/"), timeout
        self.host, self.port = parsed.hostname, parsed.port
        self.connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection

    def connection(self, timeout=None):
        return self.connection_type(self.host, self.port, timeout=timeout or self.timeout)

    def headers(self, streaming=False):
        headers = {"Content-Type": "application/json"}
        if streaming:
            headers["Accept"] = "text/event-stream"
        if SECRET:
            headers["Authorization"] = "Bearer " + SECRET
        return headers

    def json_request(self, method, path, body=None, timeout=None):
        connection = self.connection(timeout)
        try:
            connection.request(method, path, body=body, headers=self.headers())
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise ValueError(f"{method} {path}: HTTP {response.status}: {payload[:4096]!r}")
            value = json.loads(payload)
            if not isinstance(value, dict) or "error" in value:
                raise ValueError(f"{method} {path}: invalid/API error response: {value!r}")
            return value
        finally:
            connection.close()

    def served_state(self):
        state = {}
        for path in ("/v1/status", "/health"):
            try:
                state[path] = {"response": self.json_request("GET", path, timeout=min(5, self.timeout))}
            except Exception as error:
                state[path] = {"error": error_text(error)}
        return state

    def stream(self, body, directory, row):
        started = time.monotonic()
        connection = self.connection()
        network_socket, timed_out = [], threading.Event()

        def expire():
            timed_out.set()
            current = network_socket[0] if network_socket else connection.sock
            if current is not None:
                try:
                    current.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = threading.Timer(self.timeout, expire)
        timer.daemon = True
        timer.start()
        done, data = False, []
        content, reasoning, calls = [], [], {}
        row.update(content="", reasoning_content="", tool_calls=[], usage={}, timings={},
                   finish_reason=None, client_ttft_s=None, sse_chunks=0)

        def consume(payload, parsed_log):
            nonlocal done
            elapsed = time.monotonic() - started
            if payload == "[DONE]":
                done = True
                parsed_log.write(json.dumps({"elapsed_s": elapsed, "done": True}) + "\n")
                return
            chunk = json.loads(payload)
            parsed_log.write(json.dumps({"elapsed_s": elapsed, "chunk": chunk}, ensure_ascii=False) + "\n")
            parsed_log.flush()
            if not isinstance(chunk, dict) or "error" in chunk:
                raise ValueError(f"Invalid/API error SSE chunk: {chunk!r}")
            row["sse_chunks"] += 1
            if "id" in chunk:
                row["completion_id"] = chunk["id"]
            for key in ("usage", "timings"):
                if key in chunk:
                    if not isinstance(chunk[key], dict):
                        raise ValueError(f"SSE {key} must be an object")
                    row[key] = chunk[key]
            choices = chunk.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("SSE choices must be a list")
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    raise ValueError("Replay supports a single completion (index 0)")
                if choice.get("finish_reason") is not None:
                    row["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise ValueError("SSE delta must be an object")
                nonempty = False
                for key, target in (("content", content), ("reasoning_content", reasoning)):
                    value = delta.get(key)
                    if value is not None and not isinstance(value, str):
                        raise ValueError(f"Non-string {key} delta")
                    if value:
                        target.append(value)
                        nonempty = True
                deltas = delta.get("tool_calls", [])
                if not isinstance(deltas, list):
                    raise ValueError("SSE tool_calls must be a list")
                for call in deltas:
                    if not isinstance(call, dict):
                        raise ValueError("SSE tool call must be an object")
                    index = call.get("index", 0)
                    if type(index) is not int or index < 0:
                        raise ValueError("Invalid tool-call index")
                    current = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    function = call.get("function", {})
                    if not isinstance(function, dict):
                        raise ValueError("SSE tool-call function must be an object")
                    for key in ("id", "type"):
                        if call.get(key):
                            if not isinstance(call[key], str):
                                raise ValueError(f"Non-string tool-call {key}")
                            current[key] = call[key]
                            nonempty = True
                    for key in ("name", "arguments"):
                        if function.get(key):
                            if not isinstance(function[key], str):
                                raise ValueError(f"Non-string function {key}")
                            current["function"][key] += function[key]
                            nonempty = True
                if nonempty and row["client_ttft_s"] is None:
                    row["client_ttft_s"] = elapsed

        try:
            connection.request("POST", "/v1/chat/completions", body=body, headers=self.headers(streaming=True))
            if connection.sock is not None:
                network_socket.append(connection.sock)
            response = connection.getresponse()
            row["http_status"] = response.status
            if response.status != 200:
                payload = response.read(4096)
                (directory / "http-error.bin").write_bytes(payload)
                raise ValueError(f"POST /v1/chat/completions: HTTP {response.status}: {payload!r}")
            if not response.getheader("Content-Type", "").lower().startswith("text/event-stream"):
                payload = response.read(4096)
                (directory / "http-error.bin").write_bytes(payload)
                raise ValueError(f"Expected text/event-stream, got {payload!r}")
            with (directory / "response.sse").open("wb") as raw, (directory / "stream.jsonl").open("w", encoding="utf-8") as parsed:
                while not done:
                    remaining = self.timeout - (time.monotonic() - started)
                    if timed_out.is_set() or remaining <= 0:
                        raise TimeoutError("Absolute request timeout exceeded")
                    if network_socket:
                        network_socket[0].settimeout(remaining)
                    line = response.readline(2 * 1024 * 1024 + 1)
                    if not line:
                        if data:
                            consume("\n".join(data), parsed)
                        break
                    raw.write(line)
                    raw.flush()
                    if len(line) > 2 * 1024 * 1024:
                        raise ValueError("SSE line exceeds 2 MiB")
                    text = line.decode("utf-8").rstrip("\r\n")
                    if not text:
                        if data:
                            consume("\n".join(data), parsed)
                            data = []
                    elif not text.startswith(":"):
                        field, separator, value = text.partition(":")
                        if separator and value.startswith(" "):
                            value = value[1:]
                        if field == "data":
                            data.append(value)
                if timed_out.is_set():
                    raise TimeoutError("Absolute request timeout exceeded")
                if not done or row["finish_reason"] is None or not row["usage"]:
                    raise ValueError("Incomplete SSE: [DONE], finish reason, and usage are required")
        except Exception:
            if timed_out.is_set():
                raise TimeoutError("Absolute request timeout exceeded") from None
            raise
        finally:
            timer.cancel()
            connection.close()
            row.update(content="".join(content), reasoning_content="".join(reasoning),
                       tool_calls=[calls[index] for index in sorted(calls)],
                       client_total_s=time.monotonic() - started)


def frozen(entry):
    path = (ROOT / entry["path"]).resolve()
    if not path.is_relative_to(ROOT) or path.suffix != ".gz":
        raise ValueError("Frozen request path must be a .gz file inside the report")
    body = gzip.decompress(path.read_bytes())
    if sha(body) != entry["request_sha256"]:
        raise ValueError("Frozen request SHA-256 mismatch: " + str(entry["id"]))
    request = json.loads(body)
    if not isinstance(request, dict) or request.get("stream") is not True:
        raise ValueError("Frozen request must be a streaming JSON object")
    return body, request


def rates(row):
    timings = row["timings"]
    row["reused_prompt_tokens"] = timings.get("cache_n")
    row["fresh_prompt_tokens"] = timings.get("prompt_n")
    for target, count, elapsed in (("engine_prefill_tok_s", "prompt_n", "prompt_ms"),
                                   ("engine_decode_tok_s", "predicted_n", "predicted_ms")):
        n, ms = timings.get(count), timings.get(elapsed)
        row[target] = n * 1000 / ms if type(n) in (int, float) and type(ms) in (int, float) and n >= 0 and ms > 0 else None


def eligibility(entry, request, row):
    reasons, expected = [], entry["expected_prompt_tokens"]
    if row["usage"].get("prompt_tokens") != expected:
        reasons.append("usage prompt_tokens differs from frozen expectation")
    reused, fresh = row.get("reused_prompt_tokens"), row.get("fresh_prompt_tokens")
    if type(reused) is not int or type(fresh) is not int or fresh + reused != expected:
        reasons.append("timings prompt_n + cache_n differs from frozen expectation")
    measured = entry["role"] == "measured"
    if entry["suite"] == "decode":
        if measured and (type(reused) is not int or reused / expected < .999):
            reasons.append("decode reused less than 99.9% of the frozen prefix")
    elif entry["role"] != "continuation" and reused != 0:
        reasons.append("fresh request unexpectedly reused prompt tokens")
    if measured:
        generated = entry.get("expected_completion_tokens", 512)
        if row["finish_reason"] != "length" or row["usage"].get("completion_tokens") != generated or row["timings"].get("predicted_n") != generated:
            reasons.append("measured completion did not consume the exact fixed decode budget")
    if row.get("engine_prefill_tok_s") is None or row.get("engine_decode_tok_s") is None:
        reasons.append("engine throughput timings are unavailable or invalid")
    if entry["suite"] == "disk":
        if row["finish_reason"] != "stop":
            reasons.append("disk request did not finish at EOS")
        previous = entry.get("previous_prefix_tokens")
        if entry["role"] == "continuation" and previous is not None:
            ratio = reused / previous if type(reused) is int and previous > 0 else None
            row["previous_prefix_reuse_fraction"] = ratio
            if ratio is None or ratio < .99:
                reasons.append("disk continuation reused less than 99% of the previous prefix")
        if "expected_needles" in entry:
            try:
                actual = json.loads(row["content"])
            except ValueError:
                actual = None
            row["needle_grade"] = {"expected": entry["expected_needles"], "actual": actual,
                                   "passed": actual == entry["expected_needles"]}
            if not row["needle_grade"]["passed"]:
                reasons.append("disk answer differs from the eight frozen needle values")
    return reasons


def selected(entries, args):
    rows = [entry for entry in entries if entry["suite"] == args.suite]
    if args.suite == "disk":
        rows = [entry for entry in rows if entry["role"] == args.disk_phase
                and (args.depth is None or entry["depth"] == args.depth)
                and (args.round is None or entry["round"] == args.round)]
    elif args.suite == "decode":
        rows = [entry for entry in rows if args.round is None or entry["round"] == args.round]
    else:
        warmups = [entry for entry in rows if entry["role"] == "warmup"]
        measured = [entry for entry in rows if entry["role"] == "measured"
                    and (args.round is None or entry["round"] == args.round)]
        rows = warmups + measured if measured else []
    if not rows:
        raise ValueError("No requests match the selected suite/round/depth/phase")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18089")
    parser.add_argument("--suite", choices=("confirmation", "screen", "decode", "disk"))
    parser.add_argument("--round", type=int)
    parser.add_argument("--depth", type=int, help="Disk suite depth filter")
    parser.add_argument("--disk-phase", choices=("initial", "continuation"), default="initial")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request seconds, at least 1 (1M initial needs a larger value)")
    parser.add_argument("--slot-action", choices=("save", "restore"), help="Only perform this native slot action; no replay or restart")
    parser.add_argument("--slot-filename", help="Explicit safe regular-file basename inside the server's configured slot-save directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    run = {"started_at_utc": utc(), "harness_sha256": sha(Path(__file__).read_bytes()),
           "url": args.url, "suite": args.suite, "round": args.round, "depth": args.depth,
           "disk_phase": args.disk_phase, "timeout_s": args.timeout, "results": [], "status": "running"}
    try:
        if not math.isfinite(args.timeout) or args.timeout < 1:
            raise ValueError("--timeout must be finite and at least 1 second")
        client = Client(args.url, args.timeout)
        if args.slot_action:
            name = args.slot_filename or ""
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", name) or ".." in name:
                raise ValueError("--slot-filename must be an explicit safe regular-file basename")
            run["served_before"] = client.served_state()
            started = time.monotonic()
            action = {"action": args.slot_action, "filename": name, "status": "running"}
            run["slot_action"] = action
            try:
                action["response"] = client.json_request("POST", "/slots/0?action=" + args.slot_action,
                                                         json.dumps({"filename": name}).encode())
                action["status"] = "completed"
            except Exception as error:
                action.update(status="failed", error=error_text(error))
                raise
            finally:
                action["client_total_s"] = time.monotonic() - started
                run["served_after"] = client.served_state()
        else:
            if not args.suite or args.slot_filename:
                raise ValueError("Use --suite for replay, or --slot-action with --slot-filename")
            index_path = ROOT / "requests/index.json"
            index_bytes = index_path.read_bytes()
            run["index_sha256"] = sha(index_bytes)
            entries = selected(json.loads(index_bytes)["requests"], args)
            # Preflight every chosen file before sending even the warmup.
            for entry in entries:
                frozen(entry)
            for ordinal, entry in enumerate(entries):
                directory = args.out / (f"{ordinal:03d}-" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(entry["id"]))[:100])
                directory.mkdir()
                row = {"request": entry, "harness_sha256": run["harness_sha256"],
                       "started_at_utc": utc(), "status": "running", "eligible": False,
                       "served_before": client.served_state()}
                run["results"].append(row)
                try:
                    body, request = frozen(entry)
                    client.stream(body, directory, row)
                    rates(row)
                    reasons = eligibility(entry, request, row)
                    row.update(status="completed", eligible=not reasons, ineligible_reasons=reasons)
                except Exception as error:
                    row.update(status="failed", error=error_text(error), eligible=False)
                finally:
                    row.update(finished_at_utc=utc(), served_after=client.served_state())
                    save(directory / "result.json", row)
                    save(args.out / "run.json", run)
                print(f"{entry['id']}: {row['status']}, eligible={row['eligible']}", flush=True)
                if row["status"] == "failed":
                    break
            if any(not row["eligible"] for row in run["results"]):
                raise ValueError("One or more replay requests failed or were ineligible; inspect actual outputs")
        run["status"] = "completed"
    except Exception as error:
        run.update(status="failed", error=error_text(error))
        print(run["error"], file=sys.stderr)
    finally:
        run["finished_at_utc"] = utc()
        save(args.out / "run.json", run)
    return 0 if run["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
