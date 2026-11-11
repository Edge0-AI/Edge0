"""Matched OpenAI-compatible smoke benchmark; never executes generated code/tools."""
import argparse
import ctypes
import json
import threading
import time
import urllib.request
from pathlib import Path

AUTH_HEADERS = {}


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total_physical", "available_physical", "total_pagefile", "available_pagefile",
         "total_virtual", "available_virtual", "available_extended")
    ]


def available_ram():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.available_physical / 2**30


def get_json(base, route):
    req = urllib.request.Request(base + route, headers=AUTH_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def assert_idle(base):
    slots = get_json(base, "/slots")
    if not isinstance(slots, list):
        raise RuntimeError("Unrecognized slots response; cannot confirm idle")
    if any(slot.get("is_processing") for slot in slots):
        raise RuntimeError("Model is already working; not interrupting it")


def request(base, model, label, prompt, run, tools=None, *, response_format=None, reasoning_effort="low", max_tokens=1024):
    assert_idle(base)
    # Different first messages minimize shared-prefix reuse; residual cache is reported.
    payload = {
        "model": model, "messages": [{"role": "user", "content": f"Benchmark {label} run {run}.\n{prompt}"}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "repeat_penalty": 1.0, "repetition_penalty": 1.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "reasoning_effort": reasoning_effort, "seed": 42, "max_tokens": max_tokens,
        "cache_prompt": False, "strata_checkpoint": False,
        "experimental_speed_projection": False,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if tools:
        payload.update(tools=tools, tool_choice="auto")
    if response_format is not None:
        payload['response_format']=response_format
    before = available_ram()
    samples = [before]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.5):
            samples.append(available_ram())

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    first_delta = first_content = None
    text, reasoning, calls = "", "", {}
    usage, timings, finish = {}, {}, None
    error = None
    try:
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json", **AUTH_HEADERS})
        with urllib.request.urlopen(req, timeout=900) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("timings"):
                    timings = event["timings"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if content or thought or delta.get("tool_calls"):
                        first_delta = first_delta if first_delta is not None else time.perf_counter() - started
                    if content:
                        first_content = first_content if first_content is not None else time.perf_counter() - started
                    text += content
                    reasoning += thought
                    for call in delta.get("tool_calls", []):
                        target = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                        function = call.get("function", {})
                        target["name"] += function.get("name") or ""
                        target["arguments"] += function.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish
    except Exception as exc:
        error = str(exc)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        monitor.join(timeout=2)
        samples.append(available_ram())
    return {
        "case": label, "run": run, "wall_seconds": round(elapsed, 3),
        "first_token_seconds": first_delta, "first_answer_seconds": first_content,
        "usage": usage, "timings": timings, "finish_reason": finish,
        "answer": text, "reasoning": reasoning, "tool_calls": list(calls.values()),
        "available_ram_before_GiB": before, "minimum_available_ram_GiB": min(samples),
        "available_ram_after_GiB": samples[-1], "error": error,
    }


def validate(row, strict=False):
    if row["error"] or row["finish_reason"] == "length":
        return False
    if row["case"] == "tool_call":
        calls = row["tool_calls"]
        try:
            return len(calls) == 1 and calls[0]["name"] == "read_file" and json.loads(calls[0]["arguments"]) == {"path": "package.json"}
        except (ValueError, KeyError):
            return False
    raw_answer = row["answer"].strip()
    if not strict and raw_answer.startswith("```") and raw_answer.endswith("```"):
        raw_answer = "\n".join(raw_answer.splitlines()[1:-1])
    try:
        answer = json.loads(raw_answer)
    except ValueError:
        return False
    if row["case"] == "quote_math":
        expected = {"subtotal": 780, "discounted": 702, "tax": 91.26, "total": 793.26}
        return isinstance(answer, dict) and answer.keys() == expected.keys() and all(
            isinstance(answer[k], (int, float)) and abs(answer[k] - v) < 0.005 for k, v in expected.items())
    if row["case"] == "validation_logic":
        return answer == {"accepted": [0, 1.5, 3], "rejected": [None, "4", -2]}
    return answer == {"start": "CLENEX-4181", "middle": "STRATA-8081", "end": "LONDON-9060"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--model")
    parser.add_argument("--rescore", type=Path)
    parser.add_argument("--connection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new filename to preserve results")
    if args.rescore:
        results = json.loads(args.rescore.read_text(encoding="utf-8"))
        results["scoring_note"] = "Content accuracy and exact output formatting are scored separately; original evidence retained."
        for row in results["results"]:
            row["original_check_passed"] = row["check_passed"]
            row["check_passed"] = validate(row)
            row["strict_format_passed"] = validate(row, strict=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({"content_passed": sum(r["check_passed"] for r in results["results"]),
                          "strict_format_passed": sum(r["strict_format_passed"] for r in results["results"]),
                          "total": len(results["results"])}))
        return
    if args.connection:
        conn = json.loads(args.connection.read_text(encoding="utf-8-sig"))
        args.base = conn["baseURL"].removesuffix("/v1")
        args.model = conn["model"]
        AUTH_HEADERS["Authorization"] = "Bearer " + Path(conn["keyFile"]).read_text().strip()
    if not args.base or not args.model:
        parser.error("--base and --model are required for live benchmarking")
    assert_idle(args.base)
    cases = [
        ("quote_math", "An office quote includes four visits at $120 each, three floor jobs at $85 each, and $45 materials. Apply 10% discount to the subtotal then 13% tax. Return only JSON with numeric subtotal, discounted, tax and total.", None),
        ("validation_logic", 'Accept only JSON numbers which are finite and nonnegative (zero is valid); reject null, strings and negatives. Classify [0, null, "4", -2, 1.5, 3]. Return only JSON with arrays accepted and rejected, preserving their relative order.', None),
        ("tool_call", "Use the read_file tool to inspect package.json before suggesting a test command. Do not guess its contents. Call the tool exactly once; no other actions.", [{"type": "function", "function": {"name": "read_file", "description": "Read a file in the current project", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}}]),
    ]
    filler = "This is irrelevant background about routine office cleaning schedules. Ignore it when extracting marked keys.\n"
    for label, repeats in [("recall_short", 25), ("recall_long", 300)]:
        document = "START_KEY=CLENEX-4181\n" + filler * repeats + "MIDDLE_KEY=STRATA-8081\n" + filler * repeats + "END_KEY=LONDON-9060\n"
        cases.append((label, document + 'Return only JSON with start, middle and end holding the three marked key values.', None))
    results = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base": args.base,
               "model": args.model, "runs": args.runs, "results": [],
               "notes": "Same text/settings; token counts may differ. RAM is whole-system, sampled at 0.5 s. No generated tools/code are executed. TTFT includes reasoning. Low effort can mean different budgets across models. This compares whole stacks, not model or engine in isolation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, prompt, tools in cases:
        for run in range(1, args.runs + 1):
            row = request(args.base, args.model, label, prompt, run, tools)
            row["check_passed"] = validate(row)
            row["strict_format_passed"] = validate(row, strict=True)
            results["results"].append(row)
            args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps({k: row[k] for k in ("case", "run", "wall_seconds", "usage", "timings", "check_passed", "error")}), flush=True)
            if row["error"]:
                raise SystemExit("Request failed; stopping instead of retrying or restarting the server")


if __name__ == "__main__":
    main()
