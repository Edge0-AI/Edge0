#!/usr/bin/env python3
"""Build the frozen fixture bundle of the Strata standard benchmark (docs/BENCHMARK_PROTOCOL.md).

    python tools/build_standard_corpus.py --tokenizer PACK/tokenizer [--out bench/prompts/standard-v1]

`strata bench run` reads the bundle this writes and never builds prompts at run time, so every machine sends the same bytes. The
bundle has one JSON file per scheduled request (the 22 required ones and the 12 of the optional concurrency stage), `build.json`
(how it was made: the tokenizer and template hashes, the target and actual token counts) and `MANIFEST.sha256` (a `sha256sum`
listing of every other file). The SHA-256 of that manifest is the protocol's `corpus_sha256`.

The prompts are synthetic. The two record counts (for about 4,096 and 16,384 tokens) are chosen with the pinned tokenizer and the
chat template Strata ships (serve/chat_template.jinja), rendered the way the server renders a thinking request. Run it again
and the bytes are identical; a change to anything the protocol freezes needs a new protocol ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROTOCOL_ID = "strata-qwen38-thinking-v1-draft.1"
BUNDLE = "standard-v1"

SHORT_TASK = "Explain how a hash table finds a value, including collisions and one small example."
RECORD = "record{:06d}: alpha beta gamma delta epsilon zeta eta theta."
SUMMARY_QUESTION = "Summarize the structure of these records and explain how to look up a record by its ID."
TARGETS = {"prompt-4k": 4096, "prompt-16k": 16384}
TOLERANCE = 0.02
TOOL_DESCRIPTION = "Synthetic benchmark lookup; no external action is performed."

# What the template is rendered with when the runner asks for the protocol's thinking settings.
TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}

WARMUP_CAP, PERFORMANCE_CAP, SANITY_CAP, CONCURRENCY_CAP = 128, 512, 2048, 512

INSTRUCTION = {
    "I1": ("Return only a JSON object with ok set to true and count set to the integer 3. No other keys.",
           {"kind": "json", "value": {"ok": True, "count": 3}}),
    "I2": ("Sort these integers in ascending order and remove duplicates: 9, 2, -4, 2, 1. Return only a JSON array.",
           {"kind": "json", "value": [-4, 1, 2, 9]}),
    "I3": ("Project: Orion. Build: 17. Return only a JSON object with keys project (string) and build (integer).",
           {"kind": "json", "value": {"project": "Orion", "build": 17}}),
    "I4": ("Reply with exactly the word READY in uppercase and nothing else.",
           {"kind": "text", "value": "READY"}),
}

# id -> (records, inserts [(fraction, line)], question, expected)
RETRIEVAL = {
    "R1": ("prompt-4k", [(0.10, "access_code = ORBIT-731")],
           "What is access_code? Reply only with its value.", {"kind": "text", "value": "ORBIT-731"}),
    "R2": ("prompt-4k", [(0.50, "storage_label = CEDAR-284")],
           "What is storage_label? Reply only with its value.", {"kind": "text", "value": "CEDAR-284"}),
    "R3": ("prompt-16k", [(0.90, "archive_code = DELTA-906")],
           "What is archive_code? Reply only with its value.", {"kind": "text", "value": "DELTA-906"}),
    "R4": ("prompt-16k", [(0.10, "north = N-124"), (0.90, "south = S-875")],
           "Return only a JSON object containing north and south and their values.",
           {"kind": "json", "value": {"north": "N-124", "south": "S-875"}}),
}

# id -> (prompt, function name, {property: type}, expected arguments)
TOOLS = {
    "T1": ("Use lookup_record to look up record A-17.", "lookup_record", {"id": "string"}, {"id": "A-17"}),
    "T2": ("Use get_weather for Zagreb in celsius.", "get_weather", {"city": "string", "unit": "string"},
           {"city": "Zagreb", "unit": "celsius"}),
    "T3": ("Use convert_units to convert 3 km to m.", "convert_units",
           {"value": "number", "from_unit": "string", "to_unit": "string"},
           {"value": 3, "from_unit": "km", "to_unit": "m"}),
    "T4": ("Use get_build_status for project Orion, build 17.", "get_build_status",
           {"project": "string", "build": "integer"}, {"project": "Orion", "build": 17}),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(obj) -> bytes:
    """The one serialization of every fixture file: UTF-8, LF, two-space indent, key order as written."""
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def tool_schema(name: str, properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": TOOL_DESCRIPTION,
        "parameters": {"type": "object", "properties": {key: {"type": kind} for key, kind in properties.items()},
                       "required": list(properties), "additionalProperties": False}}}


def user_message(request_id: str, body: str) -> list[dict]:
    return [{"role": "user", "content": f"[case {request_id}]\n{body}"}]


def records_body(count: int, question: str, inserts: list[tuple[float, str]] = ()) -> str:
    """`count` records, one per line, with each insert placed before record index floor(fraction * count); then the question."""
    before: dict[int, list[str]] = {}
    for fraction, line in inserts:
        before.setdefault(math.floor(fraction * count), []).append(line)
    lines: list[str] = []
    for index in range(count):
        lines.extend(before.get(index, ()))
        lines.append(RECORD.format(index + 1))
    return "\n".join(lines) + "\n" + question


class Counter:
    """Token counts of a rendered request, with the tokenizer and the template the server uses."""

    def __init__(self, tokenizer_dir: Path, template_path: Path):
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        merges = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
        if merges and merges[-1] == "":
            merges.pop()
        types = json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8"))
        self.tok = ST.Tokenizer(tokens, merges, types)
        self.template = ChatTemplate(template_path)

    def ids(self, messages: list[dict], tools: list[dict] | None) -> list[int]:
        functions = [t["function"] for t in tools] if tools else None
        text = self.template.render(messages, tools=functions, **TEMPLATE_KWARGS)
        return self.tok.encode(text, parse_special=True)


def pick_record_count(counter: Counter, target: int) -> tuple[int, int]:
    """The number of complete records whose rendered performance request is closest to `target` tokens."""
    def tokens(count: int) -> int:
        body = records_body(count, SUMMARY_QUESTION)
        return len(counter.ids(user_message("performance-prompt-4k-r1", body), None))
    low, high = 20, 60
    t_low, t_high = tokens(low), tokens(high)
    slope = (t_high - t_low) / (high - low)
    count = max(1, round(low + (target - t_low) / slope))
    for _ in range(4):
        current = tokens(count)
        step = round((target - current) / slope)
        if step == 0:
            break
        count += step
    best = min(range(max(1, count - 4), count + 5), key=lambda n: (abs(tokens(n) - target), n))
    return best, tokens(best)


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build(tokenizer_dir: Path, template_path: Path, out: Path) -> dict:
    counter = Counter(tokenizer_dir, template_path)
    counts = {}
    for case, target in TARGETS.items():
        count, actual = pick_record_count(counter, target)
        if abs(actual - target) > TOLERANCE * target:
            raise SystemExit(f"{case}: {actual} tokens is not within {TOLERANCE:.0%} of {target}")
        counts[case] = {"records": count, "target_tokens": target}
        print(f"{case}: {count} records, {actual} tokens (target {target})")

    requests: list[dict] = []

    def add(request_id, phase, case_id, repetition, seed, cap, messages, *, tools=None, expected=None, category=None,
            round_=None, concurrency=1, target=None):
        requests.append({
            "id": request_id, "phase": phase, "case_id": case_id, "repetition": repetition, "round": round_,
            "seed": seed, "max_output_tokens": cap, "concurrency": concurrency, "category": category,
            "target_input_tokens": target, "messages": messages, "tools": tools,
            "tool_choice": "auto" if tools else None, "expected": expected})

    short = lambda request_id: user_message(request_id, SHORT_TASK)          # noqa: E731
    long_body = lambda case: records_body(counts[case]["records"], SUMMARY_QUESTION)   # noqa: E731
    add("warmup", "warmup", "warmup", 1, 1, WARMUP_CAP, short("warmup"))
    for repetition in (1, 2, 3):
        add(f"performance-short-r{repetition}", "performance", "short", repetition, repetition, PERFORMANCE_CAP,
            short(f"performance-short-r{repetition}"))
        for case in ("prompt-4k", "prompt-16k"):
            request_id = f"performance-{case}-r{repetition}"
            add(request_id, "performance", case, repetition, repetition, PERFORMANCE_CAP,
                user_message(request_id, long_body(case)), target=TARGETS[case])
    for name, (prompt, expected) in INSTRUCTION.items():
        add(f"sanity-{name}", "sanity", name, 1, 1, SANITY_CAP, user_message(f"sanity-{name}", prompt),
            expected=expected, category="instruction")
    for name, (case, inserts, question, expected) in RETRIEVAL.items():
        body = records_body(counts[case]["records"], question, inserts)
        add(f"sanity-{name}", "sanity", name, 1, 1, SANITY_CAP, user_message(f"sanity-{name}", body),
            expected=expected, category="retrieval", target=TARGETS[case])
    for name, (prompt, function, properties, arguments) in TOOLS.items():
        add(f"sanity-{name}", "sanity", name, 1, 1, SANITY_CAP, user_message(f"sanity-{name}", prompt),
            tools=[tool_schema(function, properties)],
            expected={"kind": "tool_call", "name": function, "arguments": arguments}, category="tool_call")
    for round_ in (1, 2, 3):
        for client in (1, 2, 3, 4):
            request_id = f"concurrency-r{round_}-c{client}"
            add(request_id, "concurrency", "short-burst", round_, 4 * (round_ - 1) + client, CONCURRENCY_CAP,
                short(request_id), round_=round_, concurrency=4)

    token_ids = {r["id"]: counter.ids(r["messages"], r["tools"]) for r in requests}
    for request in requests:
        ids = token_ids[request["id"]]
        request["input_tokens_frozen"] = len(ids)
        request["max_shared_prefix_tokens"] = max(lcp(ids, other) for key, other in token_ids.items()
                                                  if key != request["id"])
        if request["target_input_tokens"] is not None:
            deviation = abs(len(ids) - request["target_input_tokens"]) / request["target_input_tokens"]
            if deviation > TOLERANCE:
                raise SystemExit(f"{request['id']}: {len(ids)} tokens is more than {TOLERANCE:.0%} from the target")

    (out / "requests").mkdir(parents=True, exist_ok=True)
    for stale in (out / "requests").glob("*.json"):
        stale.unlink()
    files: dict[str, bytes] = {}
    for request in requests:
        files[f"requests/{request['id']}.json"] = encoded(request)
    tokenizer_files = {name: sha256_bytes((tokenizer_dir / name).read_bytes())
                       for name in ("vocab.json", "merges.txt", "token_type.json")}
    files["build.json"] = encoded({
        "protocol_id": PROTOCOL_ID, "bundle": BUNDLE,
        "profile": "qwen3.8-flash-next-thinking",
        "tokenizer_files": tokenizer_files,
        "template": {"path": "serve/chat_template.jinja", "sha256": sha256_bytes(template_path.read_bytes())},
        "template_kwargs": TEMPLATE_KWARGS,
        "record_format": RECORD, "short_task": SHORT_TASK, "summary_question": SUMMARY_QUESTION,
        "record_counts": counts, "length_tolerance": TOLERANCE,
        "requests": [{"id": r["id"], "input_tokens": r["input_tokens_frozen"],
                      "max_shared_prefix_tokens": r["max_shared_prefix_tokens"]} for r in requests],
        "note": "input_tokens counts the rendered thinking request with the generation prompt. max_shared_prefix_tokens is "
                "the longest token prefix this request shares with any other request of the bundle: cache reuse above it "
                "means the user payload was read from the cache."})
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = "".join(f"{sha256_bytes(data)}  {name}\n" for name, data in sorted(files.items())).encode("utf-8")
    (out / "MANIFEST.sha256").write_bytes(manifest)
    print(f"{len(requests)} requests, corpus_sha256 {sha256_bytes(manifest)}")
    return {"corpus_sha256": sha256_bytes(manifest), "requests": len(requests)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, type=Path,
                    help="folder with vocab.json, merges.txt and token_type.json (a pack's tokenizer/ folder)")
    ap.add_argument("--template", type=Path, default=ROOT / "serve" / "chat_template.jinja")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "prompts" / BUNDLE)
    a = ap.parse_args(argv)
    build(a.tokenizer, a.template, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
