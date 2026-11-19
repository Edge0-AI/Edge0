"""Check short answers and prompt chunks in a local latent Metal service."""

import argparse
import json
from pathlib import Path
import socket
import tempfile

import strata_bench as bench


SHORT = [
    ("What is 17 times 23? Answer with the number only.", "391"),
    ("What is the capital city of France? Answer with the city only.", "paris"),
    ("How many days are in a leap year? Answer with the number only.", "366"),
    ("What is 9 times 9? Answer with the number only.", "81"),
    ("What is the capital of Japan? Answer with the city only.", "tokyo"),
    ("Which planet is the largest in the solar system? Answer with the name only.", "jupiter"),
    ("What is the chemical symbol for gold? Answer with the symbol only.", "au"),
    ("What is 100 minus 37? Answer with the number only.", "63"),
    ("On a clear day the sky is which colour? Answer with one word.", "blue"),
    ("What is the capital of Italy? Answer with the city only.", "rome"),
]
LONG = [
    ("What is 17 times 23?  Reply with just the number.", "391"),
    ("How many days are in a leap year?  Reply with just the number.", "366"),
    ("Name the capital city of France.  Reply with just the city.", "paris"),
    ("What is 9 times 9?  Reply with just the number.", "81"),
]
FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank while the old mill wheel turns slowly "
    "and the miller counts sacks of grain in the fading afternoon light. "
)


def verify(base, output):
    words = FILLER.split()
    filler = " ".join((words * (8000 // len(words) + 1))[:8000])
    cases = [("short", question, question, expected) for question, expected in SHORT]
    cases += [
        ("long", question, filler + "\n\nQuestion: " + question + "\nAnswer:", expected)
        for question, expected in LONG
    ]
    results = []
    for kind, question, prompt, expected in cases:
        reply = bench.chat(base, "", [{"role": "user", "content": prompt}], 24, think=False)
        timing = reply.get("timings") or {}
        tokens = (reply.get("usage") or {}).get("prompt_tokens")
        answer = reply["text"].strip()
        passed = not reply["err"] and expected in answer.lower()
        if kind == "long":
            passed = passed and tokens is not None and tokens > 8192 and timing.get("cache_n") == 0
        results.append({
            "kind": kind, "question": question, "expected": expected, "answer": answer,
            "prompt_tokens": tokens, "reused_tokens": timing.get("cache_n"),
            "passed": bool(passed), "error": reply["err"],
        })
        print(f"{kind}: {tokens} prompt tokens, answer {answer!r}, passed={passed}", flush=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    return all(row["passed"] for row in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    output = options.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(options.config.read_text())
    config["log"] = str(output.with_suffix(".engine.log"))
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="strata-latent-") as temp:
        config_path = Path(temp) / "config.json"
        config_path.write_text(json.dumps(config))
        process = bench.start_server(config_path, port, None, output.with_suffix(".server.log"))
        try:
            base = bench.server_base(config.get("host"), port)
            if not bench.wait_ready(base, "", process, 240):
                raise RuntimeError("service did not load; check the engine and server logs")
            return 0 if verify(base, output) else 1
        finally:
            bench.stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
