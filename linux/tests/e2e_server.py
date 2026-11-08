"""8B conversion/LoRA/prerouter smoke. Needs real weights; no network downloads.

Run in Docker: python3 linux/tests/e2e_server.py --model-dir MODEL --output-dir LOGS
The converter's SHA gate reuses an already converted model when available.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from server_test_utils import (ROOT, completion, digest, request, require,
                               running_server, save_json, snapshot)


def convert(source, output):
    tools = ROOT / "windows/tools"
    subprocess.run([sys.executable, str(tools / "convert_mlx_to_gguf.py"),
                    "--dir", str(source), "--out", str(output)], check=True)
    subprocess.run([sys.executable, str(tools / "lora_mlx_to_gguf.py"),
                    "--dir", str(source), "--base", str(output / "edge0-8b.gguf")], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["inputs_sha256"].items():
        require(digest(source / name) == expected, f"source SHA mismatch: {name}")
    for kind, name in (("gguf", "edge0-8b.gguf"), ("lora", "lora_edge0_8b-gguf.gguf")):
        require(digest(output / name) == manifest["artifacts"][kind], f"artifact SHA mismatch: {kind}")
    require(digest(source / "lora_edge0_8b-gguf.gguf") == manifest["artifacts"]["lora"],
            "regenerated LoRA differs from converted adapter")
    return manifest["artifacts"]


def chat(port):
    result = request(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "What is 2 + 2? Reply with only the number."}],
        "temperature": 0, "seed": 4242, "max_tokens": 64, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}})
    answer = result["choices"][0]["message"]["content"]
    require(isinstance(answer, str) and answer.strip(), "chat response has no answer")
    require("4" in answer, f"unexpected arithmetic answer: {answer!r}")
    return {"answer": answer, "usage": result["usage"]}


def check_trace(log, mode):
    text = log.read_text()
    require(re.search(r"loaded .*?(147|294).*?tensors", text), "LoRA load not confirmed")
    if mode == "dormant":
        require("[pref-trace]" not in text, "dormant prerouter activated")
        return None
    if mode == "corrupt":
        require("advisory disabled" in text and "INIT ok" not in text, "corrupt sidecar not rejected")
        return None
    require("INIT ok: H=1536 E=128 K=8" in text, "8B prerouter not initialized")
    traces = re.findall(r"steps=(\d+) heads=(\d+) experts=(\d+).*?advice=(\d+) failures=(\d+)", text)
    require(traces, "no prerouter accounting")
    steps, heads, experts, advice, failures = map(int, traces[-1])
    require(steps > 0 and heads > 0 and experts > 0, "prerouter never predicted")
    require((advice > 0) == (mode == "advice") and failures == 0, "incorrect page-advice accounting")
    return dict(steps=steps, heads=heads, experts=experts, advice=advice, failures=failures)


def run_mode(args, mode, model, lora):
    log = args.output_dir / f"smoke-{mode}.log"
    features = {} if mode == "dormant" else {
        "E0_PREROUTER": str(args.model_dir / "prerouter_edge0_8b.safetensors"),
        "E0_PREFETCH": "1" if mode == "advice" else "0"}
    if mode == "corrupt":
        corrupt = args.output_dir / "corrupt.safetensors"
        corrupt.write_bytes(b"broken")
        features["E0_PREROUTER"] = str(corrupt)
    with running_server(args.binary, model, log, lora=lora, features=features) as port:
        result = chat(port)
        raw = completion(port, "Write the first ten positive integers in order:", 48)
        require(len(raw["tokens"]) == 48, "native completion did not return 48 token IDs")
        result.update(tokens=raw["tokens"], timings=raw["timings"])
    result["trace"] = check_trace(log, mode)
    print(f"PASS {mode}: answer={result['answer']!r} trace={result['trace']}", flush=True)
    return result


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


def main():
    args = arguments()
    result = {"before": snapshot(), "artifacts": convert(args.model_dir, args.converted_dir)}
    model, lora = (args.converted_dir / name for name in ("edge0-8b.gguf", "lora_edge0_8b-gguf.gguf"))
    for mode in ("dormant", "accounting", "advice", "corrupt"):
        result[mode] = run_mode(args, mode, model, lora)
    baseline = result["dormant"]["tokens"]
    require(all(result[mode]["tokens"] == baseline for mode in ("accounting", "advice", "corrupt")),
            "advisory mode changed greedy output tokens")
    require(digest(model) == result["artifacts"]["gguf"], "model bytes changed during inference")
    result["after"] = snapshot()
    save_json(args.output_dir / "smoke.json", result)
    print("PASS conversion, chat, LoRA, active predictions, mode gates, identical tokens and model SHA", flush=True)


if __name__ == "__main__":
    main()
