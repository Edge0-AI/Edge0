"""Thinking/tool formatting, CLI history, stops, and raw logprobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    model = Path(os.environ.get("MOE_TINY", "/nonexistent")) / "tiny-a"
    if not (model / "assets.tsv").exists():
        return 77
    def invoke(command, text=None):
        # Decode as UTF-8 regardless of the OS locale: the reference CLI emits UTF-8, but a
        # non-UTF-8 console (for example GBK on a Chinese Windows) would otherwise fail to decode
        # generated text. Python's -I disables PYTHONUTF8, so the encoding is pinned here.
        result = subprocess.run(command, input=text, capture_output=True, encoding='utf-8')
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, (command, result.stderr)
        return result
    invoke([args.probe])
    base = ["--model", str(model), "--device", args.backend]
    common = [*base, "--max-tokens", "4", "--temperature", "0"]
    for thinking, suffix in [("on", "<|im_start|>assistant\n<think>\n"),
                             ("off", "<|im_start|>assistant\n<think>\n\n</think>\n\n")]:
        prompt = invoke([args.cli, "prompt", "--model", str(model), "--user", "hello", "--thinking", thinking])
        assert prompt.stdout.endswith(suffix)
    description = json.loads(invoke([args.cli, "describe", *base]).stdout)
    assert description["thinking"] == dict(supported=True, default=False, enabled_supported=True)
    assert description["tools_supported"] and description["context"] == 16384
    for key, temperature, top_p in [("recommended", .7, .8), ("recommended_thinking", 1., .95)]:
        c = description[key]
        assert (c["temperature"], c["top_k"], c["top_p"], c["presence_penalty"], c["max_tokens"]) == (temperature, 20, top_p, 1.5, 32768)
    for asset in [model, Path(os.environ.get("MOE_ASSETS_35B", "/nonexistent"))]:
        if not (asset / "generation_config.json").exists():
            print("NOT EXECUTED: real asset generation_config.json missing")
            continue
        config = json.loads((asset / "generation_config.json").read_text(encoding="utf-8"))
        assert all(config[k] == description["recommended_thinking"][k] for k in ["temperature", "top_k", "top_p"])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        def run(name, extra, options=common):
            folder = root / name
            result = invoke([args.cli, "run", *options, *extra, "--out", str(folder)])
            # Single-turn run returns the answer on stdout; only scripted turns
            # create answer.txt themselves.
            if not (folder / "answer.txt").exists():
                # Match the CLI's own answer.txt: UTF-8 and LF (newline=''), so the later
                # read_bytes comparison against scripted-turn files holds on Windows too.
                (folder / "answer.txt").write_text(result.stdout, encoding="utf-8", newline="")
            return result, folder
        def row(folder):
            return json.loads((folder / "result.json").read_text(encoding="utf-8"))
        def tokens(folder):
            return json.loads((folder / "tokens.json").read_text(encoding="utf-8"))
        for thinking, temperature, top_p in [("off", .7, .8), ("on", 1., .95)]:
            result, folder = run("default-" + thinking, ["--user", "hello", "--thinking", thinking],
                                 [*base, "--max-tokens", "4"])
            r = row(folder)
            assert (r["configuration"]["temperature"], r["configuration"]["top_p"], r["configuration"]["presence_penalty"]) == (temperature, top_p, 1.5)
            assert all(r["sources"][k] == "model" for k in ["temperature", "top_k", "top_p", "presence_penalty"])
        result, folder = run("thinking", ["--user", "hello", "--thinking", "on"])
        assert row(folder)["reasoning"] and not result.stdout
        assert "[thinking]" in result.stderr and "[/thinking]" in result.stderr and "\x1b" not in result.stderr
        turns = [{"user": "hello", "thinking": False}, {"user": "again", "thinking": True},
                 {"user": "next", "thinking": False}, {"regenerate": True, "thinking": False},
                 {"reset": True}, {"user": "last", "thinking": False}]
        script = root / "turns.json"; script.write_text(json.dumps({"turns": turns}))
        _, oracle = run("oracle", ["--turns", str(script)])
        chat = root / "chat"
        result = invoke([args.cli, "chat", *common, "--out", str(chat)],
                        "hello\n/think on\nagain\n/think off\nnext\n/regen\n/new\nlast\n/quit\n")
        answers = []
        for i in range(1, 6):
            a, b = chat / str(i), oracle / str(i)
            assert tokens(a)["input"] == tokens(b)["input"] and tokens(a)["generated"] == tokens(b)["generated"]
            for field in ["reuse_method", "reused_tokens", "computed_tokens", "reasoning"]:
                assert row(a)[field] == row(b)[field], (i, field)
            assert (a / "answer.txt").read_bytes() == (b / "answer.txt").read_bytes()
            answers.append((a / "answer.txt").read_text(encoding="utf-8") + "\n")
            if i in [2, 3]:
                assert row(a)["common_prefix"] >= len(tokens(chat / str(i-1))["input"])
                assert row(a)["reuse_method"] != "cold"
        assert result.stdout == "".join(answers)
        cancelled = root / "cancel"
        invoke([args.cli, "chat", *common, "--thinking", "on", "--sigint-after-fragments", "2", "--out", str(cancelled)],
               "hello\nagain\n/quit\n")
        assert row(cancelled / "1")["stop_reason"] == "cancelled"
        assert row(cancelled / "1")["reasoning"] and not (cancelled / "1/answer.txt").read_text(encoding="utf-8")
        _, fresh = run("fresh", ["--user", "again", "--thinking", "on"])
        assert tokens(cancelled / "2")["input"] == tokens(fresh)["input"]
        assert tokens(cancelled / "2")["generated"] == tokens(fresh)["generated"]
        _, original = run("original", ["--user", "hello"])
        answer = (original / "answer.txt").read_text(encoding="utf-8")
        assert answer
        stop = answer[0]
        _, stopped = run("stopped", ["--user", "hello", "--stop", stop])
        r = row(stopped)
        assert r["stop_reason"] == "stop_string" and r["matched_stop"] == stop
        assert (stopped / "answer.txt").read_text(encoding="utf-8") == answer[:answer.index(stop)]
        generated = tokens(stopped)["generated"]
        assert generated == tokens(original)["generated"][:len(generated)]
        script.write_text(json.dumps({"turns": [{"user": "hello", "stop": [stop]}, {"user": "again"}]}))
        _, stopped_turns = run("stopped-turns", ["--turns", str(script)])
        for i in [1, 2]:
            folder = stopped_turns / str(i)
            _, replay = run(f"replay-{i}", ["--replay", str(folder / "blocks.json")])
            blocks = json.loads((folder / "blocks.json").read_text(encoding="utf-8"))["blocks"]
            selected = [j for j, block in enumerate(blocks) if block["sample"]][-len(tokens(folder)["generated"]):]
            width = row(folder)["vocabulary"] * 4
            data = (replay / "logits.f32").read_bytes()
            assert (folder / "logits.f32").read_bytes() == b"".join(data[j*width:(j+1)*width] for j in selected)
        _, folder = run("logprobs", ["--user", "hello", "--presence-penalty", "0", "--logprobs", "5"])
        r = row(folder)
        logits = np.fromfile(folder / "logits.f32", dtype=np.float32).reshape(-1, r["vocabulary"]).astype(np.float64)
        assert len(r["logprobs"]) == r["generated_count"]
        for index, p in enumerate(r["logprobs"]):
            z = logits[index]; shifted = z - z.max()
            logsoftmax = shifted - np.log(np.exp(shifted).sum())
            order = np.lexsort((np.arange(len(z)), -z))[:5]
            assert abs(p["logprob"] - logsoftmax[p["token"]]) <= 1e-9
            assert [v["token"] for v in p["top"]] == order.tolist()
            assert all(abs(v["logprob"] - logsoftmax[v["token"]]) <= 1e-9 for v in p["top"])
            if z[order[0]] != z[order[1]]:
                assert p["token"] == order[0]
        _, a = run("seed-a", ["--user", "hello", "--temperature", ".7", "--presence-penalty", "1.5", "--seed", "7"])
        _, b = run("seed-b", ["--user", "hello", "--temperature", ".7", "--presence-penalty", "1.5", "--seed", "7"])
        assert tokens(a)["generated"] == tokens(b)["generated"]
    print(json.dumps({"backend": args.backend, "cases": "a-h", "passed": True}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
