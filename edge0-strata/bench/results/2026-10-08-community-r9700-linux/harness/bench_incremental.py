#!/usr/bin/env python3
"""Time fixed short followups on retained 4K/32K prefixes; synthetic performance fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from bench_engine import save, load_tokenizer, ChatTemplate, StrataEngine, child_env, gpu_fdinfo
from resource_sampler import ResourceSampler
from single_gpu import ISOLATION, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--increments", type=int, nargs="+", default=[256, 512, 900, 2048, 4096])
    args = parser.parse_args()
    if args.repeats < 1 or args.max_new < 1 or min(args.lengths + args.increments) < 1:
        parser.error("repeats, max-new, lengths and increments must be positive")
    if len(set(args.lengths)) != len(args.lengths) or len(set(args.increments)) != len(args.increments):
        parser.error("duplicate lengths or increments")
    if any(os.environ.get(k) != v for k, v in ISOLATION.items()):
        parser.error("run through single_gpu.py")
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    flags = list(cfg["args"])
    flags[flags.index("--prompt-cache") + 1] = "6"
    tok = load_tokenizer(Path(cfg["tokenizer"]))
    tpl = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
    reset = tok.encode(tpl.render([{"role": "user", "content": "换个话题，请只回答你好。"}], enable_thinking=False), parse_special=True)
    fixtures = {f["tokens"]: f for f in json.loads(args.fixtures.read_text())["fixtures"]}
    extensions = {}
    filler = tok.encode("\n".join(f"additional record {i}: value={i % 97};" for i in range(max(args.increments))))
    for increment in args.increments:
        # A fixed, coherent assistant turn makes every arm consume the same followup IDs even when
        # generated outputs differ. This is a performance fixture, not a task-quality score.
        marker = "STRATA_INCREMENTAL_RECORDS_MARKER"
        prompt = "Each record contains an ID, a color, and an integer value. Validation must reject duplicates.<|im_end|>\n"
        prompt += tpl.render([{"role": "user", "content": f"Followup {increment}. Additional records:\n" + marker +
                              "\nNow explain how to reject duplicate IDs and invalid integers."}], enable_thinking=False)
        before, after = prompt.split(marker)
        prefix, suffix = [tok.encode(s, parse_special=True) for s in (before, after)]
        body_size = increment - len(prefix) - len(suffix)
        if body_size <= 0:
            parser.error("increment too short for the fixed template")
        extensions[increment] = prefix + filler[:body_size] + suffix
        if len(extensions[increment]) != increment:
            raise ValueError("increment token count changed")
    bases = {}
    for length in args.lengths:
        ids = fixtures[length]["ids"]
        if len(ids) != length or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != fixtures[length]["ids_sha256"]:
            raise ValueError("fixture changed")
        bases[length] = ids
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "native retained-prefix timing; fixed canonical assistant turns; not quality evidence",
                "config": cfg, "actual_args": flags, "max_new": args.max_new,
                "fixtures_path": str(args.fixtures.resolve()),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
                "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
                "extensions": extensions, "reset_ids": reset, "warmup_sequences_per_length": 1,
                "base_controls": [], "records": []}
    save(args.out / "results.json", manifest)
    env = child_env(cfg)
    if any(env.get(k) != v for k, v in ISOLATION.items()):
        raise ValueError("child environment changed GPU isolation")
    engine = StrataEngine(cfg["exe"], flags, cwd=cfg["cwd"], log=str(args.out / "engine.log"), env=env)
    manifest.update(engine_info=dict(engine.info), pid=engine.proc.pid)
    sampler = ResourceSampler(engine.proc.pid)
    sampling = {"temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 42}
    manifest["sampling"] = sampling
    try:
        for repeat in range(-1, args.repeats):
            for length, base_ids in bases.items():
                list(engine.generate(reset, 1, sampling, threading.Event()))
                for turn, (increment, extension) in enumerate(extensions.items()):
                    # Rewind to exactly the same history before each branch. Control requests are
                    # retained separately and are never counted as followup performance.
                    head = [t for t in engine.generate(base_ids, 1, sampling, threading.Event()) if t is not None]
                    control = {"base_tokens": length, "increment": increment, "repeat": repeat,
                               "output_ids": head, **engine.last}
                    manifest["base_controls"].append(control)
                    save(args.out / "results.json", manifest)
                    if len(head) != 1 or control["finish"] not in {"stop", "length"} or (turn == 0 and control["reused"] != 0):
                        raise RuntimeError("base control did not complete on the required path")
                    ids = base_ids + extension
                    begin, first, output = time.perf_counter(), None, []
                    for token in engine.generate(ids, args.max_new, sampling, threading.Event()):
                        if token is not None:
                            if first is None:
                                first = time.perf_counter()
                            output.append(token)
                    record = {"base_tokens": length, "increment": increment, "repeat": repeat,
                              "phase": "shape_warmup" if repeat < 0 else "measurement",
                              "input_tokens": len(ids), "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                              "start_monotonic_s": begin, "end_monotonic_s": time.perf_counter(),
                              "wall_s": time.perf_counter() - begin, "ttft_s": first - begin if first else None,
                              "output_ids": output, "text": tok.decode(output), **engine.last}
                    manifest["records"].append(record)
                    save(args.out / "results.json", manifest)
                    if not output or record["finish"] not in {"stop", "length"}:
                        raise RuntimeError("generation did not complete")
                    if not length - 1 <= record["reused"] <= length:
                        raise RuntimeError("followup did not retain exactly the declared history")
                    if record["prompt_tokens"] != len(ids):
                        raise RuntimeError("prompt length changed")
                    print(json.dumps({k: v for k, v in record.items() if k not in {"text", "output_ids"}}), flush=True)
        manifest["complete"] = True
        save(args.out / "results.json", manifest)
        save(args.out / "gpu-fdinfo-final.json", gpu_fdinfo(engine.proc.pid))
    finally:
        save(args.out / "resources.json", sampler.close())
        engine.close()


if __name__ == "__main__":
    main()
