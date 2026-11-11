#!/usr/bin/env python3
"""Fixed-token, single-session native engine baseline. TTFT here excludes HTTP/SSE."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from single_gpu import ISOLATION, ROOT, validate_config
from resource_sampler import ResourceSampler

sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env
from conversation_cache_parity import load_tokenizer, state_hashes


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def gpu_fdinfo(pid):
    """Per-DRM-device counters; amd-smi may list one KFD process on every card."""
    records = {}
    for path in (Path("/proc") / str(pid) / "fdinfo").iterdir():
        try:
            text = path.read_text()
            if "drm-" in text:
                records[path.name] = text
        except FileNotFoundError:
            continue
    return records


def make_fixtures(cfg, lengths):
    directory = Path(cfg["tokenizer"])
    tok = load_tokenizer(directory)
    tpl = ChatTemplate(directory / "chat_template.jinja")
    marker = "STRATA_FIXTURE_DOCUMENT_MARKER"
    template = tpl.render([{"role": "user", "content": "Read the following numbered records.\n" + marker
                           + "\nSummarize the record format, then write Python code to parse and validate it. "
                           "Explain duplicate IDs and invalid integers."}], enable_thinking=False)
    before, after = template.split(marker)
    prefix, suffix = (tok.encode(text, parse_special=True) for text in (before, after))
    corpus = "\n".join(f"record {i:06d}: color={('red', 'green', 'blue')[i % 3]}, value={i * 17 % 997};"
                       for i in range(max(lengths) // 8 + 100))
    body = tok.encode(corpus)
    fixtures = []
    for length in lengths:
        count = length - len(prefix) - len(suffix)
        if count <= 0 or count > len(body):
            raise ValueError("fixture length outside corpus bounds")
        ids = prefix + body[:count] + suffix
        fixtures.append({"name": f"records-{length}", "tokens": length, "ids": ids,
                         "text": tok.decode(ids),
                         "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    return {"schema": 1, "description": "Exact native protocol token IDs; performance fixtures, not quality gates.",
            "tokenizer": str(directory), "fixtures": fixtures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768, 131072])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--shape-warmups", type=int, default=0,
                        help="full generations per selected shape, recorded but excluded from measurement")
    args = parser.parse_args()
    if args.repeats < 1 or args.shape_warmups < 0 or args.max_new < 1:
        parser.error("repeats/max-new must be positive; shape-warmups must be nonnegative")
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    if args.make_fixtures:
        if args.fixtures.exists():
            parser.error("fixture already exists; use it unchanged or choose a new path")
        save(args.fixtures, make_fixtures(cfg, args.lengths))
        return
    if args.out is None:
        parser.error("--out is required for measurement")
    if any(os.environ.get(k) != v for k, v in ISOLATION.items()):
        parser.error("launch measurements through single_gpu.py")
    args.out.mkdir(parents=True, exist_ok=False)
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    selected = [f for f in fixtures if f["tokens"] in args.lengths]
    if {f["tokens"] for f in selected} != set(args.lengths):
        parser.error("missing requested fixture length")
    for fixture in selected:
        if (len(fixture["ids"]) != fixture["tokens"] or fixture["ids_sha256"] !=
                hashlib.sha256(json.dumps(fixture["ids"]).encode()).hexdigest()):
            raise ValueError("fixture token count or hash changed")
    env = child_env(cfg)
    if any(env.get(k) != v for k, v in ISOLATION.items()):
        raise ValueError("child environment changed GPU isolation")
    manifest = {"config": cfg, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "fixtures_path": str(args.fixtures.resolve()),
                "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
                "sampling": {"temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 42},
                "max_new": args.max_new, "timing_scope": "native engine; excludes HTTP/SSE",
                "warmup": {"short": 1, "per_shape": args.shape_warmups,
                           "max_new": args.max_new, "exclude_phases": ["short_warmup", "shape_warmup"]},
                "records": []}
    save(args.out / "results.json", manifest)
    start = time.perf_counter()
    engine = StrataEngine(cfg["exe"], cfg["args"], cwd=cfg["cwd"],
                          log=str(args.out / "engine.log"), env=env)
    manifest.update(load_s=time.perf_counter() - start, engine_info=dict(engine.info), pid=engine.proc.pid)
    save(args.out / "results.json", manifest)
    resources = ResourceSampler(engine.proc.pid)
    try:
        proc = Path(f"/proc/{engine.proc.pid}")
        (args.out / "process-status.txt").write_text((proc / "status").read_text())
        save(args.out / "thread-affinity.json", {
            path.name: {k: v.strip() for line in (path / "status").read_text().splitlines()
                        if ":" in line for k, v in [line.split(":", 1)]
                        if k in {"Name", "Cpus_allowed_list", "Mems_allowed_list"}}
            for path in (proc / "task").iterdir()})
        (args.out / "loaded-maps.txt").write_text((proc / "maps").read_text())
        (args.out / "numastat.txt").write_text(subprocess.check_output(["numastat", "-p", str(engine.proc.pid)], text=True))
        save(args.out / "gpu-fdinfo-ready.json", gpu_fdinfo(engine.proc.pid))
        tok = load_tokenizer(Path(cfg["tokenizer"]))
        tpl = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
        warm_ids = tok.encode(tpl.render([{"role": "user", "content": "请用一句话解释什么是质数。"}],
                                         enable_thinking=False), parse_special=True)
        manifest["warmup_ids"] = warm_ids
        work = [{"name": "warmup", "ids": warm_ids, "tokens": len(warm_ids), "phase": "short_warmup"}]
        work += [dict(f, repeat=n, phase="shape_warmup")
                 for n in range(args.shape_warmups) for f in selected]
        work += [dict(f, repeat=n, phase="measurement") for n in range(args.repeats) for f in selected]
        for fixture in work:
            begin, first, output = time.perf_counter(), None, []
            for token in engine.generate(fixture["ids"], args.max_new, manifest["sampling"], threading.Event()):
                if token is not None:
                    if first is None:
                        first = time.perf_counter()
                    output.append(token)
            record = {"name": fixture["name"], "repeat": fixture.get("repeat"),
                      "phase": fixture["phase"],
                      "start_monotonic_s": begin, "end_monotonic_s": time.perf_counter(),
                      "input_tokens": len(fixture["ids"]), "input_ids_sha256":
                      hashlib.sha256(json.dumps(fixture["ids"]).encode()).hexdigest(),
                      "wall_s": time.perf_counter() - begin, "ttft_s": first - begin if first else None,
                      "output_ids": output, "text": tok.decode(output), **engine.last}
            manifest["records"].append(record)
            if env.get("STRATA_STATE_HASH") is not None:
                hashes = state_hashes((args.out / "engine.log").read_text())
                if len(hashes) != len(manifest["records"]):
                    save(args.out / "results.json", manifest)
                    raise RuntimeError("requested state hash was not emitted; this engine requires --prompt-cache > 0")
                record["state"] = hashes[-1]
            save(args.out / "results.json", manifest)
            save(args.out / f"gpu-fdinfo-{len(manifest['records'])}.json", gpu_fdinfo(engine.proc.pid))
            print(json.dumps({k: v for k, v in record.items() if k not in ("text", "output_ids")}), flush=True)
            if not output or record["finish"] not in ("stop", "length"):
                raise RuntimeError("generation did not finish normally")
            if record["prompt_tokens"] != len(fixture["ids"]):
                raise RuntimeError("engine prompt length does not match fixed fixture")
            if fixture["name"] != "warmup" and record.get("reused") != 0:
                raise RuntimeError("fresh baseline reused prompt tokens")
    finally:
        save(args.out / "resources.json", resources.close())
        engine.close()


if __name__ == "__main__":
    main()
