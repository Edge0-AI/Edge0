"""bench-local/ab.py - short A/B arms against the installed IQ2_XS model: one setting changed per arm, the server
restarted for each, the same fixed requests, results appended to ab.jsonl.

  python ab.py base workers15 base cyr ...      (arm names from ARMS, run in the order given)
  python ab.py --quick v139 v139-stage0 ...     (decode only, without the prompt reads)
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_all as R  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
BASE_CFG = HERE / "base-iq2_xs.json"          # the configuration every arm starts from
RT = Path(r"D:\Strata-data\mtp\rt")
BUILD = Path(r"D:\Strata-build")

EN = ["Explain in detail how a hash map works: hashing, collisions, resizing, complexity. About 300 words.",
      "Describe how TCP establishes and closes a connection, and what each flag means. About 300 words.",
      "Compare processes and threads: memory, scheduling, communication, failure isolation. About 300 words."]
UK = ["Поясни докладно, як працює хеш-таблиця: хешування, колізії, розширення, складність. Приблизно 300 слів.",
      "Опиши, як TCP встановлює і закриває з'єднання і що означає кожен прапорець. Приблизно 300 слів."]
CODE = ["Write a complete Python module implementing an LRU cache class with get/put, a TTL option, type hints, "
        "docstrings and a small unittest suite. Code only."]


def set_arg(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    return args


def local_engine(name):
    """exe + library folders of a local build (D:\\Strata-build\\dist-<name>, unpacked from its zip)."""
    d = BUILD / f"dist-{name}" / "engine"
    return {"exe": str(d / "strata.exe"), "lib_dirs": [str(d / "rocm" / "bin")]}


QUICK = "--quick" in sys.argv                  # decode only: no 8K / 30K prompt reads (an arm in ~2 min instead of ~4)
QSETS = ("hard", "js", "sql", "regex")         # strata_eval2.py's suites an arm with "quality" runs
V139 = {"engine": "139", "root": "src-139"}


def v139s(env=None, **more):
    """A 0.1.39 arm with STRATA_SH_STREAM=0, plus this arm's own environment and settings."""
    return {**V139, **more, "env": {"STRATA_SH_STREAM": "0", **(env or {})}}

ARMS = {
    "base": {},
    "workers11": {"args": {"--pool-workers": "11"}},
    "workers15": {"args": {"--pool-workers": "15"}},
    "cyr": {"draft_vocab": "draft_vocab_cyrillic.bin"},
    "en": {"draft_vocab": "draft_vocab_en.bin"},
    "prefill16k": {"args": {"--prefill": "auto:16384"}},
    # MMQ is already on by default (prefill.cpp mmq_plan), so the A/B is turning it off: the FP16 expert path.
    # STRATA_PF_FUSED=1 is not an arm: on HIP the fused kernels fall back to MMQ (moe_fused.hpp).
    "mmq0": {"env": {"STRATA_PREFILL_MMQ": "0"}},
    "ring96": {"env": {"STRATA_PREFILL_RING": "96"}},
    "pcie30": {"args": {"--pcie-frac": "0.3"}},
    "pcie80": {"args": {"--pcie-frac": "0.8"}},
    "spec3": {"args": {"--spec": "3"}},
    "spec5": {"args": {"--spec": "5"}},
    "minp30": {"args": {"--spec-min-p": "0.3"}},
    "minp70": {"args": {"--spec-min-p": "0.7"}},
    # local builds
    "build-base": {"engine": "base"},
    "build-prs": {"engine": "prs"},
    "prs-pipeline": {"engine": "prs", "env": {"STRATA_POOL_PIPELINE": "1"}},
    "prs-kvprefetch": {"engine": "prs", "env": {"STRATA_KV_PREFETCH": "1"}},
    "prs-hcq8": {"engine": "prs", "env": {"STRATA_HC_Q8": "1"}},
    "prs-16k": {"engine": "prs", "args": {"--prefill": "auto:16384"}},
    "prs-all": {"engine": "prs", "args": {"--prefill": "auto:16384"},
                "env": {"STRATA_POOL_PIPELINE": "1", "STRATA_KV_PREFETCH": "1", "STRATA_HC_Q8": "1"}},
    # 0.1.39 (released 2026-10-04): its engine built here for gfx1030 and its own server, from D:\Strata-build\src-139
    "v139": {**V139},
    # 0.1.39 decodes slower than 0.1.38 here: its new decode steps turned off one at a time (each is default on)
    "v139-stage0": {**V139, "env": {"STRATA_IQ_STAGE_GRID": "0"}},          # i-quant grids not staged in shared memory
    "v139-decbatch0": {**V139, "env": {"STRATA_DEC_BATCH": "0"}},           # the window's rows token by token
    "v139-plebatch0": {**V139, "env": {"STRATA_PLE_BATCH": "0"}},           # the PLE block token by token
    "v139-shstream0": {**V139, "env": {"STRATA_SH_STREAM": "0"}},
    "v139-oldmmvq": {**V139, "env": {"STRATA_OLD_IQ_MMVQ": "1"}},           # the per-column i-quant kernels
    # v139s = 0.1.39 with STRATA_SH_STREAM=0 (the shared expert on the window's own stream): measured here
    # 56.4 / 43.5 / 62.7 tok/s (en / uk / code) against 42.5 / 32.5 / 50.2 with the forked stream.  The arms below
    # change one more thing each.
    "v139s": v139s(),
    "v139s-queue": v139s(par=[2, 4]),                                       # one at a time: the others wait in the queue
    "v139s-par2": v139s(cfg={"parallel": 2}, par=[1, 2, 4]),                # what 2 batch slots cost a request alone
    "v139s-par4": v139s(cfg={"parallel": 4}, par=[1, 2, 4]),
    "v139s-cyr": v139s(draft_vocab="draft_vocab_cyrillic.bin"),
    "v139s-ring0": v139s({"STRATA_RING_BYTES": "0"}),                       # 0.1.38's prompt ring (#583 off)
    "v139s-stage0": v139s({"STRATA_IQ_STAGE_GRID": "0"}),
    "v139s-nowait": v139s({"STRATA_ADAPT_NOWAIT": "1"}),
    "v139s-spec6": v139s(args={"--spec": "6"}),                             # issue #775's settings, one at a time
    "v139s-minp70": v139s(args={"--spec-min-p": "0.7"}),
    "v139s-pcie0": v139s(args={"--pcie-frac": "0.0"}),
    # the model sizes side by side on the same engine: speed, and the correctness suites with reasoning off
    "v139s-qual": v139s(quick=True, quality=True),                          # IQ2_XS, the installed size
    "v139s-q20": v139s(model="q2_0", quality=True),
    "v139s-iq3xxs": v139s(model="iq3_xxs", quality=True),
    # IQ3_XXS: 0.1.39's opt-in AVX2 gather for the IQ3 grids on the CPU (#622, bit-exact), and the Cyrillic drafts
    "v139s-iq3xxs-g": v139s({"STRATA_IQ256_GATHER": "1"}, model="iq3_xxs"),
    "v139s-iq3xxs-cyr": v139s(model="iq3_xxs", draft_vocab="draft_vocab_cyrillic.bin"),
}


def start(arm):
    spec = ARMS[arm]
    # another model size: its own config as setup wrote it (strata-<model>.json), with this script's one engine log
    base = ROOT / f"strata-{spec['model']}.json" if spec.get("model") else BASE_CFG
    cfg = json.loads(base.read_text(encoding="utf-8"))
    cfg["log"] = str(ROOT / "strata-iq2_xs.log")
    for flag, value in spec.get("args", {}).items():
        cfg["args"] = set_arg(cfg["args"], flag, value)
    if spec.get("env"):
        cfg["env"] = {**(cfg.get("env") or {}), **spec["env"]}
    if spec.get("engine"):
        cfg.update(local_engine(spec["engine"]))
    cfg.update(spec.get("cfg", {}))
    root = BUILD / spec["root"] if spec.get("root") else ROOT     # the checkout whose server and data/ the arm runs
    if root != ROOT:
        cfg["cwd"] = str(root)
        cfg["args"] = set_arg(cfg["args"], "--expert-profile", str(root / "data" / "expert-profile.bin"))
    R.stop_server()
    vocab = spec.get("draft_vocab", "draft_vocab.bin")      # the default subset unless the arm names another
    shutil.copyfile(root / "data" / vocab, RT / "draft_vocab.bin")
    R.CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return R.start_server() if root == ROOT else start_server(root)


def start_server(root):
    """R.start_server with another checkout's server (a newer version's worktree); the same wait for /health."""
    out = open(ROOT / "server.out", "w")
    err = open(ROOT / "server.err", "w")
    proc = subprocess.Popen([R.PY, "serve/server.py", "--engine", "strata", "--config", str(R.CFG), "--port", "8080"],
                            cwd=root, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    t = time.time()
    while time.time() - t < 1200:
        if R.health().get("loaded"):
            return round(time.time() - t)
        if proc.poll() is not None:
            return None
        time.sleep(3)
    return None


def measure(arm):
    rec = {"arm": arm, "time": time.strftime("%H:%M:%S")}
    for name, prompts, mt in (("en", EN, 420), ("uk", UK, 520), ("code", CODE, 700)):
        runs = [R.ask(f"[{uuid.uuid4().hex[:8]}] " + p, mt) for p in prompts]
        rec[name] = [r.get("tok_s") for r in runs]
    src = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
    rec.update({"read8k": None, "read30k": None})
    for name, kb in () if QUICK or ARMS[arm].get("quick") else (("read8k", 30), ("read30k", 125)):
        r = R.ask(f"Run {uuid.uuid4().hex}.\n\n" + src[:kb * 1000] + "\n\nIn one sentence: what is this code?", 40)
        rec[name] = r.get("prompt_tok_s")
        rec[name + "_err"] = r.get("error")
    rec.update(R.engine_facts())
    log = (ROOT / "strata-iq2_xs.log").read_text(encoding="utf-8", errors="replace")[-60000:]
    acc = [(int(a), int(b)) for a, b in __import__("re").findall(r"drafts accepted (\d+) of (\d+)", log)][-8:]
    rec["draft_accept"] = round(sum(a for a, _ in acc) / max(1, sum(b for _, b in acc)), 3)
    return rec


def main():
    if not BASE_CFG.exists():
        shutil.copyfile(R.CFG, BASE_CFG)
    for arm in [a for a in sys.argv[1:] if not a.startswith("--")]:
        secs = start(arm)
        if secs is None:
            rec = {"arm": arm, "failed_to_start": True}
            tail = (ROOT / "server.err").read_text(encoding="utf-8", errors="replace")[-400:]
            print(f"AB {arm}: FAILED TO START | {tail.splitlines()[-1] if tail.strip() else ''}", flush=True)
        else:
            rec = measure(arm)
            med = lambda v: round(statistics.median([x for x in v if x]), 1) if any(v) else None  # noqa: E731
            print(f"AB {arm}: en {med(rec['en'])} uk {med(rec['uk'])} code {med(rec['code'])} tok/s | read 8K "
                  f"{rec['read8k']} 30K {rec['read30k']} tok/s | slots {rec.get('expert_slots')} free "
                  f"{rec.get('vram_free_mib')} MiB | drafts {rec['draft_accept']}", flush=True)
            if ARMS[arm].get("par"):        # several requests at once against this arm's server: par.py's PAR lines
                subprocess.run([R.PY, str(HERE / "par.py"), *map(str, ARMS[arm]["par"])], cwd=HERE)
            if ARMS[arm].get("quality"):    # the small correctness suites (strata_eval2.py), reasoning off
                res = f"results2-{arm}.json"
                with (HERE / f"eval2-{arm}.log").open("w", encoding="utf-8") as out:
                    subprocess.run([R.PY, str(HERE / "strata_eval2.py"), "hard", "js", "sql", "regex"], cwd=HERE,
                                   stdout=out, stderr=subprocess.STDOUT,
                                   env={**os.environ, "EFFORTS": "none", "RESULTS2": res, "PYTHONIOENCODING": "utf-8"})
                q = json.loads((HERE / res).read_text(encoding="utf-8"))
                failed = [x.get("task") or x.get("q") or x.get("what") for k in QSETS for x in q[k] if not x.get("pass")]
                secs = [x["wall"] for x in q["hard"] if x.get("wall")]
                print(f"QUAL {arm}: " + ", ".join(f"{k} {sum(1 for x in q[k] if x.get('pass'))}/{len(q[k])}"
                                                   for k in QSETS) +
                      f" | hard tasks {min(secs, default=0)}-{max(secs, default=0)} s | failed: {failed}", flush=True)
        with (HERE / "ab.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if "--no-restore" in sys.argv:      # another ab.py run follows at once
        return
    start("base")                       # leave the default configuration running
    print("AB done; base configuration restored", flush=True)


if __name__ == "__main__":
    main()
