"""Tests of the patched Strata engine (branch perf/async-adapt-arena) against the release and the unpatched local build.
Every session is a fresh engine; it is stopped at the end.  Run in the foreground (<= 10 min per call).
  smoke <variant>                    one short request with STRATA_DECODE_TIMING=1; prints the async/adapt/ERR log lines
  ab <v1,v2,...> <round> <out.jsonl> 12 distinct cold prompts per session (one request each, fixed seeds)
  det <variant> <n> <out.jsonl>      greedy (temp 0), 6 prompts, n fresh engines: answers' sha256 compared
  stress <variant>                   63k prompt then 4 follow-up turns in the same conversation
  report <out.jsonl> <ref>           paired per-prompt ratios of every variant against <ref>
Local exe: env STRATA_DEV_EXE (default strata-dev\\build-vs22\\strata.exe)."""
import copy
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

STRATA = Path(r"C:\Users\urben\Documents\strata")
DEV_EXE = os.environ.get("STRATA_DEV_EXE", r"C:\Users\urben\Documents\strata-dev\build-vs22\strata.exe")
CUDA128 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
DEV = r"C:\Users\urben\Documents\strata-dev"
# the install config now runs the patched engine: variants are built from the pre-patch (release) config
BASE_CFG = json.loads((STRATA / "strata-iq3_s.json.before-async").read_text(encoding="utf-8"))
# the morning's starting point (engine 0.1.38 config before any tuning) is not reproducible: 0.1.38 is gone
FIXTURE_262K = r"C:\Users\urben\Documents\folk\bench\fixtures\out\262k.jsonl"
SCRATCH = Path(__file__).parent
PORT = 8080
LOCAL = {"exe": DEV_EXE, "lib_dirs": [CUDA128]}
ASYNC = {"--adapt-async": "1"}
# name: (config overrides, args, env)
VARS = {
    # the morning's starting point: engine 0.1.38 (kept by setup in engine/.previous) with the config before any tuning
    "base0138": ({"_base": "strata-iq3_s.json.before-optim", "exe": str(STRATA / "engine/.previous/strata.exe")}, {}, {}),
    "rel": ({}, {}, {}),
    "loc": (LOCAL, {}, {}),
    "async": (LOCAL, ASYNC, {}),
    "asyncb": (LOCAL, ASYNC, {"STRATA_ADAPT_BOUNCE": "1"}),
    "asyncbp": (LOCAL, ASYNC, {"STRATA_ADAPT_BOUNCE": "1", "STRATA_PCIE_MIN1": "1"}),
    "pmin1": (LOCAL, {}, {"STRATA_PCIE_MIN1": "1"}),
    "async_b16": (LOCAL, ASYNC, {"STRATA_ADAPT_ASYNC_BATCH": "16"}),
    "async_b80": (LOCAL, ASYNC, {"STRATA_ADAPT_ASYNC_BATCH": "80"}),
    "async_nowait": (LOCAL, ASYNC, {"STRATA_ADAPT_ASYNC_WAIT": "0"}),
    "stack2prof": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
                   {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
                    "STRATA_IQ3S_MT1": "1", "STRATA_VERIFY_PROFILE": "1", "STRATA_SPEC_DEPTH": "1"}),
    "stack3": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
               {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
                "STRATA_IQ3S_MT1": "1", "STRATA_GR_V3": "1", "STRATA_EXPERT_V2": "1", "STRATA_QFUSE": "1", "STRATA_MMVF_ROWS": "1"}),
    "r2all": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
              {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
               "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ADAPT_ASYNC_DEMAND": "1", "STRATA_GR_128": "1"}),
    "r2allprof": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
              {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
               "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ADAPT_ASYNC_DEMAND": "1", "STRATA_GR_128": "1",
               "STRATA_VERIFY_PROFILE": "1"}),
    "r2p23": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
              {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
               "STRATA_IQ3S_MT1": "1", "STRATA_ADAPT_ASYNC_DEMAND": "1", "STRATA_GR_128": "1"}),
    "r2p23prof": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
              {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
               "STRATA_IQ3S_MT1": "1", "STRATA_ADAPT_ASYNC_DEMAND": "1", "STRATA_GR_128": "1", "STRATA_VERIFY_PROFILE": "1"}),
    "p3": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
           {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
            "STRATA_IQ3S_MT1": "1", "STRATA_GR_128": "1"}),
    "p3b16": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
              {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
               "STRATA_IQ3S_MT1": "1", "STRATA_GR_128": "1", "STRATA_ADAPT_ASYNC_BATCH": "16"}),
    "p1": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
           {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
            "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ARENA_PIN_GIB": "29"}),
    "p1pm": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
             {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
              "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ARENA_PIN_GIB": "29", "STRATA_PCIE_MIN1": "1"}),
    "p1pf6": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head", "--pcie-frac": "0.6"},
              {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
               "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ARENA_PIN_GIB": "29"}),
    "p1both": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head", "--pcie-frac": "0.6"},
               {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
                "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ARENA_PIN_GIB": "29", "STRATA_PCIE_MIN1": "1"}),
    # bit-exact mode (docs/MULTI_GPU.md): STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0, the PRs' options OFF
    "x_v0142": ({"exe": DEV + r"\bin-v0142\strata.exe", "lib_dirs": [CUDA128]},
                {"--pcie-frac": "0", "--adapt-every": "0"}, {"STRATA_IQ_MT_MIN": "1"}),
    "x_pr1": ({"exe": DEV + r"\bin-pr1\strata.exe", "lib_dirs": [CUDA128]},
              {"--pcie-frac": "0", "--adapt-every": "0"}, {"STRATA_IQ_MT_MIN": "1"}),
    "x_pr2": ({"exe": DEV + r"\bin-pr2\strata.exe", "lib_dirs": [CUDA128]},
              {"--pcie-frac": "0", "--adapt-every": "0"}, {"STRATA_IQ_MT_MIN": "1"}),
    # the PR builds, options off / on
    "v0142": ({"exe": DEV + r"\bin-v0142\strata.exe", "lib_dirs": [CUDA128]}, {}, {}),
    "pr1_on": ({"exe": DEV + r"\bin-pr1\strata.exe", "lib_dirs": [CUDA128]}, ASYNC, {}),
    "pr2_on": ({"exe": DEV + r"\bin-pr2\strata.exe", "lib_dirs": [CUDA128]}, {"--pcie-frac": "0.6"},
               {"STRATA_PIN_BY_NEED": "1", "STRATA_ARENA_PIN_GIB": "29", "STRATA_PCIE_MIN1": "1"}),
    "stack1": (LOCAL, ASYNC, {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1"}),
    "stack2": (LOCAL, {**ASYNC, "--pool-tasks": "36", "--lookup-chain": "2", "--mtp-q4": "head"},
               {"STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
                "STRATA_IQ3S_MT1": "1"}),
}
VARS_X = dict(VARS)
for _n in ("v0142", "pr1", "pr2"):
    _o, _a, _e = VARS_X["x_" + _n]
    VARS["y_" + _n] = (_o, {**_a, "--expert-cache": "8000"}, _e)
COLD = [
    "Explique le fonctionnement d'une pompe à chaleur en 6 phrases.",
    "Write a Python class implementing an LRU cache with O(1) get and put, with a short usage example.",
    "Describe the main causes of the fall of the Western Roman Empire.",
    "Donne une recette détaillée de ratatouille pour 4 personnes.",
    "Explain how mRNA vaccines work, step by step.",
    "Écris une fonction JavaScript qui fait un debounce d'une autre fonction, avec des commentaires en français.",
    "Write a short project status email announcing a two-week delay and its reasons.",
    "Présente les principales caractéristiques géographiques de l'île de La Réunion.",
    "Explain the difference between permutations and combinations with three worked examples.",
    "Write a SQL query that returns the top 3 customers by revenue per country, and explain it.",
    "Résume la pensée de Descartes sur le doute méthodique.",
    "Compare TCP and UDP and give two use cases for each.",
]
WARM = ["Bonjour, présente-toi en une phrase.", "Say hello in one short sentence."]
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in "
                  r"(\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
NOTE = re.compile(r"adapt|async|ERR|error|residency|bounce|PCIE_MIN1|decode timing|K/V grown|hit rate|GPU stages|stage|profile|spec depth|lookup chain|suffix", re.I)


def free_gb() -> float:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) / 1024 / 1024


def strata_running() -> bool:
    return "strata.exe" in subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True, text=True).stdout


class Server:
    def __init__(self, name: str, variant: str, extra_env: dict | None = None):
        over, args, env = VARS[variant]
        over = dict(over)
        base = over.pop("_base", None)
        cfg = json.loads((STRATA / base).read_text(encoding="utf-8")) if base else copy.deepcopy(BASE_CFG)
        cfg.update(copy.deepcopy(over))
        a = cfg["args"]
        for flag, val in args.items():
            if flag in a:
                a[a.index(flag) + 1] = val
            else:
                a += [flag, val]
        cfg.setdefault("env", {}).update(env)
        cfg["env"].update(extra_env or {})
        self.name, self.log = name, SCRATCH / f"ab2_{name}.log"
        self.log.write_bytes(b"")
        cfg["log"] = str(self.log)
        path = STRATA / "strata-ab.json"
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        for _ in range(60):
            if not strata_running() and free_gb() > 50:
                break
            time.sleep(2)
        print(f"[{name}] start ({variant}), free RAM {free_gb():.1f} GB", flush=True)
        self.proc = subprocess.Popen([str(STRATA / ".venv/Scripts/python.exe"), "serve/server.py", "--engine", "strata",
                                      "--config", str(path), "--port", str(PORT)], cwd=STRATA,
                                     stdout=open(SCRATCH / f"ab2_{name}.out", "wb"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
                break
            except Exception:
                if self.proc.poll() is not None or time.time() - t0 > 300:
                    self.stop()
                    tail = (SCRATCH / f"ab2_{name}.out").read_bytes()[-1500:].decode("utf-8", "replace")
                    raise SystemExit(f"{name}: server did not start\n{tail}")
                time.sleep(3)
        for i, p in enumerate(WARM):
            self.ask([{"role": "user", "content": p}], seed=7 + i, max_tokens=48)

    def ask(self, messages: list, temp=0.7, seed=None, max_tokens=256) -> dict:
        req = {"model": "strata", "messages": messages, "max_tokens": max_tokens, "temperature": temp,
               "top_p": 0.8 if temp else 1.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        if seed:
            req["seed"] = seed
        start = self.log.stat().st_size
        r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(req).encode(),
                                   {"Content-Type": "application/json"})
        big = sum(len(m["content"]) for m in messages) > 20_000
        with urllib.request.urlopen(r, timeout=900 if big else 150) as resp:
            body = json.loads(resp.read())
        time.sleep(0.5)
        text = body["choices"][0]["message"].get("content") or ""
        row = dict(session=self.name, text=text, sha=hashlib.sha256(text.encode()).hexdigest()[:16], notes=[])
        for line in self.log.read_bytes()[start:].decode("utf-8", "replace").splitlines():
            m = LINE.search(line)
            if m:
                row.update(prompt_tokens=int(m[1]), prefill_tps=float(m[5]), gen=int(m[6]), decode_tps=float(m[8]),
                           acc=int(m[9]) if m[9] else 0, drafted=int(m[10]) if m[10] else 0)
            elif NOTE.search(line):
                row["notes"].append(line.strip()[:400])
        return row

    def stop(self) -> None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], capture_output=True)
        for _ in range(60):
            if not strata_running():
                break
            time.sleep(1)


def startup_notes(srv: Server) -> list:
    return [l.strip()[:300] for l in srv.log.read_bytes().decode("utf-8", "replace").splitlines()
            if re.search(r"adapt-async|asynchronous|bounce|PCIE_MIN1|is off|ERR|error|pin by need|ranked pairs|strata hc|differs|128|demand|arena", l, re.I)]


cmd = sys.argv[1]
if cmd == "smoke":
    v = sys.argv[2]
    srv = Server(f"smoke_{v}", v, {"STRATA_DECODE_TIMING": "1"})
    try:
        for l in startup_notes(srv):
            print("  start:", l)
        for i in (0, 1, 4):
            r = srv.ask([{"role": "user", "content": COLD[i]}], seed=3000 + i)
            print(f"  prompt {i}: decode {r.get('decode_tps')} tok/s, gen {r.get('gen')}, drafts {r.get('acc')}/{r.get('drafted')}")
            for n in r["notes"][-6:]:
                print("     ", n)
    finally:
        srv.stop()
elif cmd == "ab":
    out = Path(sys.argv[4])
    for v in sys.argv[2].split(","):
        srv = Server(f"{v}_r{sys.argv[3]}", v)
        try:
            vals = []
            for i, p in enumerate(COLD):
                r = srv.ask([{"role": "user", "content": p}], seed=3000 + i)
                r.update(variant=v, round=sys.argv[3], prompt_i=i)
                r.pop("text")
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if "decode_tps" in r:
                    vals.append(r["decode_tps"])
            vals.sort()
            print(f"{v:14s} round {sys.argv[3]}: median {vals[len(vals) // 2]:.1f} tok/s (min {vals[0]:.1f}, max {vals[-1]:.1f})",
                  flush=True)
            errs = [l for l in startup_notes(srv) if re.search(r"ERR|error", l)]
            if errs:
                print("   errors:", errs[:3])
        finally:
            srv.stop()
elif cmd == "det":
    v, n, out = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    shas = []
    for k in range(n):
        srv = Server(f"det_{v}_{k}", v)
        try:
            run = [srv.ask([{"role": "user", "content": COLD[i]}], temp=0)["sha"] for i in range(6)]
        finally:
            srv.stop()
        shas.append(run)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"variant": v, "engine": k, "shas": run}) + "\n")
        print(f"det {v} engine {k}: {run}", flush=True)
    same = sum(len({s[i] for s in shas}) == 1 for i in range(6))
    print(f"det {v}: {same}/6 prompts identical across {n} fresh engines", flush=True)
elif cmd == "stress":
    v = sys.argv[2]
    srv = Server(f"stress_{v}", v)
    try:
        fixture = json.loads(open(r"C:\Users\urben\Documents\folk\bench\fixtures\out\112k.jsonl", encoding="utf-8").readline())
        msgs = [{"role": "user", "content": f"[run {uuid.uuid4().hex}]\n" + fixture["messages"][0]["content"][:260_000]
                 + "\n\nIn 5 sentences: what does this code base do?"}]
        follow = ["Which module is the most complex, and why?", "List three possible bugs you would look for first.",
                  "Résume ta réponse précédente en français en 3 points.", "Write a short unit test for one of them."]
        for turn in range(5):
            r = srv.ask(msgs, max_tokens=256)
            print(f"turn {turn}: prompt {r.get('prompt_tokens')} prefill {r.get('prefill_tps')} decode {r.get('decode_tps')} "
                  f"gen {r.get('gen')} | {r['text'][:90]!r}", flush=True)
            for n in r["notes"]:
                if re.search(r"ERR|error|residency|K/V grown", n, re.I):
                    print("     ", n)
            if turn < 4:
                msgs += [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": follow[turn]}]
    finally:
        srv.stop()
elif cmd == "ctx":
    v, sizes, out = sys.argv[2], [int(x) for x in sys.argv[3].split(",")], Path(sys.argv[4])
    # the 262k fixture is ~235k tokens with this tokenizer: the 240k fixture's text follows it for the largest sizes
    text = "".join(json.loads(open(f, encoding="utf-8").readline())["messages"][0]["content"]
                   for f in (FIXTURE_262K, FIXTURE_262K.replace("262k", "240k")))
    srv = Server(f"ctx_{v}_{sizes[0]}", v)
    try:
        for k in sizes:
            prompt = (f"[run {uuid.uuid4().hex}]\n" + text[: int(k * 1024 * 3.53)]
                      + "\n\nIn 5 sentences: what does this code base do?")
            # one prompt read, then three answers with fixed seeds (the same for every variant) on the cached prefix
            reps = []
            for rep, seed in enumerate((1001, 1002, 1003)):
                t0 = time.time()
                r = srv.ask([{"role": "user", "content": prompt}], max_tokens=256, seed=seed)
                r.update(variant=v, size_k=k, rep=rep, wall_s=round(time.time() - t0, 1))
                r.pop("text")
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                reps.append(r)
            r = reps[0]
            pt, pf = r.get("prompt_tokens", 0), r.get("prefill_tps") or 1
            dec = [x["decode_tps"] for x in reps if x.get("decode_tps")]
            print(f"{v} {k}k: prompt {pt} tok, prefill {pf} tok/s ({pt / pf:.1f} s), decode "
                  f"{' / '.join(f'{d:.1f}' for d in dec)} tok/s (mean {sum(dec) / max(1, len(dec)):.1f})", flush=True)
    finally:
        srv.stop()
elif cmd == "report":
    rows = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
    ref = sys.argv[3]
    d = {}
    for r in rows:
        if "decode_tps" in r:
            d.setdefault(r["variant"], {}).setdefault(r["prompt_i"], []).append(r["decode_tps"])
    base = {i: st.mean(v) for i, v in d.get(ref, {}).items()}
    for v, per in sorted(d.items()):
        means = {i: st.mean(x) for i, x in per.items()}
        ratios = [means[i] / base[i] for i in means if i in base]
        allv = sorted(x for xs in per.values() for x in xs)
        print(f"{v:14s} median {allv[len(allv) // 2]:6.1f} tok/s (n={len(allv)}), vs {ref}: ratio median "
              f"{st.median(ratios):.3f}, prompts faster {sum(x > 1 for x in ratios)}/{len(ratios)}")
(STRATA / "strata-ab.json").unlink(missing_ok=True)
