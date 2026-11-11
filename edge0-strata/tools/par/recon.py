#!/usr/bin/env python3
"""recon.py - reconcile PAR-A's boston numbers with this harness (runs on boston, one arm after the other)."""
import json, os, subprocess, sys, time
PA = "/data/external/niko/parA"
I = "/data/external/niko/int029"
R = "/data/external/niko/par29"
W = R + "/armw"
OUT = R + "/res/recon"
PY = "/data/external/niko/t027/venv/bin/python"
os.makedirs(W, exist_ok=True)
for name, tgt in (("b5", PA + "/b5"), ("src5", PA + "/src5"), ("bint", I + "/b-int"), ("srcint", I + "/int")):
    if not os.path.lexists(W + "/" + name):
        os.symlink(tgt, W + "/" + name)
REPL = '[{"gpus":[0]},{"gpus":[1]}]'
PA_ENV = ["STRATA_QSA_SLOTS=1", "STRATA_BATCH_SELF_COMMIT=1", "STRATA_SLOT_COPY_DIRECT=1", "STRATA_CKPT_POOL=1", "STRATA_CKPT_POOL_READY=4",
          "STRATA_SLOT_CKPT_MOVE=1", "STRATA_SLOT_COPY_BATCHSYNC=1", "STRATA_REPLICA_SHARED_ARENA=1"]


def armrun(tag, build, src, loadcmd=None):
    out = f"{OUT}/{tag}"
    cmd = [PY, PA + "/armrun.py", "--out", out, "--gpus", "0,1", "--replicas", REPL, "--parallel", "8", "--build", build, "--src", src,
           "--conc", "8,16", "--rounds", "2", "--max-tokens", "128", "--port", "8191"]
    for e in PA_ENV:
        cmd += ["--env", e]
    if loadcmd:
        cmd += ["--loadcmd", loadcmd]
    env = dict(os.environ, PARA_W=W)
    t = time.time()
    subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(f"== {tag} ({time.time() - t:.0f}s)", flush=True)
    for f in ("load.txt",):
        p = f"{out}/{f}"
        if os.path.exists(p):
            print("".join(l for l in open(p).read().splitlines(True) if l.strip()[:2] in ("8 ", "16") or "tok/s" in l and l.startswith("c=")), flush=True)
            if loadcmd:
                print(open(p).read()[-700:])


def mine(tag, env, post=None, exe=None, repo=None, rounds=2):
    out = f"{OUT}/{tag}"
    cmd = [PY, R + "/h/tools/par_run.py", "--box", "boston", "--tag", tag, "--exe", exe or I + "/b-int/strata", "--repo", repo or I + "/int",
           "--gpus", "0,1", "--parallel", "8", "--server-extra", ("--replicas 2" if "A9" not in tag else ""), "--env", ";".join(env), "--rounds", "2",
           "--work", f"w2:8,16@--nonce rc --max-tokens 128 --rounds {rounds} --warm-requests 2", "--out", out]
    if post:
        cmd += ["--post-cmd", post]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"== {tag} ({time.time() - t:.0f}s)\n" + "\n".join(l for l in r.stdout.splitlines() if l.startswith(("w2", "  ", "c=", "DONE", "ABORT", " 8", " 16"))), flush=True)


THEIR_LOAD = f"{PY} {PA}/par_load.py {{url}} --conc 8,16 --rounds 2 --prompt-tokens 1000 --max-tokens 128 --out /dev/null"
MY_STACK = PA_ENV + ["STRATA_BATCH_GDN_GROUPS=1", "STRATA_BATCH_ATTN_ROWS=1"]
if "A7" in sys.argv: mine("A7_myarm_burst_rounds1", PA_ENV)
if "A8" in sys.argv: mine("A8_myarm_sustained_rounds6", PA_ENV, rounds=6)
if "A9" in sys.argv: mine("A9_layersplit_b1like_rounds1", ["STRATA_QSA_SLOTS=1"] , rounds=1, exe=I + "/b-base/strata", repo=I + "/base")
which = [x for x in sys.argv[1:] if x not in ("A7", "A8", "A9")] if len(sys.argv) > 1 else ["A1", "A2", "A3", "A4", "A5", "A6"]
if "A1" in which: armrun("A1_pa_exact_theirload", "b5", "src5")
if "A2" in which: armrun("A2_pa_exact_myload", "b5", "src5", R + "/h/tools/par/my_load.py")
if "A3" in which: armrun("A3_pa_cfg_intbuild_theirload", "bint", "srcint")
if "A4" in which: armrun("A4_pa_cfg_intbuild_myload", "bint", "srcint", R + "/h/tools/par/my_load.py")
if "A5" in which: mine("A5_myarm_theirload", MY_STACK, THEIR_LOAD)
if "A6" in which: mine("A6_myarm_paenv_theirload", PA_ENV, THEIR_LOAD)
