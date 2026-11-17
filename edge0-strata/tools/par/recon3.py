#!/usr/bin/env python3
"""recon3.py BOX ARMS... - sustained + burst for the honest headline.  Runs PAR-A's armrun.py with this harness's both.py as the loader.
BOX boston: W=armw (symlinks to the int029 / base builds); BOX r9700: W=/data/par-a (b4 / src4)."""
import os, subprocess, sys, time
box = sys.argv[1]
arms = sys.argv[2:]
HOST = ["STRATA_QSA_SLOTS=1", "STRATA_BATCH_SELF_COMMIT=1", "STRATA_SLOT_COPY_DIRECT=1", "STRATA_CKPT_POOL=1", "STRATA_CKPT_POOL_READY=4",
        "STRATA_SLOT_CKPT_MOVE=1", "STRATA_SLOT_COPY_BATCHSYNC=1"]
CUDA = ["STRATA_BATCH_GDN_GROUPS=1", "STRATA_BATCH_ATTN_ROWS=1"]
if box == "boston":
    R = "/data/external/niko/par29"; PA = "/data/external/niko/parA"; I = "/data/external/niko/int029"; W = R + "/armw"
    PY = "/data/external/niko/t027/venv/bin/python"
    for name, tgt in (("bbase", I + "/b-base"), ("srcbase", I + "/base"), ("bint", I + "/b-int"), ("srcint", I + "/int")):
        if not os.path.lexists(W + "/" + name):
            os.symlink(tgt, W + "/" + name)
    G2 = ["--gpus", "0,1"]
    REP = '[{"gpus":[0]},{"gpus":[1]}]'
    ARMS = {
        "B1": (["--parallel", "8"], [], "bbase", "srcbase"),
        "LS": (["--parallel", "8"], HOST + CUDA, "bint", "srcint"),
        "REP": (["--parallel", "8", "--replicas", REP], HOST + CUDA + ["STRATA_REPLICA_SHARED_ARENA=1", "STRATA_COALESCE_PREFILL=3", "STRATA_COALESCE_WAIT_MS=40"], "bint", "srcint"),
        "REPCR": (["--parallel", "8", "--replicas", REP, "--arg=--no-prefill-borrow"], HOST + CUDA + ["STRATA_REPLICA_SHARED_ARENA=1", "STRATA_COALESCE_PREFILL=3", "STRATA_COALESCE_WAIT_MS=40", "STRATA_BATCH_CONCURRENT_READ=1"], "bint", "srcint"),
        "REPCR_NOCO": (["--parallel", "8", "--replicas", REP, "--arg=--no-prefill-borrow"], HOST + CUDA + ["STRATA_REPLICA_SHARED_ARENA=1", "STRATA_BATCH_CONCURRENT_READ=1"], "bint", "srcint"),
        "REP_NOCO": (["--parallel", "8", "--replicas", REP], HOST + CUDA + ["STRATA_REPLICA_SHARED_ARENA=1"], "bint", "srcint"),
    }
    GP = G2; CONC = "8,16,32"
else:
    R = "/data/par29h"; PA = "/data/par-a"; W = R + "/armw"; PY = "/data/p4/venv/bin/python"
    os.makedirs(W, exist_ok=True)
    for name, tgt in (("b4", PA + "/b4"), ("src4", PA + "/src4"), ("bint", R + "/int/b-int"), ("srcint", R + "/int/src")):
        if not os.path.lexists(W + "/" + name):
            os.symlink(tgt, W + "/" + name)
    REP2 = '[{"gpus":[0]},{"gpus":[1]}]'
    REP4 = '[{"gpus":[0]},{"gpus":[1]},{"gpus":[2]},{"gpus":[3]}]'
    CO = ["STRATA_REPLICA_SHARED_ARENA=1", "STRATA_LM_HIP=1", "STRATA_COALESCE_PREFILL=3", "STRATA_COALESCE_WAIT_MS=40"]
    ARMS = {
        "B1p8_2c": (["--gpus", "0,1", "--parallel", "8"], [], "b4", "src4"),
        "B1p16_2c": (["--gpus", "0,1", "--parallel", "16"], [], "b4", "src4"),
        "REP2": (["--gpus", "0,1", "--parallel", "8", "--replicas", REP2], HOST + CO, "b4", "src4"),
        "B1_4c": (["--gpus", "0,1,2,3", "--profile4"], [], "b4", "src4"),
        "REP2CR": (["--gpus", "0,1", "--parallel", "8", "--replicas", REP2, "--arg=--no-prefill-borrow"], HOST + CO + ["STRATA_BATCH_CONCURRENT_READ=1"], "bint", "srcint"),
        "REP4CR": (["--gpus", "0,1,2,3", "--parallel", "4", "--replicas", REP4, "--arg=--no-prefill-borrow"], HOST + CO + ["STRATA_BATCH_CONCURRENT_READ=1"], "bint", "srcint"),
        "REP2I": (["--gpus", "0,1", "--parallel", "8", "--replicas", REP2], HOST + CO, "bint", "srcint"),
        "REP4": (["--gpus", "0,1,2,3", "--parallel", "4", "--replicas", REP4], HOST + CO, "b4", "src4"),
    }
    GP = None; CONC = "8,16,32"
OUT = R + "/res/recon3"
for tag in arms:
    flags, env, build, src = ARMS[tag][0], ARMS[tag][1], ARMS[tag][2], ARMS[tag][3]
    out = f"{OUT}/{box}_{tag}"
    cmd = [PY, PA + "/armrun.py", "--out", out, "--build", build, "--src", src, "--conc", CONC, "--rounds", "2", "--max-tokens", "128",
           "--port", "8197", "--loadcmd", R + "/h/tools/par/both.py"] + (GP or []) + flags
    for e in env:
        cmd += ["--env", e]
    e2 = dict(os.environ, PARA_W=W, PA_LOAD=PA + "/par_load.py", BOTH_CONC=CONC)
    t = time.time()
    subprocess.run(cmd, env=e2, capture_output=True, text=True)
    print(f"=== {box} {tag} ({time.time() - t:.0f}s)", flush=True)
    p = out + "/load.txt"
    if os.path.exists(p):
        for l in open(p).read().splitlines():
            if l.startswith(("###", "w2 c=")) or l[:3].strip().isdigit() and "tok/s" not in l and len(l.split()) > 5:
                print(l[:170], flush=True)
