#!/bin/bash
# host_seq.sh ROOT OUT - 4 prompts ONE AT A TIME (--seq) in the engine: A, A2 (stack off), H (host stack only), same fixed flags, no MTP
R=$1; O=$2; mkdir -p $O
PY=/data/external/niko/t027/venv/bin/python
export STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_CRASH_BT=$O/crash.txt
CFG=$R/res/co/cog_on-cfg.json
HOST="STRATA_CKPT_POOL=1 STRATA_CKPT_POOL_READY=4 STRATA_CKPT_THREADS=1 STRATA_SLOT_CKPT_MOVE=1 STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1"
for arm in A A2 H; do
  EXTRA=""; [ $arm = H ] && EXTRA="$HOST"
  env $EXTRA BATCH_TEST_LOG=$O/engine-$arm.log $PY $R/h/tools/batch_test.py --exe $R/h/b86/strata --config $CFG --batch 4 --n 4 --max-new 64 --pad 900 --seq --skip-solo --dump $O/dump-$arm.json > $O/bt-$arm.out 2>&1
  echo "== $arm rc=$?"; grep -E "batch: [0-9]+ slots" $O/bt-$arm.out | cut -c1-200
done
$PY - <<PY
import json
d={a:json.load(open("$O/dump-%s.json"%a))["batch"] for a in ("A","A2","H")}
for x,y in (("A","A2"),("A","H")):
    print(x,"vs",y,["same" if p==q else "DIFF@%d"%next(k for k in range(min(len(p),len(q))) if p[k]!=q[k]) for p,q in zip(d[x],d[y])])
PY
