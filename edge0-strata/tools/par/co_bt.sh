#!/bin/bash
# co_bt.sh ROOT OUT [N] - batch_test arms, dumps compared: A/A2 layer-major alone (noise), B layer-major co-read, C the usual chunk-major read
R=$1; O=$2; N=${3:-4}; mkdir -p $O
PY=/data/external/niko/t027/venv/bin/python
export STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_CRASH_BT=$O/crash.txt
CFG=$R/res/co/cog_on-cfg.json
for arm in A A2 B C; do
  EXTRA=""; [ $arm = B ] && EXTRA="--co"
  if [ $arm = C ]; then unset STRATA_PREFILL_LM_FORCE; else export STRATA_PREFILL_LM_FORCE=1; fi
  STRATA_DUMP_FIRST_LOGITS=$O/lg-$arm BATCH_TEST_LOG=$O/engine-$arm.log $PY $R/h/tools/batch_test.py --exe $R/h/b86/strata --config $CFG --batch $N --n $N --max-new 96 --pad 900 --skip-solo --dump $O/dump-$arm.json $EXTRA > $O/bt-$arm.out 2>&1
  echo "== $arm rc=$?"; grep -E "batch: [0-9]+ slots|co-read:" $O/bt-$arm.out $O/engine-$arm.log | cut -c1-240 | head -4
done
$PY - <<PY
import json
d={a:json.load(open("$O/dump-%s.json"%a))["batch"] for a in ("A","A2","B","C")}
def cmp(x,y):
    r=[]
    for p,q in zip(d[x],d[y]):
        r.append("same" if p==q else "DIFF@%d"%next(k for k in range(min(len(p),len(q))) if p[k]!=q[k]))
    return r
for x,y in (("A","A2"),("A","B"),("A","C"),("B","C")):
    print(x,"vs",y,cmp(x,y))
PY

$PY $R/h/tools/par/kl_logits.py $O lg C A A2 B 2>&1 | head -8
$PY $R/h/tools/par/kl_logits.py $O lg A A2 B 2>&1 | head -8
