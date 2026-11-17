#!/bin/bash
# overhead.sh ROOT OUT [REPS] - counters overhead: arms M (main binary), H0 (branch binary, counters off), H1 (branch, STRATA_PAR_COUNTERS=1), interleaved
R=$1; O=$2; REPS=${3:-3}; mkdir -p $O
PY=python3
for rep in $(seq 1 $REPS); do
 for arm in M H0 H1; do
  case $arm in
   M)  EXE=$R/main/b86/strata; REPO=$R/main; ENV="";;
   H0) EXE=$R/h/b86/strata;    REPO=$R/h;    ENV="";;
   H1) EXE=$R/h/b86/strata;    REPO=$R/h;    ENV="STRATA_PAR_COUNTERS=1";;
  esac
  $PY $R/h/tools/par_run.py --box boston --tag ov${rep}_$arm --exe $EXE --repo $REPO --gpus 0,1 --parallel 8 --env "$ENV" --work "w1;w2:4@--max-tokens 128 --rounds 3;w2:8@--max-tokens 128 --rounds 3" --rounds 3 --out $O > $O/ov${rep}_$arm.out 2>&1
  echo "rep $rep $arm: $(grep -E '^(w1|w2)' $O/ov${rep}_$arm.out | awk '{print $2, $3, $4, $5, $6, $7, $8, $9}' | tr '\n' '|')"
 done
done
