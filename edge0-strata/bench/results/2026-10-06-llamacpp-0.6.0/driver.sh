#!/bin/bash
# usage: driver.sh ROUNDS "cfg1 cfg2 ..."  -- one flock job per (round, config); alternate order each round
cd /data/1402-h
R=$1; shift
CFGS=($1)
for ((r=0;r<R;r++)); do
  if (( r % 2 == 1 )); then ORDER=($(printf '%s\n' "${CFGS[@]}" | tac)); else ORDER=("${CFGS[@]}"); fi
  for c in "${ORDER[@]}"; do
    flock /data/gpu.lock timeout 1790 /data/venv/bin/python bench_vs_llama.py run --machine res/$MACH --out res --rounds 1 --start-round $((r+10)) --configs $c >> drive.log 2>&1
  done
done
echo DRIVER-DONE >> drive.log
