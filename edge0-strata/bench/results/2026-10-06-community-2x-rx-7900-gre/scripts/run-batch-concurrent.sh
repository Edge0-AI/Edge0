#!/usr/bin/env bash
# The batch arms' concurrency test: N 500-token generations at once against the
# --batch N config. scripts/ab-run.sh (which measures one request at a time) runs
# first, in run-all.sh; this is the measurement that actually fills the slots.
#
# Usage: run-batch-concurrent.sh
set -u
REPORT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ROOT="$(cd -- "$REPORT/../../.." && pwd)"
DATA=$REPORT/data
S=$REPORT/scripts
PY=$ROOT/.venv/bin/python
MODEL=qwen3.8-flash-next-coder-iq1_m
LOG=$ROOT/strata-coder-iq1_m.log

step() { echo; echo "[$(date -Is)] === $* ==="; echo; }

kill_server() {
  pkill -f "serve/server.py" 2>/dev/null
  pkill -f "engine/strata --serve" 2>/dev/null
  sleep 5
}

for n in 2 4; do
  cfg=$DATA/cfg-batch$n-B.json
  if [ ! -f "$cfg" ]; then
    step "batch $n: $cfg missing, skipped"
    continue
  fi
  kill_server
  step "batch $n: $n concurrent 500-token requests"
  "$PY" "$S/batch_concurrent.py" --config "$cfg" --model "$MODEL" \
    --engine-log "$LOG" --out "$DATA/batch-concurrent-batch$n.json" --n "$n"
done

step "batch concurrency done"
kill_server
