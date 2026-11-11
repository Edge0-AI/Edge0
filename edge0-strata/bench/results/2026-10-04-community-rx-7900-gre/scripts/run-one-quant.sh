#!/usr/bin/env bash
# One arm of the community quants report: 3 fresh bench runs of one quant, then one
# 32k needle pass. The engine restarts before every run, because bench_prefill.py
# refuses a fresh prompt that reused cached tokens. Only the quant differs between
# the cfg-*.json files; every other setting is identical.
#
#   ./run-one-quant.sh coder-iq1_m | iq2_xs | iq3_xxs
#
# ROOT is this machine's Strata checkout; change it, and the absolute paths inside
# ../data/cfg-*.json, before running this elsewhere.
set -u
ROOT=/home/jason/Projects/Strata
HERE=$(cd "$(dirname "$0")" && pwd)
Q=$1
OUT=${OUT:-$HERE/out}
CFG=$HERE/../data/cfg-$Q.json
LOG=$OUT/$Q.engine.log
PY=$ROOT/.venv/bin/python
. "$HOME/.local/opt/strata-env.sh"

MODEL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_name"])' "$CFG")

stop_server() {
  pkill -f "[s]erve/server.py" 2>/dev/null
  pkill -f "[e]ngine/strata --serve" 2>/dev/null
  sleep 5
}

wait_loaded() {
  local i=0
  while [ $i -lt 900 ]; do
    curl -s http://127.0.0.1:8080/health | grep -q '"loaded": true' && return 0
    i=$((i + 2))
    sleep 2
  done
  return 1
}

mkdir -p "$OUT"
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }
rm -f "$OUT/$Q-stop"
: > "$LOG"

for run in 1 2 3; do
  stop_server
  : > "$OUT/$Q-server-run$run.out"
  nohup "$PY" "$ROOT/serve/server.py" --engine strata --config "$CFG" --port 8080 \
    > "$OUT/$Q-server-run$run.out" 2>&1 &
  if ! wait_loaded; then
    echo "$Q run $run: server never reported loaded"
    tail -20 "$OUT/$Q-server-run$run.out"
    exit 1
  fi
  echo "$Q run $run: loaded, benching"
  "$PY" "$HERE/sample_mem.py" --out "$OUT/$Q-mem-run$run.csv" --seconds 600 --stop-file "$OUT/$Q-stop" &
  sampler=$!
  "$PY" "$ROOT/tools/hip/bench_prefill.py" --model "$MODEL" \
    --url http://127.0.0.1:8080 --engine-log "$LOG" --label "$Q-run$run" \
    --output "$OUT/$Q-run$run.json" | tail -1 > /dev/null || echo "$Q run $run: BENCH FAILED"
  touch "$OUT/$Q-stop"
  wait $sampler 2>/dev/null
  rm -f "$OUT/$Q-stop"
done

echo "$Q: needle pass on the run-3 server"
"$PY" "$HERE/sample_mem.py" --out "$OUT/$Q-mem-needle.csv" --seconds 900 --stop-file "$OUT/$Q-stop" &
sampler=$!
"$PY" "$ROOT/tools/needle_bench.py" --url http://127.0.0.1:8080 \
  --lengths 32k --depths 10,50,90 --out "$OUT/$Q-needles.json" || echo "$Q: NEEDLE FAILED"
touch "$OUT/$Q-stop"
wait $sampler 2>/dev/null
rm -f "$OUT/$Q-stop"

stop_server
echo "$Q: done"
