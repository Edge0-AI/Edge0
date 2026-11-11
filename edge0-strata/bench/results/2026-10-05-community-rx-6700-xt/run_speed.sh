#!/usr/bin/env bash
# The speed matrix, with its own telemetry. (The first attempt in run_all.sh failed at start: the host venv lacked
# the `regex` module; the needle test of that run completed and is kept as needles.json.)
set -u
ROOT=/home/gpf/code/Strata
OUT=$ROOT/bench/results/2026-10-05-community-rx-6700-xt
PACK=/media/gpf/DDrive1/dev/models/packs/iq2_xs
cd "$ROOT"
python3 "$OUT/monitor_amd.py" "$OUT/telemetry.jsonl" & MON=$!
date -u +%FT%TZ > "$OUT/started.txt"
.venv/bin/python "$OUT/benchmark.py" --root "$ROOT" --pack "$PACK" --url http://127.0.0.1:8080 --out "$OUT/results" > "$OUT/benchmark.log" 2>&1
echo "benchmark exit $?" >> "$OUT/benchmark.log"
kill $MON 2>/dev/null
date -u +%FT%TZ > "$OUT/finished.txt"
