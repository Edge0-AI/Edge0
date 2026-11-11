#!/usr/bin/env bash
# The measurement run for this report: telemetry in the background, then the speed matrix, then the needle test.
set -u
ROOT=/home/gpf/code/Strata
OUT=$ROOT/bench/results/2026-10-05-community-rx-6700-xt
PACK=/media/gpf/DDrive1/dev/models/packs/iq2_xs
URL=http://127.0.0.1:8080
cd "$ROOT"
python3 "$OUT/monitor_amd.py" "$OUT/telemetry.jsonl" & MON=$!
date -u +%FT%TZ > "$OUT/started.txt"
.venv/bin/python "$OUT/benchmark.py" --root "$ROOT" --pack "$PACK" --url "$URL" --out "$OUT/results" > "$OUT/benchmark.log" 2>&1
echo "benchmark exit $?" >> "$OUT/benchmark.log"
.venv/bin/python tools/needle_bench.py --url "$URL" --lengths 32k,128k --depths 10,50,90 --out "$OUT/needles.json" > "$OUT/needles.log" 2>&1
echo "needle exit $?" >> "$OUT/needles.log"
kill $MON 2>/dev/null
date -u +%FT%TZ > "$OUT/finished.txt"
