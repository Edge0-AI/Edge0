#!/usr/bin/env bash
# Everything that runs after the measurements: a fresh hardware snapshot (the
# first one was taken before the runs), the curated engine.log, the number
# tables, and the private-path rewrite. Run it last, before the commit.
set -u
REPORT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ROOT="$(cd -- "$REPORT/../../.." && pwd)"
DATA=$REPORT/data
S=$REPORT/scripts
PY=$ROOT/.venv/bin/python

step() { echo; echo "[$(date -Is)] === $* ==="; echo; }

step "hardware snapshot"
bash "$S/collect_hardware.sh" "$DATA/hardware.txt"

step "engine.log"
"$PY" "$S/make_engine_log.py"

step "summary tables"
"$PY" "$S/summarize.py" > "$DATA/summary.md"
tail -n +1 "$DATA/summary.md"

step "private paths"
"$PY" "$S/sanitize.py"
