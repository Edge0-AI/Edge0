#!/usr/bin/env bash
# One arm of the A/B harness (scripts/ab-run.sh, the report's own copy of the
# harness), with everything it produced archived into the report's data/
# folder: the engine log slice (startup lines plus the requests), both arms'
# JSON, both server outputs and the printed summary.
#
# Usage: run-arm.sh LABEL [ab-run.sh extras ...]
# e.g.   run-arm.sh pipeline-off --arg --pipeline-windows 1
set -u
REPORT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ROOT="$(cd -- "$REPORT/../../.." && pwd)"
DATA=$REPORT/data
WORK=$DATA/ab-work
LOG=$ROOT/strata-coder-iq1_m.log
LABEL=$1; shift

mkdir -p "$DATA" "$WORK"
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
echo "[run-arm $LABEL] engine log offset $off, $(date -Is)"

# run-all.sh writes its extras the way build_cfg.py spells them ("--arg FLAG
# VALUE"), but ab-run.sh takes bare (FLAG, VALUE) pairs and crashes on a stray
# "--arg", so the markers are dropped here.
args=()
for a in "$@"; do
  [ "$a" = "--arg" ] && continue
  args+=("$a")
done

AB_OUT="$WORK" bash "$REPORT/scripts/ab-run.sh" "$LABEL" "${args[@]}" \
  > "$WORK/run.out" 2>&1
rc=$?
cp "$WORK/run.out" "$DATA/ab-$LABEL-run.out"
tail -c +$((off + 1)) "$LOG" > "$DATA/engine-ab-$LABEL.log"
for f in A B; do
  [ -f "$WORK/$f.json" ] && cp "$WORK/$f.json" "$DATA/ab-$LABEL-$f.json"
  [ -f "$WORK/server-$f.out" ] && cp "$WORK/server-$f.out" "$DATA/ab-$LABEL-server-$f.out"
done
cfgB=$WORK/cfg-$LABEL-B.json
[ -f "$cfgB" ] && cp "$cfgB" "$DATA/cfg-$LABEL-B.json"
rm -rf "$WORK"
echo "[run-arm $LABEL] ab-run.sh exit $rc, $(date -Is)"
exit $rc
