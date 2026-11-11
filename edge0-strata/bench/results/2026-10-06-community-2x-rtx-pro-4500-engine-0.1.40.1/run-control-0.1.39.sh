#!/bin/bash
# Strata 0.1.40.1: Solo-Benchmark (1k-256k), Parallel-Last (Warteschlange gegen Batch 2 / Batch 4), Tools-Test.
# Dienst wird gestoppt, trap stellt 0.1.40.1 solo (neue Produktion) wieder her; faellt das durch, 0.1.39.
MESS=/home/qni/messungen/2026-10-06-strata-0.1.40.1-dual
cd $MESS
PY=/home/qni/ai/Strata-0.1.36/.venv/bin/python
N=/home/qni/ai/Strata-0.1.39; R=/home/qni/ai/Strata-0.1.40.1
exec >> $MESS/lauf-alles.log 2>&1
stoppen() {
  pkill -TERM -f "[s]erve/server.py --engine strata"; sleep 3
  pkill -TERM -f "[e]ngine/strata --serve"; pkill -TERM -f "[e]ngine/strata-vision"; pkill -TERM -f "[e]ngine/strata "
  for i in $(seq 1 60); do
    if ! ss -ltn | grep -q ":8080 " && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; then return 0; fi
    sleep 2
  done
  pkill -KILL -f "[e]ngine/strata --serve"; pkill -KILL -f "[e]ngine/strata "; pkill -KILL -f "[s]erve/server.py --engine strata"; sleep 5
}
restore() {
  echo "== RESTORE $(date +%T)"
  stoppen
  $R/umschalten.sh solo
  echo "== RESTORE FERTIG $(date +%T)"
}
trap restore EXIT
mkcfg() {  # name vision(0|1) extra-args...   -> config-NAME.json, Penalty 0.0 in ALLEN Armen
  name=$1; vis=$2; shift 2
  $PY - "$name" "$vis" "$@" <<'PYEOF'
import json, sys
name, vis, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
c = json.load(open("/home/qni/ai/Strata-0.1.39/strata-swift-iq3_xxs.json"))
base = list(c["args"])
if vis != "1":
    base = [a for a in base if a != "--vision"]; c.pop("vision", None)
if "--vram-reserve-mib" in extra:
    i = base.index("--vram-reserve-mib"); del base[i:i + 2]
c["args"] = base + extra
c["sampling"]["presence_penalty"] = 0.0
c["log"] = "/home/qni/messungen/2026-10-06-strata-0.1.40.1-dual/server-%s.log" % name
json.dump(c, open("/home/qni/messungen/2026-10-06-strata-0.1.40.1-dual/config-%s.json" % name, "w"), indent=1)
PYEOF
}
start() {  # name vision extra...  -> 0 ok, 1 Fehler
  name=$1; vis=$2; shift 2
  echo "== ARM $name $(date +%T)  vision=$vis extra: $*"
  mkcfg $name $vis "$@"
  : > $MESS/server-$name.log
  setsid nohup $PY $N/serve/server.py --engine strata --config $MESS/config-$name.json --port 8080 > $MESS/serverout-$name.txt 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 5 http://127.0.0.1:8080/v1/models 2>/dev/null | grep -q loaded && break; sleep 3; done
  if ! curl -s -m 5 http://127.0.0.1:8080/v1/models 2>/dev/null | grep -q loaded; then
    echo "START FEHLER $name"; tail -6 $MESS/server-$name.log | cut -c1-250; stoppen; return 1
  fi
  grep -E "batch:|expert cache auto|VRAM free|resident" $MESS/server-$name.log | head -6 | cut -c1-200
  timeout 300 python3 $MESS/parlast.py $MESS/warm-$name.json 1 1 8000 300 > /dev/null 2>&1
  return 0
}
last() {  # name tag levels rounds ptok otok
  timeout 3000 python3 $MESS/parlast.py $MESS/runs-$1-$2.json $3 $4 $5 $6 > $MESS/bench-$1-$2.log 2>&1 || echo "BENCH FEHLER $1 $2"
  echo "-- $1 $2 (Strome $3, Prompt $5)"; cut -c1-130 $MESS/bench-$1-$2.log | tail -8
}
echo "== START serie $(date)"
stoppen
if start K39-solo 1 ; then
  python3 $MESS/bench.py $MESS/bench-K39-solo $MESS/server-K39-solo.log > $MESS/bench-K39-solo.log 2>&1; tail -9 $MESS/bench-K39-solo.log | cut -c1-200
  stoppen
fi
echo "== FERTIG serie39 $(date +%T)"
