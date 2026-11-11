#!/bin/bash
# The exact run behind data/. It ran on the machine that hosts the engine (lm-server), against the production engine
# that was already running (nothing restarted, nothing reconfigured). Before it started, the engine's status showed
# in_flight 0 and the last request had finished 32 minutes before the first harness request.
# benchmark.py is the unmodified file of bench/results/2026-09-30-community-rtx-5090/; run_harness.py only adds the
# Authorization header (benchmark.py has no option for one) and then runs benchmark.py as __main__.
# KEY_FILE holds the server's API key; it is never printed or stored in this folder.
set -u
KEY_FILE=/opt/hyperqwen-swift/api_key.txt
cd /root/prbench
mkdir -p out
# one GPU sample every 2 s for the telemetry in data/gpu.csv
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv -l 2 > out/gpu.csv &
GP=$!
date -u +"harness start %F %T UTC" > run.log
/opt/strata-swift/.venv/bin/python -u run_harness.py "$KEY_FILE" /root/prbench/benchmark.py \
  --root /opt/strata-bmtp --pack /opt/strata-ssd/packs/ud-q4_k_xl --url http://127.0.0.1:8000 \
  --targets 4096,25000,51000,104000,127000 --runs 3 --out /root/prbench/out >> run.log 2>&1
echo "harness exit $?" >> run.log
date -u +"harness end %F %T UTC" >> run.log
kill $GP
