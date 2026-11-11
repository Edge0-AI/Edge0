#!/bin/sh
# Strata 0.1.40.3 - Qwen3.8-Flash-Next IQ3_S on 2x TITAN RTX 24GB (sm_75)
#
#   context 262144 (the model native window)
#   --kv-resident 65536   hot KV cells per QSA layer in VRAM, the rest in pinned RAM
#   --spec 4              MTP depth
#   --ple-io ram          n-gram table in RAM
#   --vision --vram-reserve-mib 700   image encoder resident; costs VRAM the expert cache would take
#   reasoning_budget_tokens 12000     leaves room for an answer inside the reasoning cap
#   api_key <your-key>     required, the port is open to the LAN
#
# STRATA_STAGE_TRIM=1 (PR #639): each card loads only its own layers dense weights
# instead of a full copy, returning ~3.4 GB per card to the expert cache.
# Restarting takes ~1 min (about 55 GB of experts are read into RAM).
export STRATA_STAGE_TRIM=1
cd "/mnt/1TB/src/Strata" || exit 1
exec "/mnt/1TB/src/Strata/.venv/bin/python" "/mnt/1TB/src/Strata/serve/server.py" \
  --engine strata --config "/mnt/1TB/src/Strata/strata-262k-pleio.json" --port 8080
