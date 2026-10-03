#!/usr/bin/env bash
# push_models.sh - stage the three GGUF artifacts onto the device files/models/.
# Usage: bash tools/model/push_models.sh --8b | --35b | --all
#   Default sources are this repo's models/edge0-{8b,35b}-gguf/ (converter output).
#   SRC_DIR=<dir> overrides; ADB_SERIAL selects the device (default: first online).
# Triple-hop discipline: host md5 -> adb push -> run-as copy -> on-device md5.
# Any 8-hex mismatch aborts. The 21.7GB artifact takes 2-6 min; reruns are idempotent.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# Self-contained layout: sources and converted artifacts live under models/ (gitignored).
# The converter (../windows/tools/convert_mlx_to_gguf.py --dir models/edge0-{tier})
# writes models/edge0-{tier}-gguf/; names pass through unchanged.
# SRC_DIR=<dir> overrides with a single flat directory.
src_for() {  # $1 = artifact name -> host source path (convention dirs unless SRC_DIR set)
  local dir
  case "$1" in edge0-8b.gguf|lora_edge0_8b-gguf.gguf) dir=${SRC:-$REPO/models/edge0-8b-gguf} ;;
                 *)                                     dir=${SRC:-$REPO/models/edge0-35b-gguf} ;;
  esac
  echo "$dir/$1"
}
DEV=${ADB_SERIAL:-}
PKG=dev.edge0.runtime.app
T=/data/local/tmp/lgguf

# artifact -> official md5 prefix baseline (converter convention names, 2026-09)
md5_of() { case "$1" in
  edge0-8b.gguf)         echo b5f6021f ;;
  lora_edge0_8b-gguf.gguf) echo e73c92e3 ;;
  edge0-35b.gguf)        echo 7d2c2e8c ;;
  *) echo "?" ;; esac; }

FILES_8B=(edge0-8b.gguf lora_edge0_8b-gguf.gguf)
FILES_35B=(edge0-35b.gguf)
case "${1:-}" in
  --8b)  FILES=("${FILES_8B[@]}") ;;
  --35b) FILES=("${FILES_35B[@]}") ;;
  --all) FILES=("${FILES_8B[@]}" "${FILES_35B[@]}") ;;
  *) echo "usage: push_models.sh --8b|--35b|--all" >&2; exit 2 ;;
esac

ADB=(adb ${DEV:+-s "$DEV"})
"${ADB[@]}" get-state >/dev/null || { echo "ABORT: no device ($DEV)" >&2; exit 1; }
"${ADB[@]}" shell "mkdir -p $T"
for f in "${FILES[@]}"; do
  want=$(md5_of "$f"); src="$(src_for "$f")"
  [ -f "$src" ] || { echo "ABORT: missing $src (download + run converter, or place artifact)"; exit 1; }
  got=$(md5sum "$src" | cut -c1-8)
  [ "$got" = "$want" ] || { echo "ABORT: $f host md5=$got want=$want"; exit 1; }
  echo "push $f ($(du -h "$src" | cut -f1)) ..."
  MSYS_NO_PATHCONV=1 "${ADB[@]}" push "$(cygpath -w "$src" 2>/dev/null || echo "$src")" "$T/$f" >/dev/null
  # copy into the app-private dir via run-as (non-root path) + on-device verification
  "${ADB[@]}" shell "run-as $PKG mkdir -p files/models; run-as $PKG cp $T/$f files/models/$f"
  dev=$(MSYS_NO_PATHCONV=1 "${ADB[@]}" shell "run-as $PKG md5sum files/models/$f" | tr -d '\r' | cut -c1-8)
  [ "$dev" = "$want" ] || { echo "ABORT: $f device md5=$dev want=$want (truncated transfer? re-run)"; exit 1; }
  echo "OK $f md5=$dev"
done
echo "staged -> $PKG/files/models"
