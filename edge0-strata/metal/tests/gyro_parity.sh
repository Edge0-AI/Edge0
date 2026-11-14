#!/bin/bash
# metal/tests/gyro_parity.sh - the Metal kernels of agentionai's Gyro experts (TQ2_T, TQK6, TQK7 and the Hadamard rotation)
# against the CPU references, on the real tensors of a Gyro GGUF.
#
#   metal/tests/gyro_parity.sh BUILD_DIR GGUF PACK_DIR [LAYERS...] [--experts-first N]
#
# BUILD_DIR   the metal build (cmake -S metal -B build-metal; ninja expert_parity moe_mm_parity)
# GGUF        a Gyro file; a sparse copy works (header + the experts of the layers named, see docs/GYRO.md "Testing a file that
#             does not fit": the first 16 experts of each layer with --experts-first 16)
# PACK_DIR    the pack tools/iq_pack.py --compat-bf16 made from it (hadamard.txt carries the rotation's signs)
#
# 1. had-test      the Metal rotation == the CPU's hadamard_rotate, bit for bit (f32, f32 in place, f16, f16 rows)
# 2. decode-check  ref_quant's trellis decoder == the format's definition written out bit by bit (sampled blocks of every tensor)
# 3. parity        moe_plan / moe_gu / moe_down / moe_combine with the rotation of x and h, 1, 3 and 8 tokens, against double
# 4. prompt        mm_gu2 / mm_down2 (the prompt chunk's grouped GEMMs, fp16 operands) for the file's types
set -u
B=$1; G=$2; P=$3; shift 3
LAYERS=()
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --experts-first) EXTRA+=(--experts-first "$2"); shift 2 ;;
    *) LAYERS+=("$1"); shift ;;
  esac
done
[ ${#LAYERS[@]} -eq 0 ] && LAYERS=(0 47)
rc=0
run() { echo "+ $*"; "$@" || rc=1; }
run "$B/expert_parity" --had-test --had-pack "$P"
for L in "${LAYERS[@]}"; do
  run "$B/expert_parity" --gguf "$G" --layer "$L" --decode-check
  for nt in 1 3 8; do
    run "$B/expert_parity" --gguf "$G" --layer "$L" --had-pack "$P" --ntok "$nt" --seed "$nt" ${EXTRA[@]+"${EXTRA[@]}"}
  done
done
run "$B/moe_mm_parity" --v2 --gu TQK6 --dn TQK7 --T 300 --tile 16,64,16,2
run "$B/moe_mm_parity" --v2 --gu TQ2_T --dn TQ2_T --T 300 --tile 16,64,16,2
[ $rc -eq 0 ] && echo "gyro_parity: all PASS" || echo "gyro_parity: FAILED"
exit $rc
