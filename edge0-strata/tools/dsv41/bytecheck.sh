#!/bin/bash
# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
# tools/ds4/bytecheck.sh - maya-ds4's byte-identity check of two builds (a change that must not move a bit: the same
# prompt through each, every logit row dumped, the files compared byte for byte, the greedy tokens compared).
#
#   tools/ds4/bytecheck.sh <base maya-ds4> <new maya-ds4> <pack> <out dir> <ids file> [<check> ...]
#     checks: windows (DS4_CHUNK_MIN=100000: windows of 16), chunked (the prompt as one chunked run), drafter
#             (--dspark on); default: windows chunked.  64 greedy tokens after the prompt.
#   env: NEW_RESERVE_MIB (default 600, the engine's --reserve-mib default; the packed KV caches: 624, 626 with the
#        drafter - see "placement" below); GPU (default 0); CTX (default 8192)
#
# Placement.  --no-expert-cache --expert-profile none fix WHICH experts VRAM holds (by index) and keep them there, but
# not HOW MANY: that follows the free VRAM at load.  A change that frees or uses VRAM (smaller caches, a new buffer)
# moves an expert or more between VRAM and the CPU lane / the prompt path's streamed sets, and the prompt path sums a
# token's experts batch by batch - another order, other bits (the windows path only differs when a moved expert is
# picked).  So whenever a change frees or uses VRAM, give the build with more free VRAM that much more reserve:
# NEW_RESERVE_MIB = 600 + what it frees, measured, not added up from the allocations (the driver rounds them): load
# both once (the check's own first pair will do), and with E experts in VRAM each and F MiB free after the load,
# freed = (F_new - F_base) + (E_new - E_base) * the MiB of an expert (the load line: experts GB / E).  Then check that
# the load lines' "(N of M in VRAM)" agree - this script prints them.  The packed KV caches on a 32 GB V100 at 8K:
# 24 MiB (626 with the drafter's 26; the allocations' sum said 18-19, and 618 / 619 left one expert more).
#
# A prompt that a checkpoint splits (600 tokens, then 400 more from the checkpoint) is not the same bits as the 1000 in
# one chunked run: chunk boundaries change the arithmetic.  Compare a split run with the same split in the other build.
set -u
base=$1
new=$2
pack=$3
out=$4
ids=$(cat "$5")
shift 5
checks=${*:-windows chunked}
mkdir -p "$out"
common="--pack $pack --max-context ${CTX:-8192} --vision off --verbose --no-expert-cache --expert-profile none --n 64"
run() {   # <build> <check> <exe> [env ...] -- <args ...>
    local b=$1 c=$2 exe=$3
    shift 3
    local envs=()
    while [ "$1" != "--" ]; do
        envs+=("$1")
        shift
    done
    shift
    local reserve=600
    [ "$b" = new ] && reserve=${NEW_RESERVE_MIB:-600}
    env CUDA_VISIBLE_DEVICES=${GPU:-0} ${envs[@]+"${envs[@]}"} "$exe" $common --reserve-mib $reserve "$@" --ids "$ids" \
        --dump-logits "$out/${b}_$c.f32" > "$out/${b}_$c.out" 2> "$out/${b}_$c.err"
    echo "$b $c rc=$?"
}
for c in $checks; do
    for b in base new; do
        exe=$base
        [ $b = new ] && exe=$new
        case $c in
        windows) run $b $c "$exe" DS4_CHUNK_MIN=100000 -- --dspark none ;;
        chunked) run $b $c "$exe" -- --dspark none --window 1048576 ;;
        drafter) run $b $c "$exe" -- --dspark on ;;
        *) echo "unknown check $c" && exit 2 ;;
        esac
    done
    echo "== $c"
    for b in base new; do
        echo "  $b: $(grep -h 'in VRAM' "$out/${b}_$c.err" | head -1 | sed 's/.*experts/experts/')"
    done
    echo "  logits: $(cmp -s "$out/base_$c.f32" "$out/new_$c.f32" && echo IDENTICAL || echo DIFFER);" \
        "tokens: $(cmp -s <(grep -h '^[0-9]' "$out/base_$c.out") <(grep -h '^[0-9]' "$out/new_$c.out") && echo IDENTICAL || echo DIFFER)"
done
