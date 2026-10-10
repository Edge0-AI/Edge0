#!/usr/bin/env bash
# build_vendor_libs.sh - rebuild the four shared libraries from the pinned supply tree.
# Supply topology (patches/llama.cpp/README in the depot repo):
#   vendor@pin (vendor.llama.pin at the depot root; materialized by clone on first run -
#   no submodule) -> wt/and worktree with common(6)+android(14) bands replayed in order
#   -> golden tree hash. Set EDGE0_LLAMA_URL to clone from a mirror instead of GitHub.
# Usage: bash tools/llama/build_vendor_libs.sh [--replay]  (--replay = re-apply bands + verify)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# `pwd -W` is the Git-Bash (MSYS) form only; on Linux/macOS it has no -W and would abort
# the script at set -e, so fall back to `pwd -P`. EDGE0_DEPOT still wins if set.
DEPOT=${EDGE0_DEPOT:-$(cd "$REPO/.." && { pwd -W 2>/dev/null || pwd -P; })}   # depot root (this repo lives at <depot>/android)
WT=$DEPOT/wt/and
[ -f "$DEPOT/vendor.llama.pin" ] || { echo "ABORT: vendor.llama.pin missing at depot root ($DEPOT)"; exit 1; }
PIN=$(cut -d' ' -f1 "$DEPOT/vendor.llama.pin")
VENDOR=$DEPOT/vendor/llama.cpp
GOLD_TREE=e974be50ba5c184bf0ba9a4a26adbbae4235c2ab
NDK=${NDK_DIR:-${ANDROID_NDK_HOME:-}}
[ -n "$NDK" ] || { echo "ABORT: set NDK_DIR (or ANDROID_NDK_HOME) to the Android NDK r28 path"; exit 1; }
LL=$REPO/build-dl/llama-libs

# materialize the pinned vendor tree when absent (submodule-free supply)
if [ ! -d "$VENDOR/.git" ]; then
    echo "[0/3] materializing vendor: clone llama.cpp at pinned $PIN -> $VENDOR"
    mkdir -p "$DEPOT/vendor"
    git clone --filter=blob:none "${EDGE0_LLAMA_URL:-https://github.com/ggml-org/llama.cpp}" "$VENDOR"
fi
[ "$(git -C "$VENDOR" rev-parse HEAD)" = "$PIN" ] || git -C "$VENDOR" checkout --detach "$PIN"
[ -z "$(git -C "$VENDOR" status --porcelain)" ] || { echo "ABORT: vendor tree dirty - the pristine supply is never patched in place; restore with: git -C $VENDOR checkout --detach $PIN"; exit 1; }

# self-heal: a fresh clone has no consumer worktree yet - create it at the pin,
# and treat the bare tree as needing the band replay (otherwise the tree gate
# would compare an unpatched pin against the golden patched tree -> false RED)
FRESH_WT=0
if [ ! -d "$WT" ]; then
    git -C "$VENDOR" worktree add --detach "$WT" "$PIN"
    FRESH_WT=1
fi

if [ "${1:-}" = "--replay" ] || [ "$FRESH_WT" = "1" ]; then
    git -C "$WT" am --abort 2>/dev/null || true
    git -C "$WT" reset --hard "$PIN"
    for band in common android; do
        for f in $(ls "$DEPOT/patches/llama.cpp/$band/"0*.patch | sort); do
            git -C "$WT" am --3way "$f"
        done
    done
fi
T=$(git -C "$WT" rev-parse 'HEAD^{tree}')
[ "$T" = "$GOLD_TREE" ] || { echo "ABORT: tree mismatch got=$T want=$GOLD_TREE"; exit 1; }
echo "tree GREEN $T"

cmake -S "$WT" -B "$WT/build-android-cpu" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_COMMON=ON -DGGML_OPENCL=OFF -DGGML_NATIVE=OFF
ninja -C "$WT/build-android-cpu" llama ggml ggml-base ggml-cpu

mkdir -p "$LL/arm64-v8a" "$LL/include"
cp "$WT"/build-android-cpu/bin/{libllama.so,libggml.so,libggml-base.so,libggml-cpu.so} "$LL/arm64-v8a/"
# libggml-cpu DT_NEEDED libomp.so (NDK OpenMP runtime) - ship it alongside,
# otherwise the app fails dlopen at launch on a fresh build.
cp "$NDK"/toolchains/llvm/prebuilt/*/lib/clang/*/lib/linux/aarch64/libomp.so "$LL/arm64-v8a/"
cp "$WT"/include/*.h "$WT"/ggml/include/*.h "$LL/include/"
echo "OK: four libs from wt/and @$(git -C "$WT" rev-parse --short HEAD)"
