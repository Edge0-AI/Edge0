#!/bin/bash
# strata-dsv41 + the kernel parity tests for AMD (HIP) on Linux.  Default arch gfx1151 (Strix Halo, RDNA3.5); gfx1201 (R9700) works the same.
# Built and ready, NOT run on a Linux Strix box (there is none); the Windows build of the same sources is the tested one.
#
#   ROCM_ROOT=/opt/rocm   (or the TheRock wheel root, e.g. .../_rocm_sdk_devel)  tools/dsv41/build_hip_linux.sh [targets...]
#   env: STRATA_HIP_ARCHS (default gfx1151), BUILD_DIR (default build-dsv41-hip), JOBS (default nproc), STRATA_GGML_DIR (llama.cpp checkout)
# Needs: cmake, ninja, a ROCm with clang++ (ROCm >= 6.4 knows gfx1151; the wheel's device package is _rocm_sdk_libraries_gfx115X_all
# or gfx1151), and glibc matching the box.  The engine has no RPATH: run with the ROCm lib dir on LD_LIBRARY_PATH.
set -e
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
ARCH="${STRATA_HIP_ARCHS:-gfx1151}"
R="${ROCM_ROOT:-/opt/rocm}"
B="${BUILD_DIR:-$SRC/build-dsv41-hip}"
J="${JOBS:-$(nproc)}"
CLANG="$R/llvm/bin/clang++"; [ -x "$CLANG" ] || CLANG="$R/lib/llvm/bin/clang++"
GGML=""; [ -n "$STRATA_GGML_DIR" ] && GGML="-DSTRATA_GGML_DIR=$STRATA_GGML_DIR"
BC="$R/amdgcn/bitcode"; [ -d "$BC" ] || BC="$R/lib/llvm/amdgcn/bitcode"; [ -d "$BC" ] || BC="$R/llvm/amdgcn/bitcode"
export PATH="$R/bin:$PATH"
if [ ! -f "$B/build.ninja" ]; then
  cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$ARCH" -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
    -DSTRATA_NATIVE_EXPERTS=ON $GGML \
    -DCMAKE_HIP_COMPILER="$CLANG" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$R" \
    "-DCMAKE_HIP_FLAGS=--rocm-path=$R --rocm-device-lib-path=$BC" \
    "-DCMAKE_PREFIX_PATH=$R"
fi
cmake --build "$B" -j "$J" --target ${@:-strata-dsv41 ds4_iq_test} -- -k 0
echo "built in $B"
