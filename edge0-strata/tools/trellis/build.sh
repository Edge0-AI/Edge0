#!/bin/sh
# tools/trellis/build.sh [test|lib|bench] - builds the trellis encoder standalone (g++, Linux x86-64).
# Run from the repository root.  Output in ${TQ_OUT:-build-tqenc}.
set -e
OUT=${TQ_OUT:-build-tqenc}
mkdir -p "$OUT"
CXX=${CXX:-g++}
INC="-Itools/trellis -Iinclude -Isrc/kernels/cpu -Ithird_party/ggml"
OPT="-O3 -std=c++17 -mavx2 -mfma -mf16c -fno-fast-math -pthread"
case "${1:-test}" in
lib)
    $CXX $OPT -fPIC -shared $INC tools/trellis/tqenc.cpp -o "$OUT/libtqenc.so" ;;
hip)
    # needs the ROCm of /data/dsv41/rv: export ROCM_PATH=/data/dsv41/rv/lib/python3.14/site-packages/_rocm_sdk_devel PATH=/data/dsv41/rv/bin:$PATH
    # the host part of the encoder (tqenc.cpp) is built by g++: under hipcc ggml-common-tq.h's table becomes a __device__ variable
    g++ $OPT -fPIC $INC -c tools/trellis/tqenc.cpp -o "$OUT/tqenc_host.o"
    HF="-O3 --offload-arch=gfx1201 -mcumode -std=c++17 -mavx2 -mfma -mf16c"
    hipcc $HF -fPIC $INC -c tools/trellis/tqenc_hip.cpp -o "$OUT/tqenc_hip.o"
    hipcc $HF $INC -c tools/trellis/tqenc_hip_test.cpp -o "$OUT/tqenc_hip_test.o"
    hipcc --offload-arch=gfx1201 "$OUT/tqenc_hip_test.o" "$OUT/tqenc_hip.o" "$OUT/tqenc_host.o" -o "$OUT/tqenc_hip_test"
    hipcc --offload-arch=gfx1201 -shared "$OUT/tqenc_hip.o" "$OUT/tqenc_host.o" -o "$OUT/libtqenc_hip.so" ;;
test)
    $CXX $OPT $INC -c src/kernels/cpu/tq_cpu.cpp -o "$OUT/tq_cpu.o"
    $CXX $OPT $INC -c src/kernels/cpu/tq_avx2.cpp -o "$OUT/tq_avx2.o"
    $CXX -O3 -std=c++17 -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vnni -mfma -mf16c -mavx2 $INC -c src/kernels/cpu/tq_avx512.cpp -o "$OUT/tq_avx512.o"
    $CXX $OPT $INC tools/trellis/tqenc_test.cpp tools/trellis/tqenc.cpp "$OUT/tq_cpu.o" "$OUT/tq_avx2.o" "$OUT/tq_avx512.o" -o "$OUT/tqenc_test"
    ;;
esac
echo built "$OUT"
