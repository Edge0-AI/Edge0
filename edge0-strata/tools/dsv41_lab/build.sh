#!/bin/bash
# build the kernel labs on a CUDA box: tools/dsv41_lab/build.sh [arch, default sm_86]
set -e
cd "$(dirname "$0")"
ARCH=${1:-sm_86}
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
mkdir -p out
CPU=../../src/kernels/cpu
INC="-I../../include -I../../third_party/ggml -I$CPU"
g++ -O2 -std=c++17 $INC -c ../trellis/tqenc.cpp -o out/tqenc.o
g++ -O2 -std=c++17 $INC -c $CPU/tq_cpu.cpp -o out/tq_cpu.o
g++ -O2 -std=c++17 $INC -mavx2 -mfma -mf16c -c $CPU/tq_avx2.cpp -o out/tq_avx2.o
g++ -O2 -std=c++17 $INC -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi -mavx2 -mfma -mf16c -c $CPU/tq_avx512.cpp -o out/tq_avx512.o
$NVCC -O3 -std=c++17 -arch=$ARCH -lineinfo -I../../include -o out/tqk_lab tqk_lab.cu out/tqenc.o out/tq_cpu.o out/tq_avx2.o out/tq_avx512.o -lpthread
# DeepSeek kernel lab: links src/dsv41/ds4_kernels.cu unmodified
$NVCC -O3 -std=c++17 -arch=$ARCH -lineinfo -I../../include -o out/ds4_lab ds4_lab.cu ../../src/dsv41/ds4_kernels.cu
