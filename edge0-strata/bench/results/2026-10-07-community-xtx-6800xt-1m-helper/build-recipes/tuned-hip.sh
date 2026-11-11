#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${STRATA_RESEARCH_ROOT:?set STRATA_RESEARCH_ROOT to your reproduction directory}
TASK_SOURCE=$TASK_ROOT/llama.cpp-cuda/strata-100tg-ilhip-20261007
TASK_BUILD=$TASK_SOURCE/build-hip-100tg-ilhip
TASK_SDK_BASE=$TASK_ROOT/llama.cpp-cuda/strata-0139-hip-20261005/.venv102/lib/python3.12/site-packages
TASK_DEVEL=$TASK_SDK_BASE/_rocm_sdk_devel
TASK_LIBRARIES=$TASK_SDK_BASE/_rocm_sdk_libraries
export ROCM_PATH=$TASK_DEVEL HIP_PATH=$TASK_DEVEL HIP_PLATFORM=amd
export PATH=$TASK_DEVEL/llvm/bin:$PATH
export LD_LIBRARY_PATH=$TASK_DEVEL/lib:$TASK_LIBRARIES/lib:${LD_LIBRARY_PATH:-}
cmake -S "$TASK_SOURCE" -B "$TASK_BUILD" -G 'Unix Makefiles' \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=OFF \
  '-DCMAKE_HIP_ARCHITECTURES=gfx1030;gfx1100' \
  -DCMAKE_HIP_COMPILER="$TASK_DEVEL/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$TASK_DEVEL" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$TASK_DEVEL --rocm-device-lib-path=$TASK_DEVEL/amdgcn/bitcode" \
  -DCMAKE_PREFIX_PATH="$TASK_DEVEL;$TASK_LIBRARIES" \
  -DSTRATA_GGML_DIR="$TASK_ROOT/llama.cpp-cuda/strata-llama-3cf0325"
if [ "${1:-build}" != configure ]; then
  cmake --build "$TASK_BUILD" --parallel "${TASK_BUILD_JOBS:-3}" --target strata mmvq_il_parity
fi
