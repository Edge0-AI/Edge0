# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2.  The kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard
# below refuses anything older: silently building for a pre-Ampere arch would produce a binary that runs
# and is wrong about its own performance, which is the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.55 LANGUAGES CXX)
# EXPERIMENTAL: the Intel Arc engine, the SYCL port in sycl/ (PR #423; icpx + oneMKL, docs/INTEL_ARC.md). It is its
# own project that takes the unported sources from src/; with this on, nothing below is configured.
option(STRATA_ENABLE_SYCL "EXPERIMENTAL: build the Intel Arc (SYCL) engine from sycl/ - docs/INTEL_ARC.md" OFF)
if(STRATA_ENABLE_SYCL)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_ENABLE_SYCL is a separate backend; leave STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")
  endif()
  message(WARNING "STRATA_ENABLE_SYCL: the Intel Arc engine is experimental (community port, docs/INTEL_ARC.md)")
  add_subdirectory(sycl)
  return()
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Strata's own AVX2/AVX-512 kernels stay in the
# binary behind their CPU checks; the router lookahead gets an AVX1 kernel.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# #3202: Windows hands a program its command line in the system's ANSI code page (cp1251 / cp866 on a Russian Windows, cp1252 on an
# English one), but the engine reads every path as UTF-8 (the run configs and the server write UTF-8). A model folder with
# Cyrillic letters arrived as other bytes, or as "?" where the page has no such letter, and the model did not open. This manifest
# makes the process's ANSI code page UTF-8 (Windows 10 1903 and newer; older Windows ignores it and nothing changes), so the
# command line, fopen and std::ifstream all take UTF-8.
function(strata_utf8_manifest target)
  if(MSVC)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/strata-utf8.manifest)
  endif()
endfunction()

# The AVX kernels exist for x86 only.  On aarch64 (DGX Spark / GB10) portable.cpp stands in for them and
# native experts run on ggml-cpu's NEON dot products.
#
# Windows builds on an ARM64 CPU (the case this was added for): CMake can run under x64 emulation and report the PROCESSOR as
# AMD64, and an unset processor is read as x86 below, so the processor string alone is not enough there and the
# x86 kernel sources were pulled into an aarch64 build (kq_avx1.cpp failed on <immintrin.h>).  Ask the compiler
# what it targets instead; STRATA_FORCE_ARM64 overrides the answer for a shell that misdetects the host.  On x86
# and on arm64 Linux both tests give the same answer as before, so those configurations are unchanged.
option(STRATA_FORCE_ARM64 "force the aarch64 CPU path (portable.cpp) when the emulated shell misdetects the host as x86" OFF)
include(CheckCXXSourceCompiles)
set(STRATA_QUIET_SAVE "${CMAKE_REQUIRED_QUIET}")
set(CMAKE_REQUIRED_QUIET ON)
check_cxx_source_compiles("
#if !defined(_M_ARM64) && !defined(__aarch64__)
#error not aarch64
#endif
int main() { return 0; }" STRATA_COMPILER_TARGET_ARM64)
set(CMAKE_REQUIRED_QUIET "${STRATA_QUIET_SAVE}")
if(STRATA_FORCE_ARM64 OR STRATA_COMPILER_TARGET_ARM64)
  set(STRATA_CPU_X86 OFF)
elseif(NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64|i[3-6]86|x86|X86)$")   # an unset processor stays the x86 build, as before
  set(STRATA_CPU_X86 ON)
else()
  set(STRATA_CPU_X86 OFF)
endif()
# The sources' _MSC_VER branches assume x86 (intrin.h, _xgetbv, cpuid): in Arm64 builds on Windows clang-cl defines both
# _MSC_VER and __aarch64__, so the define lets the portable path skip them.  Not set on x86.
if(NOT STRATA_CPU_X86)
  add_compile_definitions(STRATA_CPU_ARM64=1)
endif()
if(STRATA_ISA_FLOOR AND NOT STRATA_CPU_X86)
  message(WARNING "STRATA_ISA_FLOOR is an x86 older-CPU build and does not apply to ${CMAKE_SYSTEM_PROCESSOR}; ignored")
  set(STRATA_ISA_FLOOR "" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
# AMD gfx906 (MI50, wave64): the same .cu sources compiled as HIP through the gfx906 compat layer.  The CUDA runtime and
# cuBLAS names are mapped by the headers in include/strata/platform/hip_compat/, which go first on the include path,
# and CUDA::cudart / CUDA::cublas become aliases of HIP's targets, so every CUDA target below stays as it is.  This is
# separate from STRATA_ENABLE_HIP (the wave32 RDNA backend): with it on, STRATA_ENABLE_CUDA is on and
# STRATA_ENABLE_HIP stays off, so every "CUDA" branch of the tree is the one taken.
option(STRATA_HIP_GFX906 "Build the CUDA targets for AMD gfx906 through HIP (wave64 compat layer)" OFF)
if(STRATA_HIP_GFX906)
  if(STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_HIP_GFX906 and STRATA_ENABLE_HIP are separate backends; enable only one")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
# The rows one verify / batch window holds (include/strata/kernels/window_limits.hpp): 8, the 0.2.7 window, or 16 (four pipelined
# groups of 16 slots = 64 slots on a four-card split, docs/BATCHING.md).  Only the CUDA and HIP engines are built from
# these sources with a wide window; the SYCL and Metal builds never read this variable and keep 8.  A window of up to
# 8 rows runs the kernels it always ran, whichever is chosen (the wide instantiations are extra code).  "auto" = 16 for
# the HIP engines (a HIP code object loads on first use, so the extra kernels cost nothing until a wide window forms)
# and 8 for CUDA, which loads every kernel at start (CUDA_MODULE_LOADING=EAGER: the wide kernels would take VRAM from
# the expert cache of every card); -DSTRATA_WINDOW_MAX_T=16 builds the wide CUDA engine.
set(STRATA_WINDOW_MAX_T "auto" CACHE STRING "rows of a verify / batch window: 8, 16 or auto (HIP 16, CUDA 8)")
set_property(CACHE STRATA_WINDOW_MAX_T PROPERTY STRINGS auto 8 16)
if(NOT STRATA_WINDOW_MAX_T STREQUAL "auto" AND NOT STRATA_WINDOW_MAX_T STREQUAL "8" AND NOT STRATA_WINDOW_MAX_T STREQUAL "16")
  message(FATAL_ERROR "STRATA_WINDOW_MAX_T must be auto, 8 or 16 (got '${STRATA_WINDOW_MAX_T}')")
endif()
if(STRATA_ENABLE_HIP OR STRATA_ENABLE_CUDA)
  if(STRATA_WINDOW_MAX_T STREQUAL "auto")
    if(STRATA_ENABLE_HIP)
      set(_strata_window_max_t 16)
    else()
      set(_strata_window_max_t 8)
    endif()
  else()
    set(_strata_window_max_t ${STRATA_WINDOW_MAX_T})
  endif()
  add_compile_definitions(STRATA_VERIFY_MAX_T=${_strata_window_max_t})
  message(STATUS "Strata: verify / batch windows hold up to ${_strata_window_max_t} rows")
endif()
# MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K (Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF
# projections, STRATA_DENSE_MMQ).  On by default since MODEL-UPDATE Phase 1: on CUDA they are their own library
# (strata_kq, loaded only for a model with these formats: the engine loads every kernel of its own binary at start,
# CUDA_MODULE_LOADING=EAGER, and these took ~3.8 MB of VRAM - 3 IQ3_XXS expert slots on a P100 - from every model); on
# HIP a code object loads on first use, so they cost other models nothing there (R9700: the same slots).  Used at run
# time with STRATA_MMQ_KQ=1 (opt-in: it changes the prompt's rounding against the FP16 path).
option(STRATA_MMQ_TQ_OVERLAY "Gyro trellis MMQ prompt kernels on an upstream llama.cpp: an overlay of its ggml-cuda headers written at configure time (cmake/strata_ggml_tq_overlay.cmake)" ON)
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files, and on HIP the dense GGUF projections (STRATA_DENSE_MMQ)" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default since MODEL-UPDATE Phase 1: one more instance per grouped kernel in iq_kernels.cu, +19 KB of sm_60 code,
# measured to leave every expert-cache slot count as it was (P100, R9700).  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
# CUDA: the pointer-table MMQ behind STRATA_MMQ_DIRECT=1 (the prompt path's MMQ reads resident experts in their cache
# slots through a Strata-side copy of llama.cpp's MMQ scheduler, src/prefill/moe_mmq_tab.cuh).  Off by default, like
# the K-quant options above: the engine loads every kernel when it starts (CUDA_MODULE_LOADING=EAGER), and these nine
# instances (+15 MB of binary) take 36-38 MiB of VRAM per GPU on sm_86 - expert slots every model would lose, with
# STRATA_MMQ_DIRECT unset too.  They also add ~70% of the upstream MMQ instances' compile time per CUDA arch (175 s
# of CPU for the nine at sm_86 alone).  Off, the build's device code is byte for byte the one without this feature.  CUDA
# only (not HIP, not gfx906); switched off at configure time when the llama.cpp tree's mmq.cuh is not one the copy
# was verified against.
option(STRATA_MMQ_TAB "Build the pointer-table prompt MMQ for STRATA_MMQ_DIRECT=1 (CUDA; ~37 MiB VRAM per GPU)" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
if(NOT STRATA_CPU_X86)
  # arm64 Linux (DGX Spark): much of the suite drives the x86 CPU kernels, which this build does not have.  A default, not a
  # prohibition: -DSTRATA_BUILD_TESTS=ON still builds what builds (classifying the tests by capability is a follow-up).
  set(_strata_tests_default OFF)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# #2027: defined here, before the GPU section reads it to register the native-expert tests (it was defined at the
# native-experts section far below, so a first configure registered four tests fewer than the second)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_split_failure_test src/core/conversation_split_failure_test.cpp)
  target_include_directories(conversation_split_failure_test PRIVATE include)
  add_test(NAME conversation_split_failure_test COMMAND conversation_split_failure_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(conversation_file_test src/core/conversation_file_test.cpp src/core/conversation_file.cpp)
  target_include_directories(conversation_file_test PRIVATE include)
  add_test(NAME conversation_file_test COMMAND conversation_file_test)
  add_executable(session_save_reclaim_test src/core/session_save_reclaim_test.cpp src/core/conversation_file.cpp)
  target_include_directories(session_save_reclaim_test PRIVATE include)
  add_test(NAME session_save_reclaim_test COMMAND session_save_reclaim_test)
endif()

if(STRATA_HIP_GFX906)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP architectures" FORCE)
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  add_library(CUDA::cudart INTERFACE IMPORTED)
  target_link_libraries(CUDA::cudart INTERFACE hip::host)
  target_compile_definitions(CUDA::cudart INTERFACE STRATA_HIP_GFX906=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(CUDA::cudart BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_library(CUDA::cublas INTERFACE IMPORTED)
  target_link_libraries(CUDA::cublas INTERFACE roc::hipblas CUDA::cudart)
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_compile_definitions(STRATA_HIP_GFX906=1)
  message(STATUS "Strata: HIP (gfx906 compat) enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_CUDA)
  # #585: CMake links CUDA code with the STATIC runtime by default, and the targets below also link CUDA::cudart (the
  # DLL's import library).  MSVC with a CUDA 12.x toolkit - the experimental Pascal/Volta build - then fails with
  # LNK1169 (every cudart symbol defined twice); GNU ld lets it pass.  That build links the shared runtime alone.  The
  # ready-made engine (CUDA 13, no STRATA_EXPERIMENTAL_SM60) is unchanged; -DCMAKE_CUDA_RUNTIME_LIBRARY overrides.
  if(MSVC AND STRATA_EXPERIMENTAL_SM60 AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  # #4023: the default below has to be set BEFORE enable_language(CUDA): since CMake 3.18 that call fills an unset
  # CMAKE_CUDA_ARCHITECTURES with nvcc's own default (sm_75 for CUDA 13), so a test after it never saw "undefined" and
  # a build with no option was an sm_75 build.  -DCMAKE_CUDA_ARCHITECTURES and the CUDAARCHS environment variable win.
  set(_strata_cuda_arch_defaulted OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    if(STRATA_BUILD_DSV41)
      # #3935: the DeepSeek engine's own release shape (sm_86, sm_89, sm_120 + PTX 120, as packaging/build_dsv41_cuda.sh), so a hand
      # build of strata-dsv41 with no architecture runs on an RTX 30, 40 or 50 card; the Qwen engine's default below is unchanged
      set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;120" CACHE STRING "CUDA architectures")
    else()
      set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
    endif()
    set(_strata_cuda_arch_defaulted ON)
  endif()
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  option(STRATA_EXPERIMENTAL_SM60 "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" OFF)
  if(_strata_cuda_arch_defaulted)
    # a recommendation, not a refusal: the release engine is built for 75;86;89;120, a hand build names its card
    message(WARNING "Strata: CMAKE_CUDA_ARCHITECTURES was not set, so this build is for ${CMAKE_CUDA_ARCHITECTURES} only (sm_120 = RTX 50). For another "
                    "card pass it: -DCMAKE_CUDA_ARCHITECTURES=native (the GPUs in this PC), 75 (RTX 20), 86 (RTX 30), "
                    "89 (RTX 40), or a list like \"86;120\".")
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Turing port: developed and measured on sm_120 (RTX 50); the tf32 mma in the QSA scorer has a
    # portable fp32-FMA fallback, so RTX 20 (75), 30 (86), RTX 40 (89) and RTX 50 (120) build too.
    # Pre-sm_75 is refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    # STRATA_EXPERIMENTAL_SM60: a Pascal (sm_60, GP100-class) or Volta (sm_70, V100 / Titan V, #236) build.
    # These sit below every architecture the kernels were written for, so two intrinsics are replaced below
    # their arch - `__dp4a` (6.1+) and `__nanosleep` (7.0+) - in include/strata/kernels/dp4a.hpp, which is
    # bit-exact for the first and merely a shorter pause for the second (sm_70 has both).  Community-tested,
    # not in the ready-made engine; RTX 20/30/40/50 cards must use the release engine instead.  It needs a CUDA
    # 12.x toolkit: CUDA 13 dropped sm_60 and sm_70 (nvcc fails CMake's compiler test with "compute_70").
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal sm_60 or Volta sm_70 build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
    if(_base LESS 52)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}': even the experimental build "
                          "needs compute capability 5.2 (sm_52, Maxwell) or newer")
    endif()
    # Maxwell (sm_52, #3218): one reporter built and ran this on two Quadro M6000 (CUDA 12.x, Qwen3.8-Flash-Next
    # IQ1, about 22 t/s).  It has no native half-precision arithmetic, so it is slower per byte than Pascal; only
    # this opt-in build gets the floor.  Warn, do not refuse.
    if(_base LESS 60)
      message(WARNING "Building for sm_${_base} (Maxwell): community-tested on one reporter's Quadro M6000 only, "
                      "no native half-precision arithmetic, CUDA 12.x toolkit required (docs/OLDER_GPUS.md).")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  if(STRATA_EXPERIMENTAL_SM60)
    # every target sees it: the Pascal / Volta code paths (the Volta prompt attention, PR #600) are compiled into this
    # build only, so the ready-made engine (sm_75+) keeps exactly the code it had
    add_compile_definitions(STRATA_EXPERIMENTAL_SM60=1)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target ${STRATA_HIP_BLAS_TARGETS})
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  add_executable(batch_telemetry_test src/core/batch_telemetry_test.cpp)
  target_include_directories(batch_telemetry_test PRIVATE include)
  add_test(NAME batch_telemetry_test COMMAND batch_telemetry_test)
  add_executable(logit_bias_test src/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE include)
  add_test(NAME logit_bias_test COMMAND logit_bias_test)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the model families (qwen4exp, qwen35moe): the guards, the geometry from metadata, the layout tables (synthetic)
  add_executable(model_family_test tests/core/model_family_test.cpp)
  target_link_libraries(model_family_test PRIVATE strata_artifact)
  add_test(NAME model_family_test COMMAND model_family_test)
  # how --batch cuts a prompt read into pieces (a layer split with no slot decoding must still reach a BYIELD)
  add_executable(batch_read_test tests/program/batch_read_test.cpp)
  target_include_directories(batch_read_test PRIVATE include)
  add_test(NAME batch_read_test COMMAND batch_read_test)
  # the model's geometry from its metadata, and the check against the kernels' compiled shape (synthetic)
  add_executable(gguf_geometry_test tests/core/gguf_geometry_test.cpp)
  target_link_libraries(gguf_geometry_test PRIVATE strata_artifact)
  add_test(NAME gguf_geometry_test COMMAND gguf_geometry_test)

  # the hipBLASLt tuning header's architecture gate: RDNA4 siblings share solution IDs, RDNA3 siblings share
  # theirs, cross-family pairs and version mismatches stay rejected (synthetic tables: no model, no GPU)
  add_executable(tuning_header_arch_family_test tests/prefill/tuning_header_arch_family_test.cpp)
  target_include_directories(tuning_header_arch_family_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
  add_test(NAME tuning_header_arch_family_test COMMAND tuning_header_arch_family_test)

  # the auto layer split's minimum stage size (#1616): header-only rule, no GPU
  add_executable(split_rules_test tests/program/split_rules_test.cpp)
  target_include_directories(split_rules_test PRIVATE include)
  add_test(NAME split_rules_test COMMAND split_rules_test)

  # #3797: the out-of-memory report at start (per card free/total, the largest other programs): the text is a pure
  # function; the driver query (NVML, loaded at run time) is only checked to be safe to call. No GPU needed.
  add_executable(gpu_report_test tests/program/gpu_report_test.cpp)
  target_include_directories(gpu_report_test PRIVATE include)
  target_link_libraries(gpu_report_test PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME gpu_report_test COMMAND gpu_report_test)

  # #3702: the later-stage reserve of the layer split search (header-only arithmetic, pinned against the old expression)
  add_executable(stage_reserve_test tests/program/stage_reserve_test.cpp)
  target_include_directories(stage_reserve_test PRIVATE include)
  add_test(NAME stage_reserve_test COMMAND stage_reserve_test)

  # the prompt planner's host-memory model (#2824): header-only, no GPU
  add_executable(prefill_plan_test tests/program/prefill_plan_test.cpp)
  target_include_directories(prefill_plan_test PRIVATE include)
  add_test(NAME prefill_plan_test COMMAND prefill_plan_test)

  # the default CPU prefill share is not armed when every expert is resident (#1595): header-only rule, no GPU
  add_executable(share_rules_test tests/prefill/share_rules_test.cpp)
  target_include_directories(share_rules_test PRIVATE include)
  add_test(NAME share_rules_test COMMAND share_rules_test)

  # the Windows HIP auto prefill chunk cap (#1630) and the commit-limit verdict (#1607): header-only, no GPU
  add_executable(host_limits_test tests/platform/host_limits_test.cpp)
  target_include_directories(host_limits_test PRIVATE include)
  add_test(NAME host_limits_test COMMAND host_limits_test)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # the DFlash2 drafter (src/core/dflash2.cpp, src/kernels/cuda/dflash2.cu) exists in the CUDA and HIP builds; the Metal and SYCL builds compile without it
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_DFLASH2=1)
  if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    # the runtime device check (src/core/device.cu) admits the cards this build was made for (#236)
    target_compile_definitions(strata_core PRIVATE STRATA_EXPERIMENTAL_SM60=1)
  endif()
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s_gemv_pair.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/route_prior.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/qsa_pattn.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/dflash2.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/cuda/probe_q8gpu.cu
                          src/kernels/cuda/latent.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # Exact gfx1100 prompt scorer (PR1825 method port): optional standalone test,
  # also available where the published source omits tests/.
  option(STRATA_PARITY_GFX1100_QSA_SCORES "Build the exact gfx1100 QSA score/ID check" OFF)
  if(STRATA_ENABLE_HIP AND (STRATA_BUILD_TESTS OR STRATA_PARITY_GFX1100_QSA_SCORES))
    set_source_files_properties(src/kernels/cuda/qsa_scores_gfx1100_test.cu
                                src/kernels/qsa_scores_gfx1100_parity.cpp PROPERTIES LANGUAGE HIP)
    add_executable(qsa_scores_gfx1100_parity src/kernels/qsa_scores_gfx1100_parity.cpp
                                           src/kernels/cuda/qsa_scores_gfx1100_test.cu)
    target_include_directories(qsa_scores_gfx1100_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(qsa_scores_gfx1100_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    enable_testing()
    add_test(NAME qsa_scores_gfx1100_parity COMMAND qsa_scores_gfx1100_parity --full)
    set_tests_properties(qsa_scores_gfx1100_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
      ENVIRONMENT "STRATA_SCORES_THREAD=1;STRATA_SCORES_THREAD_TRACE=1")
  endif()
  # perf-review D-1: the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
  # the tests, or alone with -DSTRATA_PARITY_PROMPT_ATTN=ON.  CUDA only: HIP compiles the tensor-core kernel out.
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_CUDA)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_include_directories(qsa_prompt_attn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels CUDA::cudart)
  endif()
  # PR #3943: STRATA_PATTN_SM60's score and value kernels against the decode kernels' output (the Daily's chunk kernel and
  # the merge), bitwise, on random int8 K/V (GPU, synthetic, no model).  Only in the experimental sm_60 build; it skips
  # (77) on a build without the kernels.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    add_executable(qsa_pattn_parity src/kernels/qsa_pattn_parity.cpp)
    target_include_directories(qsa_pattn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_pattn_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_pattn_parity COMMAND qsa_pattn_parity)
    set_tests_properties(qsa_pattn_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # PR #3820: STRATA_SELECT_LONG's multi-CTA top-k and shuffle-free decode scorer against qsa_block_topk and
  # qsa_block_scores, bitwise, on random scores (GPU, synthetic, no model).  The switches force the new kernels on any
  # card; the "n3" case fixes three CTAs a query.  Only in the experimental sm_60 build (skips with 77 otherwise).
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    add_executable(qsa_select_long_parity src/kernels/qsa_select_long_parity.cpp)
    target_include_directories(qsa_select_long_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_select_long_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_select_long_parity COMMAND qsa_select_long_parity)
    add_test(NAME qsa_select_long_parity_n3 COMMAND qsa_select_long_parity)
    set_tests_properties(qsa_select_long_parity qsa_select_long_parity_n3 PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(qsa_select_long_parity PROPERTIES ENVIRONMENT "STRATA_SCORES_DX=1;STRATA_TOPK_MULTI=1")
    set_tests_properties(qsa_select_long_parity_n3 PROPERTIES ENVIRONMENT "STRATA_SCORES_DX=1;STRATA_TOPK_MULTI=1;STRATA_TOPK_MULTI_N=3")
  endif()
  # PR #600: the Volta (sm_70) prompt attention's latency pipeline against the FP32 kernel it replaces.  Only in the
  # experimental sm_60/sm_70 build, where prompt_attn_v70_kernel is compiled in.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    add_executable(qsa_prompt_attn_sm70_bench bench/qsa_prompt_attn_sm70_bench.cpp)
    target_include_directories(qsa_prompt_attn_sm70_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_sm70_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  # #512: the top-k's active-block bound (Turing's dispatch, forced on any card by STRATA_TOPK_ACTIVE_ANY) against
  # the capacity rule, selected ids bitwise
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_topk_active_parity src/kernels/qsa_topk_active_parity.cpp)
    target_include_directories(qsa_topk_active_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
    set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
    # STRATA_QSA_SLOTS: a batch window's scores, top-k and attention as one launch over rows of different slots
    # against one call per row, bitwise (synthetic, no model)
    add_executable(qsa_slots_parity src/kernels/qsa_slots_parity.cpp)
    target_include_directories(qsa_slots_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_slots_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_slots_parity COMMAND qsa_slots_parity)
    add_test(NAME qsa_slots_parity_lanecell COMMAND qsa_slots_parity)
    set_tests_properties(qsa_slots_parity qsa_slots_parity_lanecell PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(qsa_slots_parity_lanecell PROPERTIES ENVIRONMENT "STRATA_ATTN_LANECELL=1")
    # fork: the Daily's decode attention (STRATA_ATTN_DAILY=1: PR #540's chunk kernel on every card and the bounded
    # merge) against upstream's kernels, bitwise; the production entries checked with the variable off and on
    add_executable(qsa_merge_daily_parity src/kernels/qsa_merge_daily_parity.cpp)
    target_include_directories(qsa_merge_daily_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_merge_daily_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_merge_daily_parity_default COMMAND qsa_merge_daily_parity)
    add_test(NAME qsa_merge_daily_parity_daily COMMAND qsa_merge_daily_parity)
    set_tests_properties(qsa_merge_daily_parity_default PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=0"
                         SKIP_RETURN_CODE 77)
    set_tests_properties(qsa_merge_daily_parity_daily PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=1"
                         SKIP_RETURN_CODE 77)
    # the prompt path's block scores on the thread-per-head kernel (Turing) against the warp kernel, bitwise
    add_executable(qsa_scores_parity src/kernels/qsa_scores_parity.cpp)
    target_include_directories(qsa_scores_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_scores_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_scores_parity COMMAND qsa_scores_parity)
    set_tests_properties(qsa_scores_parity PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "STRATA_SCORES_THREAD=1")
  endif()
  # QSAROWS (docs/QSAROWS.md): what the T rows of a verification window cost in a QSA layer's decode kernels at a long
  # context (block scores, top-k, attention over each row's selection, indexer append), row by row; GPU, synthetic, no
  # model.  Not in the default build: `cmake --build <dir> --target qsa_rows_bench`.
  if(STRATA_ENABLE_CUDA)
    add_executable(qsa_rows_bench EXCLUDE_FROM_ALL src/kernels/qsa_rows_bench.cpp)
    target_include_directories(qsa_rows_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_rows_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
  # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
  # 262144-cell capacity (the top-k dispatch follows the capacity) and exits non-zero on a failed gate
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_HIP)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    set_source_files_properties(src/kernels/qsa_select_bench.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
    # the FP32 tiled scorer (STRATA_SELECT_SIMT=1) on HIP: FP64 gate and selections against the warp kernel at 131K
    add_test(NAME qsa_select_bench_fp32_tiled COMMAND qsa_select_bench 131072 256 3 262144)
    set_tests_properties(qsa_select_bench_fp32_tiled PROPERTIES ENVIRONMENT "STRATA_SELECT_SIMT=1")
  endif()
  # the same on CUDA: the tensor-core scorer (sm_80+) or the FP32 tiled one (below sm_80, and forced on any card by
  # STRATA_QSA_WARP=1 in the second test) against the warp kernel and FP64, at 32K and 131K
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    target_include_directories(qsa_select_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
    add_test(NAME qsa_select_bench_fp32_tiled COMMAND qsa_select_bench 131072 256 3 262144)
    set_tests_properties(qsa_select_bench_fp32_tiled PROPERTIES ENVIRONMENT "STRATA_QSA_WARP=1;STRATA_SELECT_SIMT=1")
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)   # the prompt attention on a recorded selection (STRATA_QSA_DUMP)
    add_executable(pa_bench tests/cuda/pa_bench.cpp)
    target_include_directories(pa_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(pa_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_flash_attn.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "$<IF:$<BOOL:${STRATA_HIP_GFX906}>,,--use_fast_math>")
  elseif(STRATA_ENABLE_HIP)
    # Keep Q8_K's multiply and magic-bias rounding as separate FP32 operations on HIP; contraction to an
    # FMA changes tie results. The embedding gather also relies on separately rounded operations.
    set_source_files_properties(src/kernels/cuda/elementwise.cu src/kernels/cuda/quantize_act.cu
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  # This oracle is a CUDA ggml build; do not pull CUDA libraries into a HIP build.
  if(STRATA_ENABLE_CUDA AND STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${_strata_gpu_include_directories})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
    src/core/remote_expert_opt.cu
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp src/core/dflash2.cpp
    src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
    src/core/arch_defaults.cpp src/core/nv_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(host_graph_launch_test tests/core/host_graph_launch_test.cu)
    target_include_directories(host_graph_launch_test PRIVATE include)
    target_link_libraries(host_graph_launch_test PRIVATE CUDA::cudart Threads::Threads)
    add_test(NAME host_graph_launch_test COMMAND host_graph_launch_test)
    set_tests_properties(host_graph_launch_test PROPERTIES TIMEOUT 30)
  endif()
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/exchange_storage_test.cpp")
    add_executable(exchange_storage_test tests/core/exchange_storage_test.cpp)
    target_include_directories(exchange_storage_test PRIVATE include)
    target_link_libraries(exchange_storage_test PRIVATE Threads::Threads)
    add_test(NAME exchange_storage_test COMMAND exchange_storage_test)
    set_tests_properties(exchange_storage_test PROPERTIES TIMEOUT 60)
  endif()

  # available_host_bytes(): MemAvailable, cgroup limits and the operator's --memory-limit-mib cap on fixture directories
  # - no GPU, no CUDA
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/memory_guard_test.cpp")
    add_executable(memory_guard_test tests/core/memory_guard_test.cpp src/core/conversation_memory.cpp)
    target_include_directories(memory_guard_test PRIVATE include)
    add_test(NAME memory_guard_test COMMAND memory_guard_test)
    set_tests_properties(memory_guard_test PROPERTIES TIMEOUT 60)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/file_expert_source_test.cpp")
    enable_testing()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
    add_executable(vmm_test tests/core/vmm_test.cpp)
    target_link_libraries(vmm_test PRIVATE strata_engine)
    add_test(NAME vmm_test COMMAND vmm_test)
    set_tests_properties(vmm_test PROPERTIES TIMEOUT 60)
    # The resident RAM mode with a layer split: the copy keeps each stage's lend region, and the experts the
    # prompt path borrows are served from RAM (the file source reads none of them).  The device calls are
    # wrapped - conversation_transfer_test's pattern - and `pin = false` keeps the copy pageable, so this
    # runs on a CPU-only PC.
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(split_lend_complement_test tests/core/split_lend_complement_test.cpp)
      target_link_libraries(split_lend_complement_test PRIVATE strata_engine)
      target_link_options(split_lend_complement_test PRIVATE
        -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaFree -Wl,--wrap=cudaMemset -Wl,--wrap=cudaMemcpy
        -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaStreamSynchronize -Wl,--wrap=cudaDeviceSynchronize
        -Wl,--wrap=cudaGetLastError -Wl,--wrap=cudaGetDevice -Wl,--wrap=cudaSetDevice
        -Wl,--wrap=cudaMemGetInfo -Wl,--wrap=cudaHostAlloc -Wl,--wrap=cudaFreeHost)
      add_test(NAME split_lend_complement_test COMMAND split_lend_complement_test)
      set_tests_properties(split_lend_complement_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_save_test.cpp")
    enable_testing()
    add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
    target_link_libraries(expert_profile_save_test PRIVATE strata_engine)
    add_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
    set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
  endif()
  # the profile header's version is enforced: a version other than 1 is refused by its number (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_version_test.cpp")
    enable_testing()
    add_executable(expert_profile_version_test tests/core/expert_profile_version_test.cpp)
    target_link_libraries(expert_profile_version_test PRIVATE strata_engine)
    add_test(NAME expert_profile_version_test COMMAND expert_profile_version_test)
    set_tests_properties(expert_profile_version_test PROPERTIES TIMEOUT 60)
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_NATIVE_EXPERTS)
    # native_experts.txt v2-v4 (the real packs' copies and the per-role shard column) and the arena filled from
    # synthetic GGUF shards at the real expert dimensions
    add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
    target_link_libraries(expert_layout_test PRIVATE strata_engine)
    add_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
    set_tests_properties(expert_layout_test PROPERTIES TIMEOUT 60)
    # #326: a native pack's BF16 (--compat-bf16) PLE key stays in the arena; a quantized key row loads as before
    add_executable(native_dense_ple_key_test tests/core/native_dense_ple_key_test.cpp)
    target_link_libraries(native_dense_ple_key_test PRIVATE strata_engine)
    add_test(NAME native_dense_ple_key_test COMMAND native_dense_ple_key_test)
    set_tests_properties(native_dense_ple_key_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    # #369: per-layer admission on a sized (native pack) cache starts each layer in its own range
    add_executable(expert_cache_per_layer_test tests/core/expert_cache_per_layer_test.cpp)
    target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
    add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
    set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
    target_link_libraries(expert_cache_segmented_test PRIVATE strata_engine)
    add_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
    set_tests_properties(expert_cache_segmented_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    # STRATA_MMQ_DIRECT: slot_guard / slot_room / slot_tail on the uniform and the sized cache, and the default
    # (no guard) sized layout unchanged
    add_executable(expert_cache_slot_guard_test tests/core/expert_cache_slot_guard_test.cpp)
    target_link_libraries(expert_cache_slot_guard_test PRIVATE strata_engine)
    add_test(NAME expert_cache_slot_guard_test COMMAND expert_cache_slot_guard_test)
    set_tests_properties(expert_cache_slot_guard_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  if(STRATA_BUILD_CONVERSATION_TESTS)
    add_executable(conversation_snapshot_test src/core/conversation_snapshot_test.cpp)
    target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
    add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    # the same test with a GDN state over 1 MiB and the opt-in background checkpoint copy (STRATA_CKPT_ASYNC=1), alone and with
    # recent states kept in VRAM (STRATA_CKPT_VRAM_SLOTS): every checkpoint, capture and restore must give the same bytes
    add_test(NAME conversation_snapshot_test_ckpt_async COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test_ckpt_async PROPERTIES
      ENVIRONMENT "STRATA_TEST_BIG_GDN=1;STRATA_CKPT_ASYNC=1" SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_test(NAME conversation_snapshot_test_ckpt_vram_slots COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test_ckpt_vram_slots PROPERTIES
      ENVIRONMENT "STRATA_TEST_BIG_GDN=1;STRATA_CKPT_ASYNC=1;STRATA_CKPT_VRAM_SLOTS=3" SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_executable(conversation_validation_test src/core/conversation_validation_test.cpp)
    target_link_libraries(conversation_validation_test PRIVATE strata_engine)
    add_test(NAME conversation_validation_test COMMAND conversation_validation_test)
    set_tests_properties(conversation_validation_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(conversation_transfer_test src/core/conversation_validation_test.cpp)
      target_compile_definitions(conversation_transfer_test PRIVATE CONVERSATION_TEST_TRANSFERS)
      target_link_options(conversation_transfer_test PRIVATE
        -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaStreamSynchronize
        -Wl,--wrap=cudaGetDevice -Wl,--wrap=cudaStreamCreateWithFlags
        -Wl,--wrap=cudaDeviceSynchronize -Wl,--wrap=cudaGetLastError)
      target_link_libraries(conversation_transfer_test PRIVATE strata_engine)
      add_test(NAME conversation_transfer_test COMMAND conversation_transfer_test)
      set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  if(STRATA_ENABLE_HIP)
    # #313: the RDNA3 WMMA GEMMs include hip_runtime.h: AMD builds only (a CUDA build never sees this file)
    target_sources(strata_prefill PRIVATE src/prefill/wmma_gemm.cu)
  endif()
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  if(STRATA_ENABLE_CUDA)
    # v100/gdn-chunk: the chunked Volta GDN prompt recurrence.  The dispatch (src/prefill/kernels.cu) is
    # Volta-only (cc 7.0) and gated on STRATA_EXPERIMENTAL_SM60; the path itself stays opt-in (STRATA_GDN_CHUNK).
    target_sources(strata_prefill PRIVATE src/prefill/gdn_chunk.cu)
  endif()
  if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
    target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::hipblaslt)

    add_executable(tune_hipblaslt tools/hip/tune_hipblaslt.cpp)
    target_link_libraries(tune_hipblaslt PRIVATE hip::host roc::hipblas roc::hipblaslt)
  endif()
  if(STRATA_ENABLE_HIP AND STRATA_ROCBLAS_AVAILABLE)
    # the prompt path's rocBLAS solution table (STRATA_ROCBLAS_TUNING; tools/hip/tune_rocblas makes one)
    target_compile_definitions(strata_prefill PRIVATE STRATA_ROCBLAS_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::rocblas)

    add_executable(tune_rocblas tools/hip/tune_rocblas.cpp)
    target_link_libraries(tune_rocblas PRIVATE hip::host roc::rocblas)
  endif()

  # DFlash2 drafter tools (not built by default: cmake --build <dir> --target dflash2_parity dflash2_gemm_bench dflash2_attn_bench): the drafter against
  # the torch port on a synthetic case (needs the pack and a case file, tools/dflash2/mk_parity_case.py), and the timing of its Q8 GEMM and attention
  add_executable(dflash2_parity EXCLUDE_FROM_ALL src/core/dflash2_parity.cpp)
  target_include_directories(dflash2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_parity PRIVATE strata_engine strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_gemm_bench EXCLUDE_FROM_ALL src/core/dflash2_gemm_bench.cpp)
  target_include_directories(dflash2_gemm_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_gemm_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_launch_bench EXCLUDE_FROM_ALL src/core/dflash2_launch_bench.cu)
  target_include_directories(dflash2_launch_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_launch_bench PRIVATE ${_strata_gpu_runtime_target})
  add_executable(dflash2_bw_bench EXCLUDE_FROM_ALL src/core/dflash2_bw_bench.cu)
  target_include_directories(dflash2_bw_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_bw_bench PRIVATE ${_strata_gpu_runtime_target})
  add_executable(dflash2_small_bench EXCLUDE_FROM_ALL src/core/dflash2_small_bench.cpp)
  target_include_directories(dflash2_small_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_small_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(dflash2_attn_bench EXCLUDE_FROM_ALL src/core/dflash2_attn_bench.cpp)
  target_include_directories(dflash2_attn_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dflash2_attn_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata src/program/generate.cpp)
  strata_utf8_manifest(strata)
  target_include_directories(strata PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})
  if(WIN32 AND STRATA_ENABLE_HIP)
    # 0.2.1.2: the ROCm math libraries live in <package>\rocm\bin, which is on PATH only when the launcher starts the engine; load them on first
    # use (main() adds that folder to the DLL search first) so `strata.exe --version` works from the package folder.  amdhip64_7.dll and the
    # C++ runtime sit beside the exe (tools/hip/package_windows.py) and stay normal imports.
    target_link_libraries(strata PRIVATE delayimp)
    target_link_options(strata PRIVATE "LINKER:/DELAYLOAD:hipblas.dll" "LINKER:/DELAYLOAD:libhipblaslt.dll" "LINKER:/DELAYLOAD:rocblas.dll")
  endif()

  # NEW-MODELS-PLAN.md T1: Qwen3.6-35B-A3B's reference path (one GPU, the model resident, plain kernels), checked
  # against llama.cpp by tools/qwen35/; the engine itself does not run the family yet
  add_executable(strata-q36ref src/qwen35/q36ref.cu)
  target_include_directories(strata-q36ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata-q36ref PRIVATE strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  # the engine path for the same family (serve protocol, prompt chunks, decode graph): src/qwen35/q36_engine.cu, a
  # library that `strata` switches to for a qwen35moe model (the family switch at the top of its main) and that
  # strata-q36 wraps on its own.  Not on gfx906 (#3172, #3271): it calls cuBLAS entry points the wave64 compat layer does
  # not map (cublasSgemm, cublasGemmStridedBatchedEx) and its kernels assume 32-lane warps; main() guards every use with
  # STRATA_HAVE_QWEN35, so a gfx906 build links and runs the Qwen3.8 family, and `strata --families` says so
  if(NOT STRATA_HIP_GFX906)
  add_library(strata_qwen35 STATIC src/qwen35/q36_engine.cu)
  target_include_directories(strata_qwen35 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata_qwen35 PRIVATE strata_prefill strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  if(STRATA_NATIVE_EXPERTS)
    # the hybrid mode (experts that do not all fit in VRAM): the CPU expert pool on ggml-cpu's arithmetic
    target_link_libraries(strata_qwen35 PRIVATE strata_kernels_cpu)
    target_compile_definitions(strata_qwen35 PRIVATE STRATA_Q36_CPU_POOL=1)
  endif()
  add_executable(strata-q36 src/qwen35/q36_main.cpp)
  strata_utf8_manifest(strata-q36)
  target_link_libraries(strata-q36 PRIVATE strata_qwen35 ${_strata_gpu_runtime_target})
  target_link_libraries(strata PRIVATE strata_qwen35)
  target_compile_definitions(strata PRIVATE STRATA_HAVE_QWEN35=1)
  endif()

  # DeepSeek-V4.1-Flash: the third engine path (src/dsv41, from Project Maya's maya-ds4, MIT, third_party/project-maya).
  # Its own executable for now (strata-dsv41); a family switch in `strata` comes after first light.  OFF by default: it
  # changes no existing target (the vendored kernels live in strata::dsv41k, so no symbol is shared with strata_kernels).
  option(STRATA_BUILD_DSV41 "Build the DeepSeek-V4.1 engine path (strata-dsv41)" OFF)
  if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)
    add_library(strata_dsv41_kernels STATIC src/dsv41/vendor/exl3.cu src/dsv41/vendor/sampler.cu src/dsv41/vendor/moe_win.cu
                src/dsv41/vendor/exl3_cpu.cpp)
    if(NOT STRATA_ENABLE_HIP)   # (the pre-sm_80 NVIDIA decode kernels exl3.cu dispatches to)
      target_sources(strata_dsv41_kernels PRIVATE src/dsv41/vendor/exl3_legacy.cu)
    endif()
    target_include_directories(strata_dsv41_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(strata_dsv41_kernels PUBLIC ${_strata_gpu_runtime_target})
    add_executable(strata-dsv41 src/dsv41/ds4_main.cpp src/dsv41/ds4_model.cu src/dsv41/ds4_kernels.cu src/dsv41/ds4_vision.cu
                   src/dsv41/ds4_gguf.cu src/dsv41/ds4_gguf_cpu.cpp src/dsv41/ds4_iq.cu src/dsv41/ds4_iq_cpu.cpp
                   src/dsv41/ds4_tq.cu src/dsv41/ds4_tq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp src/dsv41/ds4_packinfo.cpp)
    # the commit the binary was built from, readable with --version (set -DSTRATA_BUILD_ID=... for a tree without .git)
    if(NOT STRATA_BUILD_ID AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID)   # (a git archive has no .git: the packer writes the commit here)
      file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID STRATA_BUILD_ID LIMIT_COUNT 1)
    endif()
    if(NOT STRATA_BUILD_ID)
      execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD OUTPUT_VARIABLE STRATA_BUILD_ID
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(NOT STRATA_BUILD_ID)
      set(STRATA_BUILD_ID unknown)
    endif()
    set_source_files_properties(src/dsv41/ds4_main.cpp PROPERTIES COMPILE_DEFINITIONS "DS4_BUILD_ID=\"${STRATA_BUILD_ID}\"")
    target_include_directories(strata-dsv41 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                               ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill)
    # strata_prefill: the RDNA3 WMMA dense GEMM (src/prefill/wmma_gemm.cu), which gemm() prefers over hipBLAS on gfx11 (measured
    # 4.97 -> 20-40 TFLOPS on this part: the chunked prompt's dense GEMM is 11.1 s of a 4K prompt at the library's rate)
    target_link_libraries(strata-dsv41 PRIVATE strata_prefill strata_dsv41_kernels ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target})
    find_package(Threads REQUIRED)
    target_link_libraries(strata-dsv41 PRIVATE Threads::Threads)
    if(WIN32 AND STRATA_ENABLE_HIP)   # cudaMemGetInfo is the DXGI-budgeted hip_compat::mem_get_info (src/core/device.cu): on a Strix Halo the
      # carve-out is not all the GPU may use, and the engine sizes its expert tiers by what the budget says is free
      target_link_libraries(strata-dsv41 PRIVATE strata_core)
    endif()
    if(STRATA_NATIVE_EXPERTS)   # Strata's CPU trellis kernels (tq_cpu.hpp) for the trellis experts on the CPU lane
      target_link_libraries(strata-dsv41 PRIVATE strata_kernels_cpu)
      target_compile_definitions(strata-dsv41 PRIVATE STRATA_DSV41_TQ_CPU=1)
    endif()
    if(STRATA_NATIVE_EXPERTS)
      # the CPU lane's ggml-type rows: ggml-cpu's dots against Strata's AVX2 i-quant rows (bits and GB/s); no GPU, no model
      add_executable(ds4_gg_bench tests/dsv41/gg_dot_bench.cpp src/dsv41/ds4_gg_cpu.cpp)
      target_include_directories(ds4_gg_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
      target_link_libraries(ds4_gg_bench PRIVATE strata_kernels_cpu Threads::Threads)
      target_compile_definitions(ds4_gg_bench PRIVATE STRATA_DSV41_TQ_CPU=1)
    endif()
    if(STRATA_NATIVE_EXPERTS)
      # the trellis experts (device kernels ds4_tq.cu, CPU lane ds4_tq_cpu.cpp) against a double reference; needs a GPU, no model
      # both test .cpp files launch kernels, so they compile as HIP on the AMD backend and as CUDA on the NVIDIA one
      if(STRATA_ENABLE_HIP)
        set(_ds4_test_lang HIP)
      elseif(STRATA_ENABLE_CUDA)
        set(_ds4_test_lang CUDA)
      else()
        set(_ds4_test_lang CXX)
      endif()
      set_source_files_properties(tests/dsv41/tq_expert_test.cpp PROPERTIES LANGUAGE ${_ds4_test_lang})
      add_executable(ds4_tq_test tests/dsv41/tq_expert_test.cpp src/dsv41/ds4_tq.cu src/dsv41/ds4_iq.cu src/dsv41/ds4_tq_cpu.cpp
                     src/dsv41/ds4_iq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp tools/trellis/tqenc.cpp)
      target_include_directories(ds4_tq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu)
      target_link_libraries(ds4_tq_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
      target_compile_definitions(ds4_tq_test PRIVATE STRATA_DSV41_TQ_CPU=1)
      # the same with the plain-FMA tile instead of the gfx12 WMMA one (what CUDA and gfx11 builds run)
      add_executable(ds4_tq_test_portable tests/dsv41/tq_expert_test.cpp src/dsv41/ds4_tq.cu src/dsv41/ds4_iq.cu src/dsv41/ds4_tq_cpu.cpp
                     src/dsv41/ds4_iq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp tools/trellis/tqenc.cpp)
      target_include_directories(ds4_tq_test_portable PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu)
      target_link_libraries(ds4_tq_test_portable PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
      target_compile_definitions(ds4_tq_test_portable PRIVATE STRATA_DSV41_TQ_CPU=1 DS4_TQ_PORTABLE=1)
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_tq_test_portable PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_tq_test PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      # the i-quant experts (ds4_iq.cu) against ggml's own decode of the same blocks + a double reference, device and CPU
      # lane, for every type the device takes: the gate a new type (IQ2_XS ..) has to pass before the engine uses it
      set_source_files_properties(tests/dsv41/iq_expert_test.cpp PROPERTIES LANGUAGE ${_ds4_test_lang})
      add_executable(ds4_iq_test tests/dsv41/iq_expert_test.cpp src/dsv41/ds4_iq.cu src/dsv41/ds4_iq_cpu.cpp
                     src/dsv41/ds4_gg_cpu.cpp src/dsv41/ds4_tq.cu src/dsv41/ds4_tq_cpu.cpp)
      target_include_directories(ds4_iq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu)
      target_link_libraries(ds4_iq_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
      target_compile_definitions(ds4_iq_test PRIVATE STRATA_DSV41_TQ_CPU=1)
      if(STRATA_ENABLE_HIP)
        target_compile_options(ds4_iq_test PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endif()
      # the kept conversations' .meta file and prefix matching (ds4_slotmeta.hpp): no GPU, no model
      add_executable(ds4_slot_test tests/dsv41/slot_test.cpp)
      target_include_directories(ds4_slot_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_slot_test PRIVATE cxx_std_17)
      add_executable(ds4_experts_rule_test tests/dsv41/experts_rule_test.cpp)
      target_include_directories(ds4_experts_rule_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_experts_rule_test PRIVATE cxx_std_17)
      add_executable(ds4_pick_rule_test tests/dsv41/pick_rule_test.cpp)
      target_include_directories(ds4_pick_rule_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
      target_compile_features(ds4_pick_rule_test PRIVATE cxx_std_17)
    endif()
    if(STRATA_ENABLE_HIP)
      foreach(_t strata_dsv41_kernels strata-dsv41)
        target_compile_options(${_t} PRIVATE "$<$<COMPILE_LANGUAGE:HIP,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_extra.h>")
      endforeach()
    endif()
  endif()

  if(STRATA_ENABLE_HIP)
    enable_testing()
    set(_strata_hip_tests)
    add_executable(hip_prefill_gemm tests/hip/prefill_gemm.cpp)
    target_link_libraries(hip_prefill_gemm PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_gemm COMMAND hip_prefill_gemm)
    list(APPEND _strata_hip_tests hip_prefill_gemm)
    add_executable(hip_gemm_f16_io_parity tests/hip/gemm_f16_io_parity.cpp)
    target_link_libraries(hip_gemm_f16_io_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gemm_f16_io_parity COMMAND hip_gemm_f16_io_parity)
    list(APPEND _strata_hip_tests hip_gemm_f16_io_parity)
    add_executable(hip_q2_zero tests/hip/q2_zero.cpp)
    target_link_libraries(hip_q2_zero PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_q2_zero COMMAND hip_q2_zero)
    list(APPEND _strata_hip_tests hip_q2_zero)
    add_executable(hip_expert_cache_staging tests/hip/expert_cache_staging.cpp)
    target_link_libraries(hip_expert_cache_staging PRIVATE strata_engine)
    add_test(NAME hip_expert_cache_staging COMMAND hip_expert_cache_staging)
    list(APPEND _strata_hip_tests hip_expert_cache_staging)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/intrinsics.cpp")
      add_executable(hip_intrinsics tests/hip/intrinsics.cpp)
      target_link_libraries(hip_intrinsics PRIVATE strata_hip_runtime)
      add_test(NAME hip_intrinsics COMMAND hip_intrinsics)
      list(APPEND _strata_hip_tests hip_intrinsics)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/handoff.cpp")
      add_executable(hip_handoff tests/hip/handoff.cpp)
      target_link_libraries(hip_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_handoff COMMAND hip_handoff)
      list(APPEND _strata_hip_tests hip_handoff)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/mapped_alias.cpp")
      # #325: reports what a hipHostGetDevicePointer alias is on this stack (on Windows: the host pointer, and a
      # D2D copy into it does not land - hip_handoff's timeout there). A diagnosis: exits 0 whenever it ran.
      add_executable(hip_mapped_alias tests/hip/mapped_alias.cpp)
      target_link_libraries(hip_mapped_alias PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_mapped_alias COMMAND hip_mapped_alias)
      set_tests_properties(hip_mapped_alias PROPERTIES SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_mapped_alias)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/native_qsa_score.cpp")
      add_executable(hip_native_qsa_score tests/hip/native_qsa_score.cpp)
      target_link_libraries(hip_native_qsa_score PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_native_qsa_score COMMAND hip_native_qsa_score)
      list(APPEND _strata_hip_tests hip_native_qsa_score)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_native_batch.cpp")
      add_executable(hip_prefill_native_batch tests/hip/prefill_native_batch.cpp)
      target_link_libraries(hip_prefill_native_batch PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_prefill_native_batch COMMAND hip_prefill_native_batch)
      list(APPEND _strata_hip_tests hip_prefill_native_batch)
    endif()
    # Aurora: the prompt GDN recurrence's per-head kernel against the column-split + norm kernels, bitwise
    add_executable(hip_gdn_rec_head tests/hip/gdn_rec_head.cpp)
    target_link_libraries(hip_gdn_rec_head PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gdn_rec_head COMMAND hip_gdn_rec_head)
    list(APPEND _strata_hip_tests hip_gdn_rec_head)
    add_executable(hip_gdn_chunked tests/hip/gdn_chunked.cpp)
    target_link_libraries(hip_gdn_chunked PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gdn_chunked COMMAND hip_gdn_chunked)
    add_test(NAME hip_gdn_chunked_off COMMAND hip_gdn_chunked --off)
    add_test(NAME hip_gdn_chunked_default COMMAND hip_gdn_chunked --default)
    list(APPEND _strata_hip_tests hip_gdn_chunked hip_gdn_chunked_off hip_gdn_chunked_default)
    # GDN prompt stage bench (gates + conv + recurrence, per-piece times and output/state hashes at 4K/16K/32K); a measurement
    # tool, run by hand with the STRATA_GDN_* knobs under test - no add_test
    add_executable(hip_gdn_stage_bench tests/hip/gdn_stage_bench.cpp)
    target_link_libraries(hip_gdn_stage_bench PRIVATE strata_prefill strata_hip_runtime)
    # #313: the RDNA3 / RDNA3.5 WMMA dense GEMMs (opt-in STRATA_WMMA_GEMM=1) against a reference
    add_executable(hip_prefill_wmma_gemm_parity tests/hip/prefill_wmma_gemm_parity.cpp)
    target_include_directories(hip_prefill_wmma_gemm_parity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
    target_link_libraries(hip_prefill_wmma_gemm_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_wmma_gemm_parity COMMAND hip_prefill_wmma_gemm_parity)
    set_tests_properties(hip_prefill_wmma_gemm_parity PROPERTIES SKIP_RETURN_CODE 77)   # no matrix cores: skipped
    list(APPEND _strata_hip_tests hip_prefill_wmma_gemm_parity)
    # S (STRATA_HCD_EXACT): the exact HC down kernel against hipBLASLt solution 1176 / 1177 (needs STRATA_HIPBLASLT_TUNING; exit 77 else)
    add_executable(hip_prefill_hcd_exact_parity tests/hip/prefill_hcd_exact_parity.cpp)
    target_link_libraries(hip_prefill_hcd_exact_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_hcd_exact_parity COMMAND hip_prefill_hcd_exact_parity)
    set_tests_properties(hip_prefill_hcd_exact_parity PROPERTIES SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prefill_hcd_exact_parity)
    # the HC write-back kernel timed at the prompt path's shapes with an output checksum (not a ctest; STRATA_GRW_V picks a variant)
    add_executable(hip_prefill_gr_write_bench tests/hip/prefill_gr_write_bench.cpp)
    target_link_libraries(hip_prefill_gr_write_bench PRIVATE strata_prefill strata_hip_runtime)
    # S6: the AMD fast router against the portable kernel, bitwise
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_link_libraries(hip_router_fast PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_router_fast COMMAND hip_router_fast)
    list(APPEND _strata_hip_tests hip_router_fast)
    # S6: the RDNA4 matrix-core prompt attention (STRATA_HIP_WMMA) against the FP32 kernel and FP64; skips off gfx12
    # the prompt attention on a recorded selection (STRATA_QSA_DUMP), for A/B of kernel variants (STRATA_QSA_UNION=1)
    add_executable(hip_pa_bench tests/cuda/pa_bench.cpp)
    target_link_libraries(hip_pa_bench PRIVATE strata_kernels strata_hip_runtime)
    add_executable(hip_prompt_attn_wmma src/kernels/qsa_prompt_attn_parity.cpp)
    target_link_libraries(hip_prompt_attn_wmma PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_prompt_attn_wmma COMMAND hip_prompt_attn_wmma 32768 2048 3)
    set_tests_properties(hip_prompt_attn_wmma PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prompt_attn_wmma)
    if(STRATA_HIPBLASLT_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_hipblaslt_gemm.cpp")
      add_executable(hip_prefill_hipblaslt_gemm tests/hip/prefill_hipblaslt_gemm.cpp)
      target_include_directories(hip_prefill_hipblaslt_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_hipblaslt_gemm PRIVATE strata_prefill strata_hip_runtime roc::hipblaslt roc::hipblas)
      add_test(NAME hip_prefill_hipblaslt_gemm COMMAND hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_hipblaslt_gemm)
    endif()
    if(STRATA_ROCBLAS_AVAILABLE)
      # the rocBLAS solution table of the FP16-out prompt GEMMs (gfx103x): opt-in, skips without STRATA_ROCBLAS_TUNING
      add_executable(hip_prefill_rocblas_gemm tests/hip/prefill_rocblas_gemm.cpp)
      target_include_directories(hip_prefill_rocblas_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_rocblas_gemm PRIVATE strata_prefill strata_hip_runtime roc::rocblas roc::hipblas)
      add_test(NAME hip_prefill_rocblas_gemm COMMAND hip_prefill_rocblas_gemm)
      set_tests_properties(hip_prefill_rocblas_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_rocblas_gemm)
    endif()
    if(_strata_hip_tests)
      set_tests_properties(${_strata_hip_tests} PROPERTIES TIMEOUT 60)
    endif()
    if(TARGET hip_prefill_native_batch)
      set_tests_properties(hip_prefill_native_batch PROPERTIES TIMEOUT 120)
    endif()
    if(TARGET hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
    if(TARGET hip_prefill_rocblas_gemm)
      set_tests_properties(hip_prefill_rocblas_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
  endif()

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  # TODO 24: the i-quant fixtures are generated, not shipped.  tools/iq_fixture.py needs python3 + numpy
  # and the repository's vendored gguf-py (no pip installs); it writes deterministic fixtures into the
  # BUILD directory (the source tree is not touched).  The CTest fixture makes generation-to-test order
  # explicit even under parallel execution; a fixture-generation failure fails iq_parity with it.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  # the generator needs an interpreter WITH numpy: probe the configured one, the setup's .venv, and the
  # system python3; -DIQ_FIXTURE_PYTHON=... overrides the search outright.
  set(_iq_python "")
  set(_iq_candidates "")
  if(IQ_FIXTURE_PYTHON)
    list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
  else()
    if(Python3_Interpreter_FOUND)
      list(APPEND _iq_candidates ${Python3_EXECUTABLE})
    endif()
    list(APPEND _iq_candidates "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
                                "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
                                /usr/bin/python3
                                python3)
  endif()
  foreach(_cand IN LISTS _iq_candidates)
    if(_cand STREQUAL "")
      continue()
    endif()
    execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
    if(_iq_probe EQUAL 0)
      set(_iq_python "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _iq_python STREQUAL "")
    add_test(NAME iq_parity_fixtures
             COMMAND ${_iq_python} ${CMAKE_CURRENT_SOURCE_DIR}/tools/iq_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    # exit 3: numpy or the vendored gguf-py is missing (a trimmed third_party) - a skip, not a failure
    set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
    add_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  else()
    message(STATUS "STRATA tests: no python3 with numpy was found (the configured interpreter and .venv "
                   "were probed) - the iq_parity fixtures and its test are not registered; the fixtures "
                   "need python3 with numpy and the repository's vendored gguf-py")
  endif()
  if(STRATA_ENABLE_CUDA)
    # native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
    # bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
    add_executable(mmvq_multi_parity src/kernels/mmvq_multi_parity.cpp)
    target_include_directories(mmvq_multi_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)
    if(STRATA_HIP_GFX906)   # the CUDA layout, where the negative control has power (the wave64 one is exact at every T)
      add_test(NAME mmvq_multi_parity_cuda_layout COMMAND mmvq_multi_parity)
      set_tests_properties(mmvq_multi_parity_cuda_layout PROPERTIES ENVIRONMENT "STRATA_MMVQ_WAVE=0")
    endif()
  endif()
  if(STRATA_ENABLE_CUDA)
    # fork F4: the 2-4 column dense projections from the interleaved q8_1 copy, bitwise vs native_mmvq (GPU, synthetic)
    add_executable(mmvq_il_parity src/kernels/mmvq_il_parity.cpp)
    target_include_directories(mmvq_il_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_il_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_il_parity COMMAND mmvq_il_parity)
    # sm_89's opt-in tables (#3453, STRATA_MMVQ_IL_89=1): the table check only (no case matches MMVQ_IL_ONLY)
    add_test(NAME mmvq_il_parity_89table COMMAND mmvq_il_parity)
    set_tests_properties(mmvq_il_parity_89table PROPERTIES ENVIRONMENT "STRATA_MMVQ_IL_89=1;MMVQ_IL_ONLY=no-such-case")
    # the opt-in STRATA_MMVQ_IL_PF kernels (hoisted group loads, exact integer forms): every variant bit-checked against
    # native_mmvq for every type, shape, T and rows choice
    add_test(NAME mmvq_il_parity_pf COMMAND mmvq_il_parity --pf 0,2,3,4,5,6)
    set_tests_properties(mmvq_il_parity_pf PROPERTIES TIMEOUT 900)
    # the wide dequant kernels (a thread per 8 values; sm_61's default), bitwise vs the per-group ones (GPU, synthetic,
    # no model; --bench adds old/new timings)
    add_executable(dequant_wide_parity src/kernels/dequant_wide_parity.cpp)
    target_include_directories(dequant_wide_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(dequant_wide_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME dequant_wide_parity COMMAND dequant_wide_parity)
  endif()
  # programmatic dependent launch (pdl.hpp) in a captured verify-shaped chain: bitwise vs ordinary launches over
  # replays with poisoned buffers, and only kernel-to-kernel programmatic edges (GPU, synthetic, no model; the edges
  # are only made on sm_90+, elsewhere it checks that there are none).  NVIDIA only: PDL is a CUDA feature.
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(pdl_parity src/kernels/pdl_parity.cpp)
    target_include_directories(pdl_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pdl_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME pdl_parity COMMAND pdl_parity)
    # PDL is opt-in (STRATA_DF_PDL); the test checks the opted-in path, whose PDL capture forces programmatic launches
    set_tests_properties(pdl_parity PROPERTIES ENVIRONMENT "STRATA_DF_PDL=1")
  endif()
  # the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
  # (GPU, synthetic, no model; --bench adds old/new timings)
  add_executable(iq_multi_parity src/kernels/iq_multi_parity.cpp)
  target_include_directories(iq_multi_parity PRIVATE ${_strata_gpu_include_directories})
  # strata_kernels_cpu: lat_gu_colscale_dev (iq_kernels.cu) reads the loaded layout's column scales
  target_link_libraries(iq_multi_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  add_test(NAME iq_multi_parity COMMAND iq_multi_parity)
  # q3k_real_ref: CPU-official (ggml) vs GPU on a REAL v7_l1280_q3k expert (Q3_K gu / IQ4_NL dn)
  add_executable(q3k_real_ref src/kernels/q3k_real_ref.cpp)
  target_include_directories(q3k_real_ref PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(q3k_real_ref PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  # the Gyro trellis decode kernels (TQ2_T / TQK6 / TQK7) on synthetic blocks: decode-once vs per-column bitwise, output
  # checksums to compare two builds byte for byte, --bench for timings (GPU, no model)
  add_executable(tq_pascal_bench src/kernels/tq_pascal_bench.cpp)
  target_include_directories(tq_pascal_bench PRIVATE ${_strata_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
  target_link_libraries(tq_pascal_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME tq_pascal_bench COMMAND tq_pascal_bench)
  set_tests_properties(iq_multi_parity PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1")   # the Q4_K / Q5_K / Q8_0 Split path against the old kernels
  # the same Split path on the other routes it can meet: the generic multi kernels (no V2K) and the fused gate/up + SwiGLU kernel, and the sub-16-row
  # gate/up route switched off (a K-quant row has 160 dots, not the 80 the sub-16 rows assume: STRATA_KQ_SPLIT=1 must stay bitwise on all of them)
  add_test(NAME iq_multi_parity_kq_nov2k COMMAND iq_multi_parity)
  set_tests_properties(iq_multi_parity_kq_nov2k PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1;STRATA_EXPERT_V2K=0")
  add_test(NAME iq_multi_parity_kq_swiglu COMMAND iq_multi_parity)
  set_tests_properties(iq_multi_parity_kq_swiglu PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1;STRATA_GU_SWIGLU=1")
  add_test(NAME iq_multi_parity_kq_nosub16 COMMAND iq_multi_parity)
  set_tests_properties(iq_multi_parity_kq_nosub16 PROPERTIES ENVIRONMENT "STRATA_KQ_SPLIT=1;STRATA_NO_SUB16_GU=1")
  # the prompt path's DeltaNet recurrence: gdn_rec_kh_kernel against gdn_rec_cols_pipe_kernel, bitwise (GPU,
  # synthetic, no model; --bench adds timings).  NVIDIA only: cp.async and the CUDA occupancy API (not on gfx906)
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    # Independent pinned expert uploads: byte parity and stream/event ordering.
    add_executable(dma_batch_parity tests/core/dma_batch_parity.cu)
    target_include_directories(dma_batch_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(dma_batch_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME dma_batch_parity COMMAND dma_batch_parity)
    set_tests_properties(dma_batch_parity PROPERTIES TIMEOUT 120)
    if(WIN32)
      add_test(NAME dma_batch_registered_parity COMMAND dma_batch_parity --registered)
      set_tests_properties(dma_batch_registered_parity PROPERTIES TIMEOUT 120)
    endif()
    add_executable(gdn_rec_parity src/prefill/gdn_rec_parity.cu)
    target_link_libraries(gdn_rec_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME gdn_rec_parity COMMAND gdn_rec_parity)
    # STRATA_GDN_TC=1: the tensor-core helpers of the scan, each on one warp against a double-precision reference.  Any sm_80+
    # card runs them; the scan itself needs 122 KB of shared memory a block (sm_80, sm_90), where gdn_rec_parity prints
    # "not on this card" for the tc variant
    add_test(NAME gdn_rec_parity_selftest_frag COMMAND gdn_rec_parity --selftest-frag)
    set_tests_properties(gdn_rec_parity_selftest_frag PROPERTIES SKIP_RETURN_CODE 77)
    # the production-linked GDN parity / benchmark harness (no kernel copies: it calls gdn_gates, gdn_conv and
    # gdn_recurrence of strata_prefill through the GdnProbe of include/strata/prefill/gdn_probe.hpp, against an FP64
    # oracle from tests/gdn/gdn_fixture.hpp).  Usage: docs/GDN_CONTRACT.md.  The smoke test is small and CPU-cheap.
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD OUTPUT_VARIABLE _gdn_sha
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT _gdn_sha)
      set(_gdn_sha unknown)
    endif()
    add_executable(gdn_harness src/prefill/gdn_harness.cu)
    target_include_directories(gdn_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/gdn)
    target_compile_definitions(gdn_harness PRIVATE GDN_HARNESS_SHA="${_gdn_sha}")
    target_link_libraries(gdn_harness PRIVATE strata_prefill)
    add_test(NAME gdn_harness_smoke COMMAND gdn_harness --T 400 --chunk 160 --cuts 37 --paths auto,keyhead,pipe --ref-heads 0,17)
    set_tests_properties(gdn_harness_smoke PROPERTIES TIMEOUT 300)
    # the prompt path's BF16 products against cuBLAS's own BF16 GEMM, within fp32 rounding (sm_7x runs them on the
    # FP16 tensor cores; GPU, synthetic, no model; --bench adds timings)
    add_executable(gemm_bf16_parity src/prefill/gemm_bf16_parity.cu)
    target_link_libraries(gemm_bf16_parity PRIVATE strata_prefill )
    add_test(NAME gemm_bf16_parity COMMAND gemm_bf16_parity)
    # STRATA_DEQUANT_SIDE: Gemm::native with the dense dequant on a side stream (two scratches) against the serial form,
    # bitwise (GPU, synthetic, no model; 77 without a device)
    add_executable(gemm_dequant_side_parity src/prefill/gemm_dequant_side_parity.cu)
    target_link_libraries(gemm_dequant_side_parity PRIVATE strata_prefill )
    add_test(NAME gemm_dequant_side_parity COMMAND gemm_dequant_side_parity)
    set_tests_properties(gemm_dequant_side_parity PROPERTIES SKIP_RETURN_CODE 77)
    # Pascal, opt-in STRATA_DENSE_F32=1: the dense projections' FP16 GEMMs as cublasSgemm on FP32 copies, bitwise vs
    # the cuBLAS FP16 call over a sweep of chunk sizes (GPU, synthetic, no model; --full sweeps every size, --bench
    # adds timings).  Not a ctest: whether the two agree is a property of the cuBLAS build, which this checks.
    add_executable(gemm_f32_parity src/prefill/gemm_f32_parity.cu)
    target_link_libraries(gemm_f32_parity PRIVATE strata_prefill )
    # the hyper-connection read's fused up projection + mix (gr_upmix, STRATA_HC_UPMIX) against cuBLAS + gr_mix_r,
    # bitwise where cuBLAS sums K in the same order, else within FP32 rounding and as close to FP64 (GPU, synthetic,
    # no model; --bench adds timings)
    add_executable(gr_upmix_parity src/prefill/gr_upmix_parity.cu)
    target_link_libraries(gr_upmix_parity PRIVATE strata_prefill )
    add_test(NAME gr_upmix_parity COMMAND gr_upmix_parity)
    set_tests_properties(gr_upmix_parity PROPERTIES SKIP_RETURN_CODE 77)
    # the two kernels of the layer-major prompt path (STRATA_PREFILL_LAYER_MAJOR=1, #3976): moe_shared_init and
    # moe_scatter_add against a host reference, bitwise for the scatter (GPU, synthetic, no model)
    add_executable(lm_kernels_parity src/prefill/lm_kernels_parity.cu)
    target_link_libraries(lm_kernels_parity PRIVATE strata_prefill )
    add_test(NAME lm_kernels_parity COMMAND lm_kernels_parity)
    set_tests_properties(lm_kernels_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the launches before them (GPU, synthetic,
  # no model; --bench adds v1/new timings of a window's 48 calls in a graph)
  add_executable(native_grouped_parity src/kernels/native_grouped_parity.cpp)
  target_include_directories(native_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_grouped_parity COMMAND native_grouped_parity)
  add_executable(mmvf_rows_parity src/kernels/mmvf_rows_parity.cpp)
  target_include_directories(mmvf_rows_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(mmvf_rows_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME mmvf_rows_parity COMMAND mmvf_rows_parity)
  set_tests_properties(mmvf_rows_parity PROPERTIES ENVIRONMENT "STRATA_MMVF_ROWS=1" TIMEOUT 120)
  if(STRATA_HIP_GFX906)
    add_test(NAME native_grouped_fused_parity COMMAND native_grouped_parity)
    set_tests_properties(native_grouped_fused_parity PROPERTIES
      ENVIRONMENT "STRATA_HIP_SWIGLU_FUSED=1;STRATA_EXP_MODE=7" TIMEOUT 120)
    add_executable(gfx906_gdn_rec_head tests/hip/gdn_rec_head.cpp)
    target_include_directories(gfx906_gdn_rec_head PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(gfx906_gdn_rec_head PRIVATE strata_prefill ${_strata_gpu_runtime_target})
    add_test(NAME gfx906_gdn_rec_head COMMAND gfx906_gdn_rec_head)
    set_tests_properties(gfx906_gdn_rec_head PROPERTIES TIMEOUT 120)
  endif()
  if(STRATA_ENABLE_CUDA)
    # the Volta prompt experts (gemm_iq_f16_grouped, STRATA_PF_WMMA) against a double reference; skipped below sm_70
    add_executable(pf_wmma_parity src/kernels/pf_wmma_parity.cpp)
    target_include_directories(pf_wmma_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pf_wmma_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME pf_wmma_parity COMMAND pf_wmma_parity)
    set_tests_properties(pf_wmma_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)
  # the split / batched verify-window kernels (from eddoursul's fork), bitwise vs the kernels they replace; CUDA only
  if(STRATA_ENABLE_CUDA)
    add_executable(verify_parity src/kernels/verify_parity.cpp)
    target_include_directories(verify_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(verify_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME verify_parity COMMAND verify_parity --selftest)
  endif()
  add_executable(fused_gr_bench src/kernels/fused_gr_bench.cpp)
  target_include_directories(fused_gr_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(fused_gr_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(router_chain_bench src/kernels/router_chain_bench.cpp)
  target_include_directories(router_chain_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_chain_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  # B6: the dense projections' time per verify row at the model's shapes, default vs STRATA_B6_MMVQ_ROWS, bitwise
  if(STRATA_ENABLE_CUDA)
    add_executable(b6_mmvq_bench src/kernels/b6_mmvq_bench.cpp)
    target_include_directories(b6_mmvq_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(b6_mmvq_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gdn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)
  # #3948: the same checks through the 4-blocks-a-head GDN step (the default on gfx1151), whose output-norm kernel writes the
  # STRATA_QFUSE q8_1 image on its own
  add_test(NAME gdn_parity_split COMMAND gdn_parity --selftest)
  set_tests_properties(gdn_parity_split PROPERTIES ENVIRONMENT "STRATA_GDN_SPLIT=1")

  # v100/gdn-chunk: the chunked GDN prompt recurrence vs the token recurrence and a host reference (GPU,
  # synthetic, no model).  The chunked path changes the summation order - FP32-level, not bit-exact - so the
  # test asserts measured error bounds, and --selftest pins them.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(gdn_chunk_parity src/kernels/gdn_chunk_parity.cpp)
    target_include_directories(gdn_chunk_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(gdn_chunk_parity PRIVATE strata_prefill ${_strata_gpu_runtime_target})
    add_test(NAME gdn_chunk_parity COMMAND gdn_chunk_parity --selftest)
    set_tests_properties(gdn_chunk_parity PROPERTIES SKIP_RETURN_CODE 77)

    # v100/gdn-chunk: the micro benchmark (CUDA events, real GDN shapes): chunked vs recurrent prompt path.
    add_executable(gdn_chunk_bench bench/gdn_chunk_bench.cpp)
    target_include_directories(gdn_chunk_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(gdn_chunk_bench PRIVATE strata_prefill ${_strata_gpu_runtime_target})
  endif()

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(sampler_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0")
  # the GPU verifier of probabilistic draft acceptance against the host reference (STRATA_SPEC_PROB)
  add_executable(spec_verify_parity src/kernels/spec_verify_parity.cpp)
  target_include_directories(spec_verify_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(spec_verify_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME spec_verify_parity COMMAND spec_verify_parity)
  set_tests_properties(spec_verify_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0")
  # the explicit mu / NULL verifier of the gated proposal (STRATA_SPEC_PROB=1, pick gate) against a CPU oracle
  add_executable(spec_proposal_parity src/kernels/spec_proposal_parity.cpp)
  target_include_directories(spec_proposal_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(spec_proposal_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME spec_proposal_parity COMMAND spec_proposal_parity)
  set_tests_properties(spec_proposal_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0" TIMEOUT 120)
  # the same fixtures on the other two sampled paths: the one-block kernel and the 0.1.20 kernel
  add_test(NAME sampler_parity_one_block COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_one_block PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=1")
  add_test(NAME sampler_parity_old COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_old PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=1;STRATA_SAMPLER_ONE_BLOCK=0")
  # S19: the decode QSA top-k and the greedy argmax on thread-block clusters (sm_90+) vs the one-CTA kernels, bitwise,
  # captured in a graph too (GPU, synthetic, no model; skips below sm_90); --bench: per-call times
  if(STRATA_ENABLE_CUDA)
    add_executable(decode_cluster_parity src/kernels/decode_cluster_parity.cpp)
    target_include_directories(decode_cluster_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(decode_cluster_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME decode_cluster_parity COMMAND decode_cluster_parity --selftest)
  endif()

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(rope_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  # the verify window's routing in one multi-column BF16 projection + one top-k, bitwise the per-token calls
  add_executable(route_window_parity src/kernels/route_window_parity.cpp)
  target_include_directories(route_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(route_window_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME route_window_parity COMMAND route_window_parity)

  # STRATA_ROUTE_RESIDENT's swap: the warp-per-token kernel against a scalar reference of the same semantics,
  # bitwise on the ids, the weights and the four counters (GPU, synthetic, no model)
  add_executable(route_resident_parity src/kernels/route_resident_parity.cu)
  target_include_directories(route_resident_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(route_resident_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME route_resident_parity COMMAND route_resident_parity)
  set_tests_properties(route_resident_parity PROPERTIES TIMEOUT 120)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)
  # #783 PR-c: the multi-token router and the k = 10 float4 combine, bitwise against the single-token contracts
  add_executable(native_multi_parity src/kernels/native_multi_parity.cpp)
  target_include_directories(native_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_multi_parity COMMAND native_multi_parity)
  # #783 PR-d: the batched K/V append (INT8, Q4_0, the hybrid fold) against the per-cell appends, memcmp
  add_executable(kv_steps_parity src/kernels/kv_steps_parity.cpp)
  target_include_directories(kv_steps_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_steps_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_steps_parity COMMAND kv_steps_parity)
  # #783 PR-e: the device-built expert plan (parallel grouping + prefix scan) against a host replay of its contract
  add_executable(resident_plan_parity src/kernels/resident_plan_parity.cpp)
  target_include_directories(resident_plan_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(resident_plan_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME resident_plan_parity COMMAND resident_plan_parity)
  # #783 PR-g: the multi-token hyper-connection read/write (exact-T projections under them) against the single-token calls
  add_executable(gr_multi_parity src/kernels/gr_multi_parity.cpp)
  target_include_directories(gr_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_multi_parity COMMAND gr_multi_parity)
  # STRATA_HC_PERSIST=1: the persistent hyper-connection read against the default multi read (bounded, two runs bitwise),
  # and with the variable unset it never runs (a skip where the card cannot hold its 80 blocks, and on CUDA)
  add_executable(hc_persist_parity src/kernels/hc_persist_parity.cpp)
  target_include_directories(hc_persist_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(hc_persist_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME hc_persist_parity COMMAND hc_persist_parity)
  add_test(NAME hc_persist_default_off COMMAND hc_persist_parity --off)
  if(STRATA_HIP_GFX906)
    # the S6 router (STRATA_HIP_ROUTER_FAST=1 on gfx906) against the portable kernel, bitwise, on wave64
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_include_directories(hip_router_fast PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(hip_router_fast PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  # the V100 fusion work: every fused kernel against its unfused pipeline, bitwise, and the before/after
  # CUDA-event timings at decode shapes (`--bench`).  It is a .cu because it carries the unfused reference
  # kernels it compares against.
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(fusions_parity bench/micro/fusions_parity.cu)
    target_include_directories(fusions_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(fusions_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME fusions_parity COMMAND fusions_parity)
  endif()

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # the QSA top-k the engine dispatches against the original kernel: identical ids on both sides of the register
  # kernel's reach, with tied and with equal scores (synthetic, no model)
  add_executable(qsa_topk_parity src/kernels/qsa_topk_parity.cpp)
  target_include_directories(qsa_topk_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_topk_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_topk_parity COMMAND qsa_topk_parity --selftest)
  # the register kernel from the smallest prompt batch (the line gfx1151 moves to 4096 blocks): same ids at every case
  add_test(NAME qsa_topk_parity_reg_always COMMAND qsa_topk_parity --selftest)
  set_tests_properties(qsa_topk_parity_reg_always PROPERTIES ENVIRONMENT "STRATA_TOPK_REG_MIN_BLOCKS=0")

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # STRATA_PLE_COMPACT: the raw IQ4_NL rows dequantized on the device against the host decode, bit for bit (synthetic, no model)
  add_executable(ple_iq4nl_dequant_test src/kernels/ple_iq4nl_dequant_test.cpp)
  target_include_directories(ple_iq4nl_dequant_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_iq4nl_dequant_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_iq4nl_dequant_test COMMAND ple_iq4nl_dequant_test)
  set_tests_properties(ple_iq4nl_dequant_test PROPERTIES SKIP_RETURN_CODE 77)

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # #2403: the verify window's KV calls on a streamed layer and on a ring (steps appends with the host copy, batched
  # resolve + attention, plain launches and graph replays), q4_0 / int8 / K8V4 (GPU, synthetic, no model).
  add_executable(kv_window_parity src/kernels/kv_window_parity.cpp)
  target_include_directories(kv_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_window_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_window_parity COMMAND kv_window_parity)

  # PR #120: the hybrid K8V4 KV (INT8 K + rotated Q4_0 V) vs host references (GPU, synthetic, no model).
  add_executable(kv_hybrid_parity src/kernels/kv_hybrid_parity.cpp)
  target_include_directories(kv_hybrid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_hybrid_parity COMMAND kv_hybrid_parity)
  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
  # the verify window's one-launch forms (hyper-connection write, PLE post-ops with the commit's
  # history snapshots) vs the per-token launches they replace, bitwise (GPU, synthetic, no model)
  add_executable(verify_batch_parity src/kernels/verify_batch_parity.cpp)
  target_include_directories(verify_batch_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(verify_batch_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME verify_batch_parity COMMAND verify_batch_parity)
  # #2234: the batch window's one-launch-for-all-rows kernels vs the per-row launches, bitwise (GPU, synthetic)
  add_executable(batch_rows_parity src/kernels/batch_rows_parity.cpp)
  target_include_directories(batch_rows_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(batch_rows_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME batch_rows_parity COMMAND batch_rows_parity)
  add_executable(cvec_parity src/kernels/cvec_parity.cpp)
  target_include_directories(cvec_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(cvec_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME cvec_parity COMMAND cvec_parity)

  # The rewritten per-hit and grouped expert kernels vs the previous ones, bitwise (GPU, synthetic, no model).
  # `--bench` adds previous/new timings on engine-like shapes.
  add_executable(s2_expert_grouped_parity src/kernels/s2_expert_grouped_parity.cpp)
  target_include_directories(s2_expert_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_expert_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_expert_grouped_parity COMMAND s2_expert_grouped_parity --selftest)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/native_mmvq_multi.cpp")
    add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
    target_include_directories(native_mmvq_multi PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_mmvq_multi PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # The pinned-memory headroom (pin_budget.hpp) against a simulated ceiling; skips without a GPU.
  add_executable(pin_budget_test src/core/pin_budget_test.cpp)
  target_link_libraries(pin_budget_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME pin_budget_test COMMAND pin_budget_test)

  # #3350 / #3470: the conversation copy streams and events are pooled across short-lived threads (needs a device, else skips)
  add_executable(conversation_copy_streams_test src/core/conversation_copy_streams_test.cpp)
  target_include_directories(conversation_copy_streams_test PRIVATE include)
  target_link_libraries(conversation_copy_streams_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME conversation_copy_streams_test COMMAND conversation_copy_streams_test)
  set_tests_properties(conversation_copy_streams_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pinned_shared_test src/core/pinned_shared_test.cpp)
    target_link_libraries(pinned_shared_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME pinned_shared_test COMMAND pinned_shared_test)
  endif()
  if(WIN32)   # the replicas' shared expert arena as a named section (src/core/pinned.cu): two arenas, a child process, the last handle
    add_executable(pinned_shared_win_test src/core/pinned_shared_win_test.cpp)
    target_include_directories(pinned_shared_win_test PRIVATE include)
    target_link_libraries(pinned_shared_win_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME pinned_shared_win_test COMMAND pinned_shared_win_test)
    set_tests_properties(pinned_shared_win_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
  endif()

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # #3602: the same reader under load - several clients, a tiny row cache, one in-flight slot, injected delays, rows on
  # every page phase, late-writer checks, the reader after a failed read. CPU and a temp file only; 77 = no direct reads here.
  add_executable(ple_reader_stress_test src/ngram/ple_reader_stress_test.cpp)
  target_include_directories(ple_reader_stress_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_stress_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_stress COMMAND ple_reader_stress_test)
  set_tests_properties(ple_reader_stress PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

  # 0.2.7: the PLE residency rule (table + expert arena + headroom); header-only, needs nothing
  add_executable(ple_residency_test src/core/ple_residency_test.cpp)
  target_include_directories(ple_residency_test PRIVATE include)
  add_test(NAME ple_residency_test COMMAND ple_residency_test)

  # #4474: the automatic expert cache's slot rule (the late allocation priced); header-only, needs nothing
  add_executable(auto_cache_test src/core/auto_cache_test.cpp)
  target_include_directories(auto_cache_test PRIVATE include)
  add_test(NAME auto_cache_test COMMAND auto_cache_test)

  # 0.2.7 (#3911): the qwen35 host sampler's penalties on a short (draft-vocab) logit row; header-only, CPU
  add_executable(qwen35_sampling_test src/qwen35/sampling_test.cpp)
  target_include_directories(qwen35_sampling_test PRIVATE include)
  add_test(NAME qwen35_sampling_test COMMAND qwen35_sampling_test)

  # The GPU test runs ON THE TARGET PC. `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  if(STRATA_ENABLE_CUDA)
    add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)
  else()
    add_test(NAME hip_device_selftest COMMAND strata-device --selftest)
  endif()

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/graph_registry_test.cu")
    add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
    target_include_directories(graph_registry_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(graph_registry_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME graph_registry_test COMMAND graph_registry_test)
  endif()

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/pinned_capture.cu")
    add_executable(pinned_capture bench/micro/pinned_capture.cu)
    target_include_directories(pinned_capture PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pinned_capture PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  if(STRATA_ENABLE_CUDA)
    set(STRATA_CUDA_TESTS ON)
  endif()
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
            src/spec/draft_source.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(arch_defaults_test src/core/arch_defaults_test.cpp src/core/arch_defaults.cpp)
  target_include_directories(arch_defaults_test PRIVATE include)
  add_test(NAME arch_defaults_test COMMAND arch_defaults_test)
  add_executable(nv_defaults_test src/core/nv_defaults_test.cpp src/core/nv_defaults.cpp)   # the sm_121 (GB10) table and its two off switches
  target_include_directories(nv_defaults_test PRIVATE include)
  add_test(NAME nv_defaults_test COMMAND nv_defaults_test)
  add_executable(kva_test src/prefill/kva_test.cpp)   # KVA: the switch, the automatic tail, the reasons a read is turned away
  target_include_directories(kva_test PRIVATE include)
  add_test(NAME kva_test COMMAND kva_test)
  add_executable(adapt_async_default_test src/core/adapt_async_default_test.cpp)
  target_include_directories(adapt_async_default_test PRIVATE include)
  add_test(NAME adapt_async_default_test COMMAND adapt_async_default_test)
  add_executable(uma_cache_test src/core/uma_cache_test.cpp)
  target_include_directories(uma_cache_test PRIVATE include)
  add_test(NAME uma_cache_test COMMAND uma_cache_test)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
  add_executable(spec_start_test src/spec/spec_start_test.cpp)
  target_include_directories(spec_start_test PRIVATE include)
  add_test(NAME spec_start_test COMMAND spec_start_test)
endif()
if(STRATA_BUILD_TESTS OR STRATA_BUILD_CONVERSATION_TESTS)
  add_executable(message_boundary_test src/program/message_boundary_test.cpp)
  target_include_directories(message_boundary_test PRIVATE include)
  add_test(NAME message_boundary_test COMMAND message_boundary_test)
  # a layer split's ring override and the lend slots sized with it: the RAM copy must keep every borrowed slot (CPU)
  add_executable(split_ring_test src/program/split_ring_test.cpp)
  target_include_directories(split_ring_test PRIVATE include)
  add_test(NAME split_ring_test COMMAND split_ring_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # --adapt-decay outside 0..1 falls back to 0.7 with a line (#1284; header-only, CPU)
  add_executable(adapt_decay_test src/program/adapt_decay_test.cpp)
  target_include_directories(adapt_decay_test PRIVATE include)
  add_test(NAME adapt_decay_test COMMAND adapt_decay_test)
  # the window limits (8 rows, or 16 in a wide CUDA / HIP build) and the pipelined-window probe (header-only, CPU)
  add_executable(window_limits_test src/core/window_limits_test.cpp)
  target_include_directories(window_limits_test PRIVATE include)
  add_test(NAME window_limits_test COMMAND window_limits_test)
  # which window-size graph is freed when the next one does not fit (#3702; header-only, CPU)
  add_executable(window_evict_test src/core/window_evict_test.cpp)
  target_include_directories(window_evict_test PRIVATE include)
  add_test(NAME window_evict_test COMMAND window_evict_test)
  # which path computes the draft layer's prompt K/V, and STRATA_SPLIT_MTP_BATCH's parsing (header-only, CPU)
  add_executable(draft_kv_plan_test src/program/draft_kv_plan_test.cpp)
  target_include_directories(draft_kv_plan_test PRIVATE include)
  add_test(NAME draft_kv_plan_test COMMAND draft_kv_plan_test)
  # coupled draft sampling's counter and penalty-ring arithmetic (header-only, CPU)
  add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
  target_include_directories(coupled_draft_test PRIVATE include)
  add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
  # probabilistic draft acceptance (STRATA_SPEC_PROB): rejection sampling preserves p - chi-square over 2M trials per case (CPU)
  add_executable(spec_prob_test src/core/spec_prob_test.cpp)
  target_include_directories(spec_prob_test PRIVATE include)
  add_test(NAME spec_prob_test COMMAND spec_prob_test)
  # gated proposal packets (STRATA_SPEC_PROB=1, pick gate): masked measure, NULL tail, root rescue (header-only, CPU)
  add_executable(gated_proposal_test src/core/gated_proposal_test.cpp)
  target_include_directories(gated_proposal_test PRIVATE include)
  add_test(NAME gated_proposal_test COMMAND gated_proposal_test)
  # the admission law of that packet, enumerated exactly (no Monte Carlo), in the three gate modes (CPU)
  add_executable(spec_admission_test src/core/spec_admission_test.cpp src/spec/draft_policy.cpp)
  target_include_directories(spec_admission_test PRIVATE include)
  add_test(NAME spec_admission_default COMMAND ${CMAKE_COMMAND} -E env --unset=STRATA_SPEC_PROB_GATE $<TARGET_FILE:spec_admission_test>)
  add_test(NAME spec_admission_pick COMMAND ${CMAKE_COMMAND} -E env STRATA_SPEC_PROB_GATE=pick $<TARGET_FILE:spec_admission_test> pick)
  add_test(NAME spec_admission_top COMMAND ${CMAKE_COMMAND} -E env STRATA_SPEC_PROB_GATE=top $<TARGET_FILE:spec_admission_test> top)
  add_test(NAME spec_admission_exact_top_only COMMAND ${CMAKE_COMMAND} -E env STRATA_SPEC_PROB_GATE=TOP $<TARGET_FILE:spec_admission_test> pick)
  # the per-card window cap of --draft dflash2 (header-only, CPU)
  add_executable(draft_cap_test src/core/draft_cap_test.cpp)
  target_include_directories(draft_cap_test PRIVATE include)
  add_test(NAME draft_cap_test COMMAND draft_cap_test)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test src/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  add_test(NAME bf16_bits_test COMMAND bf16_bits_test)
  # the batch window's row layout (a slot's group of proposed rows, the layer split's hand-off capacity; header-only, CPU)
  add_executable(batch_rows_test tests/core/batch_rows_test.cpp)
  target_include_directories(batch_rows_test PRIVATE include)
  add_test(NAME batch_rows_test COMMAND batch_rows_test)
  # how many pipelined slot groups --batch runs with: the 0.1.41 default, and --batch-mtp keeping it serial (header-only, CPU)
  add_executable(batch_groups_test src/program/batch_groups_test.cpp)
  target_include_directories(batch_groups_test PRIVATE include)
  add_test(NAME batch_groups_test COMMAND batch_groups_test)
endif()

if(STRATA_CPU_X86)
  add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
              src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/hadamard.cpp)
else()
  add_library(strata_kernels_cpu STATIC src/kernels/cpu/portable.cpp src/kernels/cpu/pool.cpp
              src/kernels/cpu/expert_layout.cpp src/kernels/cpu/hadamard.cpp)
endif()
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
# AVX-VNNI (Alder Lake, Sapphire Rapids and later): the AVX-2 expert kernels' vpdpwssd / vpdpbusd copies (q2_avx2.cpp,
# iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok().  GCC and Clang compile only those functions with
# target("avxvnni") - never -mavxvnni on a file, which would let the compiler use it anywhere - and MSVC takes the
# intrinsics under /arch:AVX2.  A compiler without them (GCC < 11, Clang < 12) builds the files without the copies.
include(CheckCXXSourceCompiles)
if(MSVC)
  set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
else()
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
endif()
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE AND NOT STRATA_CPU_X86)
    # arm64 (DGX Spark: Cortex-X925 + A725): ggml-cpu for any Armv8.2 CPU with dot product, FP16 and int8 matmul instead of the build
    # host's, so the engine built here does not depend on which core the compiler ran on.  Without it GGML_NATIVE OFF would leave
    # ggml-cpu at plain armv8-a, and a native pack's CPU experts would lose the NEON dot products.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(STRATA_ARM64_ARCH "armv8.2-a+dotprod+fp16+i8mm" CACHE STRING "arm64: ggml-cpu's -march for a STRATA_PORTABLE build")
    set(GGML_CPU_ARM_ARCH "${STRATA_ARM64_ARCH}" CACHE STRING "" FORCE)
  elseif(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
    # The DeepSeek engine (STRATA_BUILD_DSV41) builds ggml-cpu with BMI2 OFF: ggml's IQ1_S / IQ1_M dots use _pdep_u64 with it, and AMD
    # Zen 1/2 run PDEP in microcode (EPYC 7452, IQ1_M one-token dot 1.8 GB/s against 32 GB/s without it; IQ1_S 2.3 against 37). The
    # results are bit-identical either way (ds4_gg_bench --dump, types 19 and 29, 1-16 tokens). Other engines keep BMI2 until their
    # next rebuild is looked at: no Qwen package changes with this commit.
    if(STRATA_BUILD_DSV41)
      set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
    endif()
  elseif(STRATA_ISA_FLOOR)
    # the experimental older-CPU build (STRATA_ISA_FLOOR above): one ggml-cpu for the floor, ggml's "sandybridge"
    # feature set (SSE4.2 + AVX) or SSE4.2 alone - not the build host's, which may be newer than the PC it runs on.
    # From the Strata_Dirigo fork (rwkeyes).  ggml's defaults turn the rest on when GGML_NATIVE is off: all off here.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      set(GGML_AVX ON CACHE BOOL "" FORCE)
    else()
      set(GGML_AVX OFF CACHE BOOL "" FORCE)
    endif()
    foreach(_f GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
               GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp)
  if(STRATA_CPU_X86)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp
                                             src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/q2_avx1.cpp
                                             src/kernels/cpu/iq_avx1.cpp src/kernels/cpu/nvfp4_avx512.cpp src/kernels/cpu/nvfp4_avx2.cpp
                                             src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp src/kernels/cpu/tq_avx512.cpp)
  else()
    # the trellis dispatcher alone (the DeepSeek engine's CPU lane and Gyro): tq_cpu.cpp runs its scalar rows when the x86 units are
    # not compiled (the Metal build does the same); kq_scalar.cpp stands in for kq_avx2.cpp's latent lane, which the
    # pool and the latency path call on every host (Arm64 builds on Windows: those two calls were unresolved symbols otherwise)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/kq_scalar.cpp)
  endif()
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp
                                PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    # the router lookahead's AVX1 dot (older CPUs; called behind cpu_avx1_ok): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): 128-bit integer lanes,
    # no FMA/F16C, its own #error guard; called only behind cpu_avx1_ok() && !cpu_avx2_ok()
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): 128-bit SSSE3/SSE2 integer, SSE2 float, no FMA/F16C
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mavx512vpopcntdq;-mbmi2;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    # the trellis kernels: AVX2 + FMA, and AVX-512 F/BW/VL/VBMI/VNNI (exactly what cpu_avx512_ok() checks)
    set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512vbmi;-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): SSE4.2 + AVX with
    # AVX2/FMA/F16C explicitly off, so a -march=native build host cannot leak AVX2 code into the file that
    # runs on a Sandy Bridge Xeon
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): SSE4.2 + AVX with AVX2/FMA/F16C explicitly off, so a
    # -march=native build host cannot leak AVX2 code into the file that runs on a Sandy Bridge Xeon
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  endif()
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  if(STRATA_ISA_FLOOR STREQUAL "avx")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_AVX=1)
  elseif(STRATA_ISA_FLOOR STREQUAL "none")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_NONE=1)
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_BUILD_TESTS)
    # the Q5_1 block dequantizer (a Q5_1 PLE table) against ggml's, synthetic blocks: no model, no GPU
    add_executable(dequant_q5_1_test tests/core/dequant_q5_1_test.cpp)
    target_link_libraries(dequant_q5_1_test PRIVATE strata_artifact ggml-base)
    add_test(NAME dequant_q5_1_test COMMAND dequant_q5_1_test)
  endif()
  # the x86 kernels' parity programs and benches (to the matching endif below): not on arm64, which has none of those kernels
  if(STRATA_CPU_X86)
  # the router lookahead's dot on each rung this CPU has (AVX2, the older CPUs' AVX1) against a double reference
  add_executable(router_dot_parity src/kernels/cpu/router_dot_parity.cpp)
  target_link_libraries(router_dot_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    # a token's native expert rows are the same bits at every number of tokens that share its expert (window-invariant arithmetic)
    add_executable(native_arithmetic_parity src/kernels/cpu/native_arithmetic_parity.cpp)
    target_link_libraries(native_arithmetic_parity PRIVATE strata_kernels_cpu)
    add_test(NAME native_arithmetic_parity COMMAND native_arithmetic_parity)
    add_test(NAME native_arithmetic_parity_avx COMMAND ${CMAKE_COMMAND} -E env
             STRATA_FORCE_ISA=avx $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_kq256 COMMAND ${CMAKE_COMMAND} -E env
             STRATA_KQ256=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_mt1 COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ_MT_MIN=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_no512 COMMAND ${CMAKE_COMMAND} -E env STRATA_NO_IQ512=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_ggml COMMAND ${CMAKE_COMMAND} -E env STRATA_NO_IQ512=1 STRATA_NO_IQ256=1 STRATA_NO_IQ4NL=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_nvfp4_avx2 COMMAND ${CMAKE_COMMAND} -E env
             STRATA_NO_NVFP4_512=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_test(NAME native_arithmetic_parity_nvfp4_ggml COMMAND ${CMAKE_COMMAND} -E env
             STRATA_NO_NVFP4_512=1 STRATA_NO_NVFP4_256=1 $<TARGET_FILE:native_arithmetic_parity>)
    add_executable(pool_tasks_test src/kernels/cpu/pool_tasks_test.cpp)
    target_link_libraries(pool_tasks_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_tasks_test COMMAND pool_tasks_test)
    set_tests_properties(pool_tasks_test PROPERTIES TIMEOUT 180)
    add_test(NAME router_dot_parity COMMAND router_dot_parity)
    add_test(NAME router_dot_parity_avx COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_ISA=avx $<TARGET_FILE:router_dot_parity>)
  endif()
  # the AVX-2 expert kernels' variants (iq_avx2.cpp, q2_avx2.cpp) bit for bit against each other at 1..8 tokens and
  # against ggml-cpu's dot, on random blocks; --bench: ms per expert on one thread, the weights streamed from DRAM
  add_executable(iq_avx2_parity src/kernels/cpu/iq_avx2_parity.cpp)
  target_link_libraries(iq_avx2_parity PRIVATE strata_kernels_cpu)
  # the 512-bit i-quant rows against ggml-cpu's dot: the exact rows bit for bit at 1-4 tokens, the engine's multi-token rows to float
  # rounding, with the grid lookups as scalar loads and as gathers (SKIP without AVX-512)
  add_executable(iq512_gather_parity src/kernels/cpu/iq512_gather_parity.cpp)
  target_link_libraries(iq512_gather_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq512_gather_parity_scalar COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ_GATHER=0 $<TARGET_FILE:iq512_gather_parity>)
    add_test(NAME iq512_gather_parity_gather COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ_GATHER=1 $<TARGET_FILE:iq512_gather_parity>)
  endif()
  # MODEL-UPDATE: the pool's expert rows per format on T threads against the RAM bandwidth (a benchmark, no test)
  add_executable(kq_cpu_bench src/kernels/cpu/kq_cpu_bench.cpp)
  target_link_libraries(kq_cpu_bench PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq_avx2_parity COMMAND iq_avx2_parity)
    add_test(NAME iq_avx2_parity_iq3s_mt1 COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ3S_MT1=1 $<TARGET_FILE:iq_avx2_parity>)
  endif()
  # the older CPUs' AVX1 expert rows (iq_avx1.cpp) against ggml-cpu's dots and against their own one-token
  # rows at every width; --bench: ms per expert, the weights streamed from DRAM.  Runs on any AVX CPU:
  # on an AVX2 machine it checks the older-CPU path as an AVX-only machine would run it.
  add_executable(iq_avx1_parity src/kernels/cpu/iq_avx1_parity.cpp)
  target_link_libraries(iq_avx1_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq_avx1_parity COMMAND iq_avx1_parity)
  endif()
  # the AVX-2 Q8_K activation quantizer against ggml's, byte for byte (--bench: its time)
  add_executable(q8k_quant_parity src/kernels/cpu/q8k_quant_parity.cpp)
  target_link_libraries(q8k_quant_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME q8k_quant_parity COMMAND q8k_quant_parity)
    add_test(NAME q8k_quant_parity_avx COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_ISA=avx $<TARGET_FILE:q8k_quant_parity>)
  endif()
  endif()   # STRATA_CPU_X86
  set(_strata_mmq_types q2_0 q3_k iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)   # q3_k: the v7l1280 q3k latent pack
  # APR: the agention llama.cpp fork has MMQ instances for its trellis expert types (f62fdead0); any other ggml
  # tree leaves them to the FP16 prompt path
  if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqk6.cu")
    list(APPEND _strata_mmq_types tq2_t tqk6 tqk7)
    # Gyro-L's pair trellis (types 147-149): the same fork's ggml has their MMQ instances
    if(EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqp5.cu")
      list(APPEND _strata_mmq_types tqp5 tqp6 tqp7)
      set(_strata_mmq_tqp ON)   # moe_mmq.cu names GGML_TYPE_TQP5-7 only where the ggml has them (STRATA_MMQ_TQP)
    endif()
    set(_strata_mmq_tq ON)
  elseif(STRATA_MMQ_TQ_OVERLAY AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    # an upstream llama.cpp: the trellis MMQ instances come from an overlay of the pinned ggml-cuda headers written at
    # configure time (cmake/strata_ggml_tq_overlay.cmake); off with a message if the llama.cpp tree is not the pinned one
    if(STRATA_GGML_DIR)
      set(_strata_tq_llama "${STRATA_GGML_DIR}")
    else()
      set(_strata_tq_llama "${strata_llamacpp_SOURCE_DIR}")
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_ggml_tq_overlay.cmake)
    set(_strata_tq_overlay_dir "${CMAKE_BINARY_DIR}/ggml-tq-overlay")
    strata_ggml_tq_overlay("${_strata_tq_llama}" "${_strata_tq_overlay_dir}")
    if(STRATA_TQ_OVERLAY_OK)
      configure_file(third_party/ggml/ggml-common-tq.h "${_strata_tq_overlay_dir}/ggml-common-tq.h" COPYONLY)
      configure_file(src/kernels/cuda/tq_lut_i8.cuh "${_strata_tq_overlay_dir}/tq_lut_i8.cuh" COPYONLY)
      set(_strata_mmq_tq ON)
      set(_strata_mmq_tqp ON)   # the overlay adds the pair-trellis enumerators and instances
      set(_strata_mmq_tq_overlay ON)
      message(STATUS "trellis MMQ (Gyro prompts): overlay on the pinned ggml-cuda in ${_strata_tq_overlay_dir}")
      if(STRATA_BUILD_TESTS)
        add_test(NAME ggml_tq_overlay COMMAND ${CMAKE_COMMAND} -DLLAMA=${_strata_tq_llama}
                 -DOUT=${CMAKE_BINARY_DIR}/ggml-tq-overlay-test -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml_tq_overlay_test.cmake)
      endif()
    endif()
  endif()
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    set(_strata_quantize_cu ${_strata_mmq}/quantize.cu)   # the overlay's copy when the trellis overlay is on: it includes mmq.cuh
    if(_strata_mmq_tq_overlay)
      set(_strata_quantize_cu ${_strata_tq_overlay_dir}/quantize.cu)
    endif()
    # #136: the Q2_0 pack's prompt experts on Strata's own int8 kernels (moe_fused.cu, opt-in STRATA_PF_FUSED=1; CUDA
    # only: the HIP branch below keeps MMQ, and an sm_75 device or an sm_75-only build reports them unavailable)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                src/prefill/mmq_nvfp4_w4a8.cu
                src/prefill/ggml_cuda_host.cu ${_strata_quantize_cu})
    # M1 of the q3k latent A stage: the down half (IQ4_NL, K = 640) on moe_fused.cu's latent kernel structure
    target_sources(strata_mmq PRIVATE src/prefill/lat_q3k_down.cu)
    # M2a of the q3k latent A stage: the gate/up half (Q3_K, K = 640 features of gate and up)
    target_sources(strata_mmq PRIVATE src/prefill/lat_q3k_gu.cu)
    # M2b of the q3k latent A stage: SwiGLU + the q8_1 of H
    target_sources(strata_mmq PRIVATE src/prefill/lat_q3k_swiglu.cu)
    # M3 of the q3k latent A stage: the whole stage in one call (two launches, the down is lat_q3k_down's)
    target_sources(strata_mmq PRIVATE src/prefill/lat_q3k_fused.cu)
    # the v5 latent kernel copied whole with the q3k decode (moe_fused.cu's namespace): same target and
    # flags as its source file, which lives here too
    # the v5 latent kernel copied whole with the q3k decode (moe_fused.cu's namespace): same target and
    target_sources(strata_mmq PRIVATE src/prefill/moe_fused_q3k.cu)
    if(STRATA_EXPERIMENTAL_SM60 AND NOT STRATA_HIP_GFX906)
      # GP100 (sm_60): a resident layer's experts weight-stationary, one launch per product (moe_ws.hpp; opt-in at
      # run time, STRATA_MMQ_WS=1, and only where every visible GPU is cc 6.0)
      target_sources(strata_mmq PRIVATE src/prefill/moe_ws.cu src/prefill/moe_ws_gu_iq2.cu src/prefill/moe_ws_gu_iq3.cu
                     src/prefill/moe_ws_down.cu)
    endif()
    # q8_0: the drafter's matrices (E-9); STRATA_ORCA_Q4KS_MMQ adds q5_0 (#296, in _strata_mmq_types), and
    # STRATA_MMQ_KQUANTS q4_k q5_k q5_1 q6_k, the experts of Unsloth's UD-Q4/Q5/Q6_K_XL (docs/QUANTS.md; the dispatch
    # follows eddoursul/Strata 7706c1b).  MODEL-UPDATE Phase 1: on CUDA they are a library of their own, strata_kq
    # (below), loaded only for a model with these experts: the engine loads every kernel of its own binary at start
    # (CUDA_MODULE_LOADING=EAGER), and these four took ~3.8 MB of VRAM (3 expert slots of IQ3_XXS on a P100) from every
    # model.  gfx906 (HIP: code objects load on first use) keeps them in strata_mmq.
    set(_strata_mmq_cuda_types ${_strata_mmq_types})
    set(_strata_kq_types q4_k q5_k q5_1 q6_k)
    set(_strata_kq_dso OFF)
    if(STRATA_MMQ_KQUANTS AND STRATA_HIP_GFX906)
      list(APPEND _strata_mmq_cuda_types ${_strata_kq_types})
    elseif(STRATA_MMQ_KQUANTS)
      set(_strata_kq_dso ON)
    endif()
    foreach(_t IN LISTS _strata_mmq_cuda_types)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    if(_strata_mmq_tq_overlay)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        target_sources(strata_mmq PRIVATE ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      # first on the include path: the patched mmq.cuh / ggml.h shadow the pinned ones for this target only
      target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    endif()
    # NVFP4: W4A8 (default) in mmq_nvfp4_w4a8.cu with the plain architectures; the opt-in W4A4 needs RTX 50's FP4 MMA,
    # which exists only in the arch-specific target, so its one unit alone is built with 12x -> 12xa (the default
    # binary is unchanged and runs on sm_121; a card without that image falls back to W4A8 at run time)
    set(_strata_w4a4_archs ${CMAKE_CUDA_ARCHITECTURES})
    # `native` is first replaced by the detected architectures (e.g. 120-real), so the 12x -> 12xa rewrite below applies
    # to it too: with plain sm_120 ptxas refuses the FP4 MMA of this one unit (#4023)
    if("native" IN_LIST _strata_w4a4_archs AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
      list(REMOVE_ITEM _strata_w4a4_archs native)
      list(APPEND _strata_w4a4_archs ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    # (two steps: CMake before 3.29 rejects a \2 whose optional group did not match, "120" alone)
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_mmq_w4a4 STATIC src/prefill/mmq_nvfp4_w4a4.cu)
    set_target_properties(strata_mmq_w4a4 PROPERTIES CUDA_ARCHITECTURES "${_strata_w4a4_archs}")
    target_include_directories(strata_mmq_w4a4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq_w4a4 PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PRIVATE strata_mmq_w4a4)
    # STRATA_MMQ_TAB: the pointer-table copy of MMQ's scheduler mirrors mmq.cuh from mul_mat_q_process_tile through
    # mul_mat_q_case (struct mmq_args included).  A fingerprint of that text instead of a compile probe: nothing is
    # compiled at configure time, and a llama.cpp tree with another scheduler turns the feature off instead of
    # failing the build.  _strata_mmq_tab_known: the hashes moe_mmq_tab.cuh was verified against (3cf03257, the
    # fetched pin; the vendored third_party/llama.cpp tree has the same mmq.cuh).
    set(_strata_mmq_tab_on OFF)
    if(STRATA_MMQ_TAB AND NOT STRATA_HIP_GFX906)   # gfx906 compiles no .cu from here (its HIP glob): no tab at all
      set(_strata_mmq_tab_known 36e4dc716c504470ec09cc0852d29a6ac5eace257f6e5e72db3805ca4f043eb0)
      set(_strata_mmq_tab_hash "")
      if(EXISTS ${_strata_mmq}/mmq.cuh)
        file(READ ${_strata_mmq}/mmq.cuh _strata_mmq_src)
        string(REPLACE "\r\n" "\n" _strata_mmq_src "${_strata_mmq_src}")
        string(FIND "${_strata_mmq_src}" "static __device__ __forceinline__ void mul_mat_q_process_tile" _strata_tab_a)
        string(FIND "${_strata_mmq_src}" "#define DECL_MMQ_CASE" _strata_tab_b)
        if(_strata_tab_a GREATER -1 AND _strata_tab_b GREATER _strata_tab_a)
          math(EXPR _strata_tab_len "${_strata_tab_b} - ${_strata_tab_a}")
          string(SUBSTRING "${_strata_mmq_src}" ${_strata_tab_a} ${_strata_tab_len} _strata_tab_region)
          string(SHA256 _strata_mmq_tab_hash "${_strata_tab_region}")
        endif()
        unset(_strata_mmq_src)
        unset(_strata_tab_region)
      endif()
      if(_strata_mmq_tab_hash AND _strata_mmq_tab_hash IN_LIST _strata_mmq_tab_known)
        set(_strata_mmq_tab_on ON)
      else()
        message(STATUS "STRATA_MMQ_TAB off: mmq.cuh differs from the mirrored scheduler (${_strata_mmq_tab_hash})")
      endif()
    endif()
    if(_strata_mmq_tab_on)
      set(_strata_tab_cuh ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_mmq_tab.cuh)
      # the native packs' nine types, the ones prefill_mmq_tab_test checks bit for bit; the optional Q5_0 / K-quant
      # instances (STRATA_ORCA_Q4KS_MMQ, STRATA_MMQ_KQUANTS) get no pointer table and keep the gather
      set(_strata_mmq_tab_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
      foreach(_t IN LISTS _strata_mmq_tab_types)
        string(TOUPPER ${_t} _strata_tab_T)
        configure_file(src/prefill/mmq_tab_instance.cu.in ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu @ONLY)
        target_sources(strata_mmq PRIVATE ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu)
      endforeach()
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TAB=1)
      message(STATUS "STRATA_MMQ_TAB on: pointer-table MMQ for ${_strata_mmq_tab_types}")
    endif()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    # GGML's CUDA error reporting calls cuGetErrorString from the driver API.
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    if(TARGET CUDA::cuda_driver)   # not in the HIP (gfx906) build, whose CUDA::cudart is a stand-in
      target_link_libraries(strata_mmq PUBLIC CUDA::cuda_driver)
    endif()
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(_strata_kq_dso)
      # strata_kq: the K-quant MMQ instances with their own copy of ggml_cuda_host.cu's symbols and the static CUDA
      # runtime, next to the engine (libstrata_kq.so / strata_kq.dll); moe_mmq.cu opens it when a model asks for one
      # of these formats (STRATA_MMQ_KQ=1).  Without the file the engine runs them on the FP16 prompt path, as before.
      target_compile_definitions(strata_mmq PRIVATE STRATA_KQ_DSO=1)
      target_sources(strata_mmq PRIVATE src/prefill/mmq_kq_loader.cpp)
      if(NOT WIN32)
        target_link_libraries(strata_mmq PUBLIC ${CMAKE_DL_LIBS})
      endif()
      add_library(strata_kq SHARED src/prefill/mmq_kq_lib.cu src/prefill/ggml_cuda_host.cu)
      foreach(_t IN LISTS _strata_kq_types)
        target_sources(strata_kq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      target_include_directories(strata_kq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/include)
      target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
      if(MSVC)
        target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
      endif()
      set_target_properties(strata_kq PROPERTIES CUDA_RUNTIME_LIBRARY Static POSITION_INDEPENDENT_CODE ON
                            CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
      # ggml-base is static: position-independent so a shared library can hold it (the same code otherwise)
      set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
      target_link_libraries(strata_kq PRIVATE ggml-base)
      if(TARGET CUDA::cuda_driver)
        target_link_libraries(strata_kq PRIVATE CUDA::cuda_driver)
      endif()
      add_dependencies(strata_mmq strata_kq)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    if(_strata_mmq_tq)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQ=1)
    endif()
    if(_strata_mmq_tqp)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQP=1)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    if(STRATA_EXPERIMENTAL_SM60 AND NOT STRATA_HIP_GFX906)
      # the weight-stationary experts (moe_ws.hpp) are in strata_mmq: the prompt path may take them (STRATA_MMQ_WS=1)
      target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_WS=1)
    endif()
    if(STRATA_BUILD_TESTS)
      # #136: the fused int8 experts against a double-precision reference and the MMQ path (exit 77 below sm_80)
      add_executable(prefill_fused_moe_test tests/cuda/prefill_fused_moe_test.cpp)
      target_link_libraries(prefill_fused_moe_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_moe_test COMMAND prefill_fused_moe_test)
      set_tests_properties(prefill_fused_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # #136: the same for the native packs' formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL)
      add_executable(prefill_fused_iq_test tests/cuda/prefill_fused_iq_test.cpp)
      target_link_libraries(prefill_fused_iq_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_iq_test COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q2N=1 on its own (the test leaves STRATA_PF_FUSED unset): the native Q2_0 pack's Q2_0 / Q2_0
      # layers on sm_80+ (77 below sm_80 and without a card)
      add_test(NAME prefill_fused_iq_test_q2n COMMAND prefill_fused_iq_test "--only=Q2_0 / Q2_0" --unset-pf-fused)
      set_tests_properties(prefill_fused_iq_test_q2n PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q2N=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_NT=256 kernels (256 threads, two blocks a multiprocessor, two-pass weight loads) against the
      # same references; the batches of the test reach them through the Batch overload of experts_native (77 below sm_80)
      add_test(NAME prefill_fused_iq_test_nt256 COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_nt256 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_NT=256" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_F16=1 (f16 mma, rounding level) and STRATA_PF_FUSED_WS=1 (warp-specialized, the same F16 products;
      # about 101 KB of shared memory at its defaults, more than an sm_86 / sm_89 / sm_120 card's 99 KB, so it is set up on
      # sm_80 and sm_90 cards and the test skips (77) elsewhere)
      add_test(NAME prefill_fused_iq_test_f16 COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_f16 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_F16=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      add_test(NAME prefill_fused_iq_test_ws COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_ws PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_WS=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the MoE input quantized per token and scattered to its rows (mmq::quantize_scatter) against the per-row
      # gather: the same q8_1 bytes for every MMQ layout (--bench: both at 2K, 8K and 32K tokens)
      add_executable(prefill_mmq_scatter_test tests/cuda/prefill_mmq_scatter_test.cpp)
      target_link_libraries(prefill_mmq_scatter_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_scatter_test COMMAND prefill_mmq_scatter_test)
      set_tests_properties(prefill_mmq_scatter_test PROPERTIES TIMEOUT 120)
      if(STRATA_EXPERIMENTAL_SM60)
        # mmq::swiglu_quant: swiglu + the down product's q8_1 rows in one kernel against the two, byte for byte
        add_executable(prefill_swiglu_quant_test tests/cuda/prefill_swiglu_quant_test.cpp)
        target_link_libraries(prefill_swiglu_quant_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
        add_test(NAME prefill_swiglu_quant_test COMMAND prefill_swiglu_quant_test)
        set_tests_properties(prefill_swiglu_quant_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        # the weight-stationary experts (STRATA_MMQ_WS) against the per-group MMQ, byte for byte, on random experts; the
        # verdict is given on a compute capability 6.0 card only (77 elsewhere)
        add_executable(prefill_ws_parity_test tests/cuda/prefill_ws_parity_test.cpp)
        target_link_libraries(prefill_ws_parity_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
        add_test(NAME prefill_ws_parity_test COMMAND prefill_ws_parity_test)
        set_tests_properties(prefill_ws_parity_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      endif()
    endif()
    if(STRATA_BUILD_TESTS AND _strata_mmq_tab_on)
      # STRATA_MMQ_DIRECT: the pointer-table MMQ bit for bit against the gathered MMQ (exit 77 below sm_80)
      add_executable(prefill_mmq_tab_test tests/cuda/prefill_mmq_tab_test.cu)
      target_link_libraries(prefill_mmq_tab_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_tab_test COMMAND prefill_mmq_tab_test)
      set_tests_properties(prefill_mmq_tab_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
    if(STRATA_BUILD_TESTS AND STRATA_MMQ_KQUANTS)
      # the UD-Q4_K_XL expert formats through MMQ against ggml's dequantized weights (synthetic, a few seconds)
      add_executable(prefill_mmq_kquant_test tests/cuda/prefill_mmq_kquant_test.cpp)
      target_link_libraries(prefill_mmq_kquant_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_kquant_test COMMAND prefill_mmq_kquant_test)
      # the K-quant MMQ is opt-in at run time (STRATA_MMQ_KQ=1), and on CUDA it is strata_kq next to the test
      set_tests_properties(prefill_mmq_kquant_test PROPERTIES TIMEOUT 120 ENVIRONMENT STRATA_MMQ_KQ=1)
    endif()
    # the NVFP4 expert products (w4a8 / w4a4, STRATA_PREFILL_NVFP4) against a double-precision reference
    add_executable(mmq_nvfp4_parity src/prefill/mmq_nvfp4_parity.cpp)
    target_link_libraries(mmq_nvfp4_parity PRIVATE strata_mmq ggml-base CUDA::cudart)
    if(NOT MSVC)   # its FP16 rounding helper uses the F16C intrinsics (GCC/Clang need the target flag)
      target_compile_options(mmq_nvfp4_parity PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mf16c>)
    endif()
    # the routed-expert products (MMQ) timed at the prompt path's shapes (16 experts a launch)
    add_executable(lat_mmq_bench src/prefill/lat_mmq_bench.cpp)
    # one q3k expert through the verified MMQ products: the target the fused Q3_K dot must match
    add_executable(lat_fused_q3k_test src/prefill/lat_fused_q3k_test.cpp)
    target_link_libraries(lat_fused_q3k_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_fused_q3k_test PRIVATE include)
    # M1 of the q3k latent A stage: the down half against native_expert_grouped's own down (one unit test)
    add_executable(lat_q3k_down_test src/prefill/lat_q3k_down_test.cpp)
    target_link_libraries(lat_q3k_down_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_down_test PRIVATE include)
    # M2a: the Q3_K gate/up half against a double-precision host dequantization (+ the per-super-block decode gate)
    add_executable(lat_q3k_gu_test src/prefill/lat_q3k_gu_test.cpp)
    target_link_libraries(lat_q3k_gu_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_gu_test PRIVATE include)
    # M2b: SwiGLU + q8_1 of H against the engine's own quantizer rule on the host
    add_executable(lat_q3k_swiglu_test src/prefill/lat_q3k_swiglu_test.cpp)
    target_link_libraries(lat_q3k_swiglu_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_swiglu_test PRIVATE include)
    # M2c step 1: the gu half on the down half's four shapes (mine vs native gate/up only)
    add_executable(lat_q3k_gu_shape_test src/prefill/lat_q3k_gu_shape_test.cpp)
    target_link_libraries(lat_q3k_gu_shape_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_gu_shape_test PRIVATE include)
    # M3: the fused A stage against the serial chain (memcmp) and against native's whole step, four shapes
    add_executable(lat_q3k_fused_test src/prefill/lat_q3k_fused_test.cpp)
    add_executable(lat_cache_test src/prefill/lat_cache_test.cpp)
    target_link_libraries(lat_cache_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_cache_test PRIVATE include)
    target_link_libraries(lat_q3k_fused_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_fused_test PRIVATE include)
    # M3b: explicit per-group counts (nullptr = the old starts-derived path), scrambled starts, fused == serial
    add_executable(lat_q3k_counts_test src/prefill/lat_q3k_counts_test.cpp)
    target_link_libraries(lat_q3k_counts_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_counts_test PRIVATE include)
    # M4's unit gate: ent_tok / ent_dst (native_expert_grouped's two entry maps)
    add_executable(lat_q3k_maps_test src/prefill/lat_q3k_maps_test.cpp)
    target_link_libraries(lat_q3k_maps_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_maps_test PRIVATE include)
    # the engine's real A-stage shape (512 groups, 42450 entries, shared xl rows, non-monotonic off[])
    add_executable(lat_q3k_engine_shape_test src/prefill/lat_q3k_engine_shape_test.cpp)
    target_link_libraries(lat_q3k_engine_shape_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_engine_shape_test PRIVATE include)
    # L2a of the q3k acceptance sheet: one batch of kMaxBatch experts vs one launch a launch, same inputs
    add_executable(lat_q3k_batch_self_test src/prefill/lat_q3k_batch_self_test.cpp)
    target_link_libraries(lat_q3k_batch_self_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_batch_self_test PRIVATE include)
    # one group with thousands of rows (the engine's per-expert walk: the blockIdx.z fill is active)
    add_executable(lat_q3k_onegroup_test src/prefill/lat_q3k_onegroup_test.cpp)
    target_link_libraries(lat_q3k_onegroup_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_onegroup_test PRIVATE include)
    # the v5 kernel with the q3k decode, against the frozen chain (memcmp, two shapes)
    add_executable(lat_q3k_fusedv5_test src/prefill/lat_q3k_fusedv5_test.cpp)
    target_link_libraries(lat_q3k_fusedv5_test PRIVATE strata_mmq strata_kernels strata_kernels_cpu ggml-base CUDA::cudart)
    target_include_directories(lat_q3k_fusedv5_test PRIVATE include)
    # the latent A stage's per-layer row tables (host only, no CUDA)
    add_executable(lat_tables_test src/prefill/lat_tables_test.cpp)
    target_include_directories(lat_tables_test PRIVATE include)
    target_link_libraries(lat_mmq_bench PRIVATE strata_mmq ggml-base CUDA::cudart)
    add_executable(mmq_expert_bench src/prefill/mmq_expert_bench.cpp)
    target_link_libraries(mmq_expert_bench PRIVATE strata_mmq ggml-base CUDA::cudart)
  elseif(STRATA_PREFILL_MMQ)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    if(STRATA_GGML_DIR)
      set(_strata_gg "${STRATA_GGML_DIR}/ggml")
    else()
      set(_strata_gg "${strata_llamacpp_SOURCE_DIR}/ggml")
    endif()
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    set(_strata_quantize_cu "${_strata_mmq}/quantize.cu")   # the overlay's copy when the trellis overlay is on: it includes mmq.cuh
    if(_strata_mmq_tq_overlay)
      set(_strata_quantize_cu "${_strata_tq_overlay_dir}/quantize.cu")
    endif()
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    # Aurora: moe_fused.cu's gfx11 and moe_fused_iq.cu's gfx11 / gfx12 WMMA kernels for the prompt experts (opt-in STRATA_PF_FUSED=1)
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/moe_fused.cu
      src/prefill/moe_fused_iq.cu
      src/prefill/moe_f16_kq.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_quantize_cu}")
    # amd-hip port: the latent A stage sources of the CUDA branch.  On HIP the lat_q3k_* entry points are stubs that
    # exit ("CUDA-only"), so run with STRATA_LATENT_FUSED=0 STRATA_LATENT_FUSED_DU=0 STRATA_LAT_FUSED_Q3K=0; they are
    # compiled here so prefill.cpp links.
    set(_strata_lat_hip_sources
      src/prefill/lat_q3k_down.cu
      src/prefill/lat_q3k_gu.cu
      src/prefill/lat_q3k_swiglu.cu
      src/prefill/lat_q3k_fused.cu
      src/prefill/moe_fused_q3k.cu)
    list(APPEND _strata_mmq_sources ${_strata_lat_hip_sources})
    if(STRATA_MMQ_KQUANTS)
      # the same K-quant instances as the CUDA branch: on HIP they serve the dense GGUF projections
      # of Gemm::native (STRATA_DENSE_MMQ=1), the mixed-quant packs' attention/PLE tensors
      list(APPEND _strata_mmq_types q4_k q5_k q5_1 q6_k)
    endif()
    foreach(_t IN LISTS _strata_mmq_types)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    if(_strata_mmq_tq_overlay)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        list(APPEND _strata_mmq_sources "${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu")
      endforeach()
    endif()
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    # moe_fused.cu is Strata's own CUDA-shaped source: it takes the HIP compat headers, ggml's MMQ sources do not
    set_source_files_properties(src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu src/prefill/moe_f16_kq.cu
                                PROPERTIES INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    set_source_files_properties(${_strata_lat_hip_sources}
                                PROPERTIES INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")   # amd-hip port
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(_strata_mmq_tq_overlay)
      target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    endif()
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    if(_strata_mmq_tq)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQ=1)
    endif()
    if(_strata_mmq_tqp)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQP=1)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    # Aurora: the gfx11 fused experts against the FP64 reference and MMQ (tests/cuda's test; exit 77 off gfx11)
    add_executable(hip_prefill_fused_moe tests/cuda/prefill_fused_moe_test.cpp)
    target_include_directories(hip_prefill_fused_moe PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_moe PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_moe COMMAND hip_prefill_fused_moe)
    set_tests_properties(hip_prefill_fused_moe PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_prefill_fused_moe)
    add_executable(hip_prefill_fused_iq tests/cuda/prefill_fused_iq_test.cpp)
    target_include_directories(hip_prefill_fused_iq PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_iq PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_iq COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # the same binary on F16 weights (gfx11: the covered formats take moe_f16_kq.cu / native_f16_kernel; elsewhere it is the default run again)
    add_test(NAME hip_prefill_fused_iq_f16 COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_F16=1;STRATA_PF_F16_KQ=1")
    list(APPEND _strata_hip_tests hip_prefill_fused_iq)

    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  # the NVFP4 AVX-512 and AVX2 rows vs ggml-cpu's vec_dot, and their speed (CPU only; optional arg: a pack's experts.bin)
  add_executable(nvfp4_avx512_parity src/kernels/cpu/nvfp4_avx512_parity.cpp)
  target_link_libraries(nvfp4_avx512_parity PRIVATE strata_kernels_cpu)
  add_test(NAME nvfp4_cpu_parity COMMAND nvfp4_avx512_parity)   # random blocks; the AVX2 rows bit-equal to ggml's
  set_tests_properties(nvfp4_cpu_parity PROPERTIES TIMEOUT 120)
  if(STRATA_ENABLE_CUDA)   # the NVFP4 decode kernels vs ggml's reference
    add_executable(nvfp4_expert_gpu_parity src/kernels/nvfp4_expert_gpu_parity.cpp)
    target_link_libraries(nvfp4_expert_gpu_parity PRIVATE strata_kernels ggml-base CUDA::cudart)
    # the decode experts' weight bandwidth on random blobs of a format pair, and an output hash (bit-for-bit checks)
    add_executable(nvfp4_decode_bench src/kernels/nvfp4_decode_bench.cpp)
    target_link_libraries(nvfp4_decode_bench PRIVATE strata_kernels CUDA::cudart)

    target_sources(strata_engine PRIVATE src/core/latent_dense.cpp src/latent/latent_moe.cpp)

    # latent MoE layer0: D(Q8) -> A(native NVFP4) -> U(Q8) -> combine
    add_executable(latent_layer0_verify src/kernels/latent_layer0_verify.cpp)
    target_link_libraries(latent_layer0_verify PRIVATE strata_kernels ggml-base CUDA::cudart)

    add_executable(latent_layout_check src/kernels/latent_layout_check.cpp)
    target_link_libraries(latent_layout_check PRIVATE strata_kernels_cpu)

    add_executable(latent_source_check src/kernels/latent_source_check.cpp)
    target_link_libraries(latent_source_check PRIVATE strata_engine strata_kernels_cpu)

    add_executable(latent_dense_check src/kernels/latent_dense_check.cpp src/core/latent_dense.cpp)
    target_include_directories(latent_dense_check PRIVATE ${CMAKE_SOURCE_DIR}/include)

    add_executable(latent_layer_pack_verify src/kernels/latent_layer_pack_verify.cpp src/core/latent_dense.cpp)
    target_include_directories(latent_layer_pack_verify PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(latent_layer_pack_verify PRIVATE strata_engine strata_kernels strata_kernels_cpu CUDA::cudart)

    add_executable(latent_bench src/kernels/latent_bench.cpp src/core/latent_dense.cpp)
    target_include_directories(latent_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(latent_bench PRIVATE strata_engine strata_kernels strata_kernels_cpu CUDA::cudart)

    add_executable(latent_module_verify src/kernels/latent_module_verify.cpp src/latent/latent_moe.cpp src/core/latent_dense.cpp)
    target_include_directories(latent_module_verify PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(latent_module_verify PRIVATE strata_engine strata_kernels strata_kernels_cpu CUDA::cudart)

    add_executable(latent_u_check src/kernels/latent_u_check.cpp src/core/latent_dense.cpp)
    target_include_directories(latent_u_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(latent_u_check PRIVATE strata_engine strata_kernels CUDA::cudart)

    add_executable(latent_cpu_verify src/kernels/latent_cpu_verify.cpp src/core/latent_dense.cpp)
    target_include_directories(latent_cpu_verify PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(latent_cpu_verify PRIVATE strata_engine strata_kernels_cpu CUDA::cudart)
  endif()
  if(STRATA_ENABLE_HIP)   # amd-hip port: the CUDA block above adds these to the engine; HIP needs them too
    target_sources(strata_engine PRIVATE src/core/latent_dense.cpp src/latent/latent_moe.cpp)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    # strata_engine: the --load-check mode runs the engine's start-up checks (NativeDense, check_experts_gguf)
    target_link_libraries(native_expert_parity PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
      # STRATA_EXPERT_NIBBLE (include/strata/core/expert_nibble.hpp): the 4-bit gate/up expert slots against the
      # production kernels and the GGUF blobs, bit for bit, on a real model's layers (needs a GP100); --header: the
      # contract's host half, no GPU
      add_executable(iq_nibble_parity src/kernels/iq_nibble_parity.cpp)
      target_include_directories(iq_nibble_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(iq_nibble_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ggml-base ${_strata_gpu_runtime_target})
      if(STRATA_BUILD_TESTS)
        add_test(NAME iq_nibble_parity_header COMMAND iq_nibble_parity --header)
        # the synthetic experts of every gate/up x down format pair, no model (skips with 77 on a card that is not a GP100)
        add_test(NAME iq_nibble_parity_synthetic COMMAND iq_nibble_parity --synthetic-only)
        set_tests_properties(iq_nibble_parity_synthetic PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
      endif()
    endif()
    # MODEL-UPDATE: the dense Q8_0 GEMVs at the engine's shapes against the memory bandwidth (a benchmark, no test)
    add_executable(q8_gemv_bench src/kernels/q8_gemv_bench.cpp)
    target_include_directories(q8_gemv_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q8_gemv_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_CUDA)
      # EXPDEC (docs/EXPERTDEC.md): the grouped experts' default kernels against STRATA_EXPDEC=1 at T = 1..4, bit for bit
      # and in microseconds, on synthetic blobs read cold from DRAM
      add_executable(expdec_bench EXCLUDE_FROM_ALL src/kernels/expdec_bench.cpp)
      target_include_directories(expdec_bench PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(expdec_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    endif()
    if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
      # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
      # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
      # q4_K/q5_0 and q5_0/q8_0: a plain llama-quantize "_M" tier GGUF (not Unsloth's dynamic method) commonly
      # puts Q5_0 on down-projections (sometimes gate/up too) where this model's native_experts.txt had no GPU
      # kernel for it before; q5_0/q5_0 (both roles Q5_0) is left out of this list since stacking the same
      # lower-precision format on both roles pushes the GPU path over this test's 3% threshold (quantization
      # noise, not a kernel bug -- see the PR).
      # q4_0/q4_0 and q4_0/q4_1: a plain llama-quantize Q4_0 file (bartowski's Qwen3.8-Flash-Next-Q4_0: gate/up Q4_0,
      # down Q4_0 on 42 layers and Q4_1 on 6)
      # q6_K/q8_0: Unsloth UD-Q6_K_XL gate/up Q6_K with Q8_0 down (docs/UNSLOTH_Q6.md). Layer 2 is q8_0/q8_0, already listed.
      set(_strata_parity_pairs q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0 q4_0/q4_0 q4_0/q4_1)
      if(STRATA_Q6K_EXPERTS)
        list(APPEND _strata_parity_pairs q6_K/q8_0)
      else()   # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
        add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
        set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
      endif()
      foreach(_pair ${_strata_parity_pairs})
        string(REPLACE "/" "_" _name "${_pair}")
        add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
        set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
      endforeach()
      add_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
      # APR: the trellis expert types and the Hadamard activation rotation, when ggml comes from the agention fork
      # (the only tree with these types; marked by its ggml-cuda/tq.cuh)
      if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/tq.cuh")
        set(_strata_tq_pairs tqk6/tqk7 tq2_t/tq2_t)
        if(EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqp5.cu")   # Gyro-L's pair trellis
          list(APPEND _strata_tq_pairs tqp5/tqp5 tqp6/tqp6 tqp7/tqp7)
        endif()
        foreach(_pair ${_strata_tq_pairs})
          string(REPLACE "/" "_" _name "${_pair}")
          add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
          add_test(NAME native_expert_parity_hadamard_${_name} COMMAND native_expert_parity --synthetic-hadamard ${_pair})
          set_tests_properties(native_expert_parity_${_name} native_expert_parity_hadamard_${_name} PROPERTIES TIMEOUT 300)
        endforeach()
      endif()
      # #290: the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit
      add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    endif()
  endif()
  # #296's real-weight checks (OrcaRouter Q4_K_S files; docs/ORCA_Q4_K_S.md) - test programs, not part of a release
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
    add_executable(ple_q5_parity src/kernels/ple_q5_parity.cpp)
    target_link_libraries(ple_q5_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    add_executable(q5_projection_parity src/kernels/q5_projection_parity.cpp)
    target_include_directories(q5_projection_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q5_projection_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    # #291: the FP8 PLE table (tools/ple_fp8_pack.py) through PleTable vs the checkpoint's own values
    add_executable(ple_fp8_parity src/kernels/ple_fp8_parity.cpp)
    target_link_libraries(ple_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
endif()

if(STRATA_BUILD_TESTS)
  if(WIN32)
    add_executable(pool_affinity_test src/kernels/cpu/pool_affinity_test.cpp)
    target_link_libraries(pool_affinity_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pool_affinity_test src/kernels/cpu/pool_affinity_linux_test.cpp)
    target_link_libraries(pool_affinity_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  endif()
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)   # #93: it needs the device dequantizers and a GPU runtime
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact ${_strata_gpu_runtime_target})
  endif()

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)
  set_tests_properties(expert_multi_test PROPERTIES SKIP_RETURN_CODE 77)   # no AVX-512: skipped

  # PR #706: the opt-in AVX-2 Q2_0 bit-plane kernel (STRATA_Q2_BITPLANE=1) against the legacy one, bitwise alone/window
  if(STRATA_BUILD_TESTS AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64|X86")
    add_executable(q2_bitplane_parity src/kernels/cpu/q2_bitplane_parity.cpp)
    target_link_libraries(q2_bitplane_parity PRIVATE strata_kernels_cpu)
    add_test(NAME q2_bitplane_parity COMMAND q2_bitplane_parity)
    set_tests_properties(q2_bitplane_parity PROPERTIES ENVIRONMENT "STRATA_Q2_BITPLANE=1")
  endif()

  if(STRATA_NATIVE_EXPERTS)
    # APR port: a Hadamard-folded trellis expert (TQ2_T / TQK6 / TQK7) on the CPU path vs llama.cpp's graph for the
    # same weights (synthetic; needs STRATA_GGML_DIR = a ggml with the trellis types, else it reports and fails)
    add_executable(native_hadamard_test src/kernels/cpu/native_hadamard_test.cpp)
    target_link_libraries(native_hadamard_test PRIVATE strata_kernels_cpu strata_artifact)
    add_test(NAME native_hadamard_test COMMAND native_hadamard_test)
    set_tests_properties(native_hadamard_test PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    # the CPU trellis kernels (tq_cpu.hpp): AVX2 / AVX-512 vs the scalar kernel bit for bit, vs ggml-cpu's vec_dot
    # and to_float, plain and with the Hadamard rotation; `tq_cpu_parity --bench` is the micro-benchmark
    add_executable(tq_cpu_parity src/kernels/cpu/tq_cpu_parity.cpp)
    target_link_libraries(tq_cpu_parity PRIVATE strata_kernels_cpu)
    add_test(NAME tq_cpu_parity COMMAND tq_cpu_parity)
    add_test(NAME tq_cpu_parity_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 $<TARGET_FILE:tq_cpu_parity>)
    set_tests_properties(tq_cpu_parity tq_cpu_parity_avx2 PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    # Gyro-L's pair trellis (types 147-149) and the older trellis rows on synthetic blocks, no ggml and no model: the scalar
    # kernel against a bit-by-bit reference, every ISA bit for bit equal to scalar, the type predicates unchanged
    add_executable(tqp_cpu_test src/kernels/cpu/tqp_cpu_test.cpp)
    target_link_libraries(tqp_cpu_test PRIVATE strata_kernels_cpu)
    add_test(NAME tqp_cpu_test COMMAND tqp_cpu_test)
    add_test(NAME tqp_cpu_test_avx2 COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 $<TARGET_FILE:tqp_cpu_test>)
    set_tests_properties(tqp_cpu_test tqp_cpu_test_avx2 PROPERTIES TIMEOUT 120)
  endif()

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # where the workers are placed (one per physical core by default) and --pool-cores (real pools, no model)
  add_executable(pool_cores_test src/kernels/cpu/pool_cores_test.cpp)
  target_link_libraries(pool_cores_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_cores_test COMMAND pool_cores_test)
  # the default CPU expert worker count on mock CPUs (8c/16t, 16c/32t, 64c/128t, 2-socket 128c/256t, ...)
  add_executable(pool_plan_test src/kernels/cpu/pool_plan_test.cpp)
  target_include_directories(pool_plan_test PRIVATE include)
  add_test(NAME pool_plan_test COMMAND pool_plan_test)
  # 0.2.2: when the pool splits per NUMA node by itself (a fake sysfs node folder, mock cores; STRATA_NUMA_POOL=0, one node = unchanged)
  add_executable(numa_pool_test src/kernels/cpu/numa_pool_test.cpp)
  target_include_directories(numa_pool_test PRIVATE include)
  target_compile_features(numa_pool_test PRIVATE cxx_std_17)
  add_test(NAME numa_pool_test COMMAND numa_pool_test)
  # #3204: when the HIP start-up warm-up of hipBLAS may run (pure policy, no GPU)
  add_executable(warm_policy_test src/prefill/warm_policy_test.cpp)
  target_include_directories(warm_policy_test PRIVATE include)
  add_test(NAME warm_policy_test COMMAND warm_policy_test)
  # #4005: which streamed expert copies get a copied event (header-only, CPU)
  add_executable(copy_events_test src/prefill/copy_events_test.cpp)
  target_include_directories(copy_events_test PRIVATE include)
  add_test(NAME copy_events_test COMMAND copy_events_test)
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)

  # #795: nothing an AVX2-only CPU can reach may hold AVX-512 / AVX-VNNI-class instructions.  (1) the linked test binary
  # (every CPU kernel file plus ggml-cpu; on Linux, the kernel library) is disassembled and any function with such instructions must be a kernel of
  # a wide-ISA file, not a weak / COMDAT copy and not a probe or helper that every CPU runs; (2) with -DSTRATA_SDE=<sde>
  # (Intel SDE) the CPU tests run on an emulated AVX2-only Haswell and on an Alder Lake (AVX-VNNI, no AVX-512): SDE stops
  # at the first instruction the chip does not have.
  if(TARGET iq_avx2_parity)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      if(MSVC)
        target_link_options(iq_avx2_parity PRIVATE /MAP)
        add_test(NAME isa_guard_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_isa_guard.py
                 $<TARGET_FILE:iq_avx2_parity> --map $<TARGET_FILE_DIR:iq_avx2_parity>/iq_avx2_parity.map)
      else()
        # the library, not the binary: a Linux source build compiles ggml-cpu for the build machine on purpose
        add_test(NAME isa_guard_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_isa_guard.py
                 $<TARGET_FILE:strata_kernels_cpu>)
      endif()
      set_tests_properties(isa_guard_check PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
  endif()
  set(STRATA_SDE "" CACHE FILEPATH "Intel SDE (sde / sde64): also run the CPU tests on an emulated AVX2-only and an AVX-VNNI-only CPU")
  if(STRATA_SDE AND TARGET iq_avx2_parity)
    foreach(_chip hsw adl)
      foreach(_t iq_avx2_parity q8k_quant_parity router_dot_parity pool_tasks_test q2_bitplane_parity)
        if(TARGET ${_t})
          add_test(NAME sde_${_chip}_${_t} COMMAND ${STRATA_SDE} -${_chip} -- $<TARGET_FILE:${_t}>)
          set_tests_properties(sde_${_chip}_${_t} PROPERTIES TIMEOUT 900)
        endif()
      endforeach()
      if(TARGET q2_bitplane_parity)
        set_tests_properties(sde_${_chip}_q2_bitplane_parity PROPERTIES ENVIRONMENT "STRATA_Q2_BITPLANE=1")
      endif()
    endforeach()
  endif()

  if(UNIX AND NOT APPLE)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp src/platform/direct_file.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(direct_file_async_test PRIVATE Threads::Threads)
    add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
  # --aux-cpus: the CPU-set derivation (pure) and the placement of live threads; header-only, no GPU
  add_executable(aux_cpus_test tests/platform/aux_cpus_test.cpp)
  target_include_directories(aux_cpus_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(aux_cpus_test PRIVATE Threads::Threads)
  add_test(NAME aux_cpus_test COMMAND aux_cpus_test)
endif()

# CUDA release kernels must not spill or keep a per-thread array in local memory: every access there is a round trip
# to device memory in the hottest loops the engine has, and nothing on the build machine says so without -Xptxas -v
# or a GPU.  tools/check_cuda_spills.py reads the resource usage the compiler wrote into the kernel libraries
# (cuobjdump: STACK and LOCAL per entry function and per sm_) and fails on any kernel outside its ALLOWED table, which
# carries the measured allowance of the kernels that hold such an array on purpose.  CUDA only: HIP has no cuobjdump.
if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    set(_strata_spill_libs)
    foreach(_t strata_kernels strata_core strata_engine strata_prefill strata_qwen35 strata_mmq strata_mmq_w4a4 strata_kq)
      if(TARGET ${_t})
        list(APPEND _strata_spill_libs $<TARGET_FILE:${_t}>)
      endif()
    endforeach()
    add_test(NAME cuda_spill_guard COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_cuda_spills.py
             --cuobjdump ${CUDAToolkit_BIN_DIR}/cuobjdump${CMAKE_EXECUTABLE_SUFFIX} ${_strata_spill_libs})
    set_tests_properties(cuda_spill_guard PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    if(NOT WIN32)   # its stand-in demanglers are sh scripts
      add_test(NAME check_cuda_spills_unit COMMAND ${Python3_EXECUTABLE} -m unittest tools.test_check_cuda_spills
               WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
  else()
    message(STATUS "strata: no python3 - the cuda_spill_guard test is NOT registered")
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---- HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) is compiled as HIP ----
if(STRATA_HIP_GFX906)
  file(GLOB_RECURSE _strata_cu CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/*.cu
       ${CMAKE_CURRENT_SOURCE_DIR}/tools/vision/*.cu)
  if(_strata_mmq)
    file(GLOB _strata_mmq_cu ${_strata_mmq}/*.cu ${_strata_mmq}/template-instances/*.cu)
    list(APPEND _strata_cu ${_strata_mmq_cu})
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP)
    # #3172: strata_mmq_w4a4 is built from llama.cpp's headers too; without GGML_USE_HIP its common.cuh took the CUDA vendor header
    # (cuda.h, cuda_bf16.h: not found). Here it only reports that W4A4 (RTX 50's FP4 MMA) is not available.
    if(TARGET strata_mmq_w4a4)
      target_compile_definitions(strata_mmq_w4a4 PRIVATE GGML_USE_HIP)
    endif()
    # #3271: three more link errors of a 0.1.44+ gfx906 build, all sources that strata_mmq lists but the glob above never saw:
    # the trellis overlay's quantize.cu and instances (they are generated into the build tree, so the glob missed them: CMake drops a
    # .cu source whose language is not enabled) and moe_f16_kq.cu, which moe_fused_iq.cu calls (a stub here: no gfx11 device code, so
    # f16kq::covers() answers false and the int8 kernels run, as on every other card)
    if(_strata_mmq_tq_overlay)
      set(_strata_tq_hip ${_strata_tq_overlay_dir}/quantize.cu)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        list(APPEND _strata_tq_hip ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      list(APPEND _strata_cu ${_strata_tq_hip})
    endif()
    target_sources(strata_mmq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_f16_kq.cu)
  endif()
  set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)
endif()

if(STRATA_HIP_GFX906)
  add_executable(gfx906_qsa_query_swizzle_probe bench/gfx906_qsa_query_swizzle_probe.cpp)
  target_link_libraries(gfx906_qsa_query_swizzle_probe PRIVATE strata_kernels ${_strata_gpu_runtime_target})
endif()

# ---- DeepSeek-V4.1 on a box with no GPU toolkit ------------------------------------------------
# The whole DSV41 block above lives inside if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP), so with neither backend on
# -DSTRATA_BUILD_DSV41=ON was silently unused and the host-side benchmark could not be built at all (found on p1004,
# 2026-10-23: the CPU lane is exactly what a box without a usable GPU needs to measure).  This branch builds the one target
# that has no device code, so the CPU-lane A/B runs anywhere.
if(STRATA_BUILD_DSV41 AND NOT STRATA_ENABLE_CUDA AND NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(STATUS "STRATA_BUILD_DSV41: no GPU backend, building the host-side benchmark only (ds4_gg_bench)")
  if(STRATA_NATIVE_EXPERTS)
    add_executable(ds4_gg_bench tests/dsv41/gg_dot_bench.cpp src/dsv41/ds4_gg_cpu.cpp)
    target_include_directories(ds4_gg_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41
                               ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
    target_link_libraries(ds4_gg_bench PRIVATE strata_kernels_cpu Threads::Threads)
    target_compile_definitions(ds4_gg_bench PRIVATE STRATA_DSV41_TQ_CPU=1)
  endif()
endif()
