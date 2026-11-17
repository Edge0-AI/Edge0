# Opt-in HIP configuration. Strata's CUDA-shaped kernels target wave32 RDNA2 / RDNA3 / RDNA4 (64 KiB LDS per
# workgroup, a signed dot4 instruction). CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it
# is not on PATH.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")
endif()
# Validated on real cards: gfx1100 (RX 7900 XT / XTX) and gfx1201 (RX 9070 / 9070 XT, Radeon AI PRO R9700) by the
# maintainers; gfx1101 (RX 7800 XT, #254) and gfx1200 (RX 9060 XT, #256) by their owners. gfx1102 (RX 7600) has the
# same LDS limit and dot4 instruction and passed ctest (#192), but no model run has been reported yet. RDNA2 gfx1030 (RX 6800 / 6900) has the
# same LDS limit and wave32 but an older dot4 instruction (v_dot4_i32_i8, hip_compat/intrinsics.hpp); a community
# report ran it (#311), the maintainers have not. gfx1151 (Ryzen AI Max / Strix Halo, Radeon 8060S, RDNA3.5) is the
# same wave32 / 64 KiB LDS / sudot4 family as gfx1100, on a unified-memory APU; experimental.
set(_strata_hip_validated gfx1100 gfx1201)
set(_strata_hip_community gfx1101 gfx1200)
set(_strata_hip_unvalidated gfx1010 gfx1011 gfx1012 gfx1102 gfx1103 gfx1030 gfx1031 gfx1032 gfx1034 gfx1150 gfx1151 gfx1152)   # gfx1103: Radeon 780M iGPU, opt-in, WMMA kernels like gfx1100; gfx1152: Krackan, Radeon 860M / 840M (#1625), builds with the portable kernels, no card report yet
# CMake hands HIP a ';' list, but a -DCMAKE_HIP_ARCHITECTURES typed by hand (or ROCm's own Windows tooling) may use
# spaces, which foreach(IN LISTS) would otherwise treat as one element.
string(REPLACE " " ";" _strata_hip_norm "${CMAKE_HIP_ARCHITECTURES}")
set(STRATA_HIP_ARCH_LIST "")
foreach(_arch IN LISTS _strata_hip_norm)
  if(_arch STREQUAL "")
    continue()
  endif()
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")      # gfx1100:xnack- -> gfx1100
  if(_base IN_LIST _strata_hip_validated)
  elseif(_base IN_LIST _strata_hip_community)
    message(STATUS "Strata HIP: ${_base} was validated by community reports (docs/AMD_HIP.md)")
  elseif(_base IN_LIST _strata_hip_unvalidated)
    message(WARNING "Strata HIP: ${_base} builds, but it is not validated on a real card yet; please report results")
  else()
    message(FATAL_ERROR
      "Strata HIP supports wave32 gfx1100, gfx1101, gfx1200 and gfx1201 (unvalidated: ${_strata_hip_unvalidated}); "
      "CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
  list(APPEND STRATA_HIP_ARCH_LIST "${_base}")
endforeach()
list(REMOVE_DUPLICATES STRATA_HIP_ARCH_LIST)
if(NOT STRATA_HIP_ARCH_LIST)
  message(FATAL_ERROR "Strata HIP: CMAKE_HIP_ARCHITECTURES is empty")
endif()
# The compiled architectures reach the runtime device check (src/core/device.cu) as "gfx1100,gfx1201": a binary
# carried to a card it has no code for stops at startup with a clear message instead of "invalid device function".
string(REPLACE ";" "," STRATA_HIP_ARCHS "${STRATA_HIP_ARCH_LIST}")

enable_language(HIP)
# CMake caches CMAKE_HIP_FLAGS_<CONFIG> while it enables HIP, before it tests the compiler.  A configure that stopped
# there because HIP could not compile at all (Visual Studio 2026's <cmath> with the HIP SDK 7.2, a wrong ROCm path)
# cached them empty, and every later configure of that build folder keeps them: setup's Release build then compiled
# all kernels at -O0 (an RX 6800 decoded at 0.24-0.27 tok/s instead of 28).  Empty ones get what CMake gives a fresh
# build folder.
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(STRIP "${CMAKE_HIP_FLAGS_${_cfg}_INIT}" _strata_init)
  if("${CMAKE_HIP_FLAGS_${_cfg}}" STREQUAL "" AND NOT "${_strata_init}" STREQUAL "")
    set(CMAKE_HIP_FLAGS_${_cfg} "${_strata_init}" CACHE STRING
        "Flags used by the HIP compiler during ${_cfg} builds." FORCE)
  endif()
endforeach()
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
# Older distro hipBLAS has no workspace API. The compatibility shim uses
# rocBLAS directly for that version; newer hipBLAS keeps its existing path.
set(STRATA_HIP_BLAS_TARGETS roc::hipblas)
if(hipblas_VERSION VERSION_LESS "1.0")
  find_package(rocblas CONFIG REQUIRED)
  list(APPEND STRATA_HIP_BLAS_TARGETS roc::rocblas)
else()
  # the prompt path's rocBLAS solution table (STRATA_ROCBLAS_TUNING, gfx103x) calls rocBLAS by name
  find_package(rocblas CONFIG QUIET)
endif()
find_package(hipblaslt CONFIG QUIET)

if(NOT TARGET hip::host)
  message(FATAL_ERROR "The ROCm hip CMake package did not provide hip::host")
endif()
if(NOT TARGET roc::hipblas)
  message(FATAL_ERROR "The ROCm hipblas CMake package did not provide roc::hipblas")
endif()
if(TARGET roc::hipblaslt)
  set(STRATA_HIPBLASLT_AVAILABLE ON)
else()
  set(STRATA_HIPBLASLT_AVAILABLE OFF)
  message(STATUS "Strata: hipBLASLt not found; solution-table dispatch is unavailable")
endif()
if(TARGET roc::rocblas)
  set(STRATA_ROCBLAS_AVAILABLE ON)
else()
  set(STRATA_ROCBLAS_AVAILABLE OFF)
  message(STATUS "Strata: rocBLAS CMake package not found; the rocBLAS solution table (gfx103x) is unavailable")
endif()

# HIP's link step produces a PIE; make Strata and ggml objects PIC for the ROCm linker.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(STRATA_HIP_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat")
add_library(strata_hip_runtime INTERFACE)
target_include_directories(strata_hip_runtime BEFORE INTERFACE
  "${STRATA_HIP_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_hip_runtime INTERFACE STRATA_USE_HIP=1 "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\"")
option(STRATA_GFX1012_PORTABLE_DOT "Use the portable signed-byte dot control on gfx1012" OFF)
if(STRATA_GFX1012_PORTABLE_DOT)
  target_compile_definitions(strata_hip_runtime INTERFACE STRATA_GFX1012_PORTABLE_DOT=1)
endif()
# #313: the RDNA3 WMMA GEMM's intrinsics only compile for gfx11 targets and the host pass does not define
# __gfx11xx__, so the build supplies the macro for gfx11 configs (the kernels also gate on the device at runtime)
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx11")
  target_compile_definitions(strata_hip_runtime INTERFACE STRATA_WMMA_GFX11=1)
endif()
target_link_libraries(strata_hip_runtime INTERFACE hip::host)
# The shim renames the CUDA runtime to HIP, force-included into every host and device source. On Windows the host
# compiler is ROCm's clang++ too (tools/hip/build_windows.bat: CMake refuses to mix cl.exe with Clang HIP), which takes
# -include like it does on Linux; an MSVC-style front end (cl / clang-cl) takes /FI instead. Forward slashes, so /FI
# does not read the path's backslashes as escapes.
file(TO_CMAKE_PATH "${STRATA_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h" _strata_hip_force)
if(MSVC)
  target_compile_options(strata_hip_runtime INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/FI${_strata_hip_force}>")
else()
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-include>" "$<$<COMPILE_LANGUAGE:CXX>:${_strata_hip_force}>")
endif()
target_compile_options(strata_hip_runtime INTERFACE
  "$<$<COMPILE_LANGUAGE:HIP>:-include>" "$<$<COMPILE_LANGUAGE:HIP>:${_strata_hip_force}>")

# CMake does not infer HIP from Strata's existing CUDA-shaped .cu suffixes.
file(GLOB_RECURSE _strata_hip_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
if(_strata_hip_sources)
  set_source_files_properties(${_strata_hip_sources} PROPERTIES LANGUAGE HIP)
endif()
foreach(_source IN ITEMS tests/hip/intrinsics.cpp tests/hip/native_qsa_score.cpp)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    set_source_files_properties("${_source}" PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

message(STATUS "Strata: HIP enabled, arch ${STRATA_HIP_ARCHS}")
