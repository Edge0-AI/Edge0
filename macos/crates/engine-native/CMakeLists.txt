# edge0 native inference engine (C++/Metal on top of mlx::core).
# Built by the top-level Makefile (`make engine`); not part of the cargo workspace.
cmake_minimum_required(VERSION 3.24)
project(edge0_engine_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(E0N_VERSION "0.1.0" CACHE STRING "edge0 product version embedded in the native engine")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)

# ---- mlx::core (vendored, v0.30.6). mlx defaults to a static build; the engine needs the dylib.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
# mlx 0.30.6 turns on NAX GEMM/qmm on Apple M5 (arch gen >= 17). Those kernels
# skew edge0-8b first-token logits (fixture cosine ~0.85 vs ~0.9999 with NAX
# off). The define must be set before add_subdirectory so both mlx and this
# tree see it — device.h inlines is_nax_available().
option(E0N_MLX_NAX "Enable mlx NAX kernels (numerically wrong for edge0-8b on M5)" OFF)
if(NOT E0N_MLX_NAX)
  add_compile_definitions(MLX_METAL_NO_NAX)
endif()
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/mlx
  ${CMAKE_BINARY_DIR}/mlx
  EXCLUDE_FROM_ALL)

# nlohmann_json is reused from the copy mlx already fetches.

set(EDGE0_ABI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Tokenizer: Rust staticlib from crates/engine-tok.
set(EDGE0_TOK_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../engine-tok/Cargo.toml)
file(GLOB EDGE0_TOK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../engine-tok/src/*.rs)
string(TOLOWER ${CMAKE_BUILD_TYPE} EDGE0_TOK_PROFILE)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/rust-${EDGE0_TOK_PROFILE}/libedge0_engine_tok.a
  COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/rust
          cargo build --manifest-path ${EDGE0_TOK_MANIFEST} --${EDGE0_TOK_PROFILE}
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_BINARY_DIR}/rust/${EDGE0_TOK_PROFILE}/libedge0_engine_tok.a
          ${CMAKE_BINARY_DIR}/rust-${EDGE0_TOK_PROFILE}/libedge0_engine_tok.a
  DEPENDS ${EDGE0_TOK_SRCS} ${EDGE0_TOK_MANIFEST}
  COMMENT "engine-tok (rust staticlib, ${EDGE0_TOK_PROFILE})")
add_custom_target(engine_tok_rust
  DEPENDS ${CMAKE_BINARY_DIR}/rust-${EDGE0_TOK_PROFILE}/libedge0_engine_tok.a)
add_library(edge0_engine_tok STATIC IMPORTED GLOBAL)
add_dependencies(edge0_engine_tok engine_tok_rust)
set_target_properties(edge0_engine_tok PROPERTIES
  IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/rust-${EDGE0_TOK_PROFILE}/libedge0_engine_tok.a)

# ---- Engine core.
add_library(engine_core STATIC
  src/memstat.cpp
  src/safio.cpp
  src/expertmap.cpp
  src/hotstack.cpp
  src/routing_replay.cpp
  src/stepprefetch.cpp
  src/qgemm.cpp
  src/router.cpp
  src/registry.cpp
  src/dense.cpp
  src/prerouter.cpp
  src/fwd_common.cpp
  src/lora.cpp
  src/sampling.cpp
  src/expertio.cpp
  src/net35.cpp
  src/net8b.cpp)
target_include_directories(engine_core PUBLIC src ${EDGE0_ABI_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/_deps/metal_cpp-src)
target_link_libraries(engine_core PUBLIC mlx nlohmann_json::nlohmann_json)

# ---- Engine dylib exporting every symbol in engine-abi.h.
add_library(edge0_engine_native SHARED src/abiface.cpp)
target_compile_definitions(edge0_engine_native PRIVATE E0N_VERSION=\"${E0N_VERSION}\")
target_link_libraries(edge0_engine_native PRIVATE engine_core edge0_engine_tok)
set_target_properties(edge0_engine_native PROPERTIES
  OUTPUT_NAME edge0_engine_native
  BUILD_RPATH "@loader_path"
  BUILD_RPATH_USE_ORIGIN ON
  INSTALL_RPATH "@loader_path"
  MACOSX_RPATH ON)
