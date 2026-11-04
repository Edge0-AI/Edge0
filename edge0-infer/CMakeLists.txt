cmake_minimum_required(VERSION 3.21)
# The android-cpu preset locates the NDK toolchain through $ANDROID_NDK. A host project that adds
# this tree with add_subdirectory() has already loaded its own toolchain, so only check when this
# is the top-level source tree (PROJECT_IS_TOP_LEVEL is not available before project()).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND
   CMAKE_TOOLCHAIN_FILE MATCHES "android.toolchain.cmake$" AND "$ENV{ANDROID_NDK}" STREQUAL "")
    message(FATAL_ERROR "ANDROID_NDK must point to the Android NDK")
endif()
project(edge0 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Everything below is usable from a host project through add_subdirectory(): the CLI, tests and
# examples default to OFF there, and no output layout is imposed on the host.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
option(MOE_BUILD_CLI "Build the reference CLI" ${PROJECT_IS_TOP_LEVEL})
option(MOE_BUILD_TESTS "Build and register tests" ${PROJECT_IS_TOP_LEVEL})
option(MOE_BUILD_EXAMPLES "Build public native examples" ${PROJECT_IS_TOP_LEVEL})
include(cmake/MoeTargets.cmake)
include(cmake/BuildIdentity.cmake)
file(STRINGS "${PROJECT_SOURCE_DIR}/third_party/GGML_ORIGIN" origin)
foreach(line IN LISTS origin)
    if(line MATCHES "^commit: (.+)$")
        set(MOE_GGML_COMMIT "${CMAKE_MATCH_1}")
    elseif(line MATCHES "^ggml_version: (.+)$")
        set(MOE_GGML_VERSION "${CMAKE_MATCH_1}")
    endif()
endforeach()
if(NOT PROJECT_IS_TOP_LEVEL)
    # The engine is built and tested against static GGML (the presets set BUILD_SHARED_LIBS=OFF).
    # Without this, GGML's own option() would default BUILD_SHARED_LIBS to ON and store it in the
    # host's cache, turning the host's untyped add_library() calls into shared libraries. This
    # normal variable is scoped to this directory and does not reach the host.
    set(BUILD_SHARED_LIBS OFF)
    # Without presets, default the GGML backend options to the values of the matching preset.
    include(cmake/GgmlDefaults.cmake)
endif()
add_subdirectory(third_party/ggml)
add_subdirectory(src/platform)
add_subdirectory(src/exec)
add_subdirectory(src/weights)
add_subdirectory(src/session)
add_subdirectory(src/models/api)
add_subdirectory(src/models/edge0_35b)
add_subdirectory(src/model)
add_subdirectory(src/app)
add_library(edge0 STATIC cmake/aggregate.cpp)
add_library(edge0::edge0 ALIAS edge0)
target_include_directories(edge0 PUBLIC "${PROJECT_SOURCE_DIR}/include")
# The public headers use C++17 (std::optional, std::variant); consumers inherit the requirement.
target_compile_features(edge0 PUBLIC cxx_std_17)
target_link_libraries(edge0 PRIVATE moe_app moe_model moe_models_api moe_models_edge0_35b moe_session
    moe_weights moe_exec moe_platform)
moe_warnings(edge0)
if(MOE_BUILD_CLI)
    add_subdirectory(tools/cli)
endif()
if(MOE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(MOE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
