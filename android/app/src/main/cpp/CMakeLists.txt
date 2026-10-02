# app/src/main/cpp/CMakeLists.txt - native llama engine shell for the app.
# Links the five shared libs staged at build-dl/llama-libs/arm64-v8a (built by
# tools/llama/build_vendor_libs.sh from the pinned supply tree: vendor@7ab4ee7 with
# the common+android patch bands replayed; see patches/llama.cpp/README).
cmake_minimum_required(VERSION 3.22.1)
project(edge0llama C)

if(NOT DEFINED LLAMA_LIBS)
    set(LLAMA_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../build-dl/llama-libs/arm64-v8a")
endif()

add_library(edge0llama SHARED llama_chat.c)
target_include_directories(edge0llama PRIVATE
    "${LLAMA_LIBS}/../include"            # llama.h/ggml.h headers (synced at build time)
    "${LLAMA_LIBS}")
target_link_libraries(edge0llama
    "${LLAMA_LIBS}/libllama.so"
    "${LLAMA_LIBS}/libggml.so"
    "${LLAMA_LIBS}/libggml-base.so"
    "${LLAMA_LIBS}/libggml-cpu.so"
    android log)
