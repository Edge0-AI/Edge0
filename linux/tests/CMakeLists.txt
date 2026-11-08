cmake_minimum_required(VERSION 3.16)
project(edge0_linux_serve_tests LANGUAGES CXX)
enable_testing()

get_filename_component(EDGE0_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(EDGE0_LLAMA_SOURCE "${EDGE0_ROOT}/wt/linux" CACHE PATH "Assembled pinned llama.cpp")
set(EDGE0_LLAMA_BUILD "${EDGE0_LLAMA_SOURCE}/build-cpu" CACHE PATH "Built pinned llama.cpp")
option(EDGE0_SANITIZE "Run with address and undefined-behavior sanitizers" OFF)
find_library(GGML_BASE NAMES ggml-base PATHS "${EDGE0_LLAMA_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_LIBRARY NAMES llama PATHS "${EDGE0_LLAMA_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
find_package(Threads REQUIRED)

add_executable(edge0-serve-tests test_serve.cpp
    ../serve/posix_io.cc ../serve/mem_budget.cc ../serve/prefetch_engine.cc ../serve/expert_pool.cc)
target_compile_features(edge0-serve-tests PRIVATE cxx_std_17)
target_compile_options(edge0-serve-tests PRIVATE -Wall -Wextra -Werror)
target_include_directories(edge0-serve-tests PRIVATE ../serve
    "${EDGE0_LLAMA_SOURCE}/ggml/include" "${EDGE0_LLAMA_SOURCE}/vendor")
target_link_libraries(edge0-serve-tests PRIVATE "${GGML_BASE}" Threads::Threads)
if(EDGE0_SANITIZE)
    # Keep the test executable out of ASan's shadow range on high-entropy-ASLR
    # hosts with older GCC runtimes. This does not change the product build.
    target_compile_options(edge0-serve-tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie)
    target_link_options(edge0-serve-tests PRIVATE -fsanitize=address,undefined -no-pie)
endif()

foreach(case mib_inputs mib_precedence page_boundaries expert_bounds file_io rss_accounting
    budget_dormant budget_reclaim budget_overhead budget_unmap budget_syscall_failure budget_rss_failure budget_worker
    budget_backoff budget_backoff_in_band
    shape_8b shape_35b head_math history_math account_only advice_on advice_failure sidecar_corruption
    missing_gate invalid_expert_tensor bad_rows worker_mailbox worker_failure worker_shutdown descriptor_cleanup
    pool_bytes pool_capacity pool_bounds pool_concurrency pool_layout pool_read_failure)
    add_test(NAME ${case} COMMAND edge0-serve-tests ${case})
    set_tests_properties(${case} PROPERTIES TIMEOUT 20)
endforeach()
# Exercise exported serving hooks from the actual product shared library.
add_executable(edge0-serve-api-tests test_serve_api.cpp)
target_compile_features(edge0-serve-api-tests PRIVATE cxx_std_17)
target_compile_options(edge0-serve-api-tests PRIVATE -Wall -Wextra -Werror)
target_include_directories(edge0-serve-api-tests PRIVATE ../serve)
target_link_libraries(edge0-serve-api-tests PRIVATE "${LLAMA_LIBRARY}")
foreach(case api_dormant api_budget api_invalid)
    add_test(NAME ${case} COMMAND edge0-serve-api-tests ${case})
    set_tests_properties(${case} PROPERTIES TIMEOUT 20)
endforeach()

add_executable(edge0-rss-cost-tests test_rss_cost.cpp)
target_compile_features(edge0-rss-cost-tests PRIVATE cxx_std_17)
target_compile_options(edge0-rss-cost-tests PRIVATE -Wall -Wextra -Werror)
target_include_directories(edge0-rss-cost-tests PRIVATE ../serve)
target_link_libraries(edge0-rss-cost-tests PRIVATE "${LLAMA_LIBRARY}")
add_test(NAME rss_poll_cost COMMAND edge0-rss-cost-tests)
set_tests_properties(rss_poll_cost PROPERTIES TIMEOUT 20)
