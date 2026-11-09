function(moe_warnings target)
    if(MSVC)
        # Warnings as errors on our targets only (GGML and other third-party
        # targets keep their own settings). /utf-8 compiles sources and runs
        # with a UTF-8 execution charset (decision 2) so universal-character
        # escapes are not rejected under code page 936 (C4566). The /wd
        # suppressions cover the categories MSVC /W4 raises but our Clang/GCC
        # baseline (-Wall -Wextra -Wpedantic) does not, so /WX stops new mistakes
        # in Windows code without forcing a repo-wide cleanup of pre-existing
        # portable code (decision 1, user 2026-10-26 option A; numbers confirmed
        # by building under /WX):
        #   4244 4245 4267 4305 4310  integer, size and constant conversions
        #   4389                      signed/unsigned comparison
        #   4456                      local declaration shadows a previous one
        #   4996                      CRT "unsafe" fopen/getenv/tmpfile
        target_compile_options(${target} PRIVATE
            /W4 /WX /permissive- /utf-8
            /wd4244 /wd4245 /wd4267 /wd4305 /wd4310 /wd4389 /wd4456 /wd4996)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

function(moe_add_test)
    cmake_parse_arguments(TEST "DEVICE" "NAME;LEVEL" "COMMAND" ${ARGN})
    if(TEST_UNPARSED_ARGUMENTS OR NOT TEST_NAME OR NOT TEST_COMMAND OR
       NOT TEST_LEVEL MATCHES "^L[0-4]$")
        message(FATAL_ERROR "moe_add_test requires NAME, one LEVEL (L0-L4), and COMMAND")
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_COMMAND})
    set(labels ${TEST_LEVEL})
    if(TEST_DEVICE)
        list(APPEND labels device)
    endif()
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "${labels}" SKIP_RETURN_CODE 77
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    # Real-weight and device tests assert peak memory or share one GPU; never overlap them.
    if(TEST_DEVICE OR TEST_LEVEL MATCHES "^L[34]$")
        set_tests_properties(${TEST_NAME} PROPERTIES RUN_SERIAL TRUE)
    endif()
endfunction()
