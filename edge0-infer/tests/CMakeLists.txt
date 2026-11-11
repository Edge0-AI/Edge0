add_executable(moe_test_platform unit/platform/backend_test.cpp)
target_include_directories(moe_test_platform PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_platform PRIVATE moe_platform)
moe_warnings(moe_test_platform)
moe_add_test(NAME moe_test_platform LEVEL L1 COMMAND moe_test_platform)
add_executable(moe_test_app unit/app/runtime_test.cpp)
target_include_directories(moe_test_app PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/third_party/ggml/include" "${PROJECT_SOURCE_DIR}/third_party/ggml/src")
target_link_libraries(moe_test_app PRIVATE edge0::edge0)
target_compile_definitions(moe_test_app PRIVATE MOE_TEST_METAL=$<BOOL:${GGML_METAL}>
    MOE_TEST_GGML_VERSION="${MOE_GGML_VERSION}")
moe_warnings(moe_test_app)
moe_add_test(NAME moe_test_app LEVEL L1 COMMAND moe_test_app)
include("${PROJECT_SOURCE_DIR}/cmake/PublicHeaders.cmake")
if(WIN32)
    # Windows has no sh, and wrapping a .cmd in `cmd /c` trips cmd's quote
    # stripping when CTest passes two quoted path arguments. Invoke uv directly
    # instead (uv and git are Windows prerequisites; see getting-started.md).
    # The missing-tool skip (exit 77) that run_python.sh provides is not
    # reproduced here: a missing uv surfaces as a normal test failure.
    # -X utf8 forces Python UTF-8 Mode so the harnesses decode the CLI's UTF-8
    # output and read/write their files as UTF-8 regardless of the OS locale
    # (for example GBK on a Chinese Windows); it is spelled on the command line
    # because -I ignores PYTHONUTF8 from the environment (decision 2).
    set(moe_python uv run --no-project python -I -X utf8)
else()
    set(moe_python sh "${PROJECT_SOURCE_DIR}/tests/support/run_python.sh")
endif()
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/cli_test.py" --cli $<TARGET_FILE:edge0-infer>
        --origin "${PROJECT_SOURCE_DIR}/third_party/GGML_ORIGIN" --metal $<BOOL:${GGML_METAL}>)
    moe_add_test(NAME moe_test_metal_cli LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/cli_test.py" --cli $<TARGET_FILE:edge0-infer>
        --origin "${PROJECT_SOURCE_DIR}/third_party/GGML_ORIGIN" --metal $<BOOL:${GGML_METAL}> --device)
endif()
add_executable(moe_test_metal_runtime device/metal_test.cpp)
target_include_directories(moe_test_metal_runtime PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_metal_runtime PRIVATE edge0::edge0)
moe_warnings(moe_test_metal_runtime)
moe_add_test(NAME moe_test_metal_runtime LEVEL L2 DEVICE COMMAND moe_test_metal_runtime)

# The development repository runs check_internal_docs, which includes everything check_docs checks.
if(EXISTS "${PROJECT_SOURCE_DIR}/tools/checks/check_internal_docs.py")
    set(moe_docs_check internal_docs)
else()
    set(moe_docs_check docs)
endif()
foreach(check IN ITEMS ${moe_docs_check} language layers deps app_deps app_tokens ggml_origin scope model_terms public_options signing)
    if(check STREQUAL "docs" OR check STREQUAL "language")
        set(script "${PROJECT_SOURCE_DIR}/tools/check_${check}.py")
    else()
        set(script "${PROJECT_SOURCE_DIR}/tools/checks/check_${check}.py")
    endif()
    if(check STREQUAL "scope")
        # Without an argument the change is inferred from the wt/<change> branch.
        set(arguments "")
    else()
        set(arguments "${PROJECT_SOURCE_DIR}")
    endif()
    # Repository-workflow checks (change scope, internal documents) are not part of the published tree.
    if(EXISTS "${script}")
        moe_add_test(NAME check_${check} LEVEL L0 COMMAND ${moe_python} "${script}" ${arguments})
    endif()
endforeach()
foreach(check IN ITEMS layers deps app_deps app_tokens ggml_origin scope run_checks docs internal_docs language archive_links model_terms signing export_public)
    if(check STREQUAL "run_checks")
        set(pattern test_run_checks.py)
    elseif(check STREQUAL "archive_links")
        set(pattern test_fix_archive_links.py)
    elseif(check STREQUAL "export_public")
        set(pattern test_export_public.py)
    else()
        set(pattern test_check_${check}.py)
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/tests/unit/checks/${pattern}")
        moe_add_test(NAME test_${check} LEVEL L1 COMMAND ${moe_python} -m unittest discover
            -s "${PROJECT_SOURCE_DIR}/tests/unit/checks" -p "${pattern}")
    endif()
endforeach()

foreach(pair IN ITEMS "bpe|moe_test_text_bpe|L1" "decode|moe_test_text_decode|L1"
    "template|moe_test_edge0_35b_text|L1" "template_tokens|moe_test_edge0_35b_text_tokens|L2"
    "corpus|moe_test_text_corpus|L2")
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 mode)
    list(GET fields 1 name)
    list(GET fields 2 level)
    add_executable(${name} unit/app/text_tests.cpp)
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tests/unit")
    target_link_libraries(${name} PRIVATE moe_app)
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
    target_compile_definitions(${name} PRIVATE MOE_TEXT_MODE="${mode}" MOE_TEXT_FIXTURES="${PROJECT_SOURCE_DIR}/tests/fixtures/text")
    moe_warnings(${name})
    if(mode STREQUAL "corpus")
        moe_add_test(NAME ${name} LEVEL ${level} COMMAND ${moe_python}
            "${PROJECT_SOURCE_DIR}/tests/unit/app/check_text_fixtures.py"
            "${PROJECT_SOURCE_DIR}/tests/fixtures/text" $<TARGET_FILE:${name}>)
    elseif(mode STREQUAL "decode")
        moe_add_test(NAME ${name} LEVEL ${level} COMMAND ${moe_python}
            "${PROJECT_SOURCE_DIR}/tests/unit/app/check_decode_steps.py"
            $<TARGET_FILE:${name}> "${PROJECT_SOURCE_DIR}/tests/fixtures/text/utf8_golden.json")
    else()
        moe_add_test(NAME ${name} LEVEL ${level} COMMAND ${name})
    endif()
endforeach()
moe_add_test(NAME moe_test_text_upstream LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/app/check_text_upstream.py" "${PROJECT_SOURCE_DIR}")

foreach(case IN ITEMS catalog index adapter)
    add_executable(moe_test_model_${case} unit/model/${case}_test.cpp)
    target_include_directories(moe_test_model_${case} PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(moe_test_model_${case} PRIVATE moe_model)
    moe_warnings(moe_test_model_${case})
    moe_add_test(NAME moe_test_model_${case} LEVEL L1 COMMAND moe_test_model_${case})
endforeach()
add_executable(moe_test_edge0_35b_assets unit/models/edge0_35b/assets_test.cpp)
target_include_directories(moe_test_edge0_35b_assets PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_edge0_35b_assets PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_edge0_35b_assets)
moe_add_test(NAME moe_test_edge0_35b_assets LEVEL L1 COMMAND moe_test_edge0_35b_assets)
add_executable(moe_test_model_inspect unit/app/inspect_test.cpp)
target_include_directories(moe_test_model_inspect PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(moe_test_model_inspect PRIVATE edge0::edge0)
moe_warnings(moe_test_model_inspect)
moe_add_test(NAME moe_test_model_inspect LEVEL L1 COMMAND moe_test_model_inspect "${PROJECT_SOURCE_DIR}")

foreach(target IN ITEMS moe_test_model_catalog moe_test_model_index moe_test_model_adapter moe_test_edge0_35b_assets moe_test_model_inspect)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
endforeach()
target_sources(moe_test_model_index PRIVATE support/fread_probe.cpp)
target_link_libraries(moe_test_model_index PRIVATE ${CMAKE_DL_LIBS})
add_executable(moe_model_variant_tool unit/model/variant_tool.cpp)
target_include_directories(moe_model_variant_tool PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_model_variant_tool PRIVATE moe_model)
moe_warnings(moe_model_variant_tool)
moe_add_test(NAME moe_test_edge0_35b_manifest LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/integration/model_contract.py" --root "${PROJECT_SOURCE_DIR}"
    --profile custom --custom-tool $<TARGET_FILE:moe_test_edge0_35b_assets>)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_model_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/model_cli_test.py" $<TARGET_FILE:edge0-infer>
        $<TARGET_FILE:moe_test_model_inspect> "${PROJECT_SOURCE_DIR}")
    moe_add_test(NAME moe_test_model_tiny LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/model_contract.py" --cli $<TARGET_FILE:edge0-infer>
        --root "${PROJECT_SOURCE_DIR}" --profile tiny)
    moe_add_test(NAME moe_test_model_variants LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/model_variants_test.py" $<TARGET_FILE:edge0-infer>
        $<TARGET_FILE:moe_model_variant_tool>)
    moe_add_test(NAME moe_test_model_35b LEVEL L3 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/model_contract.py" --cli $<TARGET_FILE:edge0-infer>
        --root "${PROJECT_SOURCE_DIR}" --profile 35b)
    moe_add_test(NAME moe_test_model_35b_digest LEVEL L3 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/model_contract.py" --cli $<TARGET_FILE:edge0-infer>
        --root "${PROJECT_SOURCE_DIR}" --profile 35b --digest-only)
    foreach(profile IN ITEMS tiny 35b)
        if(profile STREQUAL "tiny")
            set(level L2)
        else()
            set(level L3)
        endif()
        moe_add_test(NAME moe_test_model_${profile}_offsets LEVEL ${level} COMMAND uv run
            --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
            "${PROJECT_SOURCE_DIR}/tests/integration/model_contract.py" --cli $<TARGET_FILE:edge0-infer>
            --root "${PROJECT_SOURCE_DIR}" --profile ${profile} --independent-reader)
    endforeach()
endif()

add_executable(moe_test_text_api integration/text_api_test.cpp)
target_link_libraries(moe_test_text_api PRIVATE edge0::edge0)
moe_warnings(moe_test_text_api)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_text_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/text_cli_test.py" $<TARGET_FILE:edge0-infer>
        $<TARGET_FILE:moe_test_text_api> "${PROJECT_SOURCE_DIR}/tests/fixtures/text")
endif()

add_executable(moe_test_text_rules unit/app/text_rules_test.cpp)
target_include_directories(moe_test_text_rules PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(moe_test_text_rules PRIVATE moe_app)
target_include_directories(moe_test_text_rules PRIVATE "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
moe_warnings(moe_test_text_rules)
moe_add_test(NAME moe_test_text_rules LEVEL L1 COMMAND moe_test_text_rules)

add_executable(moe_test_platform_runtime unit/platform/runtime_test.cpp)
target_include_directories(moe_test_platform_runtime PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_platform_runtime PRIVATE moe_platform)
moe_warnings(moe_test_platform_runtime)
moe_add_test(NAME moe_test_platform_runtime LEVEL L1 COMMAND moe_test_platform_runtime)
moe_add_test(NAME check_runtime_backends LEVEL L0 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/checks/check_runtime_backends.py" "${PROJECT_SOURCE_DIR}")
moe_add_test(NAME check_strata_files LEVEL L0 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/cli/check_strata_files.py" "${PROJECT_SOURCE_DIR}")

add_executable(moe_test_exec_budget unit/exec/budget_test.cpp)
target_include_directories(moe_test_exec_budget PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_exec_budget PRIVATE moe_exec)
moe_warnings(moe_test_exec_budget)
moe_add_test(NAME moe_test_exec_budget LEVEL L1 COMMAND moe_test_exec_budget)
add_executable(moe_test_exec_runtime unit/exec/execution_test.cpp)
target_include_directories(moe_test_exec_runtime PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_exec_runtime PRIVATE moe_exec)
moe_warnings(moe_test_exec_runtime)
moe_add_test(NAME moe_test_exec_runtime LEVEL L1 COMMAND moe_test_exec_runtime)
add_executable(moe_test_session_storage unit/session/storage_test.cpp)
target_include_directories(moe_test_session_storage PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_session_storage PRIVATE moe_session)
moe_warnings(moe_test_session_storage)
moe_add_test(NAME moe_test_session_storage LEVEL L1 COMMAND moe_test_session_storage)
add_executable(moe_test_weights_storage unit/weights/storage_test.cpp)
target_include_directories(moe_test_weights_storage PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_storage PRIVATE moe_weights)
moe_warnings(moe_test_weights_storage)
moe_add_test(NAME moe_test_weights_storage LEVEL L1 COMMAND moe_test_weights_storage)
add_executable(moe_test_weights_staged unit/weights/staged_fill_test.cpp)
target_include_directories(moe_test_weights_staged PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_staged PRIVATE moe_weights)
moe_warnings(moe_test_weights_staged)
moe_add_test(NAME moe_test_weights_staged LEVEL L1 COMMAND moe_test_weights_staged)
add_executable(moe_test_weights_vulkan_fill unit/weights/vulkan_fill_test.cpp)
target_include_directories(moe_test_weights_vulkan_fill PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_vulkan_fill PRIVATE moe_weights)
moe_warnings(moe_test_weights_vulkan_fill)
moe_add_test(NAME moe_test_weights_vulkan_fill LEVEL L1 DEVICE COMMAND moe_test_weights_vulkan_fill)
add_executable(moe_test_weights_vulkan_concurrent unit/weights/vulkan_concurrent_test.cpp)
target_include_directories(moe_test_weights_vulkan_concurrent PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_vulkan_concurrent PRIVATE moe_weights)
moe_warnings(moe_test_weights_vulkan_concurrent)
moe_add_test(NAME moe_test_weights_vulkan_concurrent LEVEL L1 DEVICE COMMAND moe_test_weights_vulkan_concurrent)
add_executable(moe_test_weights_compute unit/weights/compute_test.cpp)
target_include_directories(moe_test_weights_compute PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_compute PRIVATE moe_weights)
moe_warnings(moe_test_weights_compute)
moe_add_test(NAME moe_test_weights_compute LEVEL L2 COMMAND moe_test_weights_compute)
moe_add_test(NAME moe_test_weights_compute_device LEVEL L2 DEVICE COMMAND moe_test_weights_compute --backend metal)
add_executable(moe_test_session_attention unit/session/attention_test.cpp)
target_include_directories(moe_test_session_attention PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_session_attention PRIVATE moe_session)
moe_warnings(moe_test_session_attention)
moe_add_test(NAME moe_test_session_attention LEVEL L2 COMMAND moe_test_session_attention)
moe_add_test(NAME moe_test_session_attention_device LEVEL L2 DEVICE COMMAND moe_test_session_attention --backend metal)

moe_add_test(NAME check_model_no_compute LEVEL L0 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/checks/check_model_no_compute.py" "${PROJECT_SOURCE_DIR}")
add_executable(moe_test_edge0_35b_loading unit/models/edge0_35b/loading_test.cpp)
target_include_directories(moe_test_edge0_35b_loading PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_edge0_35b_loading PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_edge0_35b_loading)
moe_add_test(NAME moe_test_edge0_35b_estimate LEVEL L1 COMMAND moe_test_edge0_35b_loading --arithmetic)
moe_add_test(NAME moe_test_edge0_35b_loading LEVEL L2 COMMAND moe_test_edge0_35b_loading)
add_executable(moe_test_edge0_35b_admission unit/app/admission_test.cpp)
target_include_directories(moe_test_edge0_35b_admission PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src")
target_include_directories(moe_test_edge0_35b_admission PRIVATE "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_edge0_35b_admission PRIVATE edge0::edge0)
moe_warnings(moe_test_edge0_35b_admission)
moe_add_test(NAME moe_test_edge0_35b_admission LEVEL L3 COMMAND moe_test_edge0_35b_admission)
add_executable(moe_test_edge0_35b_forward unit/models/edge0_35b/forward_test.cpp)
target_include_directories(moe_test_edge0_35b_forward PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_edge0_35b_forward PRIVATE moe_models_edge0_35b)
target_compile_definitions(moe_test_edge0_35b_forward PRIVATE MOE_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
moe_warnings(moe_test_edge0_35b_forward)
moe_add_test(NAME moe_test_edge0_35b_forward LEVEL L2 COMMAND moe_test_edge0_35b_forward)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_forward_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_test.py" --cli $<TARGET_FILE:edge0-infer>)
endif()
# Answer suites compare this platform's logits against the repository answers
# (exported on macOS). -X utf8 forces Python UTF-8 Mode so the harnesses decode
# the CLI's UTF-8 output on a non-UTF-8 OS locale (for example GBK on a Chinese
# Windows). On Windows the single-backend CPU answers are not bit-reproducible
# against the macOS-exported answers, so they run in --record-only mode per
# decision 7 (differences recorded, thresholds not enforced); macOS still
# enforces. The Metal/device variants run on macOS only and stay enforced.
set(answers_utf8 -X utf8)
set(answers_record_only "")
if(WIN32)
    set(answers_record_only --record-only)
endif()
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_forward_answers LEVEL L2 COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I ${answers_utf8}
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_answers_test.py" --cli $<TARGET_FILE:edge0-infer> ${answers_record_only})
    moe_add_test(NAME moe_test_forward_answers_metal LEVEL L2 DEVICE COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_answers_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
    moe_add_test(NAME moe_test_forward_cli_metal LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
endif()
moe_add_test(NAME moe_test_edge0_35b_forward_metal LEVEL L2 DEVICE COMMAND moe_test_edge0_35b_forward --backend metal)
moe_add_test(NAME moe_test_answers_judge LEVEL L1 COMMAND uv run
    --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync pytest -p no:cacheprovider
    "${PROJECT_SOURCE_DIR}/tests/answers/test_judge.py" -q)
foreach(level IN ITEMS l1 l2)
    string(TOUPPER "${level}" label)
    moe_add_test(NAME moe_test_asset_tools_${level} LEVEL ${label} COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets"
        --offline --no-sync python "${PROJECT_SOURCE_DIR}/tools/assets/tests/run_marked.py" ${level})
endforeach()
moe_add_test(NAME moe_test_answers_inventory LEVEL L1 COMMAND uv run
    --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync pytest -p no:cacheprovider
    "${PROJECT_SOURCE_DIR}/tests/answers/test_inventory.py" -q)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_answers_export_determinism LEVEL L2 COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
        "${PROJECT_SOURCE_DIR}/tests/answers/export_determinism_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_answers_export_determinism_device LEVEL L2 DEVICE COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
        "${PROJECT_SOURCE_DIR}/tests/answers/export_determinism_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
endif()

moe_add_test(NAME check_cpu_kernels LEVEL L0 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/checks/check_cpu_kernels.py" "${PROJECT_SOURCE_DIR}")

add_executable(moe_test_session_transactions unit/session/session_test.cpp)
target_include_directories(moe_test_session_transactions PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_session_transactions PRIVATE moe_session)
moe_warnings(moe_test_session_transactions)
moe_add_test(NAME moe_test_session_transactions LEVEL L1 COMMAND moe_test_session_transactions)
add_executable(moe_test_model_generation_defaults unit/model/generation_test.cpp)
target_include_directories(moe_test_model_generation_defaults PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_model_generation_defaults PRIVATE moe_model)
moe_warnings(moe_test_model_generation_defaults)
moe_add_test(NAME moe_test_model_generation_defaults LEVEL L1 COMMAND moe_test_model_generation_defaults)
moe_add_test(NAME moe_test_model_generation_defaults_tiny LEVEL L2 COMMAND moe_test_model_generation_defaults --tiny)
target_include_directories(moe_test_session_transactions PRIVATE "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
add_executable(moe_test_session_rows unit/session/rows_test.cpp)
target_include_directories(moe_test_session_rows PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_session_rows PRIVATE moe_session)
moe_warnings(moe_test_session_rows)
moe_add_test(NAME moe_test_session_rows LEVEL L1 COMMAND moe_test_session_rows)
moe_add_test(NAME moe_test_session_rows_device LEVEL L2 DEVICE COMMAND moe_test_session_rows --backend metal)
add_executable(moe_test_edge0_35b_reuse unit/models/edge0_35b/reuse_test.cpp)
target_include_directories(moe_test_edge0_35b_reuse PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_edge0_35b_reuse PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_edge0_35b_reuse)
moe_add_test(NAME moe_test_edge0_35b_reuse LEVEL L2 COMMAND moe_test_edge0_35b_reuse)
moe_add_test(NAME moe_test_edge0_35b_reuse_device LEVEL L2 DEVICE COMMAND moe_test_edge0_35b_reuse --backend metal)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_forward_split LEVEL L2 COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python ${answers_utf8}
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_answers_test.py" --cli $<TARGET_FILE:edge0-infer> --split-last ${answers_record_only})
    moe_add_test(NAME moe_test_forward_split_device LEVEL L2 DEVICE COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_answers_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal --split-last)
endif()
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_generate_answers LEVEL L2 COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python ${answers_utf8}
        "${PROJECT_SOURCE_DIR}/tests/integration/generate_answers_test.py" --cli $<TARGET_FILE:edge0-infer> ${answers_record_only})
    moe_add_test(NAME moe_test_generate_answers_device LEVEL L2 DEVICE COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python
        "${PROJECT_SOURCE_DIR}/tests/integration/generate_answers_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
endif()
add_executable(moe_test_app_generation unit/app/generation_test.cpp)
target_include_directories(moe_test_app_generation PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_app_generation PRIVATE moe_app)
moe_warnings(moe_test_app_generation)
moe_add_test(NAME moe_test_app_generation LEVEL L1 COMMAND moe_test_app_generation)
target_include_directories(moe_test_app_generation PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_executable(moe_test_generate unit/app/generate_model_test.cpp)
target_include_directories(moe_test_generate PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_generate PRIVATE moe_app)
moe_warnings(moe_test_generate)
moe_add_test(NAME moe_test_generate LEVEL L2 COMMAND moe_test_generate)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_bench_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/bench_cli_test.py" $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_generate_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/generate_cli_test.py" $<TARGET_FILE:edge0-infer>)
endif()

if(MOE_BUILD_CLI)
    add_executable(moe_test_cli_logging unit/app/cli_logging_test.cpp "${PROJECT_SOURCE_DIR}/tools/cli/logging.cpp")
    target_include_directories(moe_test_cli_logging PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    moe_warnings(moe_test_cli_logging)
    moe_add_test(NAME moe_test_cli_logging LEVEL L1 COMMAND moe_test_cli_logging)
    moe_add_test(NAME moe_test_generate_cli_metal LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/generate_cli_test.py" $<TARGET_FILE:edge0-infer> metal)
endif()

add_executable(moe_test_weights_lifecycle unit/weights/lifecycle_test.cpp)
target_include_directories(moe_test_weights_lifecycle PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_lifecycle PRIVATE moe_weights)
moe_warnings(moe_test_weights_lifecycle)
moe_add_test(NAME moe_test_weights_lifecycle LEVEL L1 COMMAND moe_test_weights_lifecycle)
add_executable(moe_test_weights_slots unit/weights/slots_test.cpp)
target_include_directories(moe_test_weights_slots PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_slots PRIVATE moe_weights)
moe_warnings(moe_test_weights_slots)
moe_add_test(NAME moe_test_weights_slots LEVEL L1 COMMAND moe_test_weights_slots)
moe_add_test(NAME moe_test_weights_group_compute LEVEL L2 COMMAND moe_test_weights_slots)
moe_add_test(NAME moe_test_weights_group_compute_device LEVEL L2 DEVICE COMMAND moe_test_weights_slots --backend metal)
add_executable(moe_test_edge0_35b_slots unit/models/edge0_35b/slots_test.cpp)
target_include_directories(moe_test_edge0_35b_slots PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_edge0_35b_slots PRIVATE moe_models_edge0_35b)
target_compile_definitions(moe_test_edge0_35b_slots PRIVATE MOE_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
moe_warnings(moe_test_edge0_35b_slots)
moe_add_test(NAME moe_test_edge0_35b_slots LEVEL L2 COMMAND moe_test_edge0_35b_slots)
moe_add_test(NAME moe_test_edge0_35b_slots_device LEVEL L2 DEVICE COMMAND moe_test_edge0_35b_slots --backend metal)
add_executable(moe_test_model_slots_loading unit/model/slots_loading_test.cpp)
target_include_directories(moe_test_model_slots_loading PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_model_slots_loading PRIVATE moe_model)
moe_warnings(moe_test_model_slots_loading)
moe_add_test(NAME moe_test_model_slots_loading LEVEL L1 COMMAND moe_test_model_slots_loading)
add_executable(moe_test_edge0_35b_slots_failure unit/app/slots_failure_test.cpp)
target_include_directories(moe_test_edge0_35b_slots_failure PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_edge0_35b_slots_failure PRIVATE moe_app)
moe_warnings(moe_test_edge0_35b_slots_failure)
moe_add_test(NAME moe_test_edge0_35b_slots_failure LEVEL L2 COMMAND moe_test_edge0_35b_slots_failure)
add_executable(moe_test_edge0_35b_slots_config unit/models/edge0_35b/slots_config_test.cpp)
target_include_directories(moe_test_edge0_35b_slots_config PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_compile_definitions(moe_test_edge0_35b_slots_config PRIVATE MOE_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_edge0_35b_slots_config PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_edge0_35b_slots_config)
moe_add_test(NAME moe_test_edge0_35b_slots_config LEVEL L1 COMMAND moe_test_edge0_35b_slots_config)

add_executable(moe_test_platform_memory unit/platform/memory_test.cpp)
target_include_directories(moe_test_platform_memory PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_platform_memory PRIVATE moe_platform)
moe_warnings(moe_test_platform_memory)
moe_add_test(NAME moe_test_platform_memory LEVEL L1 COMMAND moe_test_platform_memory)
target_sources(moe_test_app_generation PRIVATE support/file_read_counter.cpp)
target_link_libraries(moe_test_app_generation PRIVATE ${CMAKE_DL_LIBS})

if(MOE_BUILD_CLI)
    foreach(backend IN ITEMS cpu metal)
        set(device_flag "")
        if(backend STREQUAL "metal")
            set(device_flag DEVICE)
        endif()
        foreach(mode IN ITEMS numerical memory)
            moe_add_test(NAME moe_test_single_turn_35b_${backend}_${mode} LEVEL L3 ${device_flag}
                COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
                "${PROJECT_SOURCE_DIR}/tests/integration/single_turn_35b_test.py"
                --cli $<TARGET_FILE:edge0-infer> --backend ${backend} --mode ${mode})
        endforeach()
    endforeach()
    moe_add_test(NAME moe_test_single_turn_35b_describe LEVEL L3
        COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
        "${PROJECT_SOURCE_DIR}/tests/integration/single_turn_35b_test.py"
        --cli $<TARGET_FILE:edge0-infer> --mode describe)
endif()
moe_add_test(NAME moe_test_single_turn_metrics LEVEL L1
    COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
    "${PROJECT_SOURCE_DIR}/tests/integration/single_turn_metrics_test.py")

moe_add_test(NAME moe_test_build_identity LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/build/build_identity_test.py" "${CMAKE_COMMAND}"
    "${PROJECT_SOURCE_DIR}/cmake/BuildIdentity.cmake")
if(NOT CMAKE_CROSSCOMPILING)
    # Configures a host project around this tree (add_subdirectory) and compiles one object file.
    moe_add_test(NAME moe_test_subproject LEVEL L1 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/unit/build/subproject_test.py" "${CMAKE_COMMAND}" "${CMAKE_GENERATOR}"
        "${CMAKE_MAKE_PROGRAM}" "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}" "${PROJECT_SOURCE_DIR}")
endif()

moe_add_test(NAME moe_test_abba LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/perf/abba_test.py")
moe_add_test(NAME moe_test_perf_hosts LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/perf/hosts_test.py")
moe_add_test(NAME moe_test_perf_report LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/perf/report_test.py")
moe_add_test(NAME moe_test_perf_turns LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/perf/turns_test.py")
moe_add_test(NAME moe_test_model_support LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/validation/model_support_test.py")
moe_add_test(NAME moe_test_perf_sweep LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/perf/sweep_test.py")

if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_bench_cli_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/bench_cli_test.py" $<TARGET_FILE:edge0-infer> metal)
endif()

if(MOE_BUILD_CLI AND APPLE)
    moe_add_test(NAME moe_test_tuning_direct LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/tuning_comparison_test.py"
        --a $<TARGET_FILE:edge0-infer> --b $<TARGET_FILE:edge0-infer> --backend cpu --tiny
        --b-options "[\"--model-option\",\"expert_read=direct\"]")
    moe_add_test(NAME moe_test_tuning_direct_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/tuning_comparison_test.py"
        --a $<TARGET_FILE:edge0-infer> --b $<TARGET_FILE:edge0-infer> --backend metal --tiny
        --b-options "[\"--model-option\",\"expert_read=direct\"]")
endif()

add_executable(moe_test_weights_read_batch unit/weights/read_batch_test.cpp)
target_include_directories(moe_test_weights_read_batch PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_read_batch PRIVATE moe_weights)
moe_warnings(moe_test_weights_read_batch)
moe_add_test(NAME moe_test_weights_read_batch LEVEL L1 COMMAND moe_test_weights_read_batch)

if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_tuning_ahead LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/tuning_comparison_test.py"
        --a $<TARGET_FILE:edge0-infer> --b $<TARGET_FILE:edge0-infer> --backend cpu --tiny
        --a-options "[\"--prefill-chunk\",\"3\",\"--model-option\",\"prefill_read=used\"]"
        --b-options "[\"--prefill-chunk\",\"3\",\"--model-option\",\"prefill_read=ahead\"]")
    moe_add_test(NAME moe_test_tuning_ahead_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/tuning_comparison_test.py"
        --a $<TARGET_FILE:edge0-infer> --b $<TARGET_FILE:edge0-infer> --backend metal --tiny
        --a-options "[\"--prefill-chunk\",\"3\",\"--model-option\",\"prefill_read=used\"]"
        --b-options "[\"--prefill-chunk\",\"3\",\"--model-option\",\"prefill_read=ahead\"]")
endif()

add_executable(moe_test_state_bytes unit/app/state_bytes_test.cpp)
target_include_directories(moe_test_state_bytes PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_state_bytes PRIVATE moe_app)
moe_warnings(moe_test_state_bytes)
moe_add_test(NAME moe_test_state_bytes LEVEL L3 COMMAND moe_test_state_bytes)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_forward_usage_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/forward_usage_cli_test.py" $<TARGET_FILE:edge0-infer>)
endif()
if(MOE_BUILD_CLI)
    foreach(backend IN ITEMS cpu metal)
        set(tuning_device_flag "")
        if(backend STREQUAL "metal")
            set(tuning_device_flag DEVICE)
        endif()
        moe_add_test(NAME moe_test_tuning_read_${backend} LEVEL L3 ${tuning_device_flag} COMMAND ${moe_python}
            "${PROJECT_SOURCE_DIR}/tests/integration/tuning_comparison_test.py"
            --a $<TARGET_FILE:edge0-infer> --b $<TARGET_FILE:edge0-infer> --backend ${backend} --matrix)
    endforeach()
    moe_add_test(NAME moe_test_tuning_long_ahead LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/tuning_comparison_test.py"
        --a $<TARGET_FILE:edge0-infer> --b $<TARGET_FILE:edge0-infer> --backend metal --long
        --a-options "[\"--prefill-chunk\",\"4096\",\"--model-option\",\"prefill_read=used\",\"--model-option\",\"expert_read=cached\"]"
        --b-options "[\"--prefill-chunk\",\"4096\",\"--model-option\",\"prefill_read=ahead\",\"--model-option\",\"expert_read=cached\"]")
endif()

add_executable(moe_test_weights_priority unit/weights/priority_test.cpp)
target_include_directories(moe_test_weights_priority PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_weights_priority PRIVATE moe_model)
moe_warnings(moe_test_weights_priority)
moe_add_test(NAME moe_test_weights_priority LEVEL L1 COMMAND moe_test_weights_priority)

if(MOE_BUILD_CLI)
    foreach(backend IN ITEMS cpu metal)
        set(route_device "")
        if(backend STREQUAL "metal")
            set(route_device DEVICE)
        endif()
        moe_add_test(NAME moe_test_prerouter_${backend} LEVEL L2 ${route_device} COMMAND uv run
            --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
            "${PROJECT_SOURCE_DIR}/tests/integration/prerouter_reference.py" --cli $<TARGET_FILE:edge0-infer> --backend ${backend})
        moe_add_test(NAME moe_test_routing_predicted_${backend} LEVEL L2 ${route_device} COMMAND ${moe_python}
                "${PROJECT_SOURCE_DIR}/tests/integration/routing_test.py" --cli $<TARGET_FILE:edge0-infer> --backend ${backend})
        moe_add_test(NAME moe_test_predicted_35b_${backend} LEVEL L3 ${route_device} COMMAND ${moe_python}
            "${PROJECT_SOURCE_DIR}/tests/integration/routing_test.py" --cli $<TARGET_FILE:edge0-infer> --backend ${backend} --profile 35b)
    endforeach()
    moe_add_test(NAME moe_test_prerouter_35b_cpu LEVEL L3 COMMAND uv run
        --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
        "${PROJECT_SOURCE_DIR}/tests/integration/prerouter_reference.py" --cli $<TARGET_FILE:edge0-infer> --profile 35b)
endif()

add_executable(moe_test_prerouter_state unit/models/edge0_35b/prerouter_test.cpp)
target_include_directories(moe_test_prerouter_state PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_prerouter_state PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_prerouter_state)
moe_add_test(NAME moe_test_prerouter_finish_predicted_state LEVEL L2 COMMAND moe_test_prerouter_state)
moe_add_test(NAME moe_test_prerouter_finish_predicted_state_device LEVEL L2 DEVICE COMMAND moe_test_prerouter_state --backend metal)
moe_add_test(NAME moe_test_predicted_ledger LEVEL L2 COMMAND moe_test_prerouter_state --ledger)
moe_add_test(NAME moe_test_route_trace LEVEL L1 COMMAND ${moe_python} "${PROJECT_SOURCE_DIR}/tests/perf/route_trace_test.py")
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_trace_counts LEVEL L2 COMMAND ${moe_python} "${PROJECT_SOURCE_DIR}/tests/integration/route_trace_test.py" $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_trace_counts_device LEVEL L2 DEVICE COMMAND ${moe_python} "${PROJECT_SOURCE_DIR}/tests/integration/route_trace_test.py" $<TARGET_FILE:edge0-infer> metal)
endif()
add_executable(moe_test_prerouter_admission unit/models/edge0_35b/prerouter_admission_test.cpp)
target_include_directories(moe_test_prerouter_admission PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_prerouter_admission PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_prerouter_admission)
moe_add_test(NAME moe_test_edge0_35b_prerouter_admission LEVEL L1 COMMAND moe_test_prerouter_admission "${PROJECT_SOURCE_DIR}")
moe_add_test(NAME moe_test_model_variants_prerouter_budget LEVEL L2 COMMAND moe_test_prerouter_admission "${PROJECT_SOURCE_DIR}")

if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_predicted_long_35b LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/prerouter_long_test.py" --cli $<TARGET_FILE:edge0-infer>)
endif()

add_executable(moe_test_session_reuse unit/session/reuse_test.cpp)
target_include_directories(moe_test_session_reuse PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_session_reuse PRIVATE moe_session)
moe_warnings(moe_test_session_reuse)
moe_add_test(NAME moe_test_session_reuse LEVEL L1 COMMAND moe_test_session_reuse)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_multi_turn LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_multi_turn_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
    moe_add_test(NAME moe_test_multi_turn_35b LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal --profile 35b)
endif()
add_executable(moe_template_probe unit/app/template_probe.cpp)
target_include_directories(moe_template_probe PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_template_probe PRIVATE moe_app)
moe_warnings(moe_template_probe)
moe_add_test(NAME moe_test_template_preserve LEVEL L1 COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
    "${PROJECT_SOURCE_DIR}/tests/unit/app/check_template_preserve.py" $<TARGET_FILE:moe_template_probe>)
add_executable(moe_test_history unit/app/history_test.cpp)
target_include_directories(moe_test_history PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_history PRIVATE moe_app)
moe_warnings(moe_test_history)
moe_add_test(NAME moe_test_history LEVEL L1 COMMAND moe_test_history)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_rerender LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/rerender_test.py" $<TARGET_FILE:moe_template_probe> $<TARGET_FILE:edge0-infer>)
endif()
add_executable(moe_test_checkpoint unit/models/edge0_35b/checkpoint_test.cpp)
target_include_directories(moe_test_checkpoint PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_checkpoint PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_checkpoint)
moe_add_test(NAME moe_test_checkpoint LEVEL L2 COMMAND moe_test_checkpoint)
moe_add_test(NAME moe_test_checkpoint_device LEVEL L2 DEVICE COMMAND moe_test_checkpoint --backend metal)
add_executable(moe_test_checkpoint_budget unit/app/checkpoint_budget_test.cpp)
target_include_directories(moe_test_checkpoint_budget PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_checkpoint_budget PRIVATE moe_app)
moe_warnings(moe_test_checkpoint_budget)
moe_add_test(NAME moe_test_checkpoint_budget LEVEL L2 COMMAND moe_test_checkpoint_budget)
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_multi_turn_messages LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_messages_test.py" --cli $<TARGET_FILE:edge0-infer>
        --record "${CMAKE_BINARY_DIR}/multi-turn-messages.json")
endif()

if(MOE_BUILD_CLI)
    add_executable(moe_test_chat_line unit/app/chat_line_test.cpp unit/cli/chat_lines_test.cpp "${PROJECT_SOURCE_DIR}/tools/cli/chat_line.cpp" "${PROJECT_SOURCE_DIR}/tools/cli/chat_history.cpp")
    target_include_directories(moe_test_chat_line PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    moe_warnings(moe_test_chat_line)
    moe_add_test(NAME moe_test_chat_line LEVEL L1 COMMAND moe_test_chat_line)
    moe_add_test(NAME moe_test_chat_cli LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/chat_cli_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_chat_cli_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/chat_cli_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
endif()

if(TARGET edge0-infer)
    moe_add_test(NAME moe_test_cancel LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/cancel_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_cancel_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/cancel_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
    moe_add_test(NAME moe_test_cancel_35b LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/cancel_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal --profile 35b)
endif()

add_executable(moe_json_probe unit/model/json_probe.cpp)
target_link_libraries(moe_json_probe PRIVATE moe_model)
moe_warnings(moe_json_probe)
moe_add_test(NAME moe_test_ordered_json LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/unit/model/check_json.py" $<TARGET_FILE:moe_json_probe>)
add_executable(moe_template_tools_probe unit/app/template_tools_probe.cpp)
target_link_libraries(moe_template_tools_probe PRIVATE moe_models_edge0_35b)
moe_warnings(moe_template_tools_probe)
moe_add_test(NAME moe_test_template_tools LEVEL L1 COMMAND uv run
    --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
    "${PROJECT_SOURCE_DIR}/tests/unit/app/check_template_tools.py"
    $<TARGET_FILE:moe_template_tools_probe> $<TARGET_FILE:moe_json_probe>)
add_executable(moe_test_edge0_35b_output unit/models/edge0_35b/output_parser_test.cpp)
target_include_directories(moe_test_edge0_35b_output PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_edge0_35b_output PRIVATE moe_models_edge0_35b)
moe_warnings(moe_test_edge0_35b_output)
moe_add_test(NAME moe_test_edge0_35b_output LEVEL L1 COMMAND moe_test_edge0_35b_output)
foreach(item IN ITEMS tool_results stop_match)
    add_executable(moe_test_${item} unit/app/${item}_test.cpp)
    target_include_directories(moe_test_${item} PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(moe_test_${item} PRIVATE moe_app)
    moe_warnings(moe_test_${item})
    moe_add_test(NAME moe_test_${item} LEVEL L1 COMMAND moe_test_${item})
endforeach()
add_executable(moe_test_thinking_tools_35b integration/thinking_tools_35b_test.cpp)
target_link_libraries(moe_test_thinking_tools_35b PRIVATE edge0::edge0)
target_compile_definitions(moe_test_thinking_tools_35b PRIVATE MOE_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
moe_warnings(moe_test_thinking_tools_35b)
moe_add_test(NAME moe_test_thinking_tools_35b LEVEL L3 DEVICE COMMAND moe_test_thinking_tools_35b)
add_executable(moe_thinking_rerender_probe unit/app/thinking_rerender_test.cpp)
target_include_directories(moe_thinking_rerender_probe PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(moe_thinking_rerender_probe PRIVATE moe_app)
moe_warnings(moe_thinking_rerender_probe)
foreach(backend IN ITEMS cpu metal)
    if(backend STREQUAL "metal")
        set(thinking_suffix "_device")
        set(thinking_device DEVICE)
    else()
        set(thinking_suffix "")
        set(thinking_device "")
    endif()
    moe_add_test(NAME moe_test_thinking_tools${thinking_suffix} LEVEL L2 ${thinking_device}
        COMMAND uv run --project "${PROJECT_SOURCE_DIR}/tools/assets" --offline --no-sync python -I
        "${PROJECT_SOURCE_DIR}/tests/integration/thinking_tools_test.py" --cli $<TARGET_FILE:edge0-infer>
        --backend ${backend} --probe $<TARGET_FILE:moe_thinking_rerender_probe>)
endforeach()

add_executable(moe_test_sampling_rules unit/app/sampling_test.cpp)
target_include_directories(moe_test_sampling_rules PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(moe_test_sampling_rules PRIVATE moe_app)
moe_warnings(moe_test_sampling_rules)
moe_add_test(NAME moe_test_sampling_rules LEVEL L1 COMMAND moe_test_sampling_rules)
moe_add_test(NAME moe_test_sampling LEVEL L2 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/integration/sampling_test.py" --cli $<TARGET_FILE:edge0-infer>)
moe_add_test(NAME moe_test_sampling_device LEVEL L2 DEVICE COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/integration/sampling_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)

if(MOE_BUILD_CLI)
    foreach(kind protocol engine)
        add_executable(moe_test_serve_${kind} unit/cli/serve_${kind}_test.cpp
            "${PROJECT_SOURCE_DIR}/tools/cli/serve_protocol.cpp")
        if(kind STREQUAL "engine")
            target_sources(moe_test_serve_${kind} PRIVATE "${PROJECT_SOURCE_DIR}/tools/cli/serve_engine.cpp")
        endif()
        target_include_directories(moe_test_serve_${kind} PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
        target_include_directories(moe_test_serve_${kind} SYSTEM PRIVATE
            "${PROJECT_SOURCE_DIR}/third_party/cpp-httplib" "${PROJECT_SOURCE_DIR}/third_party/nlohmann-json")
        moe_warnings(moe_test_serve_${kind})
        moe_add_test(NAME moe_test_serve_${kind} LEVEL L1 COMMAND moe_test_serve_${kind}
            "${PROJECT_SOURCE_DIR}/tests/fixtures/portal/requests.json")
    endforeach()
endif()
if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_serve LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_serve_device LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_test.py" --cli $<TARGET_FILE:edge0-infer> --backend metal)
endif()

if(MOE_BUILD_CLI)
    moe_add_test(NAME moe_test_serve_sdk_anthropic LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_sdk_runner.py" --with anthropic==2.8.0
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_sdk_anthropic_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_serve_sdk LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_sdk_runner.py"
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_sdk_test.py" --cli $<TARGET_FILE:edge0-infer>)
endif()
if(MOE_BUILD_EXAMPLES)
    moe_add_test(NAME moe_test_example_native LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/example_native_test.py" --example $<TARGET_FILE:moe_example_native_chat>)
endif()
if(MOE_BUILD_CLI)
    add_executable(moe_portal_decode integration/portal_decode.cpp)
    target_link_libraries(moe_portal_decode PRIVATE edge0::edge0)
    moe_warnings(moe_portal_decode)
    if(APPLE AND GGML_METAL)
        set(_moe_serve_35b_backend metal)
    else()
        set(_moe_serve_35b_backend cpu)
    endif()
    moe_add_test(NAME moe_test_serve_35b LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_sdk_runner.py"
        --with openai==3.26.1 --with anthropic==2.8.0
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_35b_test.py"
        --cli $<TARGET_FILE:edge0-infer> --decode $<TARGET_FILE:moe_portal_decode>
        --backend ${_moe_serve_35b_backend})
endif()
if(MOE_BUILD_CLI)
    add_executable(moe_test_delivery_failure_continue integration/portal_failure_probe.cpp)
    target_link_libraries(moe_test_delivery_failure_continue PRIVATE edge0::edge0)
    moe_warnings(moe_test_delivery_failure_continue)
    moe_add_test(NAME moe_test_delivery_failure_continue LEVEL L2 COMMAND moe_test_delivery_failure_continue)
endif()

moe_add_test(NAME moe_test_memory_metric LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/integration/memory_metric_test.py")
moe_add_test(NAME moe_test_multi_turn_counts LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_counts_test.py")
if(MOE_BUILD_CLI)
    foreach(kind threads budget)
        moe_add_test(NAME moe_test_${kind}_cli LEVEL L2 COMMAND ${moe_python}
            "${PROJECT_SOURCE_DIR}/tests/integration/${kind}_cli_test.py" --cli $<TARGET_FILE:edge0-infer>)
    endforeach()
endif()
moe_add_test(NAME moe_test_platform_android LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/platform/android_test.py")
moe_add_test(NAME moe_test_platform_android_cli LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/platform/android_cli_test.py")
moe_add_test(NAME moe_test_platform_serve_sync LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/platform/serve_sync_test.py")
if(WIN32)
    add_executable(moe_test_platform_windows unit/platform/file_windows_test.cpp)
    target_include_directories(moe_test_platform_windows PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(moe_test_platform_windows PRIVATE moe_platform)
    moe_warnings(moe_test_platform_windows)
    moe_add_test(NAME moe_test_platform_windows LEVEL L1 COMMAND moe_test_platform_windows)
endif()
add_executable(moe_test_threads_cli_selection unit/app/thread_selection_test.cpp)
target_include_directories(moe_test_threads_cli_selection PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_threads_cli_selection PRIVATE edge0::edge0)
moe_warnings(moe_test_threads_cli_selection)
moe_add_test(NAME moe_test_threads_cli_selection LEVEL L2 COMMAND moe_test_threads_cli_selection)

add_executable(moe_test_mixed_buffers unit/platform/mixed_buffers_test.cpp)
target_include_directories(moe_test_mixed_buffers PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_mixed_buffers PRIVATE moe_platform ggml)
moe_warnings(moe_test_mixed_buffers)
moe_add_test(NAME moe_test_mixed_buffers LEVEL L2 DEVICE COMMAND moe_test_mixed_buffers)

add_executable(moe_test_exec_stages unit/exec/stages_test.cpp)
target_include_directories(moe_test_exec_stages PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_exec_stages PRIVATE moe_exec)
moe_warnings(moe_test_exec_stages)
moe_add_test(NAME moe_test_exec_stages LEVEL L1 COMMAND moe_test_exec_stages)

add_executable(moe_test_exec_replicas unit/exec/replicas_test.cpp)
target_include_directories(moe_test_exec_replicas PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_exec_replicas PRIVATE moe_exec)
moe_warnings(moe_test_exec_replicas)
moe_add_test(NAME moe_test_exec_replicas LEVEL L1 COMMAND moe_test_exec_replicas)
add_executable(moe_test_exec_vulkan_replicas unit/exec/vulkan_replicas_test.cpp)
target_include_directories(moe_test_exec_vulkan_replicas PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_exec_vulkan_replicas PRIVATE moe_exec)
moe_warnings(moe_test_exec_vulkan_replicas)
moe_add_test(NAME moe_test_exec_vulkan_replicas LEVEL L1 DEVICE COMMAND moe_test_exec_vulkan_replicas)

add_executable(moe_test_weights_mixed unit/weights/mixed_storage_test.cpp)
target_include_directories(moe_test_weights_mixed PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_weights_mixed PRIVATE moe_weights)
moe_warnings(moe_test_weights_mixed)
moe_add_test(NAME moe_test_weights_mixed LEVEL L1 COMMAND moe_test_weights_mixed)

add_executable(moe_test_session_mixed_kv unit/session/mixed_kv_test.cpp)
target_include_directories(moe_test_session_mixed_kv PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_session_mixed_kv PRIVATE moe_session)
moe_warnings(moe_test_session_mixed_kv)
moe_add_test(NAME moe_test_session_mixed_kv LEVEL L1 COMMAND moe_test_session_mixed_kv)

add_executable(moe_test_mixed_stages unit/app/mixed_stages_test.cpp)
target_include_directories(moe_test_mixed_stages PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_mixed_stages PRIVATE edge0::edge0)
moe_warnings(moe_test_mixed_stages)
moe_add_test(NAME moe_test_mixed_stages LEVEL L2 COMMAND moe_test_mixed_stages)

if(MOE_BUILD_CLI AND GGML_METAL)
    moe_add_test(NAME moe_test_multi_turn_mixed LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_test.py" --cli $<TARGET_FILE:edge0-infer>
        --backend cpu --prefill-backend metal)
    moe_add_test(NAME moe_test_multi_turn_35b_mixed LEVEL L3 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_test.py" --cli $<TARGET_FILE:edge0-infer>
        --backend cpu --prefill-backend metal --profile 35b --derive-counts)
endif()
if(MOE_BUILD_CLI AND GGML_VULKAN)
    moe_add_test(NAME moe_test_multi_turn_mixed_vulkan LEVEL L2 DEVICE COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/multi_turn_test.py" --cli $<TARGET_FILE:edge0-infer>
        --backend cpu --prefill-backend vulkan)
endif()

add_executable(moe_test_mixed_metal unit/app/mixed_metal_test.cpp)
target_include_directories(moe_test_mixed_metal PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_mixed_metal PRIVATE edge0::edge0)
moe_warnings(moe_test_mixed_metal)
moe_add_test(NAME moe_test_mixed_metal LEVEL L2 DEVICE COMMAND moe_test_mixed_metal)
add_executable(moe_test_mixed_vulkan_stages unit/app/mixed_vulkan_stages_test.cpp)
target_include_directories(moe_test_mixed_vulkan_stages PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_mixed_vulkan_stages PRIVATE edge0::edge0)
moe_warnings(moe_test_mixed_vulkan_stages)
moe_add_test(NAME moe_test_mixed_vulkan_stages LEVEL L2 DEVICE COMMAND moe_test_mixed_vulkan_stages)
moe_add_test(NAME moe_test_mixed_vulkan LEVEL L2 DEVICE COMMAND moe_test_mixed_metal vulkan)

add_executable(moe_test_platform_ios_cleanup unit/cli/ios_cleanup_test.cpp)
target_include_directories(moe_test_platform_ios_cleanup PRIVATE "${PROJECT_SOURCE_DIR}")
moe_warnings(moe_test_platform_ios_cleanup)
moe_add_test(NAME moe_test_platform_ios_cleanup LEVEL L1 COMMAND moe_test_platform_ios_cleanup)
moe_add_test(NAME moe_test_platform_ios LEVEL L1 COMMAND ${moe_python} "${PROJECT_SOURCE_DIR}/tests/platform/ios_test.py")
moe_add_test(NAME moe_test_platform_ios_cli LEVEL L1 COMMAND ${moe_python} "${PROJECT_SOURCE_DIR}/tests/platform/ios_cli_test.py")

add_executable(moe_test_platform_system unit/platform/system_test.cpp)
target_include_directories(moe_test_platform_system PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_platform_system PRIVATE moe_platform ggml)
moe_warnings(moe_test_platform_system)
moe_add_test(NAME moe_test_platform_system LEVEL L1 COMMAND moe_test_platform_system)

add_executable(moe_test_observe_concurrency unit/app/observe_concurrency_test.cpp)
target_include_directories(moe_test_observe_concurrency PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/third_party/ggml/include")
target_link_libraries(moe_test_observe_concurrency PRIVATE edge0::edge0)
moe_warnings(moe_test_observe_concurrency)
moe_add_test(NAME moe_test_observe_concurrency LEVEL L2 COMMAND moe_test_observe_concurrency)

if(MOE_BUILD_CLI)
    add_executable(moe_test_serve_status unit/cli/serve_status_test.cpp "${PROJECT_SOURCE_DIR}/tools/cli/serve_status.cpp")
    target_include_directories(moe_test_serve_status PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    target_include_directories(moe_test_serve_status SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann-json")
    target_link_libraries(moe_test_serve_status PRIVATE edge0::edge0)
    moe_warnings(moe_test_serve_status)
    moe_add_test(NAME moe_test_serve_status LEVEL L1 COMMAND moe_test_serve_status)
endif()

# Delegated models (such as Edge0 145B): the native API refuses their directories;
# edge0-infer serve and chat start and proxy edge0-strata, tested with a fake engine.
add_executable(moe_test_delegated_directory unit/app/delegated_directory_test.cpp)
target_include_directories(moe_test_delegated_directory PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(moe_test_delegated_directory PRIVATE edge0::edge0)
moe_warnings(moe_test_delegated_directory)
moe_add_test(NAME moe_test_delegated_directory LEVEL L1 COMMAND moe_test_delegated_directory)
moe_add_test(NAME moe_test_fake_strata LEVEL L1 COMMAND ${moe_python}
    "${PROJECT_SOURCE_DIR}/tests/integration/fake_strata.py" --self-test)
if(MOE_BUILD_CLI)
    add_executable(moe_test_strata_registry unit/cli/strata_registry_test.cpp
        "${PROJECT_SOURCE_DIR}/tools/cli/serve_delegate_registry.cpp" "${PROJECT_SOURCE_DIR}/tools/cli/serve_delegate_status.cpp")
    target_include_directories(moe_test_strata_registry PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    target_include_directories(moe_test_strata_registry SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann-json")
    moe_warnings(moe_test_strata_registry)
    moe_add_test(NAME moe_test_strata_registry LEVEL L1 COMMAND moe_test_strata_registry)
    moe_add_test(NAME moe_test_serve_strata LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/serve_strata_test.py" --cli $<TARGET_FILE:edge0-infer>)
    moe_add_test(NAME moe_test_chat_strata LEVEL L2 COMMAND ${moe_python}
        "${PROJECT_SOURCE_DIR}/tests/integration/chat_strata_test.py" --cli $<TARGET_FILE:edge0-infer>)
endif()
