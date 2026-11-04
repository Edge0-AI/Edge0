add_library(moe_app STATIC runtime.cpp)
target_include_directories(moe_app PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/generated")
add_dependencies(moe_app moe_build_identity)
target_link_libraries(moe_app PRIVATE moe_exec moe_models_api moe_model moe_models_edge0_35b moe_session moe_platform ggml)
target_compile_definitions(moe_app PRIVATE MOE_VERSION="${PROJECT_VERSION}"
    MOE_GGML_COMMIT="${MOE_GGML_COMMIT}" MOE_GGML_VERSION="${MOE_GGML_VERSION}")
moe_warnings(moe_app)
target_sources(moe_app PRIVATE text/bpe.cpp text/decode.cpp text/unicode.cpp text/unicode-data.cpp)

target_sources(moe_app PRIVATE adapters.cpp assets.cpp)

target_sources(moe_app PRIVATE text/rules.cpp)
target_sources(moe_app PRIVATE text/api.cpp)
target_sources(moe_app PRIVATE forward_check.cpp)
target_sources(moe_app PRIVATE generation.cpp)

target_sources(moe_app PRIVATE diagnostics.cpp)
target_sources(moe_app PRIVATE submission_timing.cpp)
target_sources(moe_app PRIVATE text/history.cpp)
target_sources(moe_app PRIVATE sampling.cpp)
target_sources(moe_app PRIVATE text/tool_results.cpp text/stop.cpp)

target_sources(moe_app PRIVATE runtime_hooks.cpp stage_forward.cpp)
