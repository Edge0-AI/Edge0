add_library(moe_models_edge0_35b STATIC)
target_sources(moe_models_edge0_35b PRIVATE text.cpp output.cpp)
target_sources(moe_models_edge0_35b PRIVATE assets.cpp)
target_include_directories(moe_models_edge0_35b PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(moe_models_edge0_35b PUBLIC moe_models_api moe_model moe_session moe_weights moe_exec)
moe_warnings(moe_models_edge0_35b)

target_sources(moe_models_edge0_35b PRIVATE compute.cpp)
target_sources(moe_models_edge0_35b PRIVATE forward.cpp)
