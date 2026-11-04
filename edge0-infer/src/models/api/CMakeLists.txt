add_library(moe_models_api STATIC api.cpp)
target_include_directories(moe_models_api PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(moe_models_api PUBLIC moe_model moe_session moe_weights moe_exec)
moe_warnings(moe_models_api)
