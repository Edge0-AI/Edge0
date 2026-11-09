file(GLOB public_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/edge0/*.h")
set(header_units)
foreach(header IN LISTS public_headers)
    get_filename_component(name "${header}" NAME)
    set(unit "${CMAKE_CURRENT_BINARY_DIR}/public_${name}.cpp")
    file(WRITE "${unit}" "#include \"edge0/${name}\"\n")
    list(APPEND header_units "${unit}")
endforeach()
add_library(moe_public_headers OBJECT EXCLUDE_FROM_ALL ${header_units})
target_include_directories(moe_public_headers PRIVATE "${PROJECT_SOURCE_DIR}/include")
moe_warnings(moe_public_headers)
add_executable(moe_public_app EXCLUDE_FROM_ALL "${PROJECT_SOURCE_DIR}/tests/support/public_app.cpp")
target_link_libraries(moe_public_app PRIVATE edge0::edge0)
moe_warnings(moe_public_app)
# L0 runs before every commit and must stay within seconds: it only compiles the
# headers on their own. Linking the whole library is a separate L1 check.
add_custom_target(moe_public_header_check DEPENDS moe_public_headers)
moe_add_test(NAME moe_public_header_check LEVEL L0
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target moe_public_header_check)
add_custom_target(moe_public_app_check DEPENDS moe_public_app)
moe_add_test(NAME moe_public_app_check LEVEL L1
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target moe_public_app_check)
