add_executable(edge0-infer main.cpp)
target_link_libraries(edge0-infer PRIVATE edge0::edge0)
moe_warnings(edge0-infer)
if(WIN32)
    # Embed the UTF-8 active-code-page manifest (see edge0-infer.manifest).
    target_sources(edge0-infer PRIVATE edge0-infer.manifest)
endif()
target_sources(edge0-infer PRIVATE info.cpp inspect.cpp)
target_sources(edge0-infer PRIVATE text.cpp)

target_sources(edge0-infer PRIVATE forward.cpp)
target_sources(edge0-infer PRIVATE run.cpp)
target_sources(edge0-infer PRIVATE logging.cpp)

target_sources(edge0-infer PRIVATE weight_activity.cpp)

target_sources(edge0-infer PRIVATE diagnostics.cpp describe.cpp)

target_sources(edge0-infer PRIVATE bench.cpp)

target_sources(edge0-infer PRIVATE chat.cpp chat_line.cpp chat_history.cpp generation_summary.cpp)

target_sources(edge0-infer PRIVATE sampling.cpp)
target_sources(edge0-infer PRIVATE output.cpp)

target_sources(edge0-infer PRIVATE serve.cpp serve_protocol.cpp serve_engine.cpp serve_status.cpp)
target_include_directories(edge0-infer SYSTEM PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/cpp-httplib" "${PROJECT_SOURCE_DIR}/third_party/nlohmann-json")

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # Only the App wrapper needs a signing team; the library builds for iOS without one.
    if("$ENV{MOE_IOS_TEAM}" STREQUAL "")
        message(FATAL_ERROR "MOE_IOS_TEAM must identify your signing team (or set MOE_BUILD_CLI=OFF)")
    endif()
    enable_language(OBJCXX)
    set_source_files_properties(main.cpp PROPERTIES COMPILE_DEFINITIONS "main=moe_cli_main")
    target_sources(edge0-infer PRIVATE ios/main.mm ios/chat.mm)
    set_source_files_properties(ios/main.mm ios/chat.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(edge0-infer PRIVATE "-framework UIKit" "-framework Foundation")
    set(MOE_IOS_BUNDLE "dev.edge0.$ENV{MOE_IOS_TEAM}.edge0-infer")
    if(NOT "$ENV{MOE_IOS_BUNDLE_ID}" STREQUAL "")
        set(MOE_IOS_BUNDLE "$ENV{MOE_IOS_BUNDLE_ID}")
    endif()
    set_target_properties(edge0-infer PROPERTIES
        LINKER_LANGUAGE OBJCXX
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/ios/Info.plist.in"
        MACOSX_BUNDLE_GUI_IDENTIFIER "${MOE_IOS_BUNDLE}"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${MOE_IOS_BUNDLE}"
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "$ENV{MOE_IOS_TEAM}"
        XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
        XCODE_ATTRIBUTE_CLANG_LINK_OBJC_RUNTIME "NO"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES"
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")
endif()
