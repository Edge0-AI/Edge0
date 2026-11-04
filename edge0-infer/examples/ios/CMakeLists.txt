cmake_minimum_required(VERSION 3.22)
if("$ENV{MOE_IOS_TEAM}" STREQUAL "")
    message(FATAL_ERROR "MOE_IOS_TEAM must identify your signing team")
endif()
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET 17.0)
project(Edge0Chat LANGUAGES C CXX OBJCXX Swift)
set(CMAKE_XCODE_GENERATE_SCHEME ON)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.4-a+dotprod+fp16" CACHE STRING "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
add_subdirectory(../.. edge0)
set(CHAT_CORE ChatCore/Sources/ChatCore/ChatCore.swift ChatCore/Sources/ChatCore/ExampleFixtureCleanup.swift ChatCore/Sources/ChatCore/Status.swift ChatCore/Sources/ChatCore/Persistence.swift)
set(SHARED_APP "${CMAKE_CURRENT_SOURCE_DIR}/../shared/app.json")
set_source_files_properties("${SHARED_APP}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
add_executable(edge0-chat MACOSX_BUNDLE App/Edge0ChatApp.swift App/ChatViewModel.swift App/ContentView.swift App/Tokens.swift Edge0Bridge.mm ${CHAT_CORE} "${SHARED_APP}")
set(APP_ICONS "${CMAKE_CURRENT_SOURCE_DIR}/App/Assets.xcassets")
target_sources(edge0-chat PRIVATE "${APP_ICONS}")
set_source_files_properties("${APP_ICONS}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
set_target_properties(edge0-chat PROPERTIES XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon")
target_link_libraries(edge0-chat PRIVATE edge0::edge0)
target_compile_features(edge0-chat PRIVATE cxx_std_17)
set_source_files_properties(Edge0Bridge.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
set(MOE_IOS_EXAMPLE_BUNDLE "dev.edge0.$ENV{MOE_IOS_TEAM}.edge0-chat")
if(NOT "$ENV{MOE_IOS_EXAMPLE_BUNDLE_ID}" STREQUAL "")
    set(MOE_IOS_EXAMPLE_BUNDLE "$ENV{MOE_IOS_EXAMPLE_BUNDLE_ID}")
endif()
set_target_properties(edge0-chat PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "${MOE_IOS_EXAMPLE_BUNDLE}"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${MOE_IOS_EXAMPLE_BUNDLE}"
    XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "$ENV{MOE_IOS_TEAM}"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Edge0Bridge.h"
    XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_ENABLE_USER_SCRIPT_SANDBOXING "NO")
