# GGML options for a build in which this repository is a subproject (add_subdirectory()).
#
# The top-level build takes these options from CMakePresets.json. A host project gets the same
# values by default, so that it builds the configuration the engine is tested with: the macOS
# defaults follow the mac-metal preset, iOS the ios preset, Android the android-cpu preset, and
# every other system (Windows included) the cpu/windows presets; Vulkan stays opt-in.
#
# An option the host has already set, as a cache entry (-D) or as a normal variable before
# add_subdirectory(), is left untouched. The defaults are stored as cache entries, so they can be
# changed later like any other option.

function(moe_ggml_default name value type)
    if(NOT DEFINED ${name})
        set(${name} "${value}" CACHE ${type} "ggml: default of the Edge0 engine presets")
    endif()
endfunction()

foreach(option IN ITEMS GGML_BLAS GGML_ACCELERATE GGML_LLAMAFILE GGML_OPENMP GGML_BACKEND_DL
                        GGML_BUILD_TESTS GGML_BUILD_EXAMPLES)
    moe_ggml_default(${option} OFF BOOL)
endforeach()
moe_ggml_default(GGML_CPU ON BOOL)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    moe_ggml_default(GGML_METAL ON BOOL)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    moe_ggml_default(GGML_METAL ON BOOL)
    moe_ggml_default(GGML_NATIVE OFF BOOL)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "" OR CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        moe_ggml_default(GGML_CPU_ARM_ARCH "armv8.4-a+dotprod+fp16" STRING)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    moe_ggml_default(GGML_METAL OFF BOOL)
    moe_ggml_default(GGML_NATIVE OFF BOOL)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        moe_ggml_default(GGML_CPU_ARM_ARCH "armv8.6-a+dotprod+i8mm+fp16" STRING)
    endif()
else()
    moe_ggml_default(GGML_METAL OFF BOOL)
endif()
# Embedding the Metal library only matters when Metal is on; follow the resolved choice.
moe_ggml_default(GGML_METAL_EMBED_LIBRARY "${GGML_METAL}" BOOL)
