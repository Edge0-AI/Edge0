// Where the Metal engine finds the files it reads at run time: the .metal kernel sources it compiles when it starts,
// ggml-common.h, and the default expert-profile / draft-vocabulary tables.
//
// A build used from its own source tree finds them through the paths compiled in (STRATA_*_DEFAULT). A component package
// (strata_control/engine_component.py) is moved away from the tree that built it, so the engine also looks beside its
// own executable first:
//
//     <exe dir>/metal/kernels/*.metal            the kernel sources (metal/kernels)
//     <exe dir>/metal/engine-msl/*.metal         the engine's own kernels (metal/src/engine/msl)
//     <exe dir>/metal/ggml-common.h
//     <exe dir>/metal/data/expert-profile.bin  draft_vocab.bin
//
// Order: the environment variable (STRATA_METAL_KERNELS, ...), then that folder when it exists, then the compiled-in
// path. Nothing here touches a build run from its source tree: the folder does not exist there.
#pragma once

#include <mach-o/dyld.h>
#include <sys/stat.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace strata_metal {

inline std::string exe_dir() {
    char raw[PATH_MAX * 2];
    uint32_t n = sizeof(raw);
    if (_NSGetExecutablePath(raw, &n) != 0) return "";
    char real[PATH_MAX];
    if (!realpath(raw, real)) return "";
    std::string s(real);
    const size_t slash = s.rfind('/');
    return slash == std::string::npos ? "" : s.substr(0, slash);
}

inline bool exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// `sub` is relative to <exe dir>/metal/, `compiled` is the path compiled into this build.
inline std::string resource(const char* env, const char* sub, const char* compiled) {
    if (env) {
        if (const char* e = std::getenv(env)) return e;
    }
    const std::string dir = exe_dir();
    if (!dir.empty()) {
        const std::string beside = dir + "/metal/" + sub;
        if (exists(beside)) return beside;
    }
    return compiled;
}

}  // namespace strata_metal
