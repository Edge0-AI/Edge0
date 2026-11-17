// strata/platform/thread_name.hpp - names the engine's long-lived threads, so that a person (or a script) can tell the
// pool workers from the host thread and the helpers when looking at where they run: Linux `ps -L -o tid,psr,comm` and
// /proc/<pid>/task/<tid>/comm, Windows GetThreadDescription (Process Explorer, WinDbg, the tools/ probes).
// A name changes nothing the thread does.
//
// Linux and macOS: header-only.  Windows: the function is defined in src/kernels/cpu/pool.cpp, because that file already
// includes <windows.h> and this header must not pull it into every file that includes aux_cpus.hpp (its macros, and
// WIN32_LEAN_AND_MEAN, change what those files see).  The callers on Windows are the ones that link strata_kernels_cpu.
#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>

#include <cstdio>
#endif

namespace strata::platform {

/// Gives the calling thread a name: "strata-pool-3" for `name_this_thread("strata-pool", 3)`, "strata-host" for no index.
/// Linux keeps 15 characters, the rest is cut.  Never fails loudly: a system that cannot name a thread leaves it unnamed.
#if defined(_WIN32)
void name_this_thread(const char* base, int index = -1);
#else
inline void name_this_thread(const char* base, int index = -1) {
#if defined(__linux__) || defined(__APPLE__)
    char name[64];
    if (index >= 0) std::snprintf(name, sizeof name, "%s-%d", base, index);
    else std::snprintf(name, sizeof name, "%s", base);
#if defined(__linux__)
    name[15] = '\0';
    (void) pthread_setname_np(pthread_self(), name);
#else
    (void) pthread_setname_np(name);
#endif
#else
    (void) base;
    (void) index;
#endif
}
#endif

}  // namespace strata::platform
