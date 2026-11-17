#ifndef STRATA_METAL_DS4_OS_APPLE_H
#define STRATA_METAL_DS4_OS_APPLE_H
// ds4_os_apple.h - the Linux OS surface ds4_os.hpp uses, on macOS.
//
// ds4_os.hpp is the engine's storage/output helpers: unbuffered reads, "is this page resident", huge pages, CPU affinity.
// macOS has none of those calls under those names, and its kernel will not honour most of the requests anyway, so this
// header maps each one onto the nearest thing that works (or onto a no-op whose answer is "unknown"), which is all the
// engine's policies need.  Included at the top of ds4_os.hpp under __APPLE__.
//
// Found by compiling src/dsv41/ds4_model.cu against metal/include/metal_compat (M5 Max, 2026-10-23): 20 errors, every one
// in ds4_os.hpp.
#if defined(__APPLE__)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

// O_DIRECT has no macOS spelling: the flag that means "do not keep this in the page cache" is F_NOCACHE on the descriptor.
// A zero flag here keeps ::open() valid, and ds4_os.hpp's F_NOCACHE path (added beside it) does the work.
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

// posix_fadvise: F_RDADVISE is the read-ahead note, there is no "drop this range" call.  Both answer 0 (success).
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif
#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif
static inline int posix_fadvise(int, off_t, off_t, int) { return 0; }

// mincore does not exist.  The callers ask "is this page in RAM"; the honest macOS answer is "unknown", and the header's
// comment already defines that as false.
static inline int mincore(void*, size_t, unsigned char*) { errno = ENOSYS; return -1; }

// huge pages and NUMA binding: no equivalent, and no benefit worth an API for this engine
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif
#ifndef SYS_mbind
#define SYS_mbind 0
#endif
static inline long syscall(long, ...) { return -1; }

// CPU affinity: macOS schedules on its own; the engine only uses this to pin the token thread.  No-op, reported as success.
typedef struct { unsigned long __bits[16]; } cpu_set_t;
#ifndef CPU_ZERO
#define CPU_ZERO(set) ((void) (set))
#define CPU_SET(cpu, set) ((void) (set))
#endif
static inline int sched_setaffinity(pid_t, size_t, const cpu_set_t*) { return 0; }
static inline int pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t*) { return 0; }

#endif   // __APPLE__
#endif   // STRATA_METAL_DS4_OS_APPLE_H
