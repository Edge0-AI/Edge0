#include <cstdint>
#ifdef _WIN32
// fread interposition via dlsym(RTLD_NEXT) is POSIX-only. The Windows engine
// reads through ReadFile, not stdio fread; a boundary probe awaits a
// platform-specific port. Provide stubs so dependent tests link; callers skip
// the probe assertions on Windows.
extern "C" void model_read_probe_start(const char *, std::uint64_t) {}
extern "C" bool model_read_probe_stop() { return false; }
#else
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/stat.h>
namespace {
ino_t inode = 0;
dev_t device = 0;
std::uint64_t boundary = 0, read_bytes = 0;
bool crossed = false;
} // namespace
extern "C" void model_read_probe_start(const char *path, std::uint64_t limit) {
    struct stat s{};
    if (stat(path, &s))
        std::abort();
    inode = s.st_ino;
    device = s.st_dev;
    boundary = limit;
    read_bytes = 0;
    crossed = false;
}
extern "C" bool model_read_probe_stop() {
    inode = 0;
    return read_bytes > 0 && !crossed;
}
extern "C" size_t fread(void *out, size_t size, size_t count, FILE *file) {
    using Read = size_t (*)(void *, size_t, size_t, FILE *);
    static auto original = reinterpret_cast<Read>(dlsym(RTLD_NEXT, "fread"));
    if (!original)
        std::abort();
    if (inode) {
        struct stat s{};
        if (!fstat(fileno(file), &s) && s.st_ino == inode && s.st_dev == device) {
            auto offset = ftello(file);
            auto bytes = size * count;
            read_bytes += bytes;
            if (offset < 0 || static_cast<std::uint64_t>(offset) > boundary ||
                bytes > boundary - static_cast<std::uint64_t>(offset))
                crossed = true;
        }
    }
    return original(out, size, count, file);
}
#endif
