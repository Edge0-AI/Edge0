#include "posix_io.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace edge0 {

bool parse_mib(const char * value, size_t & bytes) {
    bytes = 0;
    if (!value || !*value) return true;
    size_t mib = 0;
    constexpr size_t maximum = std::numeric_limits<size_t>::max() / (1024 * 1024);
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || mib > (maximum - (*p - '0')) / 10) return false;
        mib = mib * 10 + (*p - '0');
    }
    bytes = mib * 1024 * 1024;
    return true;
}

bool resolve_mib(int cli_mib, const char * env, size_t & bytes) {
    if (cli_mib < 0) return false;
    return parse_mib(cli_mib ? std::to_string(cli_mib).c_str() : env, bytes);
}

size_t page_size() {
    static const size_t result = [] {
        const long value = sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<size_t>(value) : size_t(4096);
    }();
    return result;
}

bool read_rss(size_t & bytes) {
    // The resident counter includes file, anonymous and shared allocations.
    // smaps_rollup walks all page tables; repeating that after every 16 MiB
    // discard stalls large models and starves inference behind the budget lock.
    FILE * file = std::fopen("/proc/self/statm", "r");
    if (!file) return false;
    size_t total = 0, resident = 0;
    const bool ok = std::fscanf(file, "%zu %zu", &total, &resident) == 2 &&
                    !std::ferror(file) && resident <= SIZE_MAX / page_size();
    std::fclose(file);
    if (ok) bytes = resident * page_size();
    return ok;
}

bool page_range(size_t file_size, ByteRange input, ByteRange & output) {
    if (!input.length || input.offset >= file_size || input.length > file_size - input.offset) return false;
    const size_t page = page_size();
    const size_t start = input.offset - input.offset % page;
    const size_t end = input.offset + input.length;
    const size_t padding = (page - end % page) % page;
    if (end > SIZE_MAX - padding) return false;
    output = {start, end + padding - start};
    return true;
}

bool expert_range(size_t file_size, size_t offset, size_t stride,
                  size_t experts, int64_t expert, ByteRange & output) {
    if (!stride || !experts || expert < 0 || static_cast<uint64_t>(expert) >= experts ||
        offset > file_size || experts > (file_size - offset) / stride) return false;
    output = {offset + static_cast<size_t>(expert) * stride, stride};
    return true;
}

ReadOnlyFile::~ReadOnlyFile() { close(); }

bool ReadOnlyFile::open(const char * path, bool map) {
    close();
    fd_ = ::open(path, O_RDONLY | O_CLOEXEC);
    struct stat info{};
    if (fd_ < 0 || fstat(fd_, &info) || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        static_cast<uint64_t>(info.st_size) > SIZE_MAX) { close(); return false; }
    size_ = static_cast<size_t>(info.st_size);
    if (map) {
        mapping_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) { mapping_ = nullptr; close(); return false; }
    }
    return true;
}

bool ReadOnlyFile::read(size_t offset, void * destination, size_t length) const {
    if (fd_ < 0 || offset > size_ || length > size_ - offset) return false;
    size_t done = 0;
    while (done < length) {
        const size_t chunk = std::min(length - done, size_t(1) << 30);
        const ssize_t got = pread(fd_, static_cast<char *>(destination) + done, chunk,
                                  static_cast<off_t>(offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
}

bool ReadOnlyFile::advise(ByteRange range) const {
    ByteRange pages{};
    if (fd_ < 0 || !page_range(size_, range, pages)) return false;
    // fadvise reports its error directly, without setting errno.
    const int error = posix_fadvise(fd_, static_cast<off_t>(range.offset),
                                   static_cast<off_t>(range.length), POSIX_FADV_WILLNEED);
    if (error) {
        std::fprintf(stderr, "[pref-trace] posix_fadvise failed: %s\n", std::strerror(error));
        return false;
    }
    if (mapping_ && madvise(static_cast<char *>(mapping_) + pages.offset, pages.length, MADV_WILLNEED)) {
        std::fprintf(stderr, "[pref-trace] madvise(WILLNEED) failed: %s\n", std::strerror(errno));
        return false;
    }
    return true;
}

bool ReadOnlyFile::matches_mapping(const void * address, ByteRange range) const {
    if (range.offset > size_ || !range.length || range.length > size_ - range.offset) return false;
    struct stat info{};
    if (fd_ < 0 || fstat(fd_, &info)) return false;
    FILE * maps = std::fopen("/proc/self/maps", "r");
    if (!maps) return false;
    char * line = nullptr;
    size_t capacity = 0;
    bool found = false;
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(address);
    while (getline(&line, &capacity, maps) >= 0) {
        unsigned long begin, end, offset, inode;
        unsigned device_major, device_minor;
        char permissions[5]{};
        if (std::sscanf(line, "%lx-%lx %4s %lx %x:%x %lu", &begin, &end, permissions,
                        &offset, &device_major, &device_minor, &inode) != 7) continue;
        if (permissions[0] == 'r' && permissions[1] == '-' && pointer >= begin && pointer < end &&
            range.length <= end - pointer && offset <= range.offset && pointer - begin == range.offset - offset &&
            inode == info.st_ino && device_major == major(info.st_dev) && device_minor == minor(info.st_dev)) {
            found = true;
            break;
        }
    }
    std::free(line);
    std::fclose(maps);
    return found;
}

void ReadOnlyFile::close() {
    if (mapping_) munmap(mapping_, size_);
    if (fd_ >= 0) ::close(fd_);
    mapping_ = nullptr;
    fd_ = -1;
    size_ = 0;
}

} // namespace edge0
