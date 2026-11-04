#include "platform/file.h"
#include <algorithm>
#include <limits>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace edge0::platform {
namespace file_testing { std::atomic<bool> disable_direct_read{false}; std::atomic<Hooks*> hooks{nullptr}; }
bool supports_direct_read() {
#ifdef __APPLE__
    return !file_testing::disable_direct_read.load();
#else
    // Windows has no pread-style direct mode here and Android O_DIRECT requires
    // aligned offsets and buffers; GGUF experts are only 32-byte aligned. Keep
    // cached reads instead of hidden bounce buffers.
    return false;
#endif
}
bool FileIdentity::operator==(const FileIdentity& rhs) const noexcept {
    return size == rhs.size && inode == rhs.inode && device == rhs.device &&
        modified_seconds == rhs.modified_seconds && modified_nanoseconds == rhs.modified_nanoseconds;
}

#ifdef _WIN32
namespace {
Error io_error(const std::string& action) {
    return {ErrorCategory::backend_error, action + ": Windows error " + std::to_string(GetLastError())};
}
std::optional<std::wstring> widen(const std::string& path) {
    if (path.empty()) return std::wstring{};
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), static_cast<int>(path.size()), nullptr, 0);
    if (size <= 0) return std::nullopt;
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), static_cast<int>(path.size()), wide.data(), size) != size)
        return std::nullopt;
    return wide;
}
// The file is opened with FILE_FLAG_OVERLAPPED so that several reader threads
// sharing one handle issue truly concurrent reads (a synchronous handle would
// serialise all I/O on it). Each read carries its own offset and its own event;
// a thread-local manual-reset event is reused across a thread's reads. A null
// event would signal the file handle itself and race between threads.
HANDLE completion_event() {
    thread_local struct EventHolder {
        HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ~EventHolder() { if (handle) CloseHandle(handle); }
    } holder;
    return holder.handle;
}
bool read_at(void* handle, std::uint64_t offset, void* destination, std::size_t size, bool& eof) {
    auto* out = static_cast<char*>(destination);
    std::size_t done = 0;
    eof = false;
    const HANDLE event = completion_event();
    if (!event) return false;
    while (done < size) {
        const DWORD chunk = static_cast<DWORD>(std::min<std::uint64_t>(size - done, 0x40000000ULL));
        OVERLAPPED overlapped{};
        const std::uint64_t at = offset + done;
        const auto parts = split_file_offset(at);
        overlapped.Offset = parts.low;
        overlapped.OffsetHigh = parts.high;
        overlapped.hEvent = event;
        DWORD got = 0;
        if (!::ReadFile(static_cast<HANDLE>(handle), out + done, chunk, &got, &overlapped)) {
            const DWORD error = GetLastError();
            if (error == ERROR_HANDLE_EOF) { eof = true; return true; }
            if (error != ERROR_IO_PENDING) return false;
            if (!GetOverlappedResult(static_cast<HANDLE>(handle), &overlapped, &got, TRUE)) {
                if (GetLastError() == ERROR_HANDLE_EOF) { eof = true; return true; }
                return false;
            }
        }
        if (got == 0) { eof = true; return true; }
        done += got;
    }
    return true;
}
}
Result<std::shared_ptr<ReadFile>> ReadFile::open(const std::string& path, ReadMode mode) {
    if (mode == ReadMode::direct && !supports_direct_read())
        return Error{ErrorCategory::unsupported, "Direct file reads are unavailable"};
    if (auto* hooks = file_testing::hooks.load()) ++hooks->opens;
    const auto wide = widen(path);
    if (!wide) return Error{ErrorCategory::invalid_argument, "Path is not valid UTF-8: " + path};
    // Share read/write/delete to mirror POSIX open(O_RDONLY), which never locks:
    // other handles (and the engine's own writers) may append or replace the
    // file while we hold it open; identity() then reports the change.
    const HANDLE handle = CreateFileW(wide->c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return io_error("open " + path);
    auto file = std::shared_ptr<ReadFile>(new ReadFile(handle));
    auto id = file->identity();
    if (!id) return id.error();
    return file;
}
ReadFile::~ReadFile() { CloseHandle(static_cast<HANDLE>(handle_)); }
Result<FileIdentity> ReadFile::identity() const {
    if (auto* hooks = file_testing::hooks.load()) ++hooks->identities;
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(static_cast<HANDLE>(handle_), &info)) return io_error("GetFileInformationByHandle");
    if (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        return Error{ErrorCategory::invalid_argument, "Source must be a regular file"};
    const std::uint64_t size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    const std::uint64_t inode = (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    // FILETIME counts 100 ns ticks since 1601-01-01; convert to Unix seconds + ns.
    ULARGE_INTEGER ticks{};
    ticks.LowPart = info.ftLastWriteTime.dwLowDateTime;
    ticks.HighPart = info.ftLastWriteTime.dwHighDateTime;
    constexpr std::uint64_t unix_epoch_offset = 116444736000000000ULL;
    const std::uint64_t since_unix = ticks.QuadPart >= unix_epoch_offset ? ticks.QuadPart - unix_epoch_offset : 0;
    return FileIdentity{size, inode, info.dwVolumeSerialNumber,
        static_cast<std::int64_t>(since_unix / 10000000ULL),
        static_cast<std::int64_t>((since_unix % 10000000ULL) * 100ULL)};
}
std::optional<Error> ReadFile::read_exact(std::uint64_t offset, void* destination, std::size_t size, const FileIdentity* known_identity) const {
    const auto id = known_identity ? Result<FileIdentity>(*known_identity) : identity();
    if (!id) return id.error();
    if ((size && !destination) || offset > id.value().size || size > id.value().size - offset)
        return Error{ErrorCategory::invalid_argument, "File slice is out of bounds"};
    bool eof = false;
    if (!read_at(handle_, offset, destination, size, eof)) return io_error("ReadFile");
    if (eof) return Error{ErrorCategory::backend_error, "Unexpected end of file"};
    return {};
}
std::optional<Error> ReadFile::read_vector(std::uint64_t offset, const std::vector<ReadTarget>& targets,
    const FileIdentity& known_identity) const {
    std::size_t total = 0;
    for (const auto& target : targets) {
        if (!target.data || !target.bytes || target.bytes > std::numeric_limits<std::size_t>::max() - total)
            return Error{ErrorCategory::invalid_argument, "Invalid vector read target"};
        total += target.bytes;
    }
    if (targets.empty() || offset > known_identity.size || total > known_identity.size - offset)
        return Error{ErrorCategory::invalid_argument, "Vector read is out of bounds"};
    auto* hooks = file_testing::hooks.load();
    auto maximum = [](std::atomic<std::size_t>& value, std::size_t next) {
        auto old = value.load(); while (old < next && !value.compare_exchange_weak(old, next)) {}
    };
    if (hooks) { maximum(hooks->max_segments, targets.size()); maximum(hooks->max_bytes, total); }
    // Windows has no preadv; read each batch segment by segment while keeping the
    // same batch shape (<= 64 segments, capped by read_limit) so the test hooks
    // count calls like the POSIX path.
    std::size_t index = 0, within = 0, done = 0;
    while (index < targets.size()) {
        std::size_t allowance = hooks && hooks->read_limit ? hooks->read_limit : std::numeric_limits<std::size_t>::max();
        struct Segment { char* data; std::size_t bytes; };
        std::vector<Segment> batch;
        for (std::size_t i = index; i < targets.size() && batch.size() < 64 && allowance; ++i) {
            const auto start = i == index ? within : 0;
            const auto bytes = std::min(targets[i].bytes - start, allowance);
            batch.push_back({static_cast<char*>(targets[i].data) + start, bytes});
            allowance -= bytes;
        }
        if (hooks) {
            const auto call = ++hooks->vector_calls;
            if (hooks->fail_read_call && call == hooks->fail_read_call)
                return Error{ErrorCategory::backend_error, "Injected vector read failure"};
        }
        std::size_t batch_bytes = 0;
        for (const auto& segment : batch) {
            bool eof = false;
            if (!read_at(handle_, offset + done + batch_bytes, segment.data, segment.bytes, eof)) return io_error("ReadFile");
            if (eof) return Error{ErrorCategory::backend_error, "Unexpected end of file in vector read"};
            batch_bytes += segment.bytes;
        }
        done += batch_bytes;
        std::size_t remaining = batch_bytes;
        while (remaining) {
            const auto consumed = std::min(remaining, targets[index].bytes - within);
            remaining -= consumed; within += consumed;
            if (within == targets[index].bytes) { ++index; within = 0; }
        }
    }
    return {};
}
#else
namespace {
Error io_error(const std::string& action) {
    return {ErrorCategory::backend_error, action + ": " + std::strerror(errno)};
}
}
Result<std::shared_ptr<ReadFile>> ReadFile::open(const std::string& path, ReadMode mode) {
    if(mode==ReadMode::direct && !supports_direct_read())
        return Error{ErrorCategory::unsupported,"Direct file reads are unavailable"};
    if(auto* hooks=file_testing::hooks.load()) ++hooks->opens;
    int descriptor;
    do { descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC); } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) return io_error("open " + path);
    auto file = std::shared_ptr<ReadFile>(new ReadFile(descriptor));
#ifdef __APPLE__
    if(mode==ReadMode::direct) {
        if(::fcntl(descriptor,F_NOCACHE,1)!=0) return io_error("F_NOCACHE");
        if(::fcntl(descriptor,F_RDAHEAD,0)!=0) return io_error("F_RDAHEAD");
    }
#endif
    auto id = file->identity();
    if (!id) return id.error();
    return file;
}
ReadFile::~ReadFile() { ::close(descriptor_); }
Result<FileIdentity> ReadFile::identity() const {
    if(auto* hooks=file_testing::hooks.load()) ++hooks->identities;
    struct stat info{};
    if (::fstat(descriptor_, &info) != 0) return io_error("fstat");
    if (!S_ISREG(info.st_mode) || info.st_size < 0)
        return Error{ErrorCategory::invalid_argument, "Source must be a regular file"};
#ifdef __APPLE__
    const auto time = info.st_mtimespec;
#else
    const auto time = info.st_mtim;
#endif
    return FileIdentity{static_cast<std::uint64_t>(info.st_size), static_cast<std::uint64_t>(info.st_ino),
        static_cast<std::uint64_t>(info.st_dev), time.tv_sec, time.tv_nsec};
}
std::optional<Error> ReadFile::read_exact(std::uint64_t offset, void* destination, std::size_t size, const FileIdentity* known_identity) const {
    const auto id = known_identity?Result<FileIdentity>(*known_identity):identity();
    if (!id) return id.error();
    if ((size && !destination) || offset > id.value().size || size > id.value().size - offset ||
        offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - offset)
        return Error{ErrorCategory::invalid_argument, "File slice is out of bounds"};
    auto* out = static_cast<char*>(destination);
    std::size_t done = 0;
    while (done < size) {
        const auto chunk = std::min(size - done, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto count = ::pread(descriptor_, out + done, chunk, static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) return io_error("pread");
        if (count == 0) return Error{ErrorCategory::backend_error, "Unexpected end of file"};
        done += static_cast<std::size_t>(count);
    }
    return {};
}
std::optional<Error> ReadFile::read_vector(std::uint64_t offset, const std::vector<ReadTarget>& targets,
    const FileIdentity& known_identity) const {
    std::size_t total=0;
    for(const auto& target:targets) {
        if(!target.data || !target.bytes || target.bytes>std::numeric_limits<std::size_t>::max()-total)
            return Error{ErrorCategory::invalid_argument,"Invalid vector read target"};
        total+=target.bytes;
    }
    if(targets.empty() || offset>known_identity.size || total>known_identity.size-offset ||
       offset>std::uint64_t(std::numeric_limits<off_t>::max()) || total>std::uint64_t(std::numeric_limits<off_t>::max())-offset)
        return Error{ErrorCategory::invalid_argument,"Vector read is out of bounds"};
    auto* hooks=file_testing::hooks.load();
    auto maximum=[](std::atomic<std::size_t>& value,std::size_t next) {
        auto old=value.load();while(old<next && !value.compare_exchange_weak(old,next)) {}
    };
    if(hooks) { maximum(hooks->max_segments,targets.size()); maximum(hooks->max_bytes,total); }
    std::size_t index=0,within=0,done=0;
    while(index<targets.size()) {
        std::vector<iovec> vectors;
        std::size_t allowance=hooks && hooks->read_limit?hooks->read_limit:std::size_t(std::numeric_limits<ssize_t>::max());
        for(std::size_t i=index;i<targets.size() && vectors.size()<64 && allowance;++i) {
            const auto start=i==index?within:0;
            const auto bytes=std::min(targets[i].bytes-start,allowance);
            vectors.push_back({static_cast<char*>(targets[i].data)+start,bytes});allowance-=bytes;
        }
        if(hooks) {
            const auto call=++hooks->vector_calls;
            if(hooks->fail_read_call && call==hooks->fail_read_call)
                return Error{ErrorCategory::backend_error,"Injected vector read failure"};
        }
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
        const auto count=::preadv(descriptor_,vectors.data(),static_cast<int>(vectors.size()),static_cast<off_t>(offset+done));
#else
        const auto count=::pread(descriptor_,vectors.front().iov_base,vectors.front().iov_len,static_cast<off_t>(offset+done));
#endif
        if(count<0 && errno==EINTR) continue;
        if(count<0) return io_error("preadv");
        if(count==0) return Error{ErrorCategory::backend_error,"Unexpected end of file in vector read"};
        auto remaining=static_cast<std::size_t>(count);done+=remaining;
        while(remaining) {
            const auto consumed=std::min(remaining,targets[index].bytes-within);
            remaining-=consumed;within+=consumed;
            if(within==targets[index].bytes) { ++index;within=0; }
        }
    }
    return {};
}
#endif

}
