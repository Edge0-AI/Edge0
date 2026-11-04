#pragma once
#include "edge0/result.h"
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace edge0::platform {
enum class ReadMode { cached, direct };
bool supports_direct_read();
#ifdef _WIN32
// Split a 64-bit file offset into the low/high 32-bit halves an OVERLAPPED
// carries. Exposed so the high-word path is testable without a >4 GiB file.
struct FileOffsetParts { std::uint32_t low, high; };
constexpr FileOffsetParts split_file_offset(std::uint64_t offset) {
    return {static_cast<std::uint32_t>(offset & 0xFFFFFFFFULL),
            static_cast<std::uint32_t>(offset >> 32)};
}
#endif
namespace file_testing {
extern std::atomic<bool> disable_direct_read;
struct Hooks {
    std::atomic<std::size_t> opens{0}, identities{0}, vector_calls{0}, max_segments{0}, max_bytes{0};
    std::size_t read_limit = 0, fail_read_call = 0;
};
// Install only while no unrelated file operations are running; keep alive until writers join.
extern std::atomic<Hooks*> hooks;
}
struct ReadTarget { void* data; std::size_t bytes; };
struct FileIdentity {
    std::uint64_t size, inode, device;
    std::int64_t modified_seconds, modified_nanoseconds;
    bool operator==(const FileIdentity& rhs) const noexcept;
};
class ReadFile {
public:
    static Result<std::shared_ptr<ReadFile>> open(const std::string& path, ReadMode mode = ReadMode::cached);
    ~ReadFile();
    ReadFile(const ReadFile&) = delete;
    ReadFile& operator=(const ReadFile&) = delete;
    Result<FileIdentity> identity() const;
    std::optional<Error> read_exact(std::uint64_t offset, void* destination, std::size_t size, const FileIdentity* known_identity = nullptr) const;
    std::optional<Error> read_vector(std::uint64_t offset, const std::vector<ReadTarget>& targets, const FileIdentity& known_identity) const;
private:
#ifdef _WIN32
    // Stored as void* to keep <windows.h> out of this public-ish header; the
    // implementation casts back to HANDLE.
    explicit ReadFile(void* handle) : handle_(handle) {}
    void* handle_;
#else
    explicit ReadFile(int descriptor) : descriptor_(descriptor) {}
    int descriptor_;
#endif
};
}
