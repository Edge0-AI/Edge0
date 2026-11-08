// Linux serving helpers. All offsets and lengths are bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace edge0 {

// Empty/unset means zero. Only unsigned decimal MiB values are accepted.
bool parse_mib(const char * value, size_t & bytes);
bool resolve_mib(int cli_mib, const char * env, size_t & bytes);
size_t page_size();
// Approximate whole-process resident bytes from the kernel counter, not virtual size.
bool read_rss(size_t & bytes);

struct ByteRange { size_t offset, length; };
// Advice includes boundary pages, but never goes beyond the file's last page.
bool page_range(size_t file_size, ByteRange input, ByteRange & output);
bool expert_range(size_t file_size, size_t offset, size_t stride,
                  size_t experts, int64_t expert, ByteRange & output);

class ReadOnlyFile {
public:
    ReadOnlyFile() = default;
    ~ReadOnlyFile();
    ReadOnlyFile(const ReadOnlyFile &) = delete;
    ReadOnlyFile & operator=(const ReadOnlyFile &) = delete;
    bool open(const char * path, bool map);
    bool read(size_t offset, void * destination, size_t length) const;
    bool advise(ByteRange range) const;
    bool matches_mapping(const void * address, ByteRange range) const;
    void close();
    size_t size() const { return size_; }
    int descriptor() const { return fd_; }
private:
    int fd_ = -1;
    size_t size_ = 0;
    void * mapping_ = nullptr;
};

} // namespace edge0
