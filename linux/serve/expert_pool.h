#pragma once
#include "posix_io.h"
#include "prefetch_internal.h"

#include <map>
#include <mutex>

namespace edge0 {

// Immutable fill-once slots: a returned pointer stays valid until quiescent
// shutdown. At capacity, new experts use the original mmap. No slot recycling
// means concurrent ggml readers cannot observe overwritten expert bytes.
class ExpertPool {
public:
    explicit ExpertPool(const ReadOnlyFile & file);
    ~ExpertPool();
    bool start(size_t capacity);
    bool add(const void * identity, const ExpertTensor & tensor);
    const char * resolve(const void * identity, int64_t expert, size_t stride);
    bool prefetch(ByteRange range);
    size_t used();
private:
    const char * fill(const void * identity, int64_t expert, size_t stride);
    const ReadOnlyFile & file_;
    std::mutex mutex_;
    struct Entry { ExpertTensor tensor; std::vector<size_t> slots; };
    std::map<const void *, Entry> entries_;
    char * arena_ = nullptr;
    size_t capacity_ = 0, used_ = 0;
    uint64_t hits_ = 0, fills_ = 0, failures_ = 0;
};

} // namespace edge0
