#include "expert_pool.h"

#include <cstdio>
#include <sys/mman.h>

namespace edge0 {

ExpertPool::ExpertPool(const ReadOnlyFile & file) : file_(file) {}

ExpertPool::~ExpertPool() {
    if (arena_) munmap(arena_, capacity_);
    if (capacity_) std::fprintf(stderr, "[pref-trace] pool: tensors=%zu used=%zu bytes fills=%llu hits=%llu failures=%llu\n",
        entries_.size(), used_, (unsigned long long) fills_, (unsigned long long) hits_, (unsigned long long) failures_);
}

bool ExpertPool::start(size_t capacity) {
    if (arena_ || !capacity) return false;
    void * arena = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        std::fprintf(stderr, "[pref-trace] cannot reserve expert pool; using model mmap\n");
        return false;
    }
    arena_ = static_cast<char *>(arena);
    capacity_ = capacity;
    return true;
}

bool ExpertPool::add(const void * identity, const ExpertTensor & tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    ByteRange range{};
    if (!arena_ || entries_.count(identity) || !expert_range(file_.size(), tensor.offset, tensor.stride,
                                                           tensor.experts, 0, range)) return false;
    // Repacked/GPU tensors must keep their own backend layout. Enroll only
    // pointers into the exact read-only file mapping at the expected offset.
    range.length = tensor.stride * tensor.experts;
    if (!file_.matches_mapping(identity, range)) return false;
    entries_.emplace(identity, Entry{tensor, std::vector<size_t>(tensor.experts, SIZE_MAX)});
    return true;
}

const char * ExpertPool::fill(const void * identity, int64_t expert, size_t stride) {
    auto found = entries_.find(identity);
    if (found == entries_.end() || stride != found->second.tensor.stride) return nullptr;
    auto & entry = found->second;
    ByteRange range{};
    if (!expert_range(file_.size(), entry.tensor.offset, stride, entry.tensor.experts, expert, range)) return nullptr;
    size_t & slot = entry.slots[static_cast<size_t>(expert)];
    if (slot != SIZE_MAX) { ++hits_; return arena_ + slot; }
    const size_t padding = (page_size() - stride % page_size()) % page_size();
    if (stride > SIZE_MAX - padding || stride + padding > capacity_ - used_) return nullptr;
    if (!file_.read(range.offset, arena_ + used_, stride)) {
        // Drop partial data. No slot is published until a full pread succeeds.
        madvise(arena_ + used_, stride + padding, MADV_DONTNEED);
        if (!failures_++) std::fprintf(stderr, "[pref-trace] expert pool read failed; using model mmap\n");
        return nullptr;
    }
    slot = used_;
    used_ += stride + padding;
    ++fills_;
    return arena_ + slot;
}

const char * ExpertPool::resolve(const void * identity, int64_t expert, size_t stride) {
    std::lock_guard<std::mutex> lock(mutex_);
    return fill(identity, expert, stride);
}

bool ExpertPool::prefetch(ByteRange range) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & entry : entries_) {
        const auto & tensor = entry.second.tensor;
        if (range.length == tensor.stride && range.offset >= tensor.offset &&
            (range.offset - tensor.offset) % tensor.stride == 0) {
            const size_t expert = (range.offset - tensor.offset) / tensor.stride;
            if (expert < tensor.experts) return fill(entry.first, static_cast<int64_t>(expert), tensor.stride) != nullptr;
        }
    }
    return false;
}

size_t ExpertPool::used() { std::lock_guard<std::mutex> lock(mutex_); return used_; }

} // namespace edge0
