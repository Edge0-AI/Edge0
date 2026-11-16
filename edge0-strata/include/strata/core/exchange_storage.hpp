#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core::detail {

// Non-owning views: the complement and exchange allocations retain their original
// owners until FileExpertSource::close(). Ownership changes via slot IDs; the
// physical buffer addresses, bytes and CUDA aliases stay in place. Requires fully mapped/pinned expert slots.
//
// Block sizes: `initialize` is the uniform geometry (every expert block the same size). `initialize_sized` takes a
// size per expert (a pack whose layers differ, as the GGUF's mixed i-quants): the spares are kept per size, and
// since a swap exchanges two experts of one layer, the buffer a commit hands back has the size of the one it took,
// so every size keeps its number of spares for good.
class ExchangeStorage {
public:
    struct View {
        uint8_t* host = nullptr;
        const uint8_t* device = nullptr;
    };

    bool initialize(const std::vector<uint64_t>& offsets, uint8_t* resident_host,
                    const uint8_t* resident_device, uint64_t resident_bytes,
                    uint8_t* exchange_host, const uint8_t* exchange_device,
                    size_t count, size_t blob_bytes, std::string& error) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !exchange_host || !exchange_device ||
            !blob_bytes || !count || !resident_bytes || resident_bytes % blob_bytes ||
            count > std::numeric_limits<size_t>::max() / blob_bytes) {
            error = "invalid uniform mapped exchange geometry"; return false;
        }
        const size_t resident_slots = (size_t)(resident_bytes / blob_bytes);
        if (resident_slots > std::numeric_limits<size_t>::max() - count) {
            error = "exchange slot count overflow"; return false;
        }
        std::vector<bool> used(resident_slots, false);
        for (size_t i = 0; i < offsets.size(); ++i) {
            const uint64_t at = offsets[i];
            if (at == ~uint64_t{0}) continue;
            if (at >= resident_bytes || at % blob_bytes || used[(size_t)(at / blob_bytes)]) {
                error = "invalid or duplicate resident slot"; return false;
            }
            used[(size_t)(at / blob_bytes)] = true;
        }
        for (bool present : used) if (!present) {
            error = "unassigned resident slot"; return false;
        }
        const std::vector<uint64_t> sizes(offsets.size(), (uint64_t) blob_bytes);
        return initialize_sized(offsets, sizes, resident_host, resident_device, resident_bytes, exchange_host,
                                exchange_device, (uint64_t) count * (uint64_t) blob_bytes, count, 0, error);
    }

    /// The exchange allocation `initialize_sized` lays out: `count - bounce` buffers of every distinct size in
    /// `sizes` (zeros skipped), then `bounce` buffers of the largest.  0 when the geometry is empty or overflows.
    static uint64_t exchange_bytes_needed(const std::vector<uint64_t>& sizes, size_t count, size_t bounce) {
        const std::vector<size_t> classes = classes_of(sizes);
        if (classes.empty() || count == 0 || bounce > count) return 0;
        uint64_t sum = 0;
        for (const size_t c : classes) sum += c;
        const uint64_t lim = std::numeric_limits<uint64_t>::max();
        if (sum > lim / count) return 0;
        return (uint64_t) (count - bounce) * sum + (uint64_t) bounce * classes.back();
    }

    /// `sizes[i]`: expert i's block size (its layer's).  Spare ordinals `q < count - bounce` exist in every size;
    /// the last `bounce` ordinals only in the largest (buffers that are staged through and never committed).
    bool initialize_sized(const std::vector<uint64_t>& offsets, const std::vector<uint64_t>& sizes,
                          uint8_t* resident_host, const uint8_t* resident_device, uint64_t resident_bytes,
                          uint8_t* exchange_host, const uint8_t* exchange_device, uint64_t exchange_bytes,
                          size_t count, size_t bounce, std::string& error) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !exchange_host || !exchange_device || !count || bounce > count ||
            !resident_bytes || sizes.size() != offsets.size()) {
            error = "invalid mapped exchange geometry"; return false;
        }
        const std::vector<size_t> classes = classes_of(sizes);
        const uint64_t need = exchange_bytes_needed(sizes, count, bounce);
        if (classes.empty() || need == 0 || need > exchange_bytes ||
            classes.size() > std::numeric_limits<size_t>::max() / count) {
            error = "the exchange allocation does not hold the spares of every block size"; return false;
        }
        auto experts = std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<View> buffers;
        std::vector<size_t> buffer_bytes;
        std::vector<std::pair<uint64_t, uint64_t>> spans;   // (offset, size) of the resident blocks: none may overlap
        for (size_t i = 0; i < offsets.size(); ++i) {
            const uint64_t at = offsets[i];
            experts[i].store(kAbsent, std::memory_order_relaxed);
            if (at == ~uint64_t{0}) continue;
            if (sizes[i] == 0 || at >= resident_bytes || sizes[i] > resident_bytes - at) {
                error = "invalid resident slot"; return false;
            }
            experts[i].store(buffers.size(), std::memory_order_relaxed);
            buffers.push_back({resident_host + (size_t) at, resident_device + (size_t) at});
            buffer_bytes.push_back((size_t) sizes[i]);
            spans.emplace_back(at, sizes[i]);
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i)
            if (spans[i - 1].first + spans[i - 1].second > spans[i].first) {
                error = "invalid or duplicate resident slot"; return false;
            }
        std::vector<size_t> spares(classes.size() * count, kAbsent);
        uint64_t at = 0;
        for (size_t c = 0; c < classes.size(); ++c)
            for (size_t q = 0; q < count; ++q) {
                if (q >= count - bounce && c + 1 != classes.size()) continue;   // the bounce ordinals: the largest size only
                spares[c * count + q] = buffers.size();
                buffers.push_back({exchange_host + (size_t) at, exchange_device + (size_t) at});
                buffer_bytes.push_back(classes[c]);
                at += classes[c];
            }
        experts_ = std::move(experts);
        expert_count_ = offsets.size();
        expert_bytes_.assign(sizes.begin(), sizes.end());
        spares_.swap(spares);
        buffers_.swap(buffers);
        buffer_bytes_.swap(buffer_bytes);
        classes_ = classes;
        count_ = count;
        blob_bytes_ = classes.back();
        return true;
    }

    bool active() const { return blob_bytes_ != 0; }
    View resident(size_t expert) const {
        if (expert >= expert_count_) return {};
        const size_t slot = experts_[expert].load(std::memory_order_acquire);
        return slot != kAbsent ? buffers_[slot] : View{};
    }
    /// Spare `q` of the largest block size (any block fits it).
    View spare(size_t q) const { return active() ? spare(q, blob_bytes_) : View{}; }
    /// Spare `q` of blocks of exactly `bytes`.
    View spare(size_t q, size_t bytes) const {
        const size_t c = class_of(bytes);
        if (c == kAbsent || q >= count_) return {};
        const size_t id = spares_[c * count_ + q];
        return id != kAbsent ? buffers_[id] : View{};
    }

    // Only after every CPU reader and H2D copy of `in` has completed, and `out`
    // has landed in spare(q, bytes). No allocation, memcpy, or CUDA operation here.
    bool commit(size_t in, size_t out, size_t q, const uint8_t* staged, size_t bytes) {
        const size_t c = class_of(bytes);
        if (in >= expert_count_ || out >= expert_count_ || c == kAbsent || q >= count_ || !staged) return false;
        size_t& spare_id = spares_[c * count_ + q];
        const size_t incoming_slot = experts_[in].load(std::memory_order_acquire);
        if (spare_id == kAbsent || incoming_slot == kAbsent || experts_[out].load(std::memory_order_acquire) != kAbsent ||
            buffer_bytes_[incoming_slot] != bytes || expert_bytes_[out] != bytes || buffers_[spare_id].host != staged)
            return false;
        // A background router lookahead can query residency to skip redundant
        // file prefetches. Publish one atomic slot ID; the pointer/alias pair is
        // immutable. Actual compute readers still obey the completion contract.
        experts_[out].store(spare_id, std::memory_order_release);
        spare_id = incoming_slot;
        experts_[in].store(kAbsent, std::memory_order_release);
        ++exchanges_;
        avoided_bytes_ += bytes;
        return true;
    }

    uint64_t exchanges() const { return exchanges_; }
    uint64_t avoided_bytes() const { return avoided_bytes_; } // memcpy payload, not read+write total
    size_t block_sizes() const { return classes_.size(); }
    void clear() {
        experts_.reset(); expert_count_ = 0; expert_bytes_.clear(); buffers_.clear(); buffer_bytes_.clear(); spares_.clear();
        classes_.clear();
        count_ = 0; blob_bytes_ = 0; exchanges_ = avoided_bytes_ = 0;
    }

private:
    static constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
    static std::vector<size_t> classes_of(const std::vector<uint64_t>& sizes) {
        std::vector<size_t> c;
        for (const uint64_t b : sizes)
            if (b != 0 && b <= std::numeric_limits<size_t>::max()) c.push_back((size_t) b);
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        return c;
    }
    size_t class_of(size_t bytes) const {
        const auto it = std::lower_bound(classes_.begin(), classes_.end(), bytes);
        return it != classes_.end() && *it == bytes ? (size_t) (it - classes_.begin()) : kAbsent;
    }
    std::unique_ptr<std::atomic<size_t>[]> experts_;
    size_t expert_count_ = 0;
    std::vector<uint64_t> expert_bytes_; // each expert's block size, immutable
    std::vector<View> buffers_; // immutable physical addresses after initialize
    std::vector<size_t> buffer_bytes_; // each buffer's block size, immutable
    std::vector<size_t> spares_; // [size class * count_ + q]; touched only by the serialized cache manager
    std::vector<size_t> classes_; // the distinct block sizes, ascending
    size_t count_ = 0;
    size_t blob_bytes_ = 0; // the largest block size; 0 = not initialized
    uint64_t exchanges_ = 0, avoided_bytes_ = 0;
};

} // namespace strata::core::detail
