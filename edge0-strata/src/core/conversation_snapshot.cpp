#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "conversation_checked.hpp"
#include "conversation_copy.hpp"
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>

namespace strata::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode == 2 || st.kv_q4 || st.kv_int8)) {   // a streamed one reads its host copy
        error = "conversation snapshot: hybrid K8V4 cannot be a ring and needs distinct format flags";
        return false;
    }
    const auto s = strata::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        error = "conversation snapshot: invalid K/V geometry";
        return false;
    }
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t per = st.kv_q4 ? (size_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint
    // restore reconstructs that row when rewinding to an earlier prefix.
    const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V (#293,
    // STRATA_KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    l = {(st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated, cells, pooled, s.page_size, 0, 0, 0, 0, 0};
    using conversation_detail::product;
    if (!product(l.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(l.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                           int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(l.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        error = "conversation snapshot: K/V byte count overflow";
        return false;
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if (st.kv_hybrid) {
        if (!product(l.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                   (uint64_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            error = "conversation snapshot: hybrid V byte count overflow";
            return false;
        }
        l.value_scales = 0;
    }
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q4 : st.v_q4, host ? st.host.k_scale : st.k_scale,
                nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                    : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    return true;
}

// STRATA_PARK_VRAM_MIB=N (with STRATA_PARK_ASYNC): a park also keeps the whole K/V of its image in VRAM, in a slot
// of its own, while the slots' total stays within N MiB; a restore of that image copies from the slot device to
// device (~1 ms) instead of from the host buffers over PCIe, and needs no wait for the background copy.  A slot is
// released with the image or its K/V layers (their tokens), or all at once under VRAM pressure
// (conversation_park_vram_release_all).  Host thread only; the background copy only reads a slot's bytes, and every
// reuse or free of one comes after a join (conversation_park_async_begin, release_all).
struct VramKvSlot {
    void* p = nullptr;
    size_t cap = 0, used = 0;
    uint64_t tag = 0;   // 0: free (kept for reuse by size)
    int dev = -1;
    struct Part { uint64_t tag; size_t part, offset, bytes; };   // one layer's part (k, v, k_scale, v_scale, pooled)
    std::vector<Part> parts;
};
struct VramKv {
    std::vector<std::unique_ptr<VramKvSlot>> slots;
    uint64_t next_tag = 1;   // slot and layer tags, never reused
};
VramKv& vram_kv() { static VramKv v; return v; }
size_t park_vram_budget() {
    static const size_t b = [] {
        const char* v = std::getenv("STRATA_PARK_VRAM_MIB");
        const long long mib = v != nullptr ? std::atoll(v) : 0;
        return mib > 0 ? (size_t) mib << 20 : (size_t) 0;
    }();
    return b;
}
bool park_trace() { static const bool on = std::getenv("STRATA_PARK_TRACE") != nullptr; return on; }
void vram_kv_free(VramKvSlot& s) {
    int cur = s.dev;
    (void) cudaGetDevice(&cur);
    if (cur != s.dev) (void) cudaSetDevice(s.dev);
    (void) cudaFree(s.p);
    if (cur != s.dev) (void) cudaSetDevice(cur);
    (void) cudaGetLastError();
    s.p = nullptr; s.cap = 0;
}
void vram_kv_release(VramKvSlot& s) { s.tag = 0; s.used = 0; s.parts.clear(); }   // free, kept for reuse
// A slot of `need` bytes on `dev`: the best-fitting free one, else (the free ones freed first) a new one, rounded up
// to 64 MiB so a growing conversation's next parks reuse it, while the slots' total stays within the budget and
// 1.5 GB of VRAM stays free beside it.  nullptr, with `why`, otherwise.
VramKvSlot* vram_kv_take(size_t need, int dev, const char*& why) {
    VramKv& v = vram_kv();
    const size_t budget = park_vram_budget();
    if (budget == 0) { why = "off"; return nullptr; }
    VramKvSlot* best = nullptr;
    for (auto& s : v.slots)
        if (s->tag == 0 && s->dev == dev && s->cap >= need && (best == nullptr || s->cap < best->cap)) best = s.get();
    if (best == nullptr) {
        size_t held = 0;
        for (size_t i = 0; i < v.slots.size();) {
            if (v.slots[i]->tag != 0) { held += v.slots[i]->cap; ++i; continue; }
            vram_kv_free(*v.slots[i]);
            v.slots.erase(v.slots.begin() + (std::ptrdiff_t) i);
        }
        const size_t want = (need + ((size_t) 64 << 20) - 1) & ~(((size_t) 64 << 20) - 1);
        if (held > budget || want > budget - held) { why = "budget"; return nullptr; }
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess || fr < want + ((size_t) 3 << 29)) {
            (void) cudaGetLastError();
            why = "vram";
            return nullptr;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, want) != cudaSuccess) { (void) cudaGetLastError(); why = "cudaMalloc"; return nullptr; }
        auto s = std::make_unique<VramKvSlot>();
        s->p = p; s->cap = want; s->dev = dev;
        best = s.get();
        v.slots.push_back(std::move(s));
    }
    best->tag = v.next_tag++;
    best->used = 0;
    best->parts.clear();
    return best;
}
// The live slot on the current device holding parts under `tag` (nullptr: none, or tag 0)
const VramKvSlot* vram_kv_find(uint64_t tag) {
    if (tag == 0) return nullptr;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
    for (const auto& s : vram_kv().slots) {
        if (s->tag == 0 || s->dev != dev) continue;
        for (const auto& p : s->parts)
            if (p.tag == tag) return s.get();
    }
    return nullptr;
}
// Whether the slot holds, under `tag`, at least sizes[i] bytes of every part i that has any; `at` gets the offsets
bool vram_kv_locate(const VramKvSlot& s, uint64_t tag, const std::array<size_t,5>& sizes, std::array<size_t,5>& at) {
    std::array<bool,5> have{};
    for (const auto& p : s.parts)
        if (p.tag == tag && p.part < at.size() && p.bytes >= sizes[p.part]) { have[p.part] = true; at[p.part] = p.offset; }
    for (size_t i = 0; i < sizes.size(); ++i)
        if (sizes[i] > 0 && !have[i]) return false;
    return true;
}

struct ParkAsync {
    std::mutex mu;
    std::future<cudaError_t> pending;
    struct Item { uint8_t* dst; size_t n, at; };
    std::vector<Item> items;
    uint8_t* stage = nullptr;
    size_t cap = 0, used = 0;
    void* host[2] = {nullptr, nullptr};   // pinned, 4 MiB each
    cudaStream_t stream = nullptr;
    int dev = -1;
    bool active = false;
    VramKvSlot* slot = nullptr;   // STRATA_PARK_VRAM_MIB: the begun park's slot (the stage is not used then)
    uint64_t slot_tag = 0;        // its tag, kept past launch for the image's and its layers' tokens
};
ParkAsync& park_async() { static ParkAsync a; return a; }
constexpr size_t kParkPiece = (size_t) 4 << 20;
bool park_async_on() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_PARK_ASYNC"); return v != nullptr && v[0] == '1'; }();
    return on;
#endif
}
bool on_device(const void* p) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    (void) p;   // (no pointer-attribute query in the shims: the slot takes no part, as when the query fails)
    return false;
#else
    cudaPointerAttributes at{};
    if (cudaPointerGetAttributes(&at, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return at.type == cudaMemoryTypeDevice;
#endif
}
// STRATA_PARK_TRACE: where conversation_kv_save's time goes since the park's begin (ms): the pointer-attribute
// queries, the host buffer resizes and the copies (into the stage, or the synchronous ones).  Read by the park trace.
struct KvSaveTrace { double attrs = 0, resize = 0, copy = 0; };
KvSaveTrace& kv_save_trace() { static KvSaveTrace t; return t; }
bool park_trace_on() {
    static const bool on = std::getenv("STRATA_PARK_TRACE") != nullptr;
    return on;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = conversation_detail::copy_nonblocking(dst, src, n);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
} // namespace

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    // STRATA_PARK_VRAM_MIB: this layer's parts go into the park's slot under a fresh tag (0: no slot this park)
    image.dev_tag = park_async().slot != nullptr ? vram_kv().next_tag++ : 0;
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / strata::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        {   // STRATA_PARK_VRAM_MIB: with a slot, the whole part goes device to device into it (a restore of this image
            // reads it from there) and the host buffer gets only the bytes past the kept prefix, read from the slot on
            // the background thread; a part the slot cannot take is copied synchronously - never through the stage,
            // whose offsets launch would read from the slot
            ParkAsync& a = park_async();
            if (a.slot != nullptr) {
                VramKvSlot& s = *a.slot;
                const bool fits = src[i] != nullptr && sizes[i] > 0 && s.used + sizes[i] <= s.cap &&
                                  (st.kv_mode == 0 || on_device(src[i]));   // (kv_mode 0: the pools are device memory)
                if (fits) {
                    const size_t at0 = s.used;
                    const cudaError_t e = conversation_detail::copy_nonblocking(static_cast<uint8_t*>(s.p) + at0, src[i], sizes[i], true);
                    if (e != cudaSuccess) { error = std::string("conversation snapshot slot copy: ") + cudaGetErrorString(e); return false; }
                    s.parts.push_back({image.dev_tag, i, at0, sizes[i]});
                    s.used += sizes[i];
                    dst[i]->resize_for_overwrite(sizes[i]);
                    if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                            a.items.push_back({p, n, at0 + at});
                            return true;
                        })) return false;
                } else {
                    dst[i]->resize(sizes[i]);
                    if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                            return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error);
                        })) return false;
                }
                if (reused_bytes) *reused_bytes += keep;
                continue;
            }
        }
        // STRATA_PARK_ASYNC: device bytes go to the VRAM stage now and into the buffer on the background thread
        ParkAsync& a = park_async();
        const bool trace = park_trace_on();
        auto tick = [trace] { return trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}; };
        const auto t0 = tick();
        // (kv_mode 0: the pools are device memory by construction, so no pointer-attribute query)
        const bool staged = a.active && src[i] != nullptr && sizes[i] > keep && a.used + (sizes[i] - keep) <= a.cap &&
                            (st.kv_mode == 0 || on_device(src[i]));
        const auto t1 = tick();
        if (staged) dst[i]->resize_for_overwrite(sizes[i]);
        else dst[i]->resize(sizes[i]);
        const auto t2 = tick();
        const bool copied = dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                const uint8_t* s = src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr;
                if (!staged) return transfer(p, s, n, error);
                const cudaError_t e = conversation_detail::copy_nonblocking(a.stage + a.used, s, n, true);
                if (e != cudaSuccess) { error = std::string("conversation snapshot stage copy: ") + cudaGetErrorString(e); return false; }
                a.items.push_back({p, n, a.used});
                a.used += n;
                return true;
            });
        if (trace) {
            auto ms = [](std::chrono::steady_clock::time_point x, std::chrono::steady_clock::time_point y) {
                return std::chrono::duration<double, std::milli>(y - x).count(); };
            KvSaveTrace& t = kv_save_trace();
            t.attrs += ms(t0, t1); t.resize += ms(t1, t2); t.copy += ms(t2, tick());
        }
        if (!copied) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

namespace {
// After the authoritative pools were rewritten: the VRAM slots of a streamed state still hold the outgoing
// conversation, so resolve must refill them before any attention reads (shared by restore and direct copy).
bool refresh_residency(const QsaState& st, const ModelGeometry& g, int64_t upto, std::string& error) {
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    if (st.kv_mode == 2 && upto > 0) {
        auto shapes = strata::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        strata::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                        std::max<int64_t>(0, end - st.n_slots), end, st.n_slots, shapes, nullptr);
    }
    const auto status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
    return false;
}
} // namespace

bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error, bool queued) {
    Layout lf{}, lt{};
    if (!layout(from, g, upto, index, lf, error) || !valid(from, lf, upto, error) ||
        !layout(to, g, upto, index, lt, error) || !valid(to, lt, upto, error)) return false;
    if (lf.format != lt.format || lf.cells != lt.cells || lf.page_size != lt.page_size ||
        lf.pooled_rows != lt.pooled_rows || lf.data != lt.data || lf.value_data != lt.value_data ||
        lf.scales != lt.scales || lf.value_scales != lt.value_scales || lf.pooled != lt.pooled) {
        error = "conversation snapshot: direct K/V copy between different layouts";
        return false;
    }
    // The same bytes conversation_kv_save + conversation_kv_restore move, without the host image: each part
    // straight from the source's authoritative pool into the target's (device or mapped host memory), through the
    // private non-blocking stream the other snapshot copies use (complete on return, before the residency refresh).
    // `queued`: the copies are only queued, and a streamed state's refresh waits for them on the device (an event).
    const auto src = pools(from);
    const auto dst = pools(to);
    const std::array<size_t,5> sizes = {lf.data, lf.value_data, lf.scales, lf.value_scales, lf.pooled};
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!sizes[i]) continue;
        if (!src[i] || !dst[i]) { error = "conversation snapshot: missing state buffer"; return false; }
        const cudaError_t e = conversation_detail::copy_nonblocking(dst[i], src[i], sizes[i], queued);
        if (e != cudaSuccess) { error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); return false; }
    }
    if (queued && to.kv_mode != 0) {
        const cudaError_t e = conversation_detail::order_after_queued();
        if (e != cudaSuccess) { error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); return false; }
    }
    return refresh_residency(to, g, upto, error);
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    // STRATA_PARK_VRAM_MIB: from the image's VRAM slot when it holds every part (no host buffer is read: no join)
    const VramKvSlot* slot = vram_kv_find(image.dev_tag);
    if (slot == nullptr) conversation_park_join();
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto dst = pools(st);
    std::array<size_t,5> sizes{}, off{};
    for (size_t i = 0; i < src.size(); ++i) sizes[i] = src[i]->size();
    if (slot != nullptr && !vram_kv_locate(*slot, image.dev_tag, sizes, off)) { slot = nullptr; conversation_park_join(); }
    if (slot != nullptr) {
        for (size_t i = 0; i < src.size(); ++i)
            if (!transfer(dst[i], static_cast<const uint8_t*>(slot->p) + off[i], sizes[i], error)) return false;
        return refresh_residency(st, g, upto, error);
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (!src[i]->visit(0, src[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error);
            })) return false;
    return refresh_residency(st, g, upto, error);
}

bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool index, std::string& error) {
    const VramKvSlot* slot = vram_kv_find(image.dev_tag);   // (STRATA_PARK_VRAM_MIB, as in conversation_kv_restore)
    if (slot == nullptr) conversation_park_join();
    if (upto < 0 || upto > captured) {
        error = "conversation snapshot: prefix beyond the captured K/V";
        return false;
    }
    if (!conversation_kv_validate(image, st, g, captured, index, error)) return false;
    Layout l{};   // the prefix's extent; validation above checked the image against the captured one
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] > src[i]->size()) { error = "conversation snapshot: prefix beyond the K/V payload"; return false; }
    const auto dst = pools(st);
    // STRATA_PARK_VRAM_MIB: the prefix is the leading bytes of each part in the slot too
    std::array<size_t,5> off{};
    if (slot != nullptr && !vram_kv_locate(*slot, image.dev_tag, sizes, off)) { slot = nullptr; conversation_park_join(); }
    if (slot != nullptr) {
        for (size_t i = 0; i < src.size(); ++i)
            if (!transfer(dst[i], static_cast<const uint8_t*>(slot->p) + off[i], sizes[i], error)) return false;
        return refresh_residency(st, g, upto, error);
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (!src[i]->visit(0, sizes[i], [&](const uint8_t* p, size_t n, size_t at) {
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error);
            })) return false;
    return refresh_residency(st, g, upto, error);
}

bool conversation_kv_part_sizes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index,
                                std::array<uint64_t, 5>& sizes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    return true;
}

bool conversation_kv_source(SessionKvSource& out, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    SessionKvSource s;
    s.format = l.format; s.cells = l.cells; s.heads = g.n_head_kv; s.head_dim = g.head_dim;
    s.page_size = l.page_size; s.pooled_rows = l.pooled_rows; s.idx_dim = g.idx_key_dim;
    s.sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const auto src = pools(st);
    const auto sizes = s.sizes;
    auto why = std::make_shared<std::string>();
    s.read = [src, sizes, why](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > sizes[part] || n > sizes[part] - offset) { *why = "K/V read out of range"; return false; }
        return transfer(dst, src[part] ? static_cast<const uint8_t*>(src[part]) + offset : nullptr, n, *why);
    };
    s.error = [why] { return *why; };
    out = std::move(s);
    return true;
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error) {
    conversation_park_join();
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto authoritative = pools(st);
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return compare(static_cast<const uint8_t*>(authoritative[i]) + at, p, n, true);
            })) return false;
    if (st.kv_mode == 2 && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

void conversation_park_join() {
    ParkAsync& a = park_async();
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.pending.valid()) {
        const cudaError_t e = a.pending.get();
        if (e != cudaSuccess) std::fprintf(stderr, "strata: STRATA_PARK_ASYNC: a parked K/V copy failed (%s)\n", cudaGetErrorString(e));
    }
}

void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms) {
    const KvSaveTrace& t = kv_save_trace();
    attrs_ms = t.attrs; resize_ms = t.resize; copy_ms = t.copy;
}

bool conversation_park_async_begin(size_t need) {
    if (park_trace_on()) kv_save_trace() = {};   // (the sums cover this park's saves)
    conversation_park_join();   // (the stage is reused)
    ParkAsync& a = park_async();
    a.active = false;
    a.items.clear();
    a.used = 0;
    a.slot = nullptr;
    a.slot_tag = 0;
    if (!park_async_on() || need == 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    // STRATA_PARK_VRAM_MIB: a slot of the whole K/V instead of the stage, while the budget allows - taken before the
    // stage would be re-allocated, so a growing conversation's park pays no cudaFree (a device sync) when it gets one
    const char* why = "";
    a.slot = vram_kv_take(need, dev, why);
    a.slot_tag = a.slot != nullptr ? a.slot->tag : 0;
    if (park_trace() && park_vram_budget() != 0) {
        if (a.slot != nullptr)
            std::fprintf(stderr, "strata parktrace: kv slot %zu MB tag %llu\n", need >> 20, (unsigned long long) a.slot_tag);
        else
            std::fprintf(stderr, "strata parktrace: kv slot: none (%s)\n", why);
    }
    if (a.stage != nullptr && ((a.slot == nullptr && a.cap < need) || a.dev != dev)) {
        const int cur = dev;
        (void) cudaSetDevice(a.dev); (void) cudaFree(a.stage); (void) cudaSetDevice(cur);
        a.stage = nullptr; a.cap = 0;
    }
    if (a.stream != nullptr && a.dev != dev) {
        const int cur = dev;
        (void) cudaSetDevice(a.dev); (void) cudaStreamDestroy(a.stream);
        (void) cudaFreeHost(a.host[0]); (void) cudaFreeHost(a.host[1]); (void) cudaSetDevice(cur);
        a.stream = nullptr; a.host[0] = a.host[1] = nullptr;
    }
    a.dev = dev;
    if (a.stream == nullptr &&
        (cudaStreamCreateWithFlags(&a.stream, cudaStreamNonBlocking) != cudaSuccess ||
         cudaMallocHost(&a.host[0], kParkPiece) != cudaSuccess || cudaMallocHost(&a.host[1], kParkPiece) != cudaSuccess)) {
        if (a.host[0]) (void) cudaFreeHost(a.host[0]);
        if (a.host[1]) (void) cudaFreeHost(a.host[1]);
        if (a.stream) (void) cudaStreamDestroy(a.stream);
        a.stream = nullptr; a.host[0] = a.host[1] = nullptr;
        (void) cudaGetLastError();
        if (a.slot != nullptr) { vram_kv_release(*a.slot); a.slot = nullptr; a.slot_tag = 0; }   // (STRATA_PARK_VRAM_MIB)
        return false;
    }
    if (a.slot != nullptr) {   // STRATA_PARK_VRAM_MIB: the slot stands in for the stage this park
        a.active = true;
        return true;
    }
    if (a.stage == nullptr) {
        // rounded up to 64 MiB: the next parks of a growing conversation reuse it instead of re-allocating (a cudaFree
        // synchronizes the device, and the checkpoint DMA is in flight at this point)
        const size_t want = (need + ((size_t) 64 << 20) - 1) & ~(((size_t) 64 << 20) - 1);
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess || fr < want + ((size_t) 3 << 29) ||
            cudaMalloc((void**) &a.stage, want) != cudaSuccess) {
            a.stage = nullptr;
            (void) cudaGetLastError();
            return false;
        }
        a.cap = want;
    }
    a.active = true;
    return true;
}

void conversation_park_async_abandon() {
    ParkAsync& a = park_async();
    if (!a.active) return;
    a.active = false;
    a.items.clear();
    cudaStream_t s = nullptr;
    if (conversation_detail::private_stream(s) == cudaSuccess) (void) cudaStreamSynchronize(s);   // nothing reads the stage after
    (void) cudaGetLastError();
    if (a.slot != nullptr) { vram_kv_release(*a.slot); a.slot = nullptr; a.slot_tag = 0; }   // (STRATA_PARK_VRAM_MIB)
}

bool conversation_park_async_launch(std::string& error) {
    ParkAsync& a = park_async();
    if (!a.active) return true;
    a.active = false;
    cudaStream_t s = nullptr;
    cudaError_t e = conversation_detail::private_stream(s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);   // the device-to-device copies into the stage
    if (e != cudaSuccess) {
        a.items.clear();
        if (a.slot != nullptr) { vram_kv_release(*a.slot); a.slot = nullptr; a.slot_tag = 0; }   // (STRATA_PARK_VRAM_MIB)
        error = std::string("conversation snapshot stage: ") + cudaGetErrorString(e);
        return false;
    }
    // STRATA_PARK_VRAM_MIB: the host copies read the slot; it stays until the image and its K/V layers go (their tokens)
    const uint8_t* stage = a.slot != nullptr ? static_cast<const uint8_t*>(a.slot->p) : a.stage;
    a.slot = nullptr;
    if (a.items.empty()) return true;
    std::vector<ParkAsync::Item> items;
    items.swap(a.items);
    void* host[2] = {a.host[0], a.host[1]};
    cudaStream_t st = a.stream;
    const int dev = a.dev;
    std::lock_guard<std::mutex> lk(a.mu);
    // two pinned pieces in turn: the next piece's DMA runs while this one is copied into its segment
    a.pending = std::async(std::launch::async, [items = std::move(items), stage, host, st, dev] {
        conversation_detail::background_thread_placement();
        (void) cudaSetDevice(dev);
        cudaEvent_t done[2] = {nullptr, nullptr};
        cudaError_t e = cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming | cudaEventBlockingSync);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming | cudaEventBlockingSync);
        struct Piece { uint8_t* dst; size_t n; };
        Piece held[2] = {{nullptr, 0}, {nullptr, 0}};
        int b = 0;
        auto land = [&](int k) {
            if (held[k].dst == nullptr || e != cudaSuccess) return;
            e = cudaEventSynchronize(done[k]);
            if (e == cudaSuccess) std::memcpy(held[k].dst, host[k], held[k].n);
            held[k] = {nullptr, 0};
        };
        for (const auto& it : items)
            for (size_t o = 0; e == cudaSuccess && o < it.n; o += kParkPiece) {
                const size_t n = std::min(kParkPiece, it.n - o);
                land(b);
                if (e == cudaSuccess) e = cudaMemcpyAsync(host[b], stage + it.at + o, n, cudaMemcpyDeviceToHost, st);
                if (e == cudaSuccess) e = cudaEventRecord(done[b], st);
                if (e == cudaSuccess) held[b] = {it.dst + o, n};
                b ^= 1;
            }
        land(b);
        land(b ^ 1);
        for (cudaEvent_t d : done) if (d) (void) cudaEventDestroy(d);
        return e;
    });
    return true;
}

uint64_t conversation_park_slot_tag() { return park_async().slot_tag; }

void conversation_park_slot_release(uint64_t tag) {
    if (tag == 0) return;
    for (auto& s : vram_kv().slots)
        if (s->tag == tag) { vram_kv_release(*s); return; }
}

size_t conversation_park_vram_release_all() {
    if (park_vram_budget() == 0) return 0;   // off: nothing was taken, and the stage stays as it was
    conversation_park_join();   // the background copy may be reading a slot or the stage
    ParkAsync& a = park_async();
    VramKv& v = vram_kv();
    size_t freed = 0;
    for (size_t i = 0; i < v.slots.size();) {
        if (a.active && v.slots[i].get() == a.slot) { ++i; continue; }   // a park in progress keeps its own
        freed += v.slots[i]->cap;
        vram_kv_free(*v.slots[i]);
        v.slots.erase(v.slots.begin() + (std::ptrdiff_t) i);
    }
    if (a.stage != nullptr && !a.active) {
        int cur = a.dev;
        (void) cudaGetDevice(&cur);
        if (cur != a.dev) (void) cudaSetDevice(a.dev);
        (void) cudaFree(a.stage);
        if (cur != a.dev) (void) cudaSetDevice(cur);
        (void) cudaGetLastError();
        freed += a.cap;
        a.stage = nullptr; a.cap = 0;
    }
    return freed;
}

void conversation_park_join_for(const SavedConversation& image) {
    size_t hits = 0;
    for (const auto& k : image.kv) hits += vram_kv_find(k.dev_tag) != nullptr ? 1 : 0;
    if (park_trace() && park_vram_budget() != 0)
        std::fprintf(stderr, "strata parktrace: kv restore: %zu of %zu layers from the slot\n", hits, image.kv.size());
    if (hits < image.kv.size()) conversation_park_join();
}

} // namespace strata::core
