#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "conversation_copy.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot synchronize: ") + cudaGetErrorString(status);
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error, bool queued = false) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    const auto status = conversation_detail::copy_nonblocking(dst, src, bytes, queued);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

#if defined(STRATA_GPU_METAL)
}  // namespace strata::core
namespace strata::metal_engine {
// the running state lives in the Metal decode window (metal/src/engine/window.cpp), not in the session's buffers
bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, const strata::core::SessionState* ss, std::string& error);
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, const strata::core::SessionState* ss, std::string& error);
}  // namespace strata::metal_engine
namespace strata::core {
#endif

namespace {
struct CkptAsync {
    std::mutex mu;
    std::future<cudaError_t> pending;
    const void* dst = nullptr;   // the checkpoint bytes the copy in flight writes
    void* stage = nullptr;
    void* host = nullptr;   // pinned: the DMA lands here, then a plain memcpy (no driver staging beside the decode)
    cudaStream_t stream = nullptr;
    size_t cap = 0;
    int dev = -1;
};
CkptAsync& ckpt_async() { static CkptAsync a; return a; }
// STRATA_CKPT_VRAM_SLOTS=N (with STRATA_CKPT_ASYNC): the last N saved GDN states stay in VRAM (allocated only while
// more than 1.5 GB stays free); a restore of a checkpoint whose tag a slot holds copies device to device.
struct VramSlot { void* p = nullptr; size_t n = 0; uint64_t tag = 0, used = 0; int dev = -1; };
struct VramSlots { std::vector<VramSlot> s; uint64_t clock = 0, next_tag = 0; };
VramSlots& vram_slots() { static VramSlots v; return v; }
int vram_slots_n() {
    static const int n = [] { const char* v = std::getenv("STRATA_CKPT_VRAM_SLOTS"); return v ? std::max(0, std::min(16, std::atoi(v))) : 0; }();
    return n;
}
// a slot for `n` bytes on `dev`: a free one, a new one if VRAM allows, else the least recently used (nullptr: none)
VramSlot* vram_slot_for(size_t n, int dev) {
    VramSlots& v = vram_slots();
    VramSlot* best = nullptr;
    for (VramSlot& x : v.s)
        if (x.dev == dev && x.n == n && (best == nullptr || x.used < best->used)) best = &x;
    if ((int) v.s.size() < vram_slots_n()) {
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess && fr > n + ((size_t) 3 << 29)) {
            VramSlot x;
            if (cudaMalloc(&x.p, n) == cudaSuccess) { x.n = n; x.dev = dev; v.s.push_back(x); return &v.s.back(); }
        }
        (void) cudaGetLastError();
    }
    return best;
}
bool ckpt_async_on() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_CKPT_ASYNC"); return v != nullptr && v[0] == '1'; }();
    return on;
#endif
}
// STRATA_CKPT_GATHER=1 (CUDA, opt-in): the small running-state vectors (ple, then each layer's indexer tail, dead row
// and block position: 1 + 3 x layers pieces) are gathered device to device into one buffer, allocated once, and
// fetched with ONE copy into a pinned buffer, then copied into the vectors.  A small D2H copy issued while the
// checkpoint DMA (STRATA_CKPT_ASYNC) is in flight waits behind the 4 MiB piece on the copy engine: one such wait
// instead of 37.  The same bytes in the same places.  One set of buffers: the mutex serializes the saves a layer
// split runs on one thread per stage (STRATA_CKPT_THREADS), which otherwise gather through the same buffer.
struct CkptGather { std::mutex mu; void* stage = nullptr; void* host = nullptr; size_t cap = 0; int dev = -1; };
CkptGather& ckpt_gather() { static CkptGather a; return a; }
bool ckpt_gather_on() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_CKPT_GATHER"); return v != nullptr && v[0] == '1'; }();
    return on;
#endif
}
// 1: the vectors are filled; 0: no buffers (the caller copies them one by one); -1: a copy failed (`error` set)
int ckpt_gather_small(ConversationCheckpoint& c, const SessionState& ss, const ConversationStateSizes& z,
                      size_t layers, std::string& error) {
    const size_t ple = c.ple.size(), tails = c.tails.size(), dead = c.dead.size(), block_pos = c.block_pos.size();
    const size_t total = ple + tails + dead + block_pos;
    if (total == 0) return 1;
    CkptGather& a = ckpt_gather();
    std::lock_guard<std::mutex> lk(a.mu);
    int dev = 0;
    (void) cudaGetDevice(&dev);
    if (a.stage != nullptr && (a.cap < total || a.dev != dev)) {
        const int cur = dev;
        (void) cudaSetDevice(a.dev); (void) cudaFree(a.stage); (void) cudaFreeHost(a.host); (void) cudaSetDevice(cur);
        a.stage = a.host = nullptr; a.cap = 0;
    }
    if (a.stage == nullptr) {
        if (cudaMalloc(&a.stage, total) == cudaSuccess && cudaMallocHost(&a.host, total) == cudaSuccess) { a.cap = total; a.dev = dev; }
        else { if (a.stage) (void) cudaFree(a.stage); if (a.host) (void) cudaFreeHost(a.host); a.stage = a.host = nullptr; }
        (void) cudaGetLastError();
        if (a.stage == nullptr) return 0;
    }
    // the pieces in the vectors' order, queued on the private stream (in order), then the one copy to the host
    uint8_t* s = (uint8_t*) a.stage;
    if (!copy(s, ss.ple_hist, ple, error, true)) return -1;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(s + ple + j * z.tail, st.idx_tail, z.tail, error, true) ||
            !copy(s + ple + tails + j * z.dead, st.idx_dead, z.dead, error, true) ||
            !copy(s + ple + tails + dead + j * z.block_pos, st.idx_block_pos, z.block_pos, error, true)) return -1;
    }
    if (!copy(a.host, a.stage, total, error)) return -1;
    const uint8_t* h = (const uint8_t*) a.host;
    if (ple) std::memcpy(c.ple.data(), h, ple);
    if (tails) std::memcpy(c.tails.data(), h + ple, tails);
    if (dead) std::memcpy(c.dead.data(), h + ple + tails, dead);
    if (block_pos) std::memcpy(c.block_pos.data(), h + ple + tails + dead, block_pos);
    return 1;
}
}  // namespace

void conversation_checkpoint_join(const void* only) {
    CkptAsync& a = ckpt_async();
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.pending.valid() && (only == nullptr || only == a.dst)) {
        const cudaError_t e = a.pending.get();
        if (e != cudaSuccess) std::fprintf(stderr, "strata: STRATA_CKPT_ASYNC: a checkpoint copy failed (%s)\n", cudaGetErrorString(e));
    }
}

namespace {
// conversation_checkpoint_save; `staged` (STRATA_CKPT_ASYNC): the GDN bytes land in `c.gdn` on the background thread
// after this returns (conversation_checkpoint_join waits for them)
bool checkpoint_save(ConversationCheckpoint& c, const SessionState& ss, const ModelGeometry& g, std::string& error,
                     bool& staged) {
    staged = false;
    const auto j0 = std::chrono::steady_clock::now();
    conversation_checkpoint_join(nullptr);   // (the staging buffer is reused)
    if (std::getenv("STRATA_PARK_TRACE")) std::fprintf(stderr, "strata parktrace: save join %.2f ms\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - j0).count());
#if defined(STRATA_GPU_METAL)
    (void) g;
    return strata::metal_engine::window_checkpoint_save(c, &ss, error);
#endif
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    static const bool atr = std::getenv("STRATA_PARK_TRACE") != nullptr;
    const bool ready = c.gdn.size() == z.gdn;   // (a pooled checkpoint's pages are in; a fresh vector faults them all)
    const auto r0 = std::chrono::steady_clock::now();
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (atr) std::fprintf(stderr, "strata parktrace: save vectors %s, resize %.2f ms\n", ready ? "ready" : "fresh",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count());
    if (ckpt_async_on() && c.gdn.size() >= ((size_t) 1 << 20)) {
        CkptAsync& a = ckpt_async();
        int dev = 0;
        (void) cudaGetDevice(&dev);
        if (a.stage != nullptr && (a.cap < c.gdn.size() || a.dev != dev)) {
            const int cur = dev;
            (void) cudaSetDevice(a.dev); (void) cudaFree(a.stage); (void) cudaFreeHost(a.host);
            (void) cudaStreamDestroy(a.stream); (void) cudaSetDevice(cur);
            a.stage = a.host = nullptr; a.stream = nullptr; a.cap = 0;
        }
        if (a.stage == nullptr) {
            if (cudaMalloc(&a.stage, c.gdn.size()) == cudaSuccess && cudaMallocHost(&a.host, c.gdn.size()) == cudaSuccess &&
                cudaStreamCreateWithFlags(&a.stream, cudaStreamNonBlocking) == cudaSuccess) { a.cap = c.gdn.size(); a.dev = dev; }
            else { if (a.stage) (void) cudaFree(a.stage); if (a.host) (void) cudaFreeHost(a.host); a.stage = a.host = nullptr; }
        }
        (void) cudaGetLastError();
        const auto q0 = std::chrono::steady_clock::now();
        VramSlot* slot = vram_slots_n() > 0 ? vram_slot_for(c.gdn.size(), dev) : nullptr;
        if (slot != nullptr) slot->tag = 0;   // (being rewritten)
        void* dsrc = slot != nullptr ? slot->p : a.stage;
        c.dev_tag = 0;
        if (dsrc != nullptr && copy(dsrc, ss.gdn_state, c.gdn.size(), error)) {   // device to device: fast
            const auto q1 = std::chrono::steady_clock::now();
            if (slot != nullptr) {
                VramSlots& v = vram_slots();
                slot->tag = c.dev_tag = ++v.next_tag;
                slot->used = ++v.clock;
            }
            uint8_t* dst = c.gdn.data();
            const size_t n = c.gdn.size();
            void* host = a.host;
            const uint8_t* stage = (const uint8_t*) dsrc;
            cudaStream_t st = a.stream;
            if (atr) std::fprintf(stderr, "strata parktrace: async ckpt d2d %.2f ms\n",
                std::chrono::duration<double, std::milli>(q1 - q0).count());
            std::lock_guard<std::mutex> lk(a.mu);
            // the DMA in 4 MiB pieces, one in flight: a D2H copy queues behind the one before it on the copy engine, and
            // the decode's own small reads would otherwise wait for the whole 100+ MB at the link's ~1.6 GB/s
            a.dst = dst;
            a.pending = std::async(std::launch::async, [dst, host, stage, n, st, dev] {
                conversation_detail::background_thread_placement();
                (void) cudaSetDevice(dev);
                cudaEvent_t done = nullptr;
                cudaError_t e = cudaEventCreateWithFlags(&done, cudaEventDisableTiming | cudaEventBlockingSync);
                constexpr size_t PIECE = (size_t) 4 << 20;
                for (size_t o = 0; e == cudaSuccess && o < n; o += PIECE) {
                    const size_t k = std::min(PIECE, n - o);
                    e = cudaMemcpyAsync((uint8_t*) host + o, stage + o, k, cudaMemcpyDeviceToHost, st);
                    if (e == cudaSuccess) e = cudaEventRecord(done, st);
                    if (e == cudaSuccess) e = cudaEventSynchronize(done);
                    if (e == cudaSuccess) std::memcpy(dst + o, (const uint8_t*) host + o, k);
                }
                if (done) (void) cudaEventDestroy(done);
                return e;
            });
            staged = true;
        }
    }
    if (!staged && !copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error)) return false;
    const auto s0 = std::chrono::steady_clock::now();
    const bool ok = [&] {
        // STRATA_CKPT_GATHER=1: the small vectors through one device gather and one copy (else one copy each)
        if (ckpt_gather_on()) {
            const int r = ckpt_gather_small(c, ss, z, layers, error);
            if (r != 0) return r > 0;
        }
        if (!copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
        for (size_t j = 0; j < layers; ++j) {
            const auto& st = owned(ss, j);
            if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
                !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
                !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
        }
        return true;
    }();
    if (atr) std::fprintf(stderr, "strata parktrace: save small copies %.2f ms\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
    return ok;
}
}  // namespace

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
    bool staged = false;
    return checkpoint_save(c, ss, g, error, staged);
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    const VramSlot* hit = nullptr;
    if (c.dev_tag != 0) {
        int dev = 0;
        (void) cudaGetDevice(&dev);
        for (VramSlot& x : vram_slots().s)
            if (x.tag == c.dev_tag && x.dev == dev && x.n == c.gdn.size()) { x.used = ++vram_slots().clock; hit = &x; }
    }
    if (hit == nullptr) conversation_checkpoint_join(c.gdn.data());
#if defined(STRATA_GPU_METAL)
    (void) g;
    if (!strata::metal_engine::window_checkpoint_restore(c, &ss, error)) return false;
    ss.ple_prev[0] = c.ids.size() >= 2 ? c.ids[c.ids.size() - 2] : -1;
    ss.ple_prev[1] = c.ids.size() >= 1 ? c.ids[c.ids.size() - 1] : -1;
    return true;
#endif
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!sync(error)) return false;   // the copies below no longer ride the legacy stream's implicit ordering
    if (!copy(ss.gdn_state, hit != nullptr ? hit->p : (const void*) c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

bool conversation_checkpoint_copy(const SessionState& from, SessionState& to, const std::vector<int32_t>& ids,
                                  const ModelGeometry& g, std::string& error, bool queued) {
    // conversation_checkpoint_save(from) + conversation_checkpoint_restore(to) without the host image, for two
    // sessions carved alike on the same device (a stage's own session and its batch slots): the same bytes into
    // the same places, through copy() like the restore (the private non-blocking stream, complete on return unless
    // `queued`: then the caller synchronizes the device).
    ConversationStateSizes zf, zt;
    if (!checkpoint_targets(from, g, ids.size(), zf, error) || !checkpoint_targets(to, g, ids.size(), zt, error))
        return false;
    if (zf.gdn != zt.gdn || zf.tail != zt.tail || zf.dead != zt.dead || zf.block_pos != zt.block_pos ||
        owned_qsa(from) != owned_qsa(to) || (from.ple_hist != nullptr) != (to.ple_hist != nullptr))
        return fail(error, "direct copy between differently carved sessions");
    auto acopy = [&](void* dst, const void* src, size_t bytes) { return copy(dst, src, bytes, error, queued); };
    if (!acopy(to.gdn_state, from.gdn_state, zf.gdn) ||
        !acopy(to.ple_hist, from.ple_hist, from.ple_hist ? zf.ple : 0)) return false;
    for (size_t j = 0; j < owned_qsa(from); ++j) {
        const auto& a = owned(from, j);
        const auto& b = owned(to, j);
        if (!acopy(b.idx_tail, a.idx_tail, zf.tail) || !acopy(b.idx_dead, a.idx_dead, zf.dead) ||
            !acopy(b.idx_block_pos, a.idx_block_pos, zf.block_pos)) return false;
        if (!ids.empty()) {   // restore writes the saved `dead` row into the pooled row the next token fills
            const size_t row = ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!acopy(b.idx_pooled + row * g.idx_key_dim, a.idx_dead, zf.dead)) return false;
        }
    }
    const size_t tokens = ids.size();
    to.ple_prev[0] = tokens >= 2 ? ids[tokens - 2] : -1;
    to.ple_prev[1] = tokens >= 1 ? ids[tokens - 1] : -1;
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState* draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!view_validate(view, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + (draft ? 1 : 0), sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i < qsa + (draft ? 1 : 0); ++i) {
        const auto& st = i == qsa ? *draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState* draft, size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss) + (draft ? 1 : 0);
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i < owned_qsa(ss);
        const auto& st = index ? owned(ss, i) : *draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

// (conversation_snapshot.cpp) STRATA_PARK_TRACE: conversation_kv_save's time since the park's begin, by part (ms)
void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms);

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes,
                                std::vector<ConversationCheckpoint>* storage,
                                std::vector<ConversationCheckpoint>* moved) {
    static const bool ptr = std::getenv("STRATA_PARK_TRACE") != nullptr;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto pj = now();
    conversation_checkpoint_join(nullptr);   // the last turn checkpoint's DMA (STRATA_CKPT_ASYNC)
    conversation_park_join();                // the previous park's K/V DMA
    const auto p0 = now();
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    if (storage != nullptr && !storage->empty()) {   // the running state into a dropped checkpoint's vectors
        captured.live = std::move(storage->back());
        storage->pop_back();
        captured.live.stage_parts.clear();
    }
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    const auto p1 = now();
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints, storage);
    if (moved != nullptr)
        for (auto& c : *moved) {
            conversation_checkpoint_join(c.gdn.data());
            captured.checkpoints.push_back(std::move(c));
        }
    const auto p2 = now();
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss);
    captured.kv.resize(layers + (draft ? 1 : 0));
    // STRATA_PARK_VRAM_MIB: the retained layers' tokens go now (joined above, so no wait): held past the launch below,
    // their replacement would join the copy just launched
    for (auto& k : captured.kv) k.pending.reset();
    bool live_staged = false;
    const bool live_saved = checkpoint_save(captured.live, ss, g, error, live_staged);
    // STRATA_CKPT_ASYNC: the running state lands in live.gdn after the save returns; the image waits for that copy
    // before the vector is freed (an eviction within the copy's ~70 ms), as `pending` does for the K/V.  The join is
    // selective: a no-op once the copy has landed or another checkpoint's is in flight.
    if (live_staged)
        captured.live_pending = std::shared_ptr<void>(new char, [dst = (const void*) captured.live.gdn.data()](void* p) {
            conversation_checkpoint_join(dst);
            delete (char*) p;
        });
    if (!live_saved) return false;
    const auto p3 = now();
    const int64_t upto = (int64_t) view.ids.size();
    // STRATA_PARK_ASYNC: the K/V bytes these saves copy (the stage they need; an over-estimate by the kept prefix)
    size_t kv_need = 0;
    for (size_t j = 0; j < layers; ++j) kv_need += conversation_kv_bytes(owned(ss, j), g, upto, true);
    if (draft) kv_need += conversation_kv_bytes(*draft, g, upto, false);
    const bool staged = conversation_park_async_begin(kv_need);
    const auto p4 = now();
    struct Abandon { bool on; ~Abandon() { if (on) conversation_park_async_abandon(); } } abandon{staged};
    double kv_max = 0, kv_sum = 0;   // (STRATA_PARK_TRACE: the per-layer save time)
    for (size_t j = 0; j < layers; ++j) {
        const auto k0 = now();
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
        const double kt = ms(k0, now());
        kv_sum += kt; kv_max = std::max(kv_max, kt);
    }
    if (ptr) {
        size_t kvb = 0;
        for (const auto& k : captured.kv) kvb += k.bytes();
        double attrs_ms = 0, resize_ms = 0, copy_ms = 0;
        conversation_kv_save_trace(attrs_ms, resize_ms, copy_ms);   // (the main layers' saves above)
        std::fprintf(stderr, "strata parktrace: joins %.1f ms, sync %.1f ms, %zu checkpoints copied %.1f ms, live state %.1f ms, "
                             "begin %.1f ms, K/V %zu MB %.1f ms (layer max %.2f, sum %.2f; attrs %.2f, resize %.2f, copy %.2f)\n",
                     ms(pj, p0), ms(p0, p1), captured.checkpoints.size(), ms(p1, p2), ms(p2, p3), ms(p3, p4), kvb >> 20,
                     ms(p4, now()), kv_max, kv_sum, attrs_ms, resize_ms, copy_ms);
    }
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (draft && !conversation_kv_save(captured.kv.back(), *draft, g, upto, false, error,
                                       std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    abandon.on = false;
    if (staged && !conversation_park_async_launch(error)) return false;
    if (staged) {
        // STRATA_PARK_VRAM_MIB: the image's slot goes with the image, after the copy that reads it; each K/V layer
        // carries a token of its own too (the layers outlive the image when they are retained for the next park, and
        // the copy fills their buffers).  Without a slot: the image's join as before, no layer tokens.
        const uint64_t slot = conversation_park_slot_tag();
        auto token = [slot] {
            return std::shared_ptr<void>(new char, [slot](void* p) {
                conversation_park_join();
                conversation_park_slot_release(slot);
                delete (char*) p;
            });
        };
        captured.pending = token();
        if (slot != 0)
            for (auto& k : captured.kv) k.pending = token();
    }
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                   const QsaState& draft, std::string& error) {
    if (!view_validate(view, ss, g, error) || !sync(error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    // STRATA_CKPT_ASYNC: the caller writes `meta` to the session file at once, so the running state's copy lands first
    conversation_checkpoint_join(captured.live.gdn.data());
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss);
    std::vector<SessionKvSource> out(layers + 1);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_source(out.back(), draft, g, upto, false, error)) return false;
    meta = std::move(captured);
    sources = std::move(out);
    return true;
}

bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& ss, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0) return fail(error, "invalid session cells");
    const uint64_t tokens = std::min<uint64_t>(max_tokens, (uint64_t) ss.max_cells);
    const size_t layers = owned_qsa(ss);
    if ((owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    size_t tails = 0, dead = 0, block_pos = 0;
    if (!product(tails, {layers, z.tail}) || !product(dead, {layers, z.dead}) ||
        !product(block_pos, {layers, z.block_pos})) return fail(error, "indexer byte count overflow");
    SessionReadLimits l = limits;   // keeps the caller's admit / progress / max_file_bytes
    l.max_tokens = tokens;
    l.max_checkpoints = max_checkpoints;
    l.max_kv_layers = layers + 1;
    l.geometry = geometry_key(g);
    l.layer_range = std::make_pair(ss.layer_lo, ss.layer_hi);
    l.max_state_bytes = {z.gdn, ss.ple_hist ? z.ple : 0, tails, dead, block_pos};
    l.max_kv_bytes.assign(layers + 1, {});
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        const int64_t upto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(st.max_cells, 0));
        if (!conversation_kv_part_sizes(st, g, upto, true, l.max_kv_bytes[j], error)) return false;
    }
    const int64_t dupto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(draft.max_cells, 0));
    if (!conversation_kv_part_sizes(draft, g, dupto, false, l.max_kv_bytes.back(), error)) return false;
    limits = std::move(l);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error) {
    if (!image.live.stage_parts.empty())
        return fail(error, "a running checkpoint's stage parts belong in the stage images, not in live.stage_parts");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + (draft ? 1 : 0)) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return !draft || conversation_kv_validate(image.kv.back(), *draft, g, upto, false, error);
}

bool conversation_copy_stream_ready(int device) { return conversation_detail::copy_stream_ready(device); }

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error) {
    conversation_checkpoint_join(image.live.gdn.data());
    for (const auto& c : image.checkpoints) conversation_checkpoint_join(c.gdn.data());
    conversation_park_join_for(image);   // (STRATA_PARK_VRAM_MIB: no wait when the K/V comes from its slot)
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if ((draft && !conversation_kv_restore(image.kv.back(), *draft, g, upto, false, error)) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}

// the draft layer's K/V included (the stage that owns the draft head, or no layer split)
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_bytes(view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error) {
    return conversation_snapshot_capture_bytes(reuse, view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& ss,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    return conversation_snapshot_save(image, view, ss, g, &draft, error, std::move(reuse), reused_bytes);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error) {
    // the whole-session form: a layer split's image (stage images) is restored stage by stage, never through here
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    conversation_checkpoint_join(image.live.gdn.data());
    for (const auto& c : image.checkpoints) conversation_checkpoint_join(c.gdn.data());
    conversation_park_join_for(image);   // (STRATA_PARK_VRAM_MIB: no wait when the K/V comes from its slot)
    return conversation_snapshot_restore(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error) {
    conversation_checkpoint_join(image.live.gdn.data());
    for (const auto& c : image.checkpoints) conversation_checkpoint_join(c.gdn.data());
    conversation_park_join_for(image);   // (STRATA_PARK_VRAM_MIB: no wait when the K/V comes from its slot)
    return conversation_snapshot_restore_prefix(image, upto, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState* draft, std::string& error) {
    conversation_checkpoint_join(image.live.gdn.data());
    for (const auto& c : image.checkpoints) conversation_checkpoint_join(c.gdn.data());
    conversation_park_join_for(image);   // (STRATA_PARK_VRAM_MIB: no wait when the K/V comes from its slot)
    // a layer split's image (stage images) is restored stage by stage, never in part
    if (!image.stage_images.empty()) {
        fail(error, "a layer split's image restored as a single session");
        return ConversationRestore::invalid;
    }
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    const int64_t captured = (int64_t) image.live.ids.size();
    if (upto < 1 || upto > captured) {
        fail(error, "the borrowed prefix is not inside the parked conversation");
        return ConversationRestore::invalid;
    }
    if (!sync(error)) return ConversationRestore::transfer_failed;
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore_prefix(image.kv[j], owned(ss, j), g, captured, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (draft && !conversation_kv_restore_prefix(image.kv.back(), *draft, g, captured, upto, false, error))
        return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
