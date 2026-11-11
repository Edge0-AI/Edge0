// include/strata/core/expert_nibble.hpp - STRATA_EXPERT_NIBBLE=1 (opt-in, default off): the gate/up experts of
// IQ2_S / IQ3_XXS / IQ3_S layers kept as 4-bit codes in their VRAM slots, for Pascal GP100 (sm_60).
//
// **WHY.**  On a GP100 the decode window's gate/up expert kernels are not held by memory bandwidth: they use a small
// part of it and wait on each block's serial chain (the weights, the codebook lookups, the emulated dp4a, the
// reduction).  Re-encoding each 32-weight item of these three formats as sixteen bytes of 4-bit codes (one aligned
// 16-byte load) and keeping the integer scale and the fp16 d as they are makes the slot 1.25-1.68x the GGUF bytes and
// the decode of four weights 1-4 integer instructions instead of two codebook lookups.
//
// **WHY IT IS LOSSLESS.**  Their grids have only 3 (IQ2_S) or 8 (IQ3_XXS, IQ3_S) magnitudes, so each int8 weight the
// production `Split<TG>::load` decodes is one of at most 16 signed values.  At every start the transcoder decodes every
// item of every converted slot with the production load, encodes it, decodes the written code again and compares; and
// the GGUF blocks are rebuilt from every slot and compared with the blob byte for byte (`check_gguf`).  The decode
// kernels run the production integer chain and float tail in the production order: every gate / up value is the
// production kernel's, bit for bit.  The prompt path's MMQ reads the GGUF blocks rebuilt from the slot (`gather_gguf`,
// in place of the copy it does today): the same bytes, so the same bits.
//
// **OFF IS TODAY.**  Unset or 0: no layer is converted, `slot_bytes(l)` is `expert_layout().blob_bytes(l)`, nothing is
// transcoded, sealed or hashed, and none of the new branches is entered.  Only a CUDA build with
// STRATA_EXPERIMENTAL_SM60 has the kernels (STRATA_EXPERT_NIBBLE_BUILT); every other build refuses the switch at start.
#pragma once

#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(STRATA_EXPERIMENTAL_SM60) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && \
    !defined(STRATA_GPU_METAL)
#define STRATA_EXPERT_NIBBLE_BUILT 1
#else
#define STRATA_EXPERT_NIBBLE_BUILT 0
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_NIB_HD __host__ __device__
#else
#define STRATA_NIB_HD
#endif

namespace strata::core::nibble {

// ------------------------------------------------------------------------------------------------ the slot layout
//
// A converted layer's expert slot, in order (the host arena and the pack keep the GGUF blocks):
//
//   codes   kRows x kItems x 16 B   row r, item k at (r * kItems + k) * 16; rows 0..kNFf-1 gate, kNFf..kRows-1 up
//   scales  kRows x kItems x 1 B    row r, item k at kScOff + r * kItems + k (the production scale step's input)
//   d       kRows x kSuper x 2 B    row r, superblock s at kDOff + (r * kSuper + s) * 2: the block's fp16 d, bit for bit
//   ------  kGuBytes = 1,766,400 B (6900 x 256: the down part starts 256-aligned)
//   down    the GGUF down rows, copied verbatim from the raw blob's [down_off, bytes)
//   ------  slot_bytes = round_up(kGuBytes + down bytes, kSlotAlign)
//
// One item = 32 weights of a row, one 16-byte code (four 32-bit words, see `encode_item`).  Only the expert shape
// n_embd 2560, n_ff 640 is converted: every other layer keeps its GGUF blob and its kernels.

inline constexpr int64_t kNEmbd = 2560;                 ///< the gate/up rows' width (the activation)
inline constexpr int64_t kNFf = 640;                    ///< gate rows = up rows = n_ff
inline constexpr int64_t kRows = 2 * kNFf;              ///< 1280: gate, then up
inline constexpr int64_t kItems = kNEmbd / 32;          ///< 80 items of 32 weights per row
inline constexpr int64_t kSuper = kNEmbd / 256;         ///< 10 superblocks (one fp16 d each) per row
inline constexpr uint64_t kItemBytes = 16;              ///< one item's code: four words, one 16-byte load

inline constexpr uint64_t kCodesOff = 0;
inline constexpr uint64_t kUpCodesOff = (uint64_t) kNFf * kItems * kItemBytes;          ///<   819,200: row kNFf
inline constexpr uint64_t kScOff = (uint64_t) kRows * kItems * kItemBytes;              ///< 1,638,400
inline constexpr uint64_t kDOff = kScOff + (uint64_t) kRows * kItems;                   ///< 1,740,800
inline constexpr uint64_t kGuBytes = kDOff + (uint64_t) kRows * kSuper * 2;             ///< 1,766,400
inline constexpr uint64_t kDownOff = kGuBytes;                                          ///< the down rows
inline constexpr uint64_t kSlotAlign = 256;                                             ///< every slot's size

static_assert(kGuBytes == 1766400, "the slot layout above");
static_assert(kGuBytes % kSlotAlign == 0, "the down part starts 256-aligned");
static_assert(kScOff % 16 == 0 && kDOff % 16 == 0, "each region starts 16-aligned");

/// Byte offset of row `row` (0..kRows-1; up rows are kNFf + r) item `k`'s code inside the slot.
STRATA_NIB_HD constexpr uint64_t code_off(int64_t row, int64_t k) {
    return kCodesOff + ((uint64_t) row * kItems + (uint64_t) k) * kItemBytes;
}
/// Byte offset of row `row` item `k`'s scale byte.
STRATA_NIB_HD constexpr uint64_t scale_off(int64_t row, int64_t k) { return kScOff + (uint64_t) row * kItems + (uint64_t) k; }
/// Byte offset of row `row` superblock `s`'s fp16 d (s = k / 8 for item k).
STRATA_NIB_HD constexpr uint64_t d_off(int64_t row, int64_t s) { return kDOff + ((uint64_t) row * kSuper + (uint64_t) s) * 2; }
/// A converted slot's size for a raw blob whose down rows are `down_bytes` long.
STRATA_NIB_HD constexpr uint64_t slot_bytes_for(uint64_t down_bytes) {
    return (kGuBytes + down_bytes + kSlotAlign - 1) / kSlotAlign * kSlotAlign;
}

// ------------------------------------------------------------------------------------------------ the formats

inline constexpr int kIQ3_XXS = 18;
inline constexpr int kIQ3_S = 21;
inline constexpr int kIQ2_S = 22;

/// Whether `gu_type` is converted (the three formats whose weights take at most 16 values).
STRATA_NIB_HD constexpr bool fmt_built(int gu_type) { return gu_type == kIQ2_S || gu_type == kIQ3_XXS || gu_type == kIQ3_S; }
/// The GGUF gate/up row bytes of `gu_type` at kNEmbd (a converted layer's raw blob must agree; 0 = not a format here).
STRATA_NIB_HD constexpr uint64_t raw_gu_row(int gu_type) {
    return gu_type == kIQ2_S ? 820 : gu_type == kIQ3_XXS ? 980 : gu_type == kIQ3_S ? 1100 : 0;
}
inline const char* fmt_name(int gu_type) {
    switch (gu_type) {
        case kIQ3_XXS: return "IQ3_XXS";
        case kIQ3_S: return "IQ3_S";
        case kIQ2_S: return "IQ2_S";
        default: return "?";
    }
}

// ------------------------------------------------------------------------------------------------ the item code
//
// `g[32]` below is one item as the production `Split<TG>::load` leaves it: g[4 * j + b] is byte b of W.g[j]
// (j = 0..7, the dp4a order against the q8_1 block's qs words).  Word m (0..3) of the 16-byte code holds W.g[2m]
// and W.g[2m + 1]:
//
//   IQ2_S    bits 4b..4b+3 = c(g[8m + b]), bits 16+4b.. = c(g[8m + 4 + b]), b = 0..3
//            c = magnitude index of {8, 25, 43} | sign << 2: the PRMT selector into {8,25,43,0} {-8,-25,-43,0}
//   IQ3_XXS  same nibble positions; c = index of {4, 12, 20, 28, 36, 44, 52, 62} | sign << 3
//   IQ3_S    byte b = u(g[8m + b]) | u(g[8m + 4 + b]) << 4, u = (v + 15) / 2 (v odd, -15..15); v = 2u - 15
//
// The scale byte keeps the production integer scale step's input:
//   IQ2_S    the block's scales byte for the item: ls0 | ls1 << 4 (W.ls0, W.ls1)
//   IQ3_XXS  aux32 >> 28 (0..15, W.ls)
//   IQ3_S    1 + 2 * nibble (1..31, W.ls)

/// Encodes one item; false when a weight is outside the format's code set (the conversion then refuses to start).
inline bool encode_item(int gu_type, const int8_t g[32], uint32_t q[4]) {
    bool ok = true;
    auto code = [&](int8_t v) -> uint32_t {
        const int a = v < 0 ? -(int) v : (int) v;
        if (gu_type == kIQ3_S) {
            if ((v & 1) == 0 || v < -15 || v > 15) { ok = false; return 0; }
            return (uint32_t) (((int) v + 15) / 2);
        }
        if (gu_type == kIQ3_XXS) {
            static constexpr int mag[8] = { 4, 12, 20, 28, 36, 44, 52, 62 };
            for (uint32_t i = 0; i < 8; ++i)
                if (a == mag[i]) return i | (v < 0 ? 8u : 0u);
            ok = false;
            return 0;
        }
        const uint32_t m = a == 8 ? 0u : a == 25 ? 1u : a == 43 ? 2u : 3u;   // IQ2_S
        if (m == 3u) { ok = false; return 0; }
        return m | (v < 0 ? 4u : 0u);
    };
    for (int m = 0; m < 4; ++m) {
        uint32_t w = 0;
        for (int b = 0; b < 4; ++b) {
            const uint32_t c0 = code(g[8 * m + b]), c1 = code(g[8 * m + 4 + b]);
            if (gu_type == kIQ3_S) w |= (c0 | c1 << 4) << (8 * b);
            else w |= c0 << (4 * b) | c1 << (16 + 4 * b);
        }
        q[m] = w;
    }
    return ok;
}

/// Decodes one item's code back to the production W.g bytes (the host reference the parity program compares with;
/// the device decoders give the same bytes).  Codes outside the set decode as the PRMT does: IQ2_S index 3 -> 0.
inline void decode_item(int gu_type, const uint32_t q[4], int8_t g[32]) {
    static constexpr int8_t iq2[8] = { 8, 25, 43, 0, -8, -25, -43, 0 };
    static constexpr int8_t iq3[8] = { 4, 12, 20, 28, 36, 44, 52, 62 };
    for (int m = 0; m < 4; ++m)
        for (int b = 0; b < 4; ++b) {
            uint32_t c0, c1;
            if (gu_type == kIQ3_S) {
                c0 = q[m] >> (8 * b) & 15u;
                c1 = q[m] >> (8 * b + 4) & 15u;
            } else {
                c0 = q[m] >> (4 * b) & 15u;
                c1 = q[m] >> (16 + 4 * b) & 15u;
            }
            auto val = [&](uint32_t c) -> int8_t {
                if (gu_type == kIQ3_S) return (int8_t) (2 * (int) c - 15);
                if (gu_type == kIQ3_XXS) return (int8_t) ((c & 8u) ? -iq3[c & 7u] : iq3[c & 7u]);
                return iq2[c & 7u];
            };
            g[8 * m + b] = val(c0);
            g[8 * m + 4 + b] = val(c1);
        }
}

// ------------------------------------------------------------------------------------------------ the slot hash
//
// STRATA_EXPERT_NIBBLE_RECHECK's fingerprint of a slot: over the slot's first `hashed` bytes (kGuBytes plus the down
// rows; the alignment tail is not written and not hashed) as little-endian 32-bit words w_i, the last one
// zero-extended,  H = sum_i (uint64) (w_i ^ i) * kHashMul  mod 2^64.  A sum, so the device's order of adding does not
// change it; the index term makes a moved word count.

inline constexpr uint64_t kHashMul = 0x9E3779B97F4A7C15ull;

/// The host reference of the slot hash (the device hash gives the same value for the same bytes).
inline uint64_t hash_host(const void* p, uint64_t bytes) {
    const uint8_t* s = (const uint8_t*) p;
    uint64_t h = 0;
    const uint64_t n = (bytes + 3) / 4;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t w = 0;
        const uint64_t at = 4 * i, take = bytes - at < 4 ? bytes - at : 4;
        std::memcpy(&w, s + at, (size_t) take);   // little-endian hosts only
        h += (uint64_t) (w ^ (uint32_t) i) * kHashMul;
    }
    return h;
}

// ------------------------------------------------------------------------------------------------ the switches

struct Config {
    bool on = false;                       ///< STRATA_EXPERT_NIBBLE=1
    bool recheck = false;                  ///< STRATA_EXPERT_NIBBLE_RECHECK=1 (debug: every slot's hash per request)
    std::vector<int> fmts{ kIQ2_S, kIQ3_XXS, kIQ3_S };   ///< STRATA_EXPERT_NIBBLE_FMTS (default 22,18,21)
    std::vector<int64_t> skip;             ///< STRATA_EXPERT_NIBBLE_SKIP: layers kept in their GGUF blocks
    std::string error;                     ///< non-empty: a malformed value - the start is refused
};

namespace detail {
/// "a,b,c" -> integers; false on anything else (empty = an empty list).
inline bool parse_list(const char* s, std::vector<int64_t>& out) {
    out.clear();
    while (s != nullptr && *s != '\0') {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s) return false;
        out.push_back((int64_t) v);
        s = end;
        if (*s == ',') ++s;
        else if (*s != '\0') return false;
    }
    return true;
}
}  // namespace detail

/// The switches, read once (generate.cpp reads them before anything is allocated).
inline const Config& config() {
    static const Config c = [] {
        Config r;
        const char* e = std::getenv("STRATA_EXPERT_NIBBLE");
        if (e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0) {
            if (std::strcmp(e, "1") == 0) r.on = true;
            else r.error = std::string("STRATA_EXPERT_NIBBLE=") + e + ": expected 0 or 1";
        }
        const char* rc = std::getenv("STRATA_EXPERT_NIBBLE_RECHECK");
        r.recheck = rc != nullptr && std::strcmp(rc, "1") == 0;
        std::vector<int64_t> v;
        if (const char* f = std::getenv("STRATA_EXPERT_NIBBLE_FMTS"); f != nullptr && f[0] != '\0') {
            if (!detail::parse_list(f, v)) {
                if (r.error.empty()) r.error = std::string("STRATA_EXPERT_NIBBLE_FMTS=") + f + ": expected ggml types, e.g. 22,18,21";
            } else {
                r.fmts.clear();
                for (const int64_t t : v) {
                    if (!fmt_built((int) t) && r.error.empty())
                        r.error = "STRATA_EXPERT_NIBBLE_FMTS: type " + std::to_string(t) +
                                  " is not converted (22 IQ2_S, 18 IQ3_XXS, 21 IQ3_S)";
                    r.fmts.push_back((int) t);
                }
            }
        }
        if (const char* k = std::getenv("STRATA_EXPERT_NIBBLE_SKIP"); k != nullptr && k[0] != '\0') {
            if (!detail::parse_list(k, r.skip) && r.error.empty())
                r.error = std::string("STRATA_EXPERT_NIBBLE_SKIP=") + k + ": expected layer numbers, e.g. 40,44";
        }
        return r;
    }();
    return c;
}

/// With the switch on, the first environment setting it refuses, as a message (empty = none): the paths that would
/// read a converted slot as GGUF blocks, write GGUF bytes into one, or run other kernels for the converted layers
/// than the ones the 4-bit kernels are bit for bit.  The command-line half (--peer-device, remote expert caches,
/// --kv-grow, --vram-elastic), the cc 6.0 check, full residency and no loan are generate.cpp's checks; the prompt
/// path's own (MMQ on every converted layer) is Prefill::nibble_prompt_check.
inline std::string env_conflicts() {
    auto set = [](const char* n) -> const char* {
        const char* v = std::getenv(n);
        return v != nullptr && v[0] != '\0' ? v : nullptr;
    };
    const char* v = set("STRATA_MMQ_DIRECT");
    if (v != nullptr && std::atoi(v) != 0)
        return "STRATA_MMQ_DIRECT (the prompt MMQ would read a converted slot in place as GGUF blocks)";
    v = set("STRATA_PREFILL_MMQ");
    if (v != nullptr && std::atoi(v) == 0) return "STRATA_PREFILL_MMQ=0 (the FP16 prompt path reads GGUF blocks)";
    v = set("STRATA_PF_FUSED");
    if (v != nullptr && v[0] == '1') return "STRATA_PF_FUSED=1 (the fused prompt kernels read GGUF blocks)";
    v = set("STRATA_FS_SLOTS");
    if (v != nullptr && std::atoi(v) > 0) return "STRATA_FS_SLOTS (Foresight swaps GGUF blobs into slots)";
    v = set("STRATA_EXPERT_V2");
    if (v != nullptr && v[0] == '1') return "STRATA_EXPERT_V2=1 (the S26 kernels, not the CUDA layout, on IQ3_S)";
    v = set("STRATA_KV_GROW");
    if (v != nullptr && v[0] != '0') return "STRATA_KV_GROW (the elastic K/V moves slots at their GGUF size)";
    v = set("STRATA_EXP_MODE");
    if (v != nullptr && std::atoi(v) != 0 && std::atoi(v) != 8)
        return "STRATA_EXP_MODE other than 0 or 8 (another layout's gate/up kernels)";
    return {};
}

// ------------------------------------------------------------------------------------------------ the VRAM table

/// One layer's slots.  `on` false: the GGUF blob, `slot_bytes == raw_bytes`, the production kernels.
struct LayerVram {
    bool on = false;
    int gu_type = -1, d_type = -1;
    uint64_t down_off = 0;     ///< in the slot: kDownOff when on, else the raw blob's down_off
    uint64_t down_bytes = 0;   ///< the down rows (raw_bytes - raw_down_off)
    uint64_t hashed = 0;       ///< bytes the slot hash covers: kGuBytes + down_bytes when on, else 0
    uint64_t slot_bytes = 0;   ///< what the expert cache allocates per slot (`open_sized`, every estimate)
    // the raw blob as the arena and the pack hold it (expert_layout().fmt[l]); the transcoder reads these
    uint64_t raw_bytes = 0, raw_gu_row = 0, raw_up_off = 0, raw_down_off = 0;
};

/// One layer's entry from its GGUF format and the switches (pure; `plan_layers` calls it per layer).
inline LayerVram plan_layer(int64_t l, const strata::kernels::cpu::NativeFmt& f, const Config& c) {
    LayerVram v;
    v.gu_type = f.gu_type;
    v.d_type = f.d_type;
    v.raw_bytes = f.bytes;
    v.raw_gu_row = f.gu_row;
    v.raw_up_off = f.up_off;
    v.raw_down_off = f.down_off;
    v.down_bytes = f.bytes - f.down_off;
    v.down_off = f.down_off;
    v.slot_bytes = f.bytes;
    bool listed = false, skipped = false;
    for (const int t : c.fmts) listed = listed || t == f.gu_type;
    for (const int64_t s : c.skip) skipped = skipped || s == l;
    // [gate | up | down] exactly, no NVFP4 tail, the down rows a whole number of 16-byte words (the gathers copy them so)
    const bool shape = f.n_embd == kNEmbd && f.n_ff == kNFf && f.gu_row == raw_gu_row(f.gu_type) &&
                       f.up_off == (uint64_t) kNFf * f.gu_row && f.down_off == 2 * f.up_off && f.bytes > f.down_off &&
                       f.tail_off == 0 && (f.bytes - f.down_off) % 16 == 0;
    if (c.on && c.error.empty() && listed && !skipped && fmt_built(f.gu_type) && shape) {
        v.on = true;
        v.down_off = kDownOff;
        v.hashed = kGuBytes + v.down_bytes;
        v.slot_bytes = slot_bytes_for(v.down_bytes);
    }
    return v;
}

/// Every layer's entry for this pack (all off for a Q2_0 pack or with the switch off).  `err`: a SKIP layer out of
/// range, or the switch on with a Q2_0 pack.
inline std::vector<LayerVram> plan_layers(const strata::kernels::cpu::ExpertLayout& lay, std::string& err) {
    const Config& c = config();
    std::vector<LayerVram> out((size_t) (lay.n_layers > 0 ? lay.n_layers : 0));
    for (int64_t l = 0; l < (int64_t) out.size(); ++l) {
        if (lay.native && (size_t) l < lay.fmt.size()) out[(size_t) l] = plan_layer(l, lay.fmt[(size_t) l], c);
        else out[(size_t) l].slot_bytes = out[(size_t) l].raw_bytes = lay.blob_bytes(l);
    }
    if (c.on && !lay.native) err = "STRATA_EXPERT_NIBBLE needs a native pack (native_experts.txt)";
    for (const int64_t s : c.skip)
        if (c.on && (s < 0 || s >= lay.n_layers) && err.empty())
            err = "STRATA_EXPERT_NIBBLE_SKIP: layer " + std::to_string(s) + " is not in the model";
    return out;
}

namespace detail {
inline std::vector<LayerVram>& table() {
    static std::vector<LayerVram> t;
    return t;
}
}  // namespace detail

/// generate.cpp only: the table, once, after `expert_layout_load` and BEFORE the first slot estimate (placement
/// search, can_start, the cache's auto size, every `open_sized`).  Not thread-safe: set before any other thread.
inline void set_layers(std::vector<LayerVram> v) { detail::table() = std::move(v); }

/// Layer `l`'s entry (an off entry when the table is not set or `l` is outside it).  Read-only after `set_layers`.
inline const LayerVram& layer(int64_t l) {
    static const LayerVram off;
    const auto& t = detail::table();
    return l >= 0 && (size_t) l < t.size() ? t[(size_t) l] : off;
}

/// Whether any layer is converted (the switch on and the table set).
inline bool any_on() {
    for (const LayerVram& v : detail::table())
        if (v.on) return true;
    return false;
}

/// Bytes of one of layer `l`'s VRAM slots: the converted size for a converted layer, else the layout's blob bytes
/// (exactly `expert_layout().blob_bytes(l)`: with the switch off every caller sizes what it sized before).
inline uint64_t slot_bytes(int64_t l) {
    const auto& t = detail::table();
    if (l >= 0 && (size_t) l < t.size()) return t[(size_t) l].slot_bytes;
    return strata::kernels::cpu::expert_layout().blob_bytes(l);
}

// ------------------------------------------------------------------------------------------------ fatal paths

/// A converted layer reached a path that would compute or copy it as GGUF blocks: say where and stop.  Never a
/// silent wrong answer.
[[noreturn]] inline void fatal(int64_t l, const char* where, const char* why) {
    std::fprintf(stderr, "strata: STRATA_EXPERT_NIBBLE: layer %lld (%s, 4-bit slots) reached %s: %s - stopping\n",
                 (long long) l, fmt_name(layer(l).gu_type), where != nullptr ? where : "?", why != nullptr ? why : "");
    std::fflush(stderr);
    std::abort();
}

/// Before each raw slot copy that bypasses the ExpertCache API (the adaptive tier's copies, the elastic K/V's slot
/// moves): fatal when `l` is a converted layer.  The startup refusals keep these paths off; this is the backstop
/// (with `ExpertCache::seal()` for the API's own fills).
inline void guard_raw(int64_t l, const char* where) {
    if (layer(l).on) fatal(l, where, "a raw slot copy at the GGUF size");
}

// ------------------------------------------------------------------------------------------------ the decode kernels
//
// Two skeletons in the shape of native_gu_multi_kernel (16 lanes a row, a group's entries GRP_NC at a time), both bit
// for bit the production gate/up: S stages the entries' q8_1 blocks in shared memory and decodes each item where it
// is used (IQ3_S: v = 2u - 15 restored per byte, then the production apply); P loads the lane's codes, scales and d
// before staging the qs words as 16-byte rows (IQ3_S: sum = 2 dot(u, x) - 15 sum(x), the same int32).  Per format the
// faster one on 4x GP100 at T = 1..6 (native_expert_bench --nibble): IQ2_S P, IQ3_XXS S, IQ3_S S.

inline constexpr int kSkelS = 0, kSkelP = 1;
STRATA_NIB_HD constexpr int gu_skeleton(int gu_type) { return gu_type == kIQ2_S ? kSkelP : kSkelS; }

// ------------------------------------------------------------------------------------------------ iq_kernels.cu
//
// Device addresses are `unsigned long long` (as native_expert_grouped's grp_ptr), streams `void*`; each returns the
// CUDA error code (0 = success) so the caller can name the layer.  A build without STRATA_EXPERT_NIBBLE_BUILT
// returns cudaErrorInvalidValue (1) from each.

/// The reverse codebooks (grid magnitudes -> GGUF grid index) `check_gguf` and `gather_gguf` use, built on the CURRENT
/// device (once per device; call it at load, before any capture).  Non-zero also when a grid entry is not unique.
int prepare(void* stream);

/// Converts `n` raw blobs of layer `L` (device `staging`, `L.raw_bytes` apart, as the host arena holds them) into
/// the converted slots at `slot_ptrs[0..n)` (a device array): codes, scales, d.  The down rows are not touched (the
/// caller copies them to `slot + kDownOff`).  Every item is decoded with the production `Split<TG>::load`, encoded,
/// decoded from the written code again and compared with W.g[0..7], the scale and d; each item that differs or holds
/// a weight outside the code set adds 1 to `*bad` (device).  L.on must be true.
int transcode(const LayerVram& L, const void* staging, int64_t n, const unsigned long long* slot_ptrs,
              unsigned long long* bad, void* stream);

/// The GGUF gate/up blocks rebuilt from each of the `n` (<= 64) converted slots at `slot_ptrs` and its down rows,
/// compared with the raw blobs in `staging` (`L.raw_bytes` apart) byte for byte: each 32-bit word that differs adds 1
/// to `*bad`.  `prepare` first.
int check_gguf(const LayerVram& L, const void* staging, int64_t n, const unsigned long long* slot_ptrs,
               unsigned long long* bad, void* stream);

/// The prompt path's gather of a converted layer's experts (prefill.cpp, MMQ): for q in [first, n) (n <= 16), the
/// GGUF [gate | up] blocks rebuilt from slot `slots[q]` (host array of device addresses) into gu_dst + q * gu_stride -
/// the bytes the raw blob holds, byte for byte - and its down rows copied to d_dst + q * d_stride; `zero_bytes` zeroed
/// after expert n - 1's gate/up and down (MMQ's tail).  Every address, stride and size 16-byte aligned.  `prepare`
/// on this device first.
int gather_gguf(const LayerVram& L, const uint8_t* const* slots, int first, int n, void* gu_dst, uint64_t gu_stride,
                void* d_dst, uint64_t d_stride, uint64_t zero_bytes, void* stream);

/// The slot hash (`hash_host`'s value) of the `L.hashed` bytes at each `slot_ptrs[i]`, into device `out[0..n)`.
int hash_slots(const LayerVram& L, const unsigned long long* slot_ptrs, int64_t n, unsigned long long* out,
               void* stream);

}  // namespace strata::core::nibble
