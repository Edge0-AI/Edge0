// src/kernels/iq_nibble_parity.cpp - STRATA_EXPERT_NIBBLE (strata/core/expert_nibble.hpp): the 4-bit gate/up expert
// slots against the production kernels and the GGUF blobs, bit for bit.
//
//     build/iq_nibble_parity --header                    the host half of expert_nibble.hpp (no GPU)
//     build/iq_nibble_parity <shard.gguf> [<shard.gguf> ...] [--layers-per-fmt 2] [--layers 1,17] [--ne 512]
//                            [--plans 8] [--synthetic 64] [--no-ggml] [--negative]
//     build/iq_nibble_parity --synthetic-only [--synthetic 64] [--plans 8] [--no-ggml]      no model, no shard
//
// Each model (any shard of it; its other shards are opened beside it) gives `--layers-per-fmt` layers of each
// converted format (IQ2_S, IQ3_XXS, IQ3_S; or `--layers`), `--ne` experts each, read from the GGUF tensors into the
// blob the pack holds (gate rows, up rows, down rows).  Per layer:
//   (1) nibble::transcode, 64 blobs a batch as the startup fill runs it: 0 items that do not round-trip;
//   (2) every gate/up byte of a slot is written (8 slots transcoded again into another pre-fill byte);
//   (3) nibble::hash_slots against nibble::hash_host on every slot read back (and a flipped byte must change it);
//   (4) the host codec on every item: decode_item then encode_item gives the code back;
//   (5) ggml's reference dequantizer (to_float) on every GGUF row against the weight the 4-bit slot gives through
//       dequantize_row_iq2_s / _iq3_xxs / _iq3_s's own expression (code, scale byte, fp16 d): every weight, bit for bit;
//   (6) nibble::check_gguf (the start-up check: the GGUF blocks rebuilt from every slot against the blob): 0 words
//       differ; nibble::gather_gguf (the prompt path's MMQ gather) in groups of 16 and of 1..15 from a later first
//       position: the rebuilt [gate | up] and the down rows byte for byte the blob's, the MMQ tail after the group zero;
//   (7) native_expert_grouped on the converted slots (native_expert_layout_nib) against the same call on the GGUF
//       blobs (native_expert_layout): hq and out byte for byte for every entry, and gate / up where the reference
//       layout writes them - in the card's default layout (a GP100: mode 8, whose gate/up kernel writes hq) and in the
//       CUDA layout (native_expert_set_mode(0)), each with native_grouped_set_v1 off and on (h too, in mode 0 + v1);
//       top-10 routing T = 1..6 at 0% and 50% sharing (`--plans` each), one group of T entries (G = 1) per T, plans of
//       groups with 0..11 entries (more than one GRP_NC pass, empty groups); grid_groups 0 (cap) and 4 (the PCIe
//       call's stride); the default kernel per format and each skeleton forced (S, P); q8_1 activations random, all
//       +127, all -127 and alternating +-127 (the sum of x at its extremes, +-4064).
// `--synthetic N`: per format, N experts whose gate/up blocks are random bytes (every grid index, sign byte and scale
// nibble the format has; the coverage is counted and must be complete) and whose fp16 d cycle through +-the smallest
// subnormal, +-65504, +-1, the smallest normal and two ordinary values; checks (1)-(7) on them.
// `--synthetic-only`: the synthetic experts alone, for every gate/up format x both down types (IQ4_NL, Q2_0), the down rows
// random bytes with ordinary fp16 scales: checks (1)-(7) with no model file (a GP100 is still needed).
// `--negative`: the control - one code byte per slot changed after the transcode; (6) and (7) must then report differences.
// Exit 0 when every check is equal; 1 otherwise; 2 on a usage or setup error.  A build without STRATA_EXPERT_NIBBLE_BUILT
// (the CUDA build with STRATA_EXPERIMENTAL_SM60) has no 4-bit kernels: --header only.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_nibble.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace nib = strata::core::nibble;
namespace K = strata::kernels;
namespace cpu = strata::kernels::cpu;

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}

// Every value the format holds at every one of an item's 32 positions (the other positions cycle through the set),
// encoded, decoded and compared; and one value outside the set must be refused.
void round_trip(int fmt, const std::vector<int8_t>& set, int8_t outside) {
    int8_t g[32], back[32];
    uint32_t q[4];
    for (size_t shift = 0; shift < set.size(); ++shift) {
        for (int i = 0; i < 32; ++i) g[i] = set[(shift + (size_t) i) % set.size()];
        const bool ok = nib::encode_item(fmt, g, q);
        nib::decode_item(fmt, q, back);
        expect(ok && std::memcmp(g, back, 32) == 0, (std::string(nib::fmt_name(fmt)) + " round trip").c_str());
    }
    for (int i = 0; i < 32; ++i) g[i] = set[0];
    g[17] = outside;
    expect(!nib::encode_item(fmt, g, q), (std::string(nib::fmt_name(fmt)) + " refuses a value outside the set").c_str());
}

int header_check() {
    expect(nib::kGuBytes == 1766400 && nib::kScOff == 1638400 && nib::kDOff == 1740800, "layout sizes");
    expect(nib::code_off(nib::kNFf, 0) == nib::kUpCodesOff, "up rows' codes");
    expect(nib::slot_bytes_for(921600) % nib::kSlotAlign == 0, "slot alignment");
    round_trip(nib::kIQ2_S, { 8, 25, 43, -8, -25, -43 }, 9);
    round_trip(nib::kIQ3_XXS, { 4, 12, 20, 28, 36, 44, 52, 62, -4, -12, -20, -28, -36, -44, -52, -62 }, 5);
    std::vector<int8_t> s3;
    for (int v = -15; v <= 15; v += 2) s3.push_back((int8_t) v);
    round_trip(nib::kIQ3_S, s3, 2);
    // IQ2_S's nibble is the decode kernel's PRMT selector into {8,25,43,0} {-8,-25,-43,0}
    {
        int8_t g[32] = {}, back[32];
        uint32_t q[4];
        for (int i = 0; i < 32; ++i) g[i] = (int8_t) (i % 2 ? -25 : 43);
        nib::encode_item(nib::kIQ2_S, g, q);
        expect((q[0] & 0xFu) == 2u && (q[0] >> 4 & 0xFu) == 5u, "IQ2_S selector nibbles");
        nib::decode_item(nib::kIQ2_S, q, back);
        expect(std::memcmp(g, back, 32) == 0, "IQ2_S selector decode");
    }
    expect(nib::gu_skeleton(nib::kIQ2_S) == nib::kSkelP && nib::gu_skeleton(nib::kIQ3_XXS) == nib::kSkelS &&
               nib::gu_skeleton(nib::kIQ3_S) == nib::kSkelS,
           "kernel table");
    {
        const uint8_t b[6] = { 1, 2, 3, 4, 5, 6 };
        const uint64_t h = nib::hash_host(b, 6);
        const uint64_t want = (uint64_t) (0x04030201u ^ 0u) * nib::kHashMul + (uint64_t) (0x00000605u ^ 1u) * nib::kHashMul;
        expect(h == want, "hash_host");
    }
    std::printf("%s (%d failures)\n", fails == 0 ? "header OK" : "header FAILED", fails);
    return fails == 0 ? 0 : 1;
}

// ------------------------------------------------------------------------------------------------ the GPU parity

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                 \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

struct Q81 {   // block_q8_1: fp16 d, fp16 s (= d * sum), 32 int8
    uint16_t d, s;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "block_q8_1");

constexpr int64_t kEmbd = nib::kNEmbd, kFf = nib::kNFf;
constexpr int kTok = 12;   // activation rows: T <= 6 for routing plans, 12 for the 0..11-entry plans

// One layer: `ne` GGUF blobs (gate rows | up rows | down rows, as the pack and the host arena hold them) on the host
// and the device, and their converted slots.
struct Layer {
    std::string tag;
    int layer = -1;
    cpu::NativeFmt f;
    nib::LayerVram V;
    K::NativeExpertLayout Lr, Ln;
    int ne = 0;
    std::vector<uint8_t> host;   // ne x f.bytes
    uint8_t* raw = nullptr;      // device copy of `host`
    uint8_t* slots = nullptr;    // ne x V.slot_bytes (the last one only `hashed` long: nothing past it is readable)
};

bool layer_fmt(const strata::GgufModel& m, int l, int& gt, int& dt) {
    const auto* g = m.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
    const auto* d = m.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
    if (g == nullptr || d == nullptr) return false;
    gt = (int) g->type;
    dt = (int) d->type;
    return true;
}

// The blobs of experts [0, ne) of layer l.
bool read_layer(const strata::GgufModel& m, int l, int ne, Layer& Y) {
    const char* roles[3] = { "gate", "up", "down" };
    const strata::TensorInfo* t[3] = {};
    size_t sh[3] = {};
    for (int r = 0; r < 3; ++r) t[r] = m.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &sh[r]);
    if (!t[0] || !t[1] || !t[2]) return false;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, kEmbd, kFf, Y.f, err)) {
        std::printf("layer %d: %s\n", l, err.c_str());
        return false;
    }
    const size_t gu_role = Y.f.up_off, dn_role = Y.f.bytes - Y.f.down_off;
    Y.layer = l;
    Y.ne = ne;
    Y.host.resize((size_t) ne * Y.f.bytes);
    const uint8_t* G = m.shard(sh[0]).tensor_data(*t[0]);
    const uint8_t* U = m.shard(sh[1]).tensor_data(*t[1]);
    const uint8_t* D = m.shard(sh[2]).tensor_data(*t[2]);
    for (int e = 0; e < ne; ++e) {
        uint8_t* b = Y.host.data() + (size_t) e * Y.f.bytes;
        std::memcpy(b, G + (size_t) e * gu_role, gu_role);
        std::memcpy(b + Y.f.up_off, U + (size_t) e * gu_role, gu_role);
        std::memcpy(b + Y.f.down_off, D + (size_t) e * dn_role, dn_role);
    }
    return true;
}

// check (6): nibble::check_gguf over every slot (the start-up check), and nibble::gather_gguf (the prompt path's gather)
// in groups of 16 and from later first positions, against the blobs byte for byte (and the MMQ tail zeroed).
// `expect_diff`: the negative control (only the report says so: the counts must then be > 0).
long long gguf_parity(const Layer& Y, const unsigned long long* dp, bool expect_diff) {
    long long bad_total = 0;
    const int ne = Y.ne;
    unsigned long long* dbad = nullptr;
    CK(cudaMalloc(&dbad, 8));
    CK(cudaMemset(dbad, 0, 8));
    for (int e0 = 0; e0 < ne; e0 += 64) {
        const int rc = nib::check_gguf(Y.V, Y.raw + (size_t) e0 * Y.V.raw_bytes, std::min(64, ne - e0), dp + e0, dbad, nullptr);
        if (rc != 0) {
            std::printf("check_gguf: %s\n", cudaGetErrorString((cudaError_t) rc));
            std::exit(2);
        }
    }
    CK(cudaDeviceSynchronize());
    unsigned long long cbad = 0;
    CK(cudaMemcpy(&cbad, dbad, 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(dbad));
    // the gathers: MMQ's group buffers (16 experts of [gate | up] and of down, each followed by the 4096-byte tail)
    const uint64_t gub = (uint64_t) nib::kRows * Y.f.gu_row, db = Y.V.down_bytes, zero = 4096;
    uint8_t *ggu = nullptr, *gd = nullptr;
    CK(cudaMalloc(&ggu, 16 * gub + zero));
    CK(cudaMalloc(&gd, 16 * db + zero));
    std::vector<uint8_t> hgu(16 * gub + zero), hd(16 * db + zero);
    std::vector<unsigned long long> hp(ne);
    CK(cudaMemcpy(hp.data(), dp, (size_t) ne * 8, cudaMemcpyDeviceToHost));
    long long gbad = 0, groups = 0;
    for (int e0 = 0; e0 < ne; e0 += 16) {
        const int first = (e0 / 16) % 4 == 3 ? 1 + (e0 / 64) % 15 : 0;   // every 4th group from a later position
        const int n = std::min(16, ne - e0);
        if (first >= n) continue;
        const uint8_t* sl[16] = {};
        for (int q = first; q < n; ++q) sl[q] = (const uint8_t*) (uintptr_t) hp[(size_t) (e0 + q)];
        CK(cudaMemset(ggu, 0xA5, 16 * gub + zero));
        CK(cudaMemset(gd, 0xA5, 16 * db + zero));
        const int rc = nib::gather_gguf(Y.V, sl, first, n, ggu, gub, gd, db, zero, nullptr);
        if (rc != 0) {
            std::printf("gather_gguf: %s\n", cudaGetErrorString((cudaError_t) rc));
            std::exit(2);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hgu.data(), ggu, hgu.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hd.data(), gd, hd.size(), cudaMemcpyDeviceToHost));
        for (int q = first; q < n; ++q) {
            const uint8_t* raw = Y.host.data() + (size_t) (e0 + q) * Y.f.bytes;
            gbad += std::memcmp(hgu.data() + (size_t) q * gub, raw, gub) != 0;
            gbad += std::memcmp(hd.data() + (size_t) q * db, raw + Y.f.down_off, db) != 0;
        }
        for (uint64_t i = 0; i < zero; ++i) gbad += hgu[(size_t) n * gub + i] != 0 || hd[(size_t) n * db + i] != 0 ? 1 : 0;
        ++groups;
    }
    CK(cudaFree(ggu));
    CK(cudaFree(gd));
    std::printf("    check_gguf: %llu words differ; gather_gguf (%lld groups, the tail zeroed): %lld matrices or tail bytes differ%s\n",
                cbad, groups, gbad, expect_diff ? " (NEGATIVE CONTROL: both must be > 0)" : "");
    bad_total += (long long) cbad + gbad;   // the negative control fails the run, as (7) does: it must
    return bad_total;
}

// Plans the layer, uploads the blobs, transcodes them into slots (checks 1-6).
long long convert(Layer& Y, bool ggml_check) {
    long long bad_total = 0;
    nib::Config c;
    c.on = true;
    Y.V = nib::plan_layer(Y.layer, Y.f, c);
    Y.Lr = K::native_expert_layout(Y.f.gu_type, Y.f.d_type, kEmbd, kFf);
    Y.Ln = K::native_expert_layout_nib(Y.f.gu_type, Y.f.d_type, kEmbd, kFf);
    if (!Y.V.on || Y.V.slot_bytes != Y.Ln.bytes || Y.V.down_off != Y.Ln.down_off || Y.Lr.bytes != Y.f.bytes) {
        std::printf("layer %d: plan_layer / native_expert_layout_nib disagree (on %d, slot %llu vs %zu)\n", Y.layer,
                    (int) Y.V.on, (unsigned long long) Y.V.slot_bytes, Y.Ln.bytes);
        return 1;
    }
    const int ne = Y.ne;
    const size_t slots_bytes = (size_t) (ne - 1) * Y.V.slot_bytes + Y.V.hashed;
    CK(cudaMalloc(&Y.raw, Y.host.size()));
    CK(cudaMemcpy(Y.raw, Y.host.data(), Y.host.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Y.slots, slots_bytes));
    CK(cudaMemset(Y.slots, 0xA5, slots_bytes));
    std::vector<unsigned long long> hp(ne);
    for (int e = 0; e < ne; ++e) hp[e] = (unsigned long long) (Y.slots + (size_t) e * Y.V.slot_bytes);
    unsigned long long *dp = nullptr, *dbad = nullptr, *dhash = nullptr;
    CK(cudaMalloc(&dp, (size_t) ne * 8));
    CK(cudaMalloc(&dbad, 8));
    CK(cudaMalloc(&dhash, (size_t) ne * 8));
    CK(cudaMemcpy(dp, hp.data(), (size_t) ne * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dbad, 0, 8));
    // (1) the startup fill's batches: 64 blobs at a time, then the down rows behind the codes
    for (int e0 = 0; e0 < ne; e0 += 64) {
        const int n = std::min(64, ne - e0);
        const int rc = nib::transcode(Y.V, Y.raw + (size_t) e0 * Y.V.raw_bytes, n, dp + e0, dbad, nullptr);
        if (rc != 0) {
            std::printf("transcode: %s\n", cudaGetErrorString((cudaError_t) rc));
            std::exit(2);
        }
    }
    for (int e = 0; e < ne; ++e)
        CK(cudaMemcpyAsync(Y.slots + (size_t) e * Y.V.slot_bytes + nib::kDownOff,
                           Y.raw + (size_t) e * Y.V.raw_bytes + Y.V.raw_down_off, Y.V.down_bytes, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    unsigned long long bad = 0;
    CK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
    std::printf("%s layer %2d %-7s / down %-6s: %d experts, blob %zu B, slot %llu B (x%.3f); transcode: %llu of %lld items do not round-trip\n",
                Y.tag.c_str(), Y.layer, nib::fmt_name(Y.f.gu_type), ggml_type_name((ggml_type) Y.f.d_type), ne, Y.f.bytes,
                (unsigned long long) Y.V.slot_bytes, (double) Y.V.slot_bytes / (double) Y.f.bytes, bad,
                (long long) ne * nib::kRows * nib::kItems);
    bad_total += (long long) bad;
    bad_total += gguf_parity(Y, dp, false);
    // (2) every gate/up byte written: the first 8 blobs again into slots pre-filled with 0x5A
    {
        const int n8 = std::min(8, ne);
        uint8_t* alt = nullptr;
        unsigned long long* ap = nullptr;
        CK(cudaMalloc(&alt, (size_t) n8 * Y.V.slot_bytes));
        CK(cudaMemset(alt, 0x5A, (size_t) n8 * Y.V.slot_bytes));
        std::vector<unsigned long long> ah(n8);
        for (int e = 0; e < n8; ++e) ah[e] = (unsigned long long) (alt + (size_t) e * Y.V.slot_bytes);
        CK(cudaMalloc(&ap, (size_t) n8 * 8));
        CK(cudaMemcpy(ap, ah.data(), (size_t) n8 * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(dbad, 0, 8));
        if (nib::transcode(Y.V, Y.raw, n8, ap, dbad, nullptr) != 0) std::exit(2);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> a(nib::kGuBytes), b(nib::kGuBytes);
        long long wbad = 0;
        for (int e = 0; e < n8; ++e) {
            CK(cudaMemcpy(a.data(), Y.slots + (size_t) e * Y.V.slot_bytes, nib::kGuBytes, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), alt + (size_t) e * Y.V.slot_bytes, nib::kGuBytes, cudaMemcpyDeviceToHost));
            wbad += std::memcmp(a.data(), b.data(), nib::kGuBytes) != 0;
        }
        std::printf("    every gate/up byte written (0xA5 vs 0x5A pre-fill, %d slots): %lld differ\n", n8, wbad);
        bad_total += wbad;
        CK(cudaFree(alt));
        CK(cudaFree(ap));
    }
    // (3) the device hash of every slot against the host reference over the slot read back
    const int rc = nib::hash_slots(Y.V, dp, ne, dhash, nullptr);
    if (rc != 0) {
        std::printf("hash_slots: %s\n", cudaGetErrorString((cudaError_t) rc));
        std::exit(2);
    }
    std::vector<unsigned long long> hh(ne);
    CK(cudaMemcpy(hh.data(), dhash, (size_t) ne * 8, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> sl(slots_bytes);
    CK(cudaMemcpy(sl.data(), Y.slots, slots_bytes, cudaMemcpyDeviceToHost));
    std::atomic<long long> hbad{ 0 }, cbad{ 0 }, dnbad{ 0 }, gbad{ 0 };
    std::atomic<long long> grows{ 0 };
    std::string first_ggml;
    std::atomic<int> first_set{ 0 };
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) Y.f.gu_type);
    const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned ti = 0; ti < nth; ++ti)
        th.emplace_back([&, ti]() {
            std::vector<float> ref((size_t) kEmbd), got((size_t) kEmbd);
            for (int e = (int) ti; e < ne; e += (int) nth) {
                const uint8_t* s = sl.data() + (size_t) e * Y.V.slot_bytes;
                const uint8_t* raw = Y.host.data() + (size_t) e * Y.f.bytes;
                hbad += nib::hash_host(s, Y.V.hashed) != hh[e];
                if (e == 0) {   // the hash sees every byte: one flipped must change it
                    std::vector<uint8_t> t2(s, s + Y.V.hashed);
                    t2[Y.V.hashed - 1] ^= 1;
                    hbad += nib::hash_host(t2.data(), Y.V.hashed) == hh[e];
                }
                dnbad += std::memcmp(s + nib::kDownOff, raw + Y.V.raw_down_off, Y.V.down_bytes) != 0;
                for (int64_t r = 0; r < nib::kRows; ++r) {
                    // (4) the host codec
                    for (int64_t k = 0; k < nib::kItems; ++k) {
                        uint32_t q[4], q2[4];
                        int8_t g[32];
                        std::memcpy(q, s + nib::code_off(r, k), 16);
                        nib::decode_item(Y.f.gu_type, q, g);
                        const bool ok = nib::encode_item(Y.f.gu_type, g, q2);
                        cbad += !ok || std::memcmp(q, q2, 16) != 0;
                    }
                    if (!ggml_check) continue;
                    // (5) ggml's dequantizer of the GGUF row against the slot's weights through its expression
                    const uint8_t* row = raw + (r < kFf ? (size_t) r * Y.f.gu_row : Y.f.up_off + (size_t) (r - kFf) * Y.f.gu_row);
                    tr->to_float(row, ref.data(), kEmbd);
                    for (int64_t k = 0; k < nib::kItems; ++k) {
                        uint32_t q[4];
                        std::memcpy(q, s + nib::code_off(r, k), 16);
                        int8_t gv[32];
                        nib::decode_item(Y.f.gu_type, q, gv);
                        const int sc = s[nib::scale_off(r, k)];
                        ggml_fp16_t dh;
                        std::memcpy(&dh, s + nib::d_off(r, k / 8), sizeof dh);
                        const float d = ggml_fp16_to_fp32(dh);
                        for (int j = 0; j < 32; ++j) {
                            const float db = Y.f.gu_type == nib::kIQ2_S   ? d * (0.5f + ((j < 16 ? sc : sc >> 4) & 0xf)) * 0.25f
                                           : Y.f.gu_type == nib::kIQ3_XXS ? d * (0.5f + sc) * 0.5f
                                                                          : d * sc;
                            const int a = gv[j] < 0 ? -(int) gv[j] : (int) gv[j];
                            float v = db * a;
                            if (gv[j] < 0) v = -v;
                            got[(size_t) (k * 32 + j)] = v;
                        }
                    }
                    ++grows;
                    if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
                        ++gbad;
                        if (first_set.exchange(1) == 0) {
                            size_t cc = 0;
                            while (cc < ref.size() && std::memcmp(&ref[cc], &got[cc], 4) == 0) ++cc;
                            char buf[200];
                            std::snprintf(buf, sizeof buf, "expert %d row %lld column %zu: slot %.9g, GGUF %.9g", e,
                                          (long long) r, cc, (double) got[cc], (double) ref[cc]);
                            first_ggml = buf;
                        }
                    }
                }
            }
        });
    for (auto& t : th) t.join();
    std::printf("    hash_slots vs hash_host: %lld of %d slots differ (+1 flipped-byte check); down rows: %lld of %d slots differ;"
                " host codec: %lld of %lld items differ\n",
                hbad.load(), ne, dnbad.load(), ne, cbad.load(), (long long) ne * nib::kRows * nib::kItems);
    if (ggml_check)
        std::printf("    ggml to_float of the GGUF rows vs the slot's weights: %lld of %lld rows differ%s%s\n", gbad.load(),
                    grows.load(), first_ggml.empty() ? "" : "; first: ", first_ggml.c_str());
    bad_total += hbad + dnbad + cbad + gbad;
    CK(cudaFree(dp));
    CK(cudaFree(dbad));
    CK(cudaFree(dhash));
    return bad_total;
}

// ---- plans: synthetic top-10 routing and resident_plan_kernel's grouping (groups by first appearance)
struct Plan {
    int ng = 0, ne = 0, ntok = 0, maxn = 0;
    std::vector<int> gid, start, tok, dst;
};

void make_routing(std::mt19937& rng, int NE, int T, int share_pct, std::vector<int>& ids) {
    ids.clear();
    std::vector<int> prev, cur;
    for (int t = 0; t < T; ++t) {
        cur.clear();
        const bool sh = t > 0 && share_pct > 0;
        if (sh) {
            std::vector<int> p = prev;
            std::shuffle(p.begin(), p.end(), rng);
            cur.assign(p.begin(), p.begin() + 10 * share_pct / 100);
        }
        while ((int) cur.size() < 10) {
            const int e = (int) (rng() % (unsigned) NE);
            if (std::find(cur.begin(), cur.end(), e) != cur.end()) continue;
            if (sh && std::find(prev.begin(), prev.end(), e) != prev.end()) continue;
            cur.push_back(e);
        }
        std::shuffle(cur.begin(), cur.end(), rng);
        ids.insert(ids.end(), cur.begin(), cur.end());
        prev = cur;
    }
}

Plan build_plan(const std::vector<int>& ids) {
    const int n = (int) ids.size();
    Plan p;
    std::vector<int> first(n), cnt, grp_of(n, -1);
    for (int j = 0; j < n; ++j) {
        int f = j;
        for (int i = 0; i < j; ++i)
            if (ids[i] == ids[j]) { f = i; break; }
        first[j] = f;
    }
    for (int j = 0; j < n; ++j)
        if (first[j] == j) {
            grp_of[j] = p.ng++;
            p.gid.push_back(ids[j]);
            cnt.push_back(0);
        }
    for (int j = 0; j < n; ++j) cnt[grp_of[first[j]]]++;
    p.start.assign(p.ng + 1, 0);
    int acc = 0;
    for (int g = 0; g < p.ng; ++g) {
        p.start[g] = acc;
        acc += cnt[g];
        p.maxn = std::max(p.maxn, cnt[g]);
    }
    p.start[p.ng] = acc;
    p.tok.assign(n, 0);
    p.dst.assign(n, 0);
    std::vector<int> fill(p.ng, 0);
    for (int j = 0; j < n; ++j) {
        const int g = grp_of[first[j]];
        const int at = p.start[g] + fill[g]++;
        p.tok[at] = j / 10;
        p.dst[at] = j;
    }
    p.ne = n;
    p.ntok = n / 10;
    return p;
}

// one group of T entries (G = 1): an expert every token of the window routes to
Plan single_group(std::mt19937& rng, int NE, int T) {
    Plan p;
    p.ntok = T;
    p.ng = 1;
    p.gid.push_back((int) (rng() % (unsigned) NE));
    p.start = { 0, T };
    for (int t = 0; t < T; ++t) {
        p.tok.push_back(t);
        p.dst.push_back(T - 1 - t);
    }
    p.ne = T;
    p.maxn = T;
    return p;
}

// groups of 0..11 entries (more than one GRP_NC pass, empty groups), 12 tokens, distinct tokens within a group,
// destination rows in any order
Plan synthetic_plan(std::mt19937& rng, int NE) {
    Plan p;
    p.ntok = kTok;
    std::vector<int> ids(NE);
    for (int i = 0; i < NE; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    const int sizes[] = { 3, 0, 11, 1, 5, 8, 0, 2, 4, 9, 6, 7, 10, 1 };
    int acc = 0;
    for (int g = 0; g < (int) (sizeof sizes / sizeof sizes[0]); ++g) {
        p.gid.push_back(ids[(size_t) g % ids.size()]);
        p.start.push_back(acc);
        std::vector<int> toks(kTok);
        for (int i = 0; i < kTok; ++i) toks[i] = i;
        std::shuffle(toks.begin(), toks.end(), rng);
        std::sort(toks.begin(), toks.begin() + sizes[g]);
        for (int i = 0; i < sizes[g]; ++i) {
            p.tok.push_back(toks[i]);
            p.dst.push_back(acc + i);
        }
        acc += sizes[g];
        p.maxn = std::max(p.maxn, sizes[g]);
    }
    p.start.push_back(acc);
    p.ng = (int) p.gid.size();
    p.ne = acc;
    std::vector<int> perm(acc);
    for (int i = 0; i < acc; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < acc; ++i) p.dst[i] = perm[i];
    return p;
}

std::vector<Q81> make_x(std::mt19937& rng) {
    std::vector<Q81> x((size_t) kTok * (kEmbd / 32));
    for (size_t i = 0; i < x.size(); ++i) {
        Q81& b = x[i];
        const int tok = (int) (i / (kEmbd / 32));
        const float d = 0.002f + 0.03f * (float) (rng() % 1000) / 1000.0f;
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            int q = (int) (rng() % 255) - 127;
            if (tok == 1) q = 127;   // the sum of x at its extremes (+-4064)
            if (tok == 2) q = -127;
            if (tok == 3 && (i % (kEmbd / 32)) < 8) q = (j & 1) ? 127 : -127;
            b.qs[j] = (int8_t) q;
            sum += q;
        }
        b.d = ggml_fp32_to_fp16(d);
        b.s = ggml_fp32_to_fp16(d * (float) sum);
    }
    return x;
}

struct Dev {
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *ng = nullptr, *tok = nullptr, *dst = nullptr;
};
Dev upload(const Plan& p, const Layer& Y, bool conv) {
    Dev d;
    std::vector<unsigned long long> hp(std::max(1, p.ng));
    for (int g = 0; g < p.ng; ++g)
        hp[g] = conv ? (unsigned long long) (Y.slots + (size_t) p.gid[g] * Y.V.slot_bytes)
                     : (unsigned long long) (Y.raw + (size_t) p.gid[g] * Y.f.bytes);
    CK(cudaMalloc(&d.ptr, (size_t) std::max(1, p.ng) * 8));
    CK(cudaMalloc(&d.start, (size_t) (p.ng + 1) * 4));
    CK(cudaMalloc(&d.ng, 4));
    CK(cudaMalloc(&d.tok, (size_t) std::max(1, p.ne) * 4));
    CK(cudaMalloc(&d.dst, (size_t) std::max(1, p.ne) * 4));
    CK(cudaMemcpy(d.ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.start, p.start.data(), (size_t) (p.ng + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.ng, &p.ng, 4, cudaMemcpyHostToDevice));
    if (p.ne) {
        CK(cudaMemcpy(d.tok, p.tok.data(), (size_t) p.ne * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.dst, p.dst.data(), (size_t) p.ne * 4, cudaMemcpyHostToDevice));
    }
    return d;
}
void release(Dev& d) {
    CK(cudaFree(d.ptr));
    CK(cudaFree(d.start));
    CK(cudaFree(d.ng));
    CK(cudaFree(d.tok));
    CK(cudaFree(d.dst));
}

struct Outs {
    std::vector<uint8_t> gate, up, h, hq, out;
};
// one native_expert_grouped call; scratch and out pre-filled with `fill` so an unwritten value shows
Outs run(const Layer& Y, const Plan& p, bool conv, const void* dx, void* scr, float* dout, int64_t gy, int mode,
         uint8_t fill) {
    // the window's capacity, as verify sizes it: T x 10 groups and entries (the blocks past the plan's groups idle)
    const int64_t cap_e = std::max<int64_t>((int64_t) p.ntok * 10, std::max(p.ne, p.ng)), cap_g = cap_e;
    Dev d = upload(p, Y, conv);
    CK(cudaMemset(scr, fill, K::native_expert_scratch_bytes(cap_e, kFf)));
    CK(cudaMemset(dout, fill, (size_t) cap_e * kEmbd * 4));
    K::native_expert_set_mode(mode, 0);
    K::native_expert_grouped(conv ? Y.Ln : Y.Lr, d.ptr, d.start, d.ng, d.dst, d.tok, cap_g, cap_e, dx, scr, dout, nullptr, gy);
    K::native_expert_set_mode(-1, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const size_t f = (size_t) cap_e * kFf * 4, fa = (f + 255) & ~(size_t) 255;
    Outs o;
    o.gate.resize((size_t) p.ne * kFf * 4);
    o.up.resize(o.gate.size());
    o.h.resize(o.gate.size());
    o.hq.resize((size_t) p.ne * (kFf / 32) * 36);
    o.out.resize((size_t) p.ne * kEmbd * 4);
    const uint8_t* s = (const uint8_t*) scr;
    if (p.ne) {
        CK(cudaMemcpy(o.gate.data(), s, o.gate.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o.up.data(), s + fa, o.up.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o.h.data(), s + 2 * fa, o.h.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o.hq.data(), s + 3 * fa, o.hq.size(), cudaMemcpyDeviceToHost));
        std::vector<uint8_t> all((size_t) cap_e * kEmbd * 4);
        CK(cudaMemcpy(all.data(), dout, all.size(), cudaMemcpyDeviceToHost));
        for (int e = 0; e < p.ne; ++e)
            std::memcpy(o.out.data() + (size_t) e * kEmbd * 4, all.data() + (size_t) p.dst[e] * kEmbd * 4, kEmbd * 4);
    }
    release(d);
    return o;
}

long long diff_rows(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t row) {
    long long n = 0;
    for (size_t i = 0; i + row <= a.size(); i += row) n += std::memcmp(a.data() + i, b.data() + i, row) != 0;
    return n;
}

// check (7): every plan, both grid strides, every kernel choice, both layouts, v1 off / on, against the GGUF blobs
// through the production kernels
long long grouped_parity(const Layer& Y, std::mt19937& rng, int plans, const void* dx, void* scr, float* dout) {
    struct Case {
        Plan p;
        int T;
    };
    std::vector<Case> cases;
    for (int T = 1; T <= 6; ++T) {
        for (int share : { 0, 50 })
            for (int i = 0; i < plans; ++i) {
                std::vector<int> ids;
                make_routing(rng, Y.ne, T, share, ids);
                cases.push_back({ build_plan(ids), T });
            }
        cases.push_back({ single_group(rng, Y.ne, T), T });
    }
    for (int i = 0; i < 4; ++i) cases.push_back({ synthetic_plan(rng, Y.ne), kTok });
    int hist[13] = {};
    for (const Case& cs : cases)
        for (int g = 0; g < cs.p.ng; ++g) hist[std::min(12, cs.p.start[g + 1] - cs.p.start[g])]++;
    struct Var {
        int v, mode;
        bool v1;
        const char* name;
    };
    // mode -1: the card's default layout (a GP100: mode 8 - its reference gate/up kernel writes hq, not gate / up);
    // mode 0: the CUDA layout
    const Var vars[] = { { -1, -1, false, "default, card" }, { 0, -1, false, "S, card" }, { 1, -1, false, "P, card" },
                         { -1, -1, true, "default, card, V1" }, { -1, 0, false, "default, mode 0" },
                         { 0, 0, false, "S, mode 0" }, { 1, 0, false, "P, mode 0" }, { -1, 0, true, "default, mode 0, V1" },
                         // STRATA_NIB_GU_PROTO (nib_gu_proto.cuh): the staged-once kernel, id 31
                         { 31, -1, false, "gu-proto 31, card" }, { 31, 0, false, "gu-proto 31, mode 0" },
                         { 31, 0, true, "gu-proto 31, mode 0, V1" } };
    constexpr int NV = (int) (sizeof vars / sizeof vars[0]);
    long long rows[NV] = {}, bad[NV][5] = {}, total = 0;
    for (const Case& cs : cases)
        for (int64_t gy : { (int64_t) 0, (int64_t) 4 }) {
            Outs ref[2][2];   // [mode 0][v1]
            bool have[2][2] = {};
            for (int vi = 0; vi < NV; ++vi) {
                const int m0 = vars[vi].mode == 0 ? 1 : 0, r1 = vars[vi].v1 ? 1 : 0;
                K::native_grouped_set_v1(vars[vi].v1);
                if (!have[m0][r1]) {
                    K::native_expert_nib_set_variant(-1);
                    ref[m0][r1] = run(Y, cs.p, false, dx, scr, dout, gy, vars[vi].mode, 0xFF);
                    have[m0][r1] = true;
                }
                K::native_expert_nib_set_variant(vars[vi].v);
                const Outs o = run(Y, cs.p, true, dx, scr, dout, gy, vars[vi].mode, 0x00);
                rows[vi] += cs.p.ne;
                // gate / up: written by both arms in the CUDA layout, and by every arm of a striding call (gy 4: mode 0)
                const bool gu = vars[vi].mode == 0 || gy == 4;
                if (gu) {
                    bad[vi][0] += diff_rows(ref[m0][r1].gate, o.gate, kFf * 4);
                    bad[vi][1] += diff_rows(ref[m0][r1].up, o.up, kFf * 4);
                }
                if (vars[vi].v1 && gu) bad[vi][2] += diff_rows(ref[m0][r1].h, o.h, kFf * 4);
                bad[vi][3] += diff_rows(ref[m0][r1].hq, o.hq, (kFf / 32) * 36);
                bad[vi][4] += diff_rows(ref[m0][r1].out, o.out, kEmbd * 4);
            }
        }
    K::native_expert_nib_set_variant(-1);
    K::native_grouped_set_v1(false);
    std::printf("    plans: %zu (T 1..6 x sharing 0/50%% x %d, G=1 per T, 4 of 0..11 entries); groups by entries:", cases.size(),
                plans);
    for (int n = 0; n <= 12; ++n) std::printf(" %d:%d", n, hist[n]);
    std::printf("\n");
    for (int vi = 0; vi < NV; ++vi) {
        const long long b = bad[vi][0] + bad[vi][1] + bad[vi][2] + bad[vi][3] + bad[vi][4];
        std::printf("    native_expert_grouped %-20s: %6lld entries x 2 strides, rows differing: gate %lld, up %lld, h %s, hq %lld, out %lld\n",
                    vars[vi].name, rows[vi], bad[vi][0], bad[vi][1],
                    vars[vi].v1 ? std::to_string(bad[vi][2]).c_str() : "-", bad[vi][3], bad[vi][4]);
        total += b;
    }
    return total;
}

// --synthetic: random gate/up blocks, d cycling through the fp16 extremes; the coverage of each field is counted
long long synthesize(Layer& Y, std::mt19937& rng) {
    static const uint16_t kD[] = { 0x0001, 0x8001, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00, 0x0400, 0x1A00, 0x2400 };
    const int gt = Y.f.gu_type;
    const size_t bb = gt == nib::kIQ2_S ? 82 : gt == nib::kIQ3_XXS ? 98 : 110;   // block_iq2_s / iq3_xxs / iq3_s
    const size_t nblk = (size_t) nib::kRows * nib::kSuper;
    std::set<int> grid, sign, scale, dset;
    for (int e = 0; e < Y.ne; ++e) {
        uint8_t* b = Y.host.data() + (size_t) e * Y.f.bytes;
        for (size_t i = 0; i < 2 * Y.f.up_off; ++i) b[i] = (uint8_t) rng();
        for (size_t k = 0; k < nblk; ++k) {
            uint8_t* blk = b + k * bb;
            const uint16_t d = kD[((size_t) e * nblk + k) % (sizeof kD / sizeof kD[0])];
            std::memcpy(blk, &d, 2);
            dset.insert(d);
            const uint8_t* qs = blk + 2;
            for (int ib = 0; ib < 8; ++ib) {
                if (gt == nib::kIQ2_S) {
                    const uint8_t* qh = blk + 66;
                    const uint8_t* sc = blk + 74;
                    for (int l = 0; l < 4; ++l) {
                        grid.insert(qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300));
                        sign.insert(qs[32 + 4 * ib + l]);
                    }
                    scale.insert(sc[ib] & 15);
                    scale.insert(sc[ib] >> 4);
                } else if (gt == nib::kIQ3_XXS) {
                    uint32_t aux;
                    std::memcpy(&aux, blk + 66 + 4 * ib, 4);
                    for (int l = 0; l < 8; ++l) grid.insert(qs[8 * ib + l]);
                    for (int l = 0; l < 4; ++l) sign.insert((int) ((aux >> (7 * l)) & 127));
                    scale.insert((int) (aux >> 28));
                } else {
                    const uint8_t qh = blk[66 + ib];
                    const uint8_t* sg = blk + 74;
                    const uint8_t* sc = blk + 106;
                    for (int l = 0; l < 4; ++l) {
                        grid.insert(qs[8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256));
                        grid.insert(qs[8 * ib + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
                        sign.insert(sg[4 * ib + l]);
                    }
                    scale.insert((sc[ib / 2] >> (4 * (ib & 1))) & 15);
                }
            }
        }
    }
    const size_t want_grid = gt == nib::kIQ2_S ? 1024 : gt == nib::kIQ3_XXS ? 256 : 512;
    const size_t want_sign = gt == nib::kIQ3_XXS ? 128 : 256;
    const bool ok = grid.size() == want_grid && sign.size() == want_sign && scale.size() == 16 &&
                    dset.size() == sizeof kD / sizeof kD[0];
    std::printf("synthetic %-7s: %d experts, coverage: grid indices %zu/%zu, sign %s %zu/%zu, scale nibbles %zu/16, d values %zu/%zu"
                " (+-min subnormal, +-65504, +-1, min normal, 2 ordinary) %s\n",
                nib::fmt_name(gt), Y.ne, grid.size(), want_grid, gt == nib::kIQ3_XXS ? "indices" : "bytes", sign.size(),
                want_sign, scale.size(), dset.size(), sizeof kD / sizeof kD[0], ok ? "complete" : "INCOMPLETE");
    return ok ? 0 : 1;
}

std::vector<int> split_ints(const std::string& s) {
    std::vector<int> v;
    size_t p = 0;
    while (p < s.size()) {
        v.push_back(std::atoi(s.c_str() + p));
        p = s.find(',', p);
        if (p == std::string::npos) break;
        ++p;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 && std::strcmp(argv[1], "--header") == 0) return header_check();
    std::vector<std::string> shards;
    int per_fmt = 2, NE = 512, plans = 8, syn = 64;
    bool ggml_check = true, negative = false, syn_only = false;
    std::vector<int> only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--layers-per-fmt") per_fmt = std::atoi(next());
        else if (a == "--layers") only = split_ints(next());
        else if (a == "--ne") NE = std::atoi(next());
        else if (a == "--plans") plans = std::atoi(next());
        else if (a == "--synthetic") syn = std::atoi(next());
        else if (a == "--no-ggml") ggml_check = false;
        else if (a == "--negative") negative = true;
        else if (a == "--synthetic-only") syn_only = true;
        else if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf") shards.push_back(a);
        else {
            std::fprintf(stderr,
                         "usage: iq_nibble_parity --header\n"
                         "       iq_nibble_parity <shard.gguf> [<shard.gguf> ...] [--layers-per-fmt 2] [--layers 1,17] [--ne 512]\n"
                         "                        [--plans 8] [--synthetic 64] [--no-ggml] [--negative]\n"
                         "       iq_nibble_parity --synthetic-only [--synthetic 64] [--plans 8] [--no-ggml]\n");
            return 2;
        }
    }
    if ((shards.empty() && !syn_only) || NE < 16 || NE > 512) {
        std::fprintf(stderr, "iq_nibble_parity: give at least one model shard (.gguf); --ne 16..512\n");
        return 2;
    }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    char bus[64] = {};
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    std::printf("device 0: %s, bus %s, cc %d.%d, %d SMs\n", prop.name, bus, prop.major, prop.minor, prop.multiProcessorCount);
    if (prop.major != 6 || prop.minor != 0) {
        std::fprintf(stderr, "iq_nibble_parity: the 4-bit kernels are the sm_60 build's; this GPU is cc %d.%d (skipped)\n",
                     prop.major, prop.minor);
        return 77;
    }
    if (!STRATA_EXPERT_NIBBLE_BUILT || nib::prepare(nullptr) != 0) {
        std::fprintf(stderr, "iq_nibble_parity: no 4-bit kernels in this build, or the GGUF codebooks are not unique\n");
        return 2;
    }
    std::mt19937 rng(20261019);
    void* scr = nullptr;
    float* dout = nullptr;
    void* dx = nullptr;
    const int64_t capmax = (int64_t) kTok * 10;
    CK(cudaMalloc(&scr, K::native_expert_scratch_bytes(capmax, kFf)));
    CK(cudaMalloc((void**) &dout, (size_t) capmax * kEmbd * 4));
    const std::vector<Q81> hx = make_x(rng);
    CK(cudaMalloc(&dx, hx.size() * sizeof(Q81)));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(Q81), cudaMemcpyHostToDevice));

    if (syn_only) {   // no model: every gate/up format with each down type, random experts
        long long total_s = 0;
        int done = 0;
        for (const int gt : {nib::kIQ2_S, nib::kIQ3_XXS, nib::kIQ3_S})
            for (const int dt : {20, 42}) {   // the packs' down types: IQ4_NL, Q2_0
                Layer S;
                std::string err;
                if (!cpu::native_fmt(gt, dt, kEmbd, kFf, S.f, err)) {
                    std::printf("%s / down type %d: %s\n", nib::fmt_name(gt), dt, err.c_str());
                    ++total_s;
                    continue;
                }
                S.tag = "syn";
                S.layer = 0;
                S.ne = std::max(16, std::min(syn, NE));
                S.host.resize((size_t) S.ne * S.f.bytes);
                for (auto& b : S.host) b = (uint8_t) rng();
                // block_iq4_nl and block_q2_0 are 18 bytes, the fp16 scale first: an ordinary scale in every block of the down rows
                for (int e = 0; e < S.ne; ++e)
                    for (size_t o = S.f.down_off; o + 18 <= S.f.bytes; o += 18) {
                        const uint16_t sc = (uint16_t) ((9u << 10) | (rng() & 0x3FFu));
                        std::memcpy(S.host.data() + (size_t) e * S.f.bytes + o, &sc, 2);
                    }
                long long sb = synthesize(S, rng);
                sb += convert(S, ggml_check);
                sb += grouped_parity(S, rng, plans, dx, scr, dout);
                total_s += sb;
                ++done;
                CK(cudaFree(S.raw));
                CK(cudaFree(S.slots));
            }
        std::printf("\n%s: %d synthetic format pairs, %lld differences\n", total_s == 0 && done == 6 ? "PASS" : "FAIL", done, total_s);
        return total_s == 0 && done == 6 ? 0 : 1;
    }
    long long total = 0;
    int layers_done = 0;
    std::vector<int> syn_done;
    for (size_t si = 0; si < shards.size(); ++si) {
        std::unique_ptr<strata::GgufModel> model;
        try {
            model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(shards[si]));
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "iq_nibble_parity: %s\n", ex.what());
            return 2;
        }
        const std::string tag(1, (char) ('A' + si));
        std::printf("\n#### model %s: %s (%zu shards)\n", tag.c_str(), shards[si].c_str(), model->size());
        std::vector<int> pick;
        int count[64] = {};
        for (int l = 0; l < 256; ++l) {
            int gt = -1, dt = -1;
            if (!layer_fmt(*model, l, gt, dt)) break;
            if (!nib::fmt_built(gt)) continue;
            const bool want = only.empty() ? count[gt]++ < per_fmt : std::find(only.begin(), only.end(), l) != only.end();
            if (want) pick.push_back(l);
        }
        for (int l : pick) {
            Layer Y;
            Y.tag = tag;
            if (!read_layer(*model, l, NE, Y)) {
                std::printf("layer %d: cannot read its experts\n", l);
                ++total;
                continue;
            }
            long long b = convert(Y, ggml_check);
            if (negative) {   // the control: one code byte changed per slot (item 3 of row 5), which (6) must see
                for (int e = 0; e < Y.ne; ++e) {
                    uint8_t* at = Y.slots + (size_t) e * Y.V.slot_bytes + nib::code_off(5, 3);
                    uint8_t v = 0;
                    CK(cudaMemcpy(&v, at, 1, cudaMemcpyDeviceToHost));
                    v ^= 0x11;
                    CK(cudaMemcpy(at, &v, 1, cudaMemcpyHostToDevice));
                }
                std::printf("    NEGATIVE CONTROL: one code byte changed per slot - the rows below must differ\n");
                std::vector<unsigned long long> hp(Y.ne);
                for (int e = 0; e < Y.ne; ++e) hp[e] = (unsigned long long) (Y.slots + (size_t) e * Y.V.slot_bytes);
                unsigned long long* dp = nullptr;
                CK(cudaMalloc(&dp, (size_t) Y.ne * 8));
                CK(cudaMemcpy(dp, hp.data(), (size_t) Y.ne * 8, cudaMemcpyHostToDevice));
                b += gguf_parity(Y, dp, true);
                CK(cudaFree(dp));
            }
            b += grouped_parity(Y, rng, plans, dx, scr, dout);
            total += b;
            ++layers_done;
            // the synthetic experts of this format, once: this layer's blobs with random gate/up blocks
            if (syn > 0 && std::find(syn_done.begin(), syn_done.end(), Y.f.gu_type) == syn_done.end() && !negative) {
                syn_done.push_back(Y.f.gu_type);
                Layer S;
                S.tag = "syn";
                S.layer = Y.layer;
                S.f = Y.f;
                S.ne = std::min(syn, Y.ne);
                S.host.assign(Y.host.begin(), Y.host.begin() + (ptrdiff_t) ((size_t) S.ne * Y.f.bytes));
                long long sb = synthesize(S, rng);
                sb += convert(S, ggml_check);
                sb += grouped_parity(S, rng, std::max(1, plans / 2), dx, scr, dout);
                total += sb;
                CK(cudaFree(S.raw));
                CK(cudaFree(S.slots));
            }
            CK(cudaFree(Y.raw));
            CK(cudaFree(Y.slots));
        }
    }
    std::printf("\n%s: %d layers (+ %zu synthetic formats), %lld differences\n", total == 0 ? "PASS" : "FAIL", layers_done,
                syn_done.size(), total);
    return total == 0 ? 0 : 1;
}
