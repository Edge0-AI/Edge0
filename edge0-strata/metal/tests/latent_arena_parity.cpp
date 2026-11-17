// metal/tests/latent_arena_parity.cpp - the latent layout's routed-expert kernels reading their experts from the
// SSD-streaming arena (function constant 20: smap[e] -> a 16 KiB-page slot in one of four chunks, metal/src/engine/stream.cpp)
// against the same kernels reading one whole experts buffer (experts.bin as it is).  The arena only moves the bytes, so
// every output must be the same, byte for byte.  Random weights and inputs, no model.
//
//   latent_arena_parity [--width 1280|1792] [--window-only] [--misplace]
//
// The window path (latent_routed.metal): latent_gate_up, latent_gate_up_8, latent_gate_up_4, latent_down, latent_down_4
// for windows of 1, 5 and 8 tokens.  The prompt chain (moe_mm.metal, latent_prompt_q3_stage.metal) with tm 32: below 2048
// rows mm_gu2 Q3_K 32_64_32_2 reads the experts, from 2048 rows latent_prompt_q3_stage stages the layer's gate/up rows to
// fp16 first (in the arena only the experts that are in it: the others are skipped) and latent_prompt_gu_f16 reads them;
// mm_down2 IQ4_NL 32_64_32_2 reads the down rows in both.  The prompt's routed experts are drawn from 400 of the 512, so
// 112 are not in the arena (smap ~0).  --misplace points one routed expert's smap at another slot: every case that reads
// that expert must then differ (the check of the check).  Both widths when --width is not given.  Exit 0 = all identical.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {
constexpr int NE = 512, NFF = 640, TOPK = 10, GROUPS = 8, HIDDEN = 2560, TM = 32, PAGE = 16384, CHUNKS = 4;
struct LatentRoutedArgs {
    uint32_t n_tok, groups, hidden, latent, ff, topk, plan_token_entry;
    uint32_t u_row, gu_row, down_row, up_off, down_off, expert_bytes;
};
struct MmPlanArgs { uint32_t n_ent, topk, n_expert, tm, n_blocks; };
struct MmArgs { uint64_t gate_off, up_off, down_off, gu_stride, d_stride; uint32_t gu_row, d_row, n_embd, n_ff; };

int failures = 0, cases = 0;
void report(const std::string& what, const mtl::Buffer& a, const mtl::Buffer& b, size_t bytes) {
    size_t diff = 0, nonzero = 0;
    const uint8_t *x = a.as<uint8_t>(), *y = b.as<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        diff += x[i] != y[i];
        nonzero += x[i] != 0;
    }
    ++cases;
    const bool empty = nonzero == 0;   // an output never written would compare equal: it is a failure too
    failures += diff != 0 || empty;
    std::printf("%-58s %s", what.c_str(), empty ? "EMPTY (nothing written)" : diff ? "DIFFERENT" : "identical");
    if (diff) std::printf(" (%zu of %zu bytes)", diff, bytes);
    std::printf("  [%zu of %zu bytes non-zero]\n", nonzero, bytes);
}

struct Arena {
    std::vector<mtl::Buffer> chunk;
    mtl::Buffer smap;     // 2 layers x NE: the kernels are bound at layer 1 (offset NE * 4), as the engine binds layer l
    uint32_t* map() const { return smap.as<uint32_t>() + NE; }
};

int run_width(mtl::Context& cx, int W, bool window_only, bool misplace) {
    const size_t gu_row = ref::row_bytes(ref::Q3_K, W), d_row = ref::row_bytes(ref::IQ4_NL, NFF);
    const size_t up_off = gu_row * NFF, down_off = 2 * up_off, blob = down_off + d_row * W;
    const size_t pages = (blob + PAGE - 1) / PAGE, slots = 4 * 136;
    std::printf("width %d: expert blob %zu bytes (%zu pages); %d experts, %zu arena slots in %d chunks\n", W, blob, pages, NE,
                slots, CHUNKS);
    // the whole buffer: expert e at e * blob, every expert different
    mtl::Buffer whole = cx.alloc(blob * NE);
    for (int e = 0; e < NE; ++e) {
        ref::random_blocks(ref::Q3_K, whole.as<uint8_t>() + e * blob, 2 * NFF, W, 1000 + e, 0.02f);
        ref::random_blocks(ref::IQ4_NL, whole.as<uint8_t>() + e * blob + down_off, W, NFF, 5000 + e, 0.02f);
    }
    // the arena: the experts in shuffled slots across the four chunks; the unused slots hold noise
    Arena ar;
    const size_t per = slots / CHUNKS;
    for (int c = 0; c < CHUNKS; ++c) {
        ar.chunk.push_back(cx.alloc(per * pages * PAGE));
        ref::Rng noise(77 + c);
        uint32_t* p = ar.chunk.back().as<uint32_t>();
        for (size_t i = 0; i < per * pages * PAGE / 4; ++i) p[i] = (uint32_t) noise.next();
    }
    ar.smap = cx.alloc(4 * 2 * NE);
    std::vector<int> order(slots);
    std::iota(order.begin(), order.end(), 0);
    ref::Rng rs(11 + W);
    for (size_t i = slots - 1; i > 0; --i) std::swap(order[i], order[rs.next() % (i + 1)]);
    auto code = [&](int slot) { return (uint32_t) (slot / per) << 28 | (uint32_t) ((slot % per) * pages); };
    auto host = [&](int slot) { return ar.chunk[(size_t) slot / per].as<uint8_t>() + (slot % per) * pages * PAGE; };
    auto place = [&](const std::vector<bool>& in) {
        std::memset(ar.smap.ptr, 0xFF, ar.smap.bytes);
        for (int e = 0; e < NE; ++e)
            if (in[(size_t) e]) {
                std::memcpy(host(order[(size_t) e]), whole.as<uint8_t>() + e * blob, blob);
                ar.map()[e] = code(order[(size_t) e]);
            }
    };
    auto arena_args = [&](std::vector<mtl::Arg>& a) {
        a.push_back(mtl::Arg::B(ar.smap, 4 * NE));
        for (int c = 1; c < CHUNKS; ++c) a.push_back(mtl::Arg::B(ar.chunk[(size_t) c]));
    };
    const mtl::Constants on = mtl::Constants().b(20, true);
    ref::Rng r(3 + W);

    // ---- the window path
    {
        const int lib = cx.compile({"quant.metal", "units.metal", "latent_routed.metal"});
        place(std::vector<bool>(NE, true));
        mtl::Buffer labels = cx.alloc(4 * NE);
        for (int e = 0; e < NE; ++e) labels.as<int32_t>()[e] = (int32_t) (r.next() % GROUPS);
        for (const int T : {1, 5, 8}) {
            const int entries = T * TOPK;
            mtl::Buffer ids = cx.alloc(4 * entries), proj = cx.alloc(4 * (size_t) T * GROUPS * W);
            for (int t = 0; t < T; ++t) {
                std::set<int> used;
                for (int k = 0; k < TOPK; ++k) {
                    int e;
                    do e = (int) (r.next() % NE); while (used.count(e));
                    used.insert(e);
                    ids.as<int32_t>()[t * TOPK + k] = e;
                }
            }
            if (misplace) ar.map()[ids.as<int32_t>()[0]] = code(order[(size_t) ids.as<int32_t>()[1]]);
            for (size_t i = 0; i < (size_t) T * GROUPS * W; ++i) proj.as<float>()[i] = r.sym();
            LatentRoutedArgs la{(uint32_t) T, GROUPS, HIDDEN, (uint32_t) W, NFF, TOPK, 0, 0, (uint32_t) gu_row, (uint32_t) d_row,
                                (uint32_t) up_off, (uint32_t) down_off, (uint32_t) blob};
            mtl::Buffer mid[2] = {cx.alloc(4 * (size_t) entries * NFF), cx.alloc(4 * (size_t) entries * NFF)};
            mtl::Buffer out[2] = {cx.alloc(4 * (size_t) entries * W), cx.alloc(4 * (size_t) entries * W)};
            for (const auto& [name, rows] : std::vector<std::pair<std::string, int>>{{"latent_gate_up", 4}, {"latent_gate_up_8", 16},
                                                                                     {"latent_gate_up_4", 32}}) {
                mtl::Pipeline p[2] = {cx.pipeline(lib, name), cx.pipeline(lib, name, on)};
                for (int k = 0; k < 2; ++k) {
                    std::memset(mid[k].ptr, 0, mid[k].bytes);
                    mtl::Dispatch d;
                    d.p = &p[k];
                    d.args = {mtl::Arg::B(k ? ar.chunk[0] : whole), mtl::Arg::B(labels), mtl::Arg::B(ids), mtl::Arg::B(proj),
                              mtl::Arg::B(mid[k]), mtl::Arg::V(la)};
                    if (k) arena_args(d.args);
                    d.grid[0] = NFF / rows;
                    d.grid[1] = entries;
                    d.tg[0] = 128;
                    cx.run({d});
                }
                report("window T=" + std::to_string(T) + " " + name, mid[0], mid[1], mid[0].bytes);
            }
            for (const auto& [name, rows] : std::vector<std::pair<std::string, int>>{{"latent_down", 4}, {"latent_down_4", 32}}) {
                mtl::Pipeline p[2] = {cx.pipeline(lib, name), cx.pipeline(lib, name, on)};
                for (int k = 0; k < 2; ++k) {
                    std::memset(out[k].ptr, 0, out[k].bytes);
                    mtl::Dispatch d;
                    d.p = &p[k];
                    d.args = {mtl::Arg::B(k ? ar.chunk[0] : whole), mtl::Arg::B(ids), mtl::Arg::B(mid[0]), mtl::Arg::B(out[k]),
                              mtl::Arg::V(la)};
                    if (k) arena_args(d.args);
                    d.grid[0] = W / rows;
                    d.grid[1] = entries;
                    d.tg[0] = 128;
                    cx.run({d});
                }
                report("window T=" + std::to_string(T) + " " + name, out[0], out[1], out[0].bytes);
            }
            if (misplace) ar.map()[ids.as<int32_t>()[0]] = code(order[(size_t) ids.as<int32_t>()[0]]);
        }
    }
    if (window_only) return 0;

    // ---- the prompt chain: the experts of 400 of the 512 are routed and in the arena; the other 112 are not
    {
        const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "moe_mm.metal", "latent_prompt_q3_stage.metal"});
        std::vector<bool> in(NE, false);
        for (int e = 0; e < 400; ++e) in[(size_t) e] = true;
        place(in);
        mtl::Pipeline pclear = cx.pipeline(lib, "mm_clear"), phist = cx.pipeline(lib, "mm_hist"), pscan = cx.pipeline(lib, "mm_scan"),
                      pplace = cx.pipeline(lib, "mm_place");
        const std::string gun = "mm_gu2_" + std::to_string(ref::Q3_K) + "_32_64_32_2",
                          dnn = "mm_down2_" + std::to_string(ref::IQ4_NL) + "_32_64_32_2";
        const mtl::Constants p0 = mtl::Constants().i(5, 0), p0a = mtl::Constants().i(5, 0).b(20, true);
        mtl::Pipeline pgu[2] = {cx.pipeline(lib, gun, p0), cx.pipeline(lib, gun, p0a)};
        mtl::Pipeline pdn[2] = {cx.pipeline(lib, dnn), cx.pipeline(lib, dnn, on)};
        mtl::Pipeline pst[2] = {cx.pipeline(lib, "latent_prompt_q3_stage"), cx.pipeline(lib, "latent_prompt_q3_stage", on)};
        mtl::Pipeline pgu16 = cx.pipeline(lib, "latent_prompt_gu_f16");
        mtl::Buffer w16 = cx.alloc(2ull * NE * 2 * NFF * W);
        for (const int T : {256, 2048}) {
            const bool staged = T >= 2048;
            const int nent = T * TOPK, nblk = (nent + 1023) / 1024, maxrows = nent + NE * TM;
            mtl::Buffer ids = cx.alloc(4 * nent), w = cx.alloc(4 * nent), bcnt = cx.alloc(4 * (size_t) nblk * NE),
                        boff = cx.alloc(4 * (size_t) nblk * NE), eoff = cx.alloc(4 * NE), tiles = cx.alloc(8 * (size_t) (maxrows / TM + 1)),
                        meta = cx.alloc(64), stok = cx.alloc(4 * (size_t) maxrows), sw = cx.alloc(4 * (size_t) maxrows),
                        pos = cx.alloc(4 * nent), XS = cx.alloc(2 * (size_t) maxrows * W);
            mtl::Buffer HS[2] = {cx.alloc(2 * (size_t) maxrows * NFF), cx.alloc(2 * (size_t) maxrows * NFF)};
            mtl::Buffer YE[2] = {cx.alloc(4 * (size_t) maxrows * W), cx.alloc(4 * (size_t) maxrows * W)};
            for (int t = 0; t < T; ++t) {
                std::set<int> used;
                for (int k = 0; k < TOPK; ++k) {
                    int e;
                    do e = (int) (r.next() % 400); while (used.count(e));
                    used.insert(e);
                    ids.as<int32_t>()[t * TOPK + k] = e;
                    w.as<float>()[t * TOPK + k] = 0.1f;
                }
            }
            for (size_t i = 0; i < (size_t) maxrows * W; ++i) XS.as<uint16_t>()[i] = ref::f32_to_f16(r.sym());
            {   // the plan: tiles of TM rows per expert (meta[0] tiles)
                MmPlanArgs pa{(uint32_t) nent, TOPK, NE, TM, (uint32_t) nblk};
                const uint32_t nclr = (uint32_t) maxrows;
                meta.as<uint32_t>()[2] = NFF / 64;
                std::vector<mtl::Dispatch> ds(4);
                ds[0].p = &pclear; ds[0].args = {mtl::Arg::B(stok), mtl::Arg::V(nclr)}; ds[0].grid[0] = (maxrows + 255) / 256; ds[0].tg[0] = 256;
                ds[1].p = &phist; ds[1].args = {mtl::Arg::B(ids), mtl::Arg::B(bcnt), mtl::Arg::V(pa)}; ds[1].grid[0] = nblk; ds[1].tg[0] = 1024;
                ds[2].p = &pscan; ds[2].args = {mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(tiles), mtl::Arg::B(meta),
                                                mtl::Arg::V(pa)}; ds[2].tg[0] = NE;
                ds[3].p = &pplace; ds[3].args = {mtl::Arg::B(ids), mtl::Arg::B(w), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(stok),
                                                 mtl::Arg::B(sw), mtl::Arg::B(pos), mtl::Arg::V(pa)}; ds[3].grid[0] = nblk; ds[3].tg[0] = 1024;
                cx.run(ds);
            }
            const uint32_t ntiles = meta.as<uint32_t>()[0];
            if (misplace) ar.map()[ids.as<int32_t>()[0]] = code(order[(size_t) ids.as<int32_t>()[1]]);
            const MmArgs ma{0, up_off, down_off, blob, blob, (uint32_t) gu_row, (uint32_t) d_row, (uint32_t) W, NFF};
            for (int k = 0; k < 2; ++k) {
                std::vector<mtl::Dispatch> ds;
                if (staged) {
                    std::memset(w16.ptr, 0x7B, w16.bytes);   // what an arena run does not stage stays this (large finite halves)
                    const uint32_t q3[4] = {(uint32_t) W, NFF, (uint32_t) gu_row, (uint32_t) blob};
                    mtl::Dispatch s;
                    s.p = &pst[k];
                    s.args = {mtl::Arg::B(k ? ar.chunk[0] : whole), mtl::Arg::B(w16), mtl::Arg::V(q3)};
                    if (k) arena_args(s.args);
                    s.grid[0] = (2 * NFF * (W / 32) + 127) / 128;
                    s.grid[1] = NE;
                    s.tg[0] = 128;
                    ds.push_back(s);
                    mtl::Dispatch g;
                    g.p = &pgu16;
                    g.args = {mtl::Arg::B(w16), mtl::Arg::B(XS), mtl::Arg::B(HS[k]), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
                    g.grid[0] = NFF / 64;
                    g.grid[1] = ntiles;
                    g.tg[0] = 64;
                    ds.push_back(g);
                } else {
                    mtl::Dispatch g;
                    g.p = &pgu[k];
                    g.args = {mtl::Arg::B(k ? ar.chunk[0] : whole), mtl::Arg::B(XS), mtl::Arg::B(HS[k]), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
                    if (k) arena_args(g.args);
                    g.grid[0] = NFF / 64;
                    g.grid[1] = ntiles;
                    g.tg[0] = 64;
                    ds.push_back(g);
                }
                cx.run(ds);
                mtl::Dispatch dd;   // the down GEMM, on the whole run's middle rows (the gate/up result is compared above)
                dd.p = &pdn[k];
                dd.args = {mtl::Arg::B(k ? ar.chunk[0] : whole), mtl::Arg::B(HS[0]), mtl::Arg::B(YE[k]), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
                if (k) arena_args(dd.args);
                dd.grid[0] = W / 64;
                dd.grid[1] = ntiles;
                dd.tg[0] = 64;
                cx.run({dd});
            }
            const std::string rows = "prompt " + std::to_string(T) + " rows ";
            report(rows + (staged ? "latent_prompt_q3_stage + latent_prompt_gu_f16" : gun), HS[0], HS[1], HS[0].bytes);
            report(rows + dnn, YE[0], YE[1], YE[0].bytes);
            if (misplace) ar.map()[ids.as<int32_t>()[0]] = code(order[(size_t) ids.as<int32_t>()[0]]);
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> widths;
    bool window_only = false, misplace = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--width" && i + 1 < argc) widths.push_back(std::atoi(argv[++i]));
        else if (a == "--window-only") window_only = true;
        else if (a == "--misplace") misplace = true;
        else { std::fprintf(stderr, "usage: latent_arena_parity [--width 1280|1792] [--window-only] [--misplace]\n"); return 2; }
    }
    if (widths.empty()) widths = {1280, 1792};
    mtl::Context cx;
    std::printf("device %s%s\n", cx.device_name().c_str(), misplace ? "; --misplace: one routed expert's smap points at another slot" : "");
    for (int w : widths) {
        if (w != 1280 && w != 1792) { std::fprintf(stderr, "width %d: 1280 or 1792\n", w); return 2; }
        run_width(cx, w, window_only, misplace);
    }
    std::printf("%d cases, %d different: %s\n", cases, failures, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
