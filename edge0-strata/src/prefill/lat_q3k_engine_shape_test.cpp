// src/prefill/lat_q3k_engine_shape_test.cpp - "our output differs by how much", per shape.
// ONE input, two paths, byte for byte the same arguments: the frozen serial chain
// (lat_q3k_gu -> lat_q3k_swiglu_q8 -> lat_q3k_down) and lat_q3k_fused.  hbuf is compared on its own, so a difference
// points at the fused epilogue (hbuf) or at the down (out).
// Shapes: the engine's own 512-group table from the dump (dump_ref/g-off.bin, g-cnt.bin, g-tok.bin), the single groups
// the engine failed on, nb=2, the tile boundaries, and the 128 x 4 regression.
#include "strata/prefill/lat_q3k_fused.hpp"
#include "strata/prefill/lat_q3k_gu.hpp"
#include "strata/prefill/lat_q3k_swiglu.hpp"
#include "strata/prefill/lat_q3k_down.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

namespace {
constexpr int64_t ELL = 1280, FF = 640;
constexpr int64_t GUB = 1280 * 550, DNB = ELL * FF / 32 * 18;
constexpr int kXq = (int) (ELL / 32) * 36;
constexpr int kHq = (int) strata::prefill::lat_q3k_fused_hbuf_row;
constexpr int NBLOB = 4;
constexpr int E = 42450;          // the engine's entry space (the dumps' off[NG])
constexpr int NG = 512;
const std::string DUMP = "/home/user/lat-s26/q3k/dump_ref/";

template <class T>
std::vector<T> rd(const char* name, size_t expect) {
    std::vector<T> v(expect);
    std::FILE* f = std::fopen((DUMP + name).c_str(), "rb");
    if (f == nullptr) { std::printf("FAIL cannot open %s\n", name); std::exit(1); }
    const size_t n = std::fread(v.data(), sizeof(T), expect, f);
    std::fclose(f);
    if (n != expect) { std::printf("FAIL %s: %zu of %zu\n", name, n, expect); std::exit(1); }
    return v;
}
}  // namespace

struct Shape {
    std::string tag;
    std::vector<int32_t> starts;   // ng + 1
    std::vector<int32_t> counts;   // ng
    int ng() const { return (int) counts.size(); }
};

static bool hbuf_at_ok(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t at) {
    return a[at] == b[at];
}

int main() {
    std::vector<int32_t> off = rd<int32_t>("g-off.bin", NG + 1);
    std::vector<int32_t> cnt = rd<int32_t>("g-cnt.bin", NG);
    std::vector<int32_t> tok = rd<int32_t>("g-tok.bin", (size_t) E);
    if (off[NG] != E) { std::printf("FAIL off[NG] %d != %d\n", off[NG], E); return 1; }

    // 4 different experts (never one blob replicated: a wrong row range would still write the right values)
    std::mt19937 rng(91);
    std::normal_distribution<float> nw(0.f, 0.05f);
    std::vector<uint8_t> blob((size_t) NBLOB * (GUB + DNB));
    for (int i = 0; i < NBLOB; ++i) {
        uint8_t* b = blob.data() + (size_t) i * (GUB + DNB);
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), b, 0, 1280, ELL, nullptr) != (size_t) GUB) {
            std::printf("FAIL gu quantize\n"); return 1;
        }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), b + GUB, 0, ELL, FF, nullptr) != (size_t) DNB) {
            std::printf("FAIL dn quantize\n"); return 1;
        }
    }
    const size_t EB = (size_t) (GUB + DNB);
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, blob.size()));
    CK(cudaMemcpy(dall, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    std::vector<const uint8_t*> bp((size_t) NG);
    for (int g = 0; g < NG; ++g) bp[(size_t) g] = dall + (size_t) (g % NBLOB) * EB;
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) NG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) NG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    int ntok = 0;
    for (int32_t t : tok) ntok = std::max(ntok, (int) t);
    ++ntok;
    std::vector<uint8_t> xq((size_t) ntok * kXq);
    {
        std::uniform_int_distribution<int> u8(-127, 127);
        for (size_t i = 0; i < xq.size(); ++i) xq[i] = (uint8_t) u8(rng);
        for (int t = 0; t < ntok; ++t)
            for (int k = 0; k < (int) (ELL / 32); ++k) {
                uint8_t* b = xq.data() + (size_t) t * kXq + 36 * k;
                const uint16_t d16 = ggml_fp32_to_fp16(0.01f + 1e-4f * ((t + k) % 97)), s16 = 0;
                std::memcpy(b, &d16, 2); std::memcpy(b + 2, &s16, 2);
            }
    }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, xq.size()));
    CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    int32_t* dtok = nullptr;
    CK(cudaMalloc(&dtok, (size_t) E * 4));
    CK(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice));
    std::vector<int32_t> ident((size_t) E);
    for (int e = 0; e < E; ++e) ident[(size_t) e] = e;
    int32_t* dident = nullptr;
    CK(cudaMalloc(&dident, (size_t) E * 4));
    CK(cudaMemcpy(dident, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice));

    int32_t *dgs = nullptr, *dcnt = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (NG + 1) * 4));
    CK(cudaMalloc(&dcnt, (size_t) NG * 4));
    CK(cudaMalloc(&dng, 4));
    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) E * FF * 4)); CK(cudaMalloc(&dup, (size_t) E * FF * 4));
    uint8_t* dhq = nullptr; uint8_t* dhbuf = nullptr;
    CK(cudaMalloc(&dhq, (size_t) E * kHq)); CK(cudaMalloc(&dhbuf, (size_t) E * kHq));
    float* dfu = nullptr; float* dse = nullptr;
    CK(cudaMalloc(&dfu, (size_t) E * ELL * 4)); CK(cudaMalloc(&dse, (size_t) E * ELL * 4));

    int sms = 0;
    { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); }
    const int kItems = (int) (ELL / 256);   // the launchers' fb count (n_embd / 256 for the down, 5 here)
    int gz_last = 1;

    std::vector<float> yfu((size_t) E * ELL), yse((size_t) E * ELL);
    std::vector<uint8_t> hfu((size_t) E * kHq), hse((size_t) E * kHq);

    // one shape: identical arguments to both paths, hbuf and out compared separately
    auto run = [&](const Shape& sh, bool print_only) -> int {
        const int ng = sh.ng();
        CK(cudaMemcpy(dgs, sh.starts.data(), (size_t) (ng + 1) * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcnt, sh.counts.data(), (size_t) ng * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice));
        gz_last = (sms > 0 && ng * kItems < sms) ? 8 : 1;
        CK(cudaMemset(dhbuf, 0x00, (size_t) E * kHq));
        CK(cudaMemset(dhq, 0x00, (size_t) E * kHq));
        CK(cudaMemset(dfu, 0, (size_t) E * ELL * 4));
        CK(cudaMemset(dse, 0, (size_t) E * ELL * 4));
        strata::prefill::lat_q3k_fused(dblob, dgs, ng, dxq, dhbuf, dfu, ELL, nullptr, dcnt, dtok, dident);
        strata::prefill::lat_q3k_gu(dblob, dgs, ng, dxq, dgate, dup, FF, nullptr, dcnt, dtok);
        strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        strata::prefill::lat_q3k_down(dblob, dgs, ng, dhq, dse, ELL, nullptr, dcnt, dident);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(yfu.data(), dfu, yfu.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yse.data(), dse, yse.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hfu.data(), dhbuf, hfu.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hse.data(), dhq, hse.size(), cudaMemcpyDeviceToHost));

        // the rows this shape owns
        std::vector<char> mine((size_t) E, 0);
        for (int g = 0; g < ng; ++g)
            for (int e = sh.starts[(size_t) g]; e < sh.starts[(size_t) g] + sh.counts[(size_t) g]; ++e) mine[(size_t) e] = 1;

        // hbuf: per entry, the 68 bytes the down reads (codes + the two d's)
        long long hb_diff = 0, hb_tot = 0, hb_pad = 0;
        int hb_first_e = -1, hb_first_c = -1, hb_first_b = -1, hb_zdiff = 0;
        for (int e = 0; e < E; ++e) {
            if (!mine[(size_t) e]) continue;
            for (int c = 0; c < 10; ++c)
                for (int b = 0; b < 80; ++b) {
                    const size_t at = (size_t) e * kHq + (size_t) c * 80 + (size_t) b;
                    ++hb_tot;
                    if (hbuf_at_ok(hfu, hse, at)) continue;
                    if (b >= 68) { ++hb_pad; continue; }
                    ++hb_diff;
                    if (hb_first_e < 0) { hb_first_e = e; hb_first_c = c; hb_first_b = b; }
                }
        }
        // out: 1280 floats an entry
        long long o_diff = 0, o_tot = 0;
        double o_max = 0, o_abs = 0, o_ref = 0;
        int o_first_e = -1, o_first_r = -1;
        for (int e = 0; e < E; ++e) {
            if (!mine[(size_t) e]) continue;
            for (int r = 0; r < ELL; ++r) {
                const size_t at = (size_t) e * ELL + r;
                const double a = yfu[at], b2 = yse[at];
                ++o_tot;
                if (a == b2) continue;
                ++o_diff;
                const double d = std::fabs(a - b2);
                o_max = std::max(o_max, d); o_abs += d; o_ref += std::fabs(b2);
                if (o_first_e < 0) { o_first_e = e; o_first_r = r; }
            }
        }
        // where the first hbuf difference sits in the tiles (which z slice would own that row)
        int rowin = -1, zs = -1, gid = -1;
        if (hb_first_e >= 0)
            for (int g = 0; g < ng; ++g)
                if (hb_first_e >= sh.starts[(size_t) g] && hb_first_e < sh.starts[(size_t) g] + sh.counts[(size_t) g]) {
                    gid = g; rowin = hb_first_e - sh.starts[(size_t) g]; zs = (rowin / 64) % gz_last; break;
                }
        std::printf("%-34s ng=%-4d rows=%-6zu gz=%d\n", sh.tag.c_str(), ng, (size_t) std::count(mine.begin(), mine.end(), 1),
                    gz_last);
        std::printf("   hbuf: differing %lld / %lld bytes (pad %lld)  first at entry %d chunk %d byte %d"
                    " (group %d, row %d, z %d)\n", hb_diff, hb_tot, hb_pad, hb_first_e, hb_first_c, hb_first_b, gid,
                    rowin, zs);
        std::printf("   out : differing %lld / %lld values  max|d| %.3e  rel %.3e  first at entry %d row %d\n", o_diff,
                    o_tot, o_max, o_ref > 0 ? o_abs / o_ref : 0.0, o_first_e, o_first_r);
        return (hb_diff == 0 && o_diff == 0) ? 0 : 1;
    };

    // ---- the shape table ---------------------------------------------------------------------------------------
    std::vector<Shape> shapes;
    {   // 1. the engine's own table
        Shape s; s.tag = "engine table (512 groups, dump)";
        s.starts = off; s.counts = cnt;
        shapes.push_back(s);
    }
    {   // 2. the single groups the engine failed on
        const int32_t one[][2] = {{468, 30567}, {41, 31035}, {2749, 32028}, {400, 34777}, {2376, 37469}};
        for (const auto& v : one) {
            Shape s;
            char t[64]; std::snprintf(t, sizeof t, "nb=1 cnt=%d start=%d", v[0], v[1]);
            s.tag = t; s.starts = {v[1], v[1] + v[0]}; s.counts = {v[0]};
            shapes.push_back(s);
        }
    }
    {   // 3. nb=2
        Shape s; s.tag = "nb=2 cnt={468,41} start={30567,31035}";
        s.starts = {30567, 31035, 31076}; s.counts = {468, 41};
        shapes.push_back(s);
    }
    {   // 4. every count from 1 to 64, then the boundaries and the engine's big sizes
        std::vector<int> sizes;
        for (int c = 1; c <= 64; ++c) sizes.push_back(c);
        for (int c : {65, 128, 256, 512, 1024, 1375, 2062, 2748, 2749, 3098}) sizes.push_back(c);
        for (int c : sizes) {
            Shape s;
            char t[64]; std::snprintf(t, sizeof t, "nb=1 sweep cnt=%d", c);
            s.tag = t; s.starts = {0, c}; s.counts = {c};
            shapes.push_back(s);
        }
    }
    {   // 5. the 128 x 4 regression
        Shape s; s.tag = "regression 128 groups x 4 entries";
        for (int g = 0; g <= 128; ++g) s.starts.push_back(g * 4);
        s.counts.assign(128, 4);
        shapes.push_back(s);
    }

    int fails = 0;
    for (const Shape& sh : shapes) fails += run(sh, false);
    std::printf("%s: %zu shapes, %d with a difference\n", fails == 0 ? "PASS" : "FAIL", shapes.size(), fails);
    return fails == 0 ? 0 : 1;
}
