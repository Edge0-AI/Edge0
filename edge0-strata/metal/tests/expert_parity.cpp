// metal/tests/expert_parity.cpp - the grouped routed-expert kernels (moe.metal) against a double-precision CPU
// reference, and their speed.
//
//   expert_parity [--gu Q2_0] [--dn Q2_0] [--ntok 1..8] [--gucfg L,R,S] [--dncfg L,R,S]     parity (synthetic)
//   expert_parity --bench [--gu ..] [--dn ..] [--ntok N] [--layers 8] [--reps 96] [--sweep]  speed
//
// Gyro (agentionai's trellis experts, TQ2_T / TQK6 / TQK7):
//   expert_parity --gu TQK6 --dn TQK7 [--had-pack <pack dir>]       synthetic blocks; with a pack, the Hadamard rotation of
//                                                                   x (CPU) and of h (the Metal had128 kernel) is part of it
//   expert_parity --gguf <Gyro file> --layer L [--had-pack <pack dir>] [--ntok N] [--seed S]   REAL experts of layer L (64 of
//                                                                   the 512, random) instead of random blocks
//   expert_parity --had-test [--had-pack <pack dir>]                the Metal had128 kernels (f32, f16, f16 rows) against the CPU rotation, bitwise
//
// Parity: random valid blocks for 64 experts, random activations x ~ U(-1,1), random distinct top-10 per token,
// weights summing to 1.  Checked: the plan (groups by expert id, entries in (token, k) order), h = silu(gate) * up
// per entry and y = sum_k w * down . h, each against double; reported as max |err| / max |ref| and relative L2.
// Bench: `layers` distinct copies of a 512-expert arena (no reuse across reps beyond chance; the SLC is ~32 MB),
// a different random routing per rep; GB/s = bytes of the distinct experts the kernel must read / GPU time.
#include "mtl_rt.h"
#include "ref_quant.h"
#include "gguf_lite.h"
// Gyro: the engine's CPU Hadamard rotation (the reference of the Metal one) and hadamard.txt's parser
#include "../../src/kernels/cpu/hadamard.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-common-tq.h"
const uint16_t* tq_lut_for_test() { return tq_lut_f16; }

namespace {

constexpr int N_EMBD = 2560, N_FF = 640, TOPK = 10;
// plan layout (moe.metal)
constexpr int P_NG = 0, P_IGU = 4, P_IDN = 8, P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE,
              P_ETOK = P_GSTART + P_MAXE + 4, P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE,
              P_INTS = P_TOKENT + P_MAXE;

struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};

int type_of(const std::string& s) {
    for (int t : {ref::TQ2_T, ref::TQK6, ref::TQK7, ref::Q2_0, ref::IQ3_XXS, ref::IQ2_XXS, ref::IQ2_XS, ref::IQ2_S, ref::IQ3_S, ref::IQ4_NL, ref::IQ4_XS, ref::Q4_K, ref::Q5_K, ref::Q6_K, ref::Q8_0, ref::Q5_1})
        if (s == ref::type_name(t)) return t;
    std::fprintf(stderr, "unknown type %s\n", s.c_str());
    std::exit(2);
}

struct Cfg { int L, R, S; };
Cfg parse_cfg(const char* s) {
    Cfg c{};
    if (std::sscanf(s, "%d,%d,%d", &c.L, &c.R, &c.S) != 3) { std::fprintf(stderr, "bad cfg %s\n", s); std::exit(2); }
    return c;
}
std::string kname(const char* k, int t, Cfg c) {
    return std::string(k) + "_" + std::to_string(t) + "_" + std::to_string(c.L) + "_" + std::to_string(c.R) + "_" +
           std::to_string(c.S);
}
int rows_per_tg(Cfg c) { return c.S * (32 / c.L) * c.R; }

struct Layout {
    int gu, dn;
    size_t gu_row, d_row, up_off, down_off, blob;
};
Layout layout(int gu, int dn) {
    Layout L{gu, dn};
    L.gu_row = ref::row_bytes(gu, N_EMBD);
    L.d_row = ref::row_bytes(dn, N_FF);
    L.up_off = (size_t)N_FF * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.blob = L.down_off + (size_t)N_EMBD * L.d_row;
    return L;
}

void random_routing(ref::Rng& r, int ntok, int n_expert, std::vector<int>& ids, std::vector<float>& w,
                    double share = 0.0, const std::vector<int>* prev = nullptr) {
    ids.assign(ntok * TOPK, 0);
    w.assign(ntok * TOPK, 0.0f);
    for (int t = 0; t < ntok; ++t) {
        std::set<int> used;
        float sum = 0;
        for (int k = 0; k < TOPK; ++k) {
            int e;
            do {
                // with probability `share`, reuse an expert of the previous token (verify windows share experts)
                if (t > 0 && share > 0 && r.uni() < share) e = ids[(t - 1) * TOPK + (r.next() % TOPK)];
                else e = (int)(r.next() % n_expert);
            } while (used.count(e));
            used.insert(e);
            ids[t * TOPK + k] = e;
            w[t * TOPK + k] = 0.2f + r.uni();
            sum += w[t * TOPK + k];
        }
        for (int k = 0; k < TOPK; ++k) w[t * TOPK + k] /= sum;
    }
    (void)prev;
}

struct Kernels {
    mtl::Pipeline plan, gu, dn, comb, prep;
    Cfg cgu, cdn;
};
int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}
constexpr int CAPE = 8 * TOPK;                     // entries capacity of h
// prepared buffers: x [8][N_EMBD] -> XPX, h [CAPE][N_FF] -> XPH (4 planes + S16)
constexpr size_t PLANE_X = 8 * N_EMBD, PLANE_H = (size_t)CAPE * N_FF;

mtl::Dispatch prep_dispatch(const mtl::Pipeline& p, const mtl::Buffer& x, const mtl::Buffer& xp, uint32_t n, uint32_t nt) {
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(x), mtl::Arg::B(xp), mtl::Arg::V(n), mtl::Arg::V(nt)};
    d.tg[0] = 256;
    d.grid[0] = (n / 16 * nt + 255) / 256;
    return d;
}

int g_gu_type = 0, g_dn_type = 0;
// Gyro: the rotation of h between the gate/up and the down dispatch (the Metal had128_f32 kernel, signs of width n_ff)
struct HadArgs { uint32_t width, x_stride, y_stride; float scale; uint32_t has_signs, pad; };
const mtl::Pipeline* g_had32 = nullptr;
const mtl::Buffer* g_hsh = nullptr;
bool g_had_h = false;
bool g_dn_prep = false;   // Q2_0 down: prepared h (an xprep dispatch) instead of the raw-x decoder
const mtl::Buffer* g_xpx = nullptr;
const mtl::Buffer* g_xph = nullptr;

void encode_layer(std::vector<mtl::Dispatch>& ds, const Kernels& K, const mtl::Buffer& W, size_t w_off,
                  const mtl::Buffer& x, const mtl::Buffer& h, const mtl::Buffer& ye, const mtl::Buffer& y,
                  const mtl::Buffer& ids, const mtl::Buffer& wts, const mtl::Buffer& plan, size_t plan_off,
                  const MoeArgs& a, bool with_plan, bool with_gu, bool with_dn, bool with_comb) {
    const mtl::Buffer& XPX = *g_xpx;
    const mtl::Buffer& XPH = *g_xph;
    if (with_plan) {
        mtl::Dispatch d;
        d.p = &K.plan;
        d.args = {mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::B(plan, plan_off), mtl::Arg::V(a)};
        d.tg[0] = 512;
        ds.push_back(d);
    }
    if (with_gu) {
        mtl::Dispatch d;
        d.p = &K.gu;
        d.args = {mtl::Arg::B(W, w_off), mtl::Arg::B(XPX, 4 * PLANE_X * xkind(g_gu_type)), mtl::Arg::B(h),
                  mtl::Arg::B(plan, plan_off), mtl::Arg::V(a), mtl::Arg::B(XPX, 4 * PLANE_X * 4)};
        d.tg[0] = K.cgu.S * 32;
        d.indirect = &plan;
        d.ind_off = plan_off + P_IGU * 4;
        ds.push_back(d);
        if (g_had_h) {   // H h in place, one row per entry (the entries are the window's token x k)
            static HadArgs ha;
            ha = HadArgs{(uint32_t)N_FF, (uint32_t)N_FF, (uint32_t)N_FF, 1.0f / std::sqrt(128.0f), g_hsh ? 1u : 0u, 0u};
            mtl::Dispatch hd;
            hd.p = g_had32;
            hd.args = {mtl::Arg::B(h), mtl::Arg::B(h), mtl::Arg::B(g_hsh ? *g_hsh : h), mtl::Arg::V(ha)};
            hd.grid[0] = N_FF / 128;
            hd.grid[1] = a.n_tok * TOPK;
            hd.tg[0] = 128;
            ds.push_back(hd);
        }
    }
    if (with_dn) {
        const int dk = g_dn_prep ? xkind(g_dn_type) : 0;
        if (dk) ds.push_back(prep_dispatch(K.prep, h, XPH, N_FF, CAPE));
        mtl::Dispatch d;
        d.p = &K.dn;
        d.args = {mtl::Arg::B(W, w_off),
                  dk ? mtl::Arg::B(XPH, 4 * PLANE_H * dk) : mtl::Arg::B(h), mtl::Arg::B(ye),
                  mtl::Arg::B(plan, plan_off), mtl::Arg::V(a), mtl::Arg::B(XPH, 4 * PLANE_H * 4)};
        d.tg[0] = K.cdn.S * 32;
        d.indirect = &plan;
        d.ind_off = plan_off + P_IDN * 4;
        ds.push_back(d);
    }
    if (with_comb) {
        mtl::Dispatch d;
        d.p = &K.comb;
        d.args = {mtl::Arg::B(ye), mtl::Arg::B(y), mtl::Arg::B(plan, plan_off), mtl::Arg::V(a)};
        d.tg[0] = 256;
        d.grid[0] = (a.n_tok * N_EMBD + 255) / 256;
        ds.push_back(d);
    }
}

struct Err {
    double max_abs = 0, max_ref = 0, se = 0, sr = 0;
    void add(double got, double ref) {
        max_abs = std::max(max_abs, std::fabs(got - ref));
        max_ref = std::max(max_ref, std::fabs(ref));
        se += (got - ref) * (got - ref);
        sr += ref * ref;
    }
    double rel_max() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
    double rel_l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int gu = ref::Q2_0, dn = ref::Q2_0, ntok = 1, layers = 8, reps = 96;
    Cfg cgu{32, 2, 4}, cdn{8, 2, 4};
    bool bench = false, sweep = false, had_test = false, decode_check = false;
    double share = 0.0;
    std::string gguf_path, had_pack;
    int real_layer = 0, experts_first = 0;
    uint64_t seed = 1;
    bool cfg_gu_set = false, cfg_dn_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return i + 1 < argc ? argv[++i] : (std::fprintf(stderr, "%s needs a value\n", a.c_str()), std::exit(2), (char*)nullptr); };
        if (a == "--gu") gu = type_of(nx());
        else if (a == "--dn") dn = type_of(nx());
        else if (a == "--ntok") ntok = std::atoi(nx());
        else if (a == "--gucfg") { cgu = parse_cfg(nx()); cfg_gu_set = true; }
        else if (a == "--dncfg") { cdn = parse_cfg(nx()); cfg_dn_set = true; }
        else if (a == "--bench") bench = true;
        else if (a == "--sweep") sweep = true;
        else if (a == "--layers") layers = std::atoi(nx());
        else if (a == "--reps") reps = std::atoi(nx());
        else if (a == "--share") share = std::atof(nx());
        else if (a == "--dn-prep") g_dn_prep = true;
        else if (a == "--gguf") gguf_path = nx();
        else if (a == "--layer") real_layer = std::atoi(nx());
        else if (a == "--experts-first") experts_first = std::atoi(nx());   // only experts 0..N-1 of the layer are in the file (a sparse copy)
        else if (a == "--had-pack") had_pack = nx();
        else if (a == "--had-test") had_test = true;
        else if (a == "--decode-check") decode_check = true;
        else if (a == "--seed") seed = std::strtoull(nx(), nullptr, 10);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    (void)cfg_gu_set;
    (void)cfg_dn_set;
    if (dn == ref::IQ4_NL && cdn.L == 8 && !cfg_dn_set) cdn = {4, 2, 4};
    gguf::File gf;
    const gguf::Tensor *tg_ = nullptr, *tu_ = nullptr, *td_ = nullptr;
    if (!gguf_path.empty()) {   // real experts: the types come from the file
        std::string err;
        if (!gf.open(gguf_path, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        const std::string b = "blk." + std::to_string(real_layer) + ".";
        tg_ = gf.find(b + "ffn_gate_exps.weight");
        tu_ = gf.find(b + "ffn_up_exps.weight");
        td_ = gf.find(b + "ffn_down_exps.weight");
        if (!tg_ || !tu_ || !td_) { std::fprintf(stderr, "no routed experts in layer %d of %s\n", real_layer, gguf_path.c_str()); return 2; }
        gu = tg_->type;
        dn = td_->type;
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "moe.metal"});
    const Layout L = layout(gu, dn);
    // Gyro: the pack's hadamard.txt (the signs of the widths 2560 and 640) -> the engine's CPU spec (the reference)
    mtl::Buffer hsx, hsh;
    mtl::Pipeline had32 = cx.pipeline(lib, "had128_f32"), had16 = cx.pipeline(lib, "had128_f16"), had16r = cx.pipeline(lib, "had128_f16_rows");
    const float* sgx = nullptr;
    const float* sgh = nullptr;
    if (!had_pack.empty()) {
        strata::kernels::cpu::HadamardSpec spec;
        std::string err;
        if (!strata::kernels::cpu::hadamard_parse(had_pack + "/hadamard.txt", 48, spec, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
        strata::kernels::cpu::hadamard_set(std::move(spec));
        sgx = strata::kernels::cpu::hadamard_signs(N_EMBD);
        sgh = strata::kernels::cpu::hadamard_signs(N_FF);
        if (sgx) { hsx = cx.alloc(4 * N_EMBD); std::memcpy(hsx.ptr, sgx, 4 * N_EMBD); }
        if (sgh) { hsh = cx.alloc(4 * N_FF); std::memcpy(hsh.ptr, sgh, 4 * N_FF); }
        g_had32 = &had32;
        g_hsh = sgh ? &hsh : nullptr;
        g_had_h = !had_test;
        std::printf("Hadamard: block %d, signs for %s%s\n", strata::kernels::cpu::hadamard_spec().block, sgx ? "2560 " : "", sgh ? "640" : "(none)");
    }
    if (decode_check) {
        // the decoder the Metal kernels and ref_quant share the meaning of, written a third time straight from the format's
        // definition (ggml-common-tq.h): step t's state = 16 stream bits from bit (31 - t) * K, LSB first, circular modulo 32 K
        // bits, masked to 15 bits (TQ2_T: qs[(t + 31) % 32] << 8 | qs[t]); x = state * 0x9e3779b1; points lut[x >> 21] and
        // lut[(x >> 10) & 2047]; weights = fp16 values * d.  Compared bit for bit with ref::dequant on the real tensors.
        extern const uint16_t* tq_lut_for_test();
        long blocks = 0, bad = 0;
        const gguf::Tensor* ts[3] = {tg_, tu_, td_};
        if (!tg_) { std::fprintf(stderr, "--decode-check needs --gguf\n"); return 2; }
        for (const gguf::Tensor* t : ts) {
            const int ty = t->type, K = ty == ref::TQ2_T ? 8 : ty == ref::TQK6 ? 6 : 7, bb = ref::block_bytes(ty);
            const uint8_t* base = (const uint8_t*)gf.map + t->offset;
            const size_t nb = t->bytes / bb;
            const size_t step = std::max<size_t>(1, nb / 200000);   // up to 200k blocks a tensor, spread over it
            for (size_t i = 0; i < nb; i += step) {
                const uint8_t* b = base + i * bb;
                float want[128], got[128];
                const float d = ref::f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
                const uint8_t* qs = b + 2;
                for (int st = 0; st < 32; ++st) {
                    uint32_t state;
                    if (K == 8) state = (((uint32_t)qs[(st + 31) % 32] << 8) | qs[st]) & 0x7fff;
                    else {
                        const int off = ((31 - st) * K) % (32 * K);
                        uint32_t v = 0;
                        for (int bit = 0; bit < 16; ++bit) {
                            const int pos = (off + bit) % (32 * K);
                            v |= (uint32_t)((qs[pos / 8] >> (pos % 8)) & 1) << bit;
                        }
                        state = v & 0x7fff;
                    }
                    const uint32_t x = state * 0x9e3779b1u;
                    const uint16_t* lut = tq_lut_for_test();
                    const uint32_t pa = x >> 21, pb = (x >> 10) & 2047u;
                    want[4 * st] = ref::f16_to_f32(lut[2 * pa]) * d;
                    want[4 * st + 1] = ref::f16_to_f32(lut[2 * pa + 1]) * d;
                    want[4 * st + 2] = ref::f16_to_f32(lut[2 * pb]) * d;
                    want[4 * st + 3] = ref::f16_to_f32(lut[2 * pb + 1]) * d;
                }
                ref::dequant(ty, b, got, 128);
                bad += std::memcmp(want, got, sizeof want) != 0;
                ++blocks;
            }
        }
        std::printf("decode-check layer %d: %ld blocks of the real tensors, %ld differ from the format's definition  %s\n", real_layer,
                    blocks, bad, bad == 0 ? "PASS" : "FAIL");
        return bad == 0 ? 0 : 1;
    }
    if (had_test) {
        // the Metal rotation kernels against the CPU's (hadamard_rotate), bit for bit
        int bad = 0;
        ref::Rng r2(77 + seed);
        for (int width : {N_FF, N_EMBD}) {
            const int rows = 37;
            const float* sg = width == N_EMBD ? sgx : sgh;
            const mtl::Buffer& sb = width == N_EMBD ? hsx : hsh;
            const float scale = 1.0f / std::sqrt(128.0f);
            std::vector<float> in((size_t)rows * width);
            for (auto& v : in) v = r2.sym() * 3.0f;
            mtl::Buffer bx = cx.alloc(4 * in.size()), by = cx.alloc(4 * in.size());
            std::memcpy(bx.ptr, in.data(), 4 * in.size());
            HadArgs ha{(uint32_t)width, (uint32_t)width, (uint32_t)width, scale, sg ? 1u : 0u, 0u};
            auto run32 = [&](const mtl::Buffer& src, const mtl::Buffer& dst) {
                mtl::Dispatch d;
                d.p = &had32;
                d.args = {mtl::Arg::B(src), mtl::Arg::B(dst), mtl::Arg::B(sg ? sb : src), mtl::Arg::V(ha)};
                d.grid[0] = width / 128; d.grid[1] = rows; d.tg[0] = 128;
                cx.run({d});
            };
            run32(bx, by);   // out of place
            std::vector<float> want = in;
            for (int r = 0; r < rows; ++r) strata::kernels::cpu::hadamard_rotate(want.data() + (size_t)r * width, width, 128, sg);
            int b32 = 0;
            for (size_t i = 0; i < in.size(); ++i) b32 += std::memcmp(&want[i], by.as<float>() + i, 4) != 0;
            run32(bx, bx);   // in place
            int b32i = 0;
            for (size_t i = 0; i < in.size(); ++i) b32i += std::memcmp(&want[i], bx.as<float>() + i, 4) != 0;
            // f16 (no signs): out of place, and the rows kernel in place; the CPU's float result rounded to half
            std::vector<_Float16> hin(in.size());
            for (size_t i = 0; i < in.size(); ++i) hin[i] = (_Float16)in[i];
            std::vector<float> hw(in.size());
            for (size_t i = 0; i < in.size(); ++i) hw[i] = (float)hin[i];
            for (int r = 0; r < rows; ++r) strata::kernels::cpu::hadamard_rotate(hw.data() + (size_t)r * width, width, 128, nullptr);
            mtl::Buffer hx = cx.alloc(2 * in.size()), hy = cx.alloc(2 * in.size());
            std::memcpy(hx.ptr, hin.data(), 2 * in.size());
            HadArgs hb{(uint32_t)width, (uint32_t)width, (uint32_t)width, scale, 0u, 0u};
            {
                mtl::Dispatch d;
                d.p = &had16;
                d.args = {mtl::Arg::B(hx), mtl::Arg::B(hy), mtl::Arg::B(hx), mtl::Arg::V(hb)};
                d.grid[0] = width / 128; d.grid[1] = rows; d.tg[0] = 128;
                cx.run({d});
            }
            int b16 = 0;
            for (size_t i = 0; i < in.size(); ++i) {
                const _Float16 w = (_Float16)hw[i];
                b16 += std::memcmp(&w, hy.as<_Float16>() + i, 2) != 0;
            }
            {
                mtl::Dispatch d;
                d.p = &had16r;
                d.args = {mtl::Arg::B(hx), mtl::Arg::B(hx), mtl::Arg::B(hx), mtl::Arg::V(hb)};
                d.grid[0] = rows; d.tg[0] = 128;
                cx.run({d});
            }
            int b16r = 0;
            for (size_t i = 0; i < in.size(); ++i) {
                const _Float16 w = (_Float16)hw[i];
                b16r += std::memcmp(&w, hx.as<_Float16>() + i, 2) != 0;
            }
            std::printf("had-test width %d, %d rows%s: f32 %d / f32 in place %d / f16 %d / f16 rows %d of %zu values differ from the CPU\n",
                        width, rows, sg ? " (with signs)" : " (no signs)", b32, b32i, b16, b16r, in.size());
            bad += b32 + b32i + b16 + b16r;
        }
        std::printf("had-test %s\n", bad == 0 ? "PASS (bitwise)" : "FAIL");
        return bad == 0 ? 0 : 1;
    }
    std::printf("device %s; gate/up %s, down %s, blob %zu bytes; %d token(s)\n", cx.device_name().c_str(),
                ref::type_name(gu), ref::type_name(dn), L.blob, ntok);

    auto make = [&](Cfg a, Cfg b) {
        Kernels K;
        K.plan = cx.pipeline(lib, "moe_plan");
        K.prep = cx.pipeline(lib, "xprep");
        K.comb = cx.pipeline(lib, "moe_combine");
        K.gu = cx.pipeline(lib, kname("moe_gu", gu, a), mtl::Constants().i(22, std::getenv("MOE_PROBE") ? std::atoi(std::getenv("MOE_PROBE")) : 0).b(23, std::getenv("QUANT_CTAB") != nullptr));
        K.dn = cx.pipeline(lib, kname("moe_down", dn == ref::Q2_0 && !g_dn_prep ? 142 : dn, b));
        K.cgu = a;
        K.cdn = b;
        if (std::getenv("PIPE_INFO"))
            std::printf("pipelines: gu %s max threads %d tg mem %d | dn %s max threads %d tg mem %d\n", K.gu.name.c_str(), K.gu.max_threads,
                        K.gu.static_tg_mem, K.dn.name.c_str(), K.dn.max_threads, K.dn.static_tg_mem);
        return K;
    };
    auto args_for = [&](const Kernels& K, int nt, int n_expert) {
        MoeArgs a{};
        a.gate_off = 0;
        a.up_off = L.up_off;
        a.down_off = L.down_off;
        a.gu_stride = L.blob;
        a.d_stride = L.blob;
        a.gu_row = (uint32_t)L.gu_row;
        a.d_row = (uint32_t)L.d_row;
        a.n_embd = N_EMBD;
        a.n_ff = N_FF;
        a.n_tok = nt;
        a.topk = TOPK;
        a.n_expert = n_expert;
        a.gx_gu = N_FF / rows_per_tg(K.cgu);
        a.gx_dn = N_EMBD / rows_per_tg(K.cdn);
        return a;
    };

    mtl::Buffer x = cx.alloc(4 * 8 * N_EMBD), h = cx.alloc(4 * 8 * TOPK * N_FF), ye = cx.alloc(4 * 8 * TOPK * N_EMBD),
                y = cx.alloc(4 * 8 * N_EMBD);
    mtl::Buffer XPX = cx.alloc(4 * (4 * PLANE_X + PLANE_X / 16)), XPH = cx.alloc(4 * (4 * PLANE_H + PLANE_H / 16));
    g_xpx = &XPX;
    g_xph = &XPH;
    g_gu_type = gu;
    g_dn_type = dn;
    ref::Rng rng(12345 + (seed - 1));
    for (int i = 0; i < 8 * N_EMBD; ++i) x.as<float>()[i] = rng.sym();
    if (g_had_h) {   // H (s * x) per token on the GPU, checked against the CPU's; x then holds the rotated rows
        std::vector<float> want(x.as<float>(), x.as<float>() + 8 * N_EMBD);
        for (int t = 0; t < 8; ++t) strata::kernels::cpu::hadamard_rotate(want.data() + t * N_EMBD, N_EMBD, 128, sgx);
        HadArgs ha{(uint32_t)N_EMBD, (uint32_t)N_EMBD, (uint32_t)N_EMBD, 1.0f / std::sqrt(128.0f), sgx ? 1u : 0u, 0u};
        mtl::Dispatch d;
        d.p = &had32;
        d.args = {mtl::Arg::B(x), mtl::Arg::B(x), mtl::Arg::B(sgx ? hsx : x), mtl::Arg::V(ha)};
        d.grid[0] = N_EMBD / 128; d.grid[1] = 8; d.tg[0] = 128;
        cx.run({d});
        int bad = 0;
        for (int i = 0; i < 8 * N_EMBD; ++i) bad += std::memcmp(&want[i], x.as<float>() + i, 4) != 0;
        std::printf("x rotation on the GPU vs the CPU: %d of %d values differ\n", bad, 8 * N_EMBD);
    }
    {
        mtl::Pipeline pp = cx.pipeline(lib, "xprep");
        cx.run({prep_dispatch(pp, x, XPX, N_EMBD, 8)});
    }

    if (!bench) {
        // ---------------- parity ----------------
        const int NE = tg_ && experts_first ? std::max(experts_first, 2 * TOPK) : 64;
        mtl::Buffer W = cx.alloc(L.blob * NE);
        std::vector<int> pick;
        for (int e = 0; e < NE; ++e) {
            uint8_t* b = W.as<uint8_t>() + e * L.blob;
            if (tg_) {   // real experts of the file: expert `ex` of the layer's three tensors
                int ex;
                if (experts_first) ex = e % experts_first;   // (a sparse copy holds only the first experts: reused when NE is larger)
                else do ex = (int)(rng.next() % 512); while (std::find(pick.begin(), pick.end(), ex) != pick.end());
                pick.push_back(ex);
                std::memcpy(b, (const uint8_t*)gf.map + tg_->offset + (size_t)ex * L.gu_row * N_FF, L.gu_row * N_FF);
                std::memcpy(b + L.up_off, (const uint8_t*)gf.map + tu_->offset + (size_t)ex * L.gu_row * N_FF, L.gu_row * N_FF);
                std::memcpy(b + L.down_off, (const uint8_t*)gf.map + td_->offset + (size_t)ex * L.d_row * N_EMBD, L.d_row * N_EMBD);
            } else {
                ref::random_blocks(gu, b, 2 * N_FF, N_EMBD, 1000 + e + 100 * (seed - 1));
                ref::random_blocks(dn, b + L.down_off, N_EMBD, N_FF, 5000 + e + 100 * (seed - 1));
            }
        }
        std::vector<int> ids;
        std::vector<float> w;
        random_routing(rng, ntok, NE, ids, w, 0.3);
        mtl::Buffer bid = cx.alloc(4 * ids.size()), bw = cx.alloc(4 * w.size()), plan = cx.alloc(4 * P_INTS);
        std::memcpy(bid.ptr, ids.data(), 4 * ids.size());
        std::memcpy(bw.ptr, w.data(), 4 * w.size());
        Kernels K = make(cgu, cdn);
        MoeArgs a = args_for(K, ntok, 512);
        std::vector<mtl::Dispatch> ds;
        encode_layer(ds, K, W, 0, x, h, ye, y, bid, bw, plan, 0, a, true, true, true, true);
        cx.run(ds);
        // the plan
        const int* P = plan.as<int>();
        std::set<int> distinct(ids.begin(), ids.end());
        bool plan_ok = P[P_NG] == (int)distinct.size() && P[P_IGU + 1] == P[P_NG] && P[P_IDN + 1] == P[P_NG];
        {
            int gi = 0;
            for (int e : distinct) {
                if (P[P_GEXP + gi] != e) plan_ok = false;
                int j = P[P_GSTART + gi];
                for (int i = 0; i < ntok * TOPK; ++i)
                    if (ids[i] == e) {
                        if (P[P_ETOK + j] != i / TOPK || P[P_TOKENT + i] != j) plan_ok = false;
                        float fw;
                        std::memcpy(&fw, &P[P_EW + j], 4);
                        if (fw != w[i]) plan_ok = false;
                        ++j;
                    }
                if (j != P[P_GSTART + gi + 1]) plan_ok = false;
                ++gi;
            }
        }
        // the reference
        Err eh, ey;
        std::vector<float> wg(N_FF * N_EMBD), wu(N_FF * N_EMBD), wd(N_EMBD * N_FF);
        std::vector<double> yref(ntok * N_EMBD, 0.0);
        for (int i = 0; i < ntok * TOPK; ++i) {
            const int e = ids[i], t = i / TOPK;
            const uint8_t* b = W.as<uint8_t>() + e * L.blob;
            ref::dequant(gu, b, wg.data(), wg.size());
            ref::dequant(gu, b + L.up_off, wu.data(), wu.size());
            ref::dequant(dn, b + L.down_off, wd.data(), wd.size());
            const float* xt = x.as<float>() + t * N_EMBD;
            std::vector<double> hr(N_FF);
            const int ent = P[P_TOKENT + i];
            for (int r = 0; r < N_FF; ++r) {
                double g = 0, u = 0;
                for (int k = 0; k < N_EMBD; ++k) {
                    g += (double)wg[r * N_EMBD + k] * xt[k];
                    u += (double)wu[r * N_EMBD + k] * xt[k];
                }
                hr[r] = g / (1.0 + std::exp(-g)) * u;
            }
            if (g_had_h) {   // the reference of the down input: H h, in float as the engine's CPU path does
                std::vector<float> hf(N_FF);
                for (int r = 0; r < N_FF; ++r) hf[r] = (float)hr[r];
                strata::kernels::cpu::hadamard_rotate(hf.data(), N_FF, 128, sgh);
                for (int r = 0; r < N_FF; ++r) hr[r] = hf[r];
            }
            for (int r = 0; r < N_FF; ++r) eh.add(h.as<float>()[ent * N_FF + r], hr[r]);
            for (int r = 0; r < N_EMBD; ++r) {
                double s = 0;
                // the reference down uses the GPU's h (float) so that the y check isolates the down kernel ...
                for (int k = 0; k < N_FF; ++k) s += (double)wd[r * N_FF + k] * h.as<float>()[ent * N_FF + k];
                yref[t * N_EMBD + r] += w[i] * s;
            }
        }
        for (int i = 0; i < ntok * N_EMBD; ++i) ey.add(y.as<float>()[i], yref[i]);
        const bool ok = plan_ok && eh.rel_l2() < 1e-5 && ey.rel_l2() < 1e-5 && eh.rel_max() < 1e-4 && ey.rel_max() < 1e-4;
        std::printf("parity %s/%s ntok %d gu %d,%d,%d dn %d,%d,%d: plan %s (%d groups), h rel L2 %.2e max %.2e, "
                    "y rel L2 %.2e max %.2e  %s\n",
                    ref::type_name(gu), ref::type_name(dn), ntok, cgu.L, cgu.R, cgu.S, cdn.L, cdn.R, cdn.S,
                    plan_ok ? "ok" : "WRONG", P[P_NG], eh.rel_l2(), eh.rel_max(), ey.rel_l2(), ey.rel_max(),
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    // ---------------- bench ----------------
    const int NE = 512;
    const size_t layer_bytes = L.blob * NE;
    std::printf("arena: %d layers x %d experts = %.2f GB (max buffer %.1f GB)\n", layers, NE, layers * layer_bytes / 1e9,
                cx.max_buffer_length() / 1e9);
    mtl::Buffer W = cx.alloc(layer_bytes * layers);
    {
        // 64 distinct random experts, copied over the arena
        std::vector<uint8_t> tmp(L.blob * 64);
        for (int e = 0; e < 64; ++e) {
            ref::random_blocks(gu, tmp.data() + e * L.blob, 2 * N_FF, N_EMBD, 1000 + e);
            ref::random_blocks(dn, tmp.data() + e * L.blob + L.down_off, N_EMBD, N_FF, 5000 + e);
        }
        for (size_t e = 0; e < (size_t)NE * layers; ++e)
            std::memcpy(W.as<uint8_t>() + e * L.blob, tmp.data() + (e % 64) * L.blob, L.blob);
    }
    std::vector<std::vector<int>> rids(reps);
    std::vector<std::vector<float>> rw(reps);
    mtl::Buffer bid = cx.alloc(4 * 8 * TOPK * reps), bw = cx.alloc(4 * 8 * TOPK * reps), plan = cx.alloc(4 * P_INTS * reps);
    auto set_routing = [&](int nt) {
        double groups = 0;
        for (int r = 0; r < reps; ++r) {
            random_routing(rng, nt, NE, rids[r], rw[r], share);
            std::memcpy(bid.as<int>() + r * 8 * TOPK, rids[r].data(), 4 * rids[r].size());
            std::memcpy(bw.as<float>() + r * 8 * TOPK, rw[r].data(), 4 * rw[r].size());
            groups += std::set<int>(rids[r].begin(), rids[r].end()).size();
        }
        return groups / reps;
    };
    auto run_cfg = [&](const Kernels& K, int nt, double groups, bool print) {
        const MoeArgs a = args_for(K, nt, NE);
        std::vector<mtl::Dispatch> dp, dg, dd, dall;
        for (int r = 0; r < reps; ++r) {
            const size_t woff = (size_t)(r % layers) * layer_bytes;
            std::vector<mtl::Dispatch> one;
            encode_layer(one, K, W, woff, x, h, ye, y, bid, bw, plan, (size_t)r * P_INTS * 4, a, true, false, false, false);
            one[0].args[0] = mtl::Arg::B(bid, (size_t)r * 8 * TOPK * 4);
            one[0].args[1] = mtl::Arg::B(bw, (size_t)r * 8 * TOPK * 4);
            dp.push_back(one[0]);
            std::vector<mtl::Dispatch> g, d;
            encode_layer(g, K, W, woff, x, h, ye, y, bid, bw, plan, (size_t)r * P_INTS * 4, a, false, true, false, false);
            encode_layer(d, K, W, woff, x, h, ye, y, bid, bw, plan, (size_t)r * P_INTS * 4, a, false, false, true, false);
            dg.push_back(g[0]);
            for (auto& z : d) dd.push_back(z);
            dall.push_back(one[0]);
            dall.push_back(g[0]);
            for (auto& z : d) dall.push_back(z);
        }
        cx.run(dp);   // plans in place
        cx.warm(dall, 0.1);
        std::vector<double> tp, tg, td, ta;
        for (int it = 0; it < 7; ++it) {
            tp.push_back(cx.run(dp) / reps);
            tg.push_back(cx.run(dg) / reps);
            td.push_back(cx.run(dd) / reps);
            ta.push_back(cx.run(dall) / reps);
        }
        const double mp = median(tp), mg = median(tg), md = median(td), ma = median(ta);
        const double bg = groups * 2.0 * N_FF * L.gu_row, bd = groups * N_EMBD * L.d_row;
        if (print)
            std::printf("ntok %d groups %5.1f | gu %2d,%d,%d %7.1f us %6.1f GB/s | dn %2d,%d,%d %7.1f us %6.1f GB/s | "
                        "plan %5.1f us | layer %7.1f us = %6.1f GB/s (%4.1f%% of 585)\n",
                        nt, groups, K.cgu.L, K.cgu.R, K.cgu.S, mg * 1e6, bg / mg / 1e9, K.cdn.L, K.cdn.R, K.cdn.S,
                        md * 1e6, bd / md / 1e9, mp * 1e6, ma * 1e6, (bg + bd) / ma / 1e9,
                        (bg + bd) / ma / 1e9 / 585 * 100);
        return std::make_pair(mg, md);
    };
    const std::vector<Cfg> gus = {{32, 1, 4}, {32, 2, 4}, {32, 4, 4}, {32, 2, 2}, {32, 2, 8}, {32, 1, 8},
                                  {16, 1, 4}, {16, 2, 4}, {8, 1, 4},  {8, 2, 4},  {8, 1, 2}};
    const std::vector<Cfg> dns = {{8, 1, 4}, {8, 2, 4}, {8, 4, 4}, {8, 1, 8},  {8, 2, 8}, {4, 1, 4},
                                  {4, 2, 4}, {4, 4, 4}, {16, 1, 4}, {16, 2, 4}, {32, 1, 4}, {2, 2, 4}};
    std::vector<int> ntoks = {ntok};
    if (sweep) ntoks = {1, 2, 3, 4, 8};
    for (int nt : ntoks) {
        const double groups = set_routing(nt);
        if (sweep) {
            // gate/up sweep with the default down, then down sweep with the best gate/up
            Cfg bgu = cgu, bdn = cdn;
            double best = 1e9;
            for (Cfg c : gus) {
                if (N_FF % rows_per_tg(c)) continue;
                auto t = run_cfg(make(c, cdn), nt, groups, false);
                std::printf("  gu %2d,%d,%d: %7.1f us  %6.1f GB/s\n", c.L, c.R, c.S, t.first * 1e6,
                            groups * 2.0 * N_FF * L.gu_row / t.first / 1e9);
                if (t.first < best) { best = t.first; bgu = c; }
            }
            best = 1e9;
            for (Cfg c : dns) {
                if (N_EMBD % rows_per_tg(c)) continue;
                auto t = run_cfg(make(bgu, c), nt, groups, false);
                std::printf("  dn %2d,%d,%d: %7.1f us  %6.1f GB/s\n", c.L, c.R, c.S, t.second * 1e6,
                            groups * N_EMBD * L.d_row / t.second / 1e9);
                if (t.second < best) { best = t.second; bdn = c; }
            }
            run_cfg(make(bgu, bdn), nt, groups, true);
        } else {
            run_cfg(make(cgu, cdn), nt, groups, true);
        }
    }
    return 0;
}
