// src/kernels/cpu/native_hadamard_test.cpp - APR port, stage 3: one Hadamard-folded trellis expert (TQ2_T / TQK6 /
// TQK7, ggml types 144-146) on Strata's CPU path against llama.cpp's own graph for the same weights and input.
//
// Strata's path:  native_quant_act (signs, FWHT, Q8_0) -> native_gu_rows (ggml-cpu vec_dot + SwiGLU)
//                 -> native_quant_h (signs, FWHT, Q8_0) -> native_down_rows
// The reference:  the ops llama.cpp's build_lora_mm_id builds for a prism.hadamard weight - ggml_mul(x, signs),
//                 llama_mul_mat_hadamard (reshape to [128, n/128], ggml_mul_mat(rot, .)), ggml_mul_mat(W', .) -
//                 computed by ggml-cpu from the same checkout (STRATA_GGML_DIR = the agention llama.cpp fork).
// The rotation itself is computed two different ways (FWHT here, a dense 128x128 matmul there), so the two outputs
// agree to float rounding plus the odd Q8_0 rounding flip, not bit for bit; the test bounds the relative error.
//
//   native_hadamard_test                         synthetic: random W, rotated and quantized here; also checks the
//                                                folded expert against the UNROTATED float expert (W x) and that
//                                                skipping the rotation is caught
//   native_hadamard_test --pack <dir> --gguf <file> [layer expert]...   real experts of a prism.hadamard GGUF
//                                                (Gyro-S / Gyro-M) through the pack tools/iq_pack.py made of it
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels::cpu;

namespace {

constexpr int H = 2560, FF = 640, B = 128;

struct Expert {
    int gu_type = 0, d_type = 0;
    std::vector<uint8_t> gate, up, down;   // ggml rows: gate/up [FF rows of H], down [H rows of FF]
    std::vector<float> s_h, s_ff;          // sign vectors (empty: identity)
};

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1));
}

float silu(float g) { return g / (1.f + std::exp(-g)); }

// Strata's CPU path for one token; `loaded` = the layer's NativeFmt as expert_layout_load made it from a pack
std::vector<float> strata_expert(const Expert& e, const std::vector<float>& x, bool rotate,
                                 const NativeFmt* loaded = nullptr) {
    NativeFmt f;
    std::string err;
    if (!native_fmt(e.gu_type, e.d_type, H, FF, f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); std::exit(1); }
    if (rotate && loaded) {
        f = *loaded;
    } else if (rotate) {
        f.had_block = B;
        f.had_gu = f.had_d = true;
        f.gu_signs = e.s_h.empty() ? nullptr : e.s_h.data();
        f.d_signs = e.s_ff.empty() ? nullptr : e.s_ff.data();
    }
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data(), e.gate.data(), e.gate.size());
    std::memcpy(blob.data() + f.up_off, e.up.data(), e.up.size());
    std::memcpy(blob.data() + f.down_off, e.down.data(), e.down.size());
    std::vector<uint8_t> act(kNativeActBytes), hq(kNativeHBytes);
    std::vector<float> ff(FF), out(H);
    native_quant_act(f, x.data(), act.data());
    const void* a[1] = {act.data()};
    float* ffp[1] = {ff.data()};
    native_gu_rows(f, blob.data(), a, 1, ffp, 0, FF);
    native_quant_h(f, ff.data(), hq.data());
    const void* hp[1] = {hq.data()};
    float* op[1] = {out.data()};
    native_down_rows(f, blob.data(), hp, 1, op, 0, H);
    return out;
}

// llama.cpp's graph for the same expert (the fork's build_lora_mm + llama_mul_mat_hadamard), one token
std::vector<float> llama_expert(const Expert& e, const std::vector<float>& x) {
    ggml_init_params ip = {(size_t) 256 << 20, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    // the normalized Sylvester-Walsh-Hadamard matrix exactly as llama_model_base builds prism.hadamard.128
    ggml_tensor* rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, B, B);
    float* R = (float*) rot->data;
    const float scale = 1.0f / sqrtf((float) B);
    for (uint32_t r = 0; r < (uint32_t) B; ++r)
        for (uint32_t c = 0; c < (uint32_t) B; ++c) {
            uint32_t p = r & c;
            p ^= p >> 16; p ^= p >> 8; p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
            R[r * B + c] = (p & 1) ? -scale : scale;
        }
    auto hadamard = [&](ggml_tensor* cur, const std::vector<float>& signs) {
        if (!signs.empty()) {
            ggml_tensor* s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) signs.size());
            std::memcpy(s->data, signs.data(), signs.size() * sizeof(float));
            cur = ggml_mul(ctx, cur, s);
        }
        ggml_tensor* res = ggml_reshape_2d(ctx, cur, B, ggml_nelements(cur) / B);
        res = ggml_mul_mat(ctx, rot, res);
        return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    };
    ggml_tensor* wg = ggml_new_tensor_2d(ctx, (ggml_type) e.gu_type, H, FF);
    ggml_tensor* wu = ggml_new_tensor_2d(ctx, (ggml_type) e.gu_type, H, FF);
    ggml_tensor* wd = ggml_new_tensor_2d(ctx, (ggml_type) e.d_type, FF, H);
    std::memcpy(wg->data, e.gate.data(), e.gate.size());
    std::memcpy(wu->data, e.up.data(), e.up.size());
    std::memcpy(wd->data, e.down.data(), e.down.size());
    ggml_tensor* xt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    std::memcpy(xt->data, x.data(), H * sizeof(float));
    ggml_tensor* xr = hadamard(xt, e.s_h);
    ggml_tensor* g = ggml_mul_mat(ctx, wg, xr);
    ggml_tensor* u = ggml_mul_mat(ctx, wu, xr);
    ggml_tensor* h = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ggml_tensor* out = ggml_mul_mat(ctx, wd, hadamard(h, e.s_ff));
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    std::vector<float> y(H);
    std::memcpy(y.data(), out->data, H * sizeof(float));
    ggml_free(ctx);
    return y;
}

// W' = W diag(s) H^T per 128-column block (what the APR encoder folds), then quantized with ggml
std::vector<uint8_t> fold_and_quantize(const std::vector<float>& w, int rows, int cols, const std::vector<float>& s,
                                       int type, std::vector<float>& folded) {
    folded.assign(w.begin(), w.end());
    for (int r = 0; r < rows; ++r) hadamard_rotate(folded.data() + (size_t) r * cols, cols, B, s.data());
    // (H symmetric and orthonormal: row-wise H (s * w_r) is exactly the fold W diag(s) H^T)
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, cols) * rows);
    ggml_quantize_chunk((ggml_type) type, folded.data(), q.data(), 0, rows, cols, nullptr);
    return q;
}

int synthetic(int gu_type, int d_type) {
    std::mt19937 rng(1234 + gu_type * 7 + d_type);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> coin(0, 1);
    Expert e;
    e.gu_type = gu_type;
    e.d_type = d_type;
    std::vector<float> wg((size_t) FF * H), wu((size_t) FF * H), wd((size_t) H * FF), x(H);
    for (auto& v : wg) v = nd(rng) * 0.02f;
    for (auto& v : wu) v = nd(rng) * 0.02f;
    for (auto& v : wd) v = nd(rng) * 0.02f;
    for (auto& v : x) v = nd(rng);
    x[17] = 40.f;   // a massive channel, as the real residual stream has: the case the rotation exists for
    e.s_h.resize(H);
    e.s_ff.resize(FF);
    for (auto& v : e.s_h) v = coin(rng) ? 1.f : -1.f;
    for (auto& v : e.s_ff) v = coin(rng) ? 1.f : -1.f;
    std::vector<float> fg, fu, fd;
    e.gate = fold_and_quantize(wg, FF, H, e.s_h, gu_type, fg);
    e.up = fold_and_quantize(wu, FF, H, e.s_h, gu_type, fu);
    e.down = fold_and_quantize(wd, H, FF, e.s_ff, d_type, fd);

    // the unrotated float expert W x: what the folded, quantized expert must approximate
    std::vector<float> h(FF), ref(H);
    for (int r = 0; r < FF; ++r) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) { g += (double) wg[(size_t) r * H + c] * x[c]; u += (double) wu[(size_t) r * H + c] * x[c]; }
        h[r] = silu((float) g) * (float) u;
    }
    for (int r = 0; r < H; ++r) {
        double o = 0;
        for (int c = 0; c < FF; ++c) o += (double) wd[(size_t) r * FF + c] * h[c];
        ref[r] = (float) o;
    }
    const auto ours = strata_expert(e, x, true);
    const auto llama = llama_expert(e, x);
    const auto unrot = strata_expert(e, x, false);
    const double e_ll = rel(ours, llama), e_ref = rel(ours, ref), e_llref = rel(llama, ref), e_un = rel(unrot, ref);
    std::printf("synthetic %s/%s: strata vs llama.cpp graph %.2e | vs unrotated float expert: strata %.4f, llama.cpp "
                "%.4f | strata WITHOUT the rotation %.4f\n", ggml_type_name((ggml_type) gu_type),
                ggml_type_name((ggml_type) d_type), e_ll, e_ref, e_llref, e_un);
    // strata == llama.cpp to rounding, and both approximate W x equally well.  With Q8_0 weights the quantization
    // error is ~1%, so the fold's convention itself is checked: the rotated path must reproduce the unrotated float
    // expert and the unrotated path must not.  (The trellis types' own error at 1.6-2.1 bpw on Gaussian weights,
    // compounded through SwiGLU, is large with this test's placeholder ggml encoder; there only agreement counts.)
    const bool hi = gu_type == 8 && d_type == 8;
    const bool ok = e_ll < 2e-3 && std::fabs(e_ref - e_llref) < 0.01 && (!hi || (e_ref < 0.03 && e_un > 0.8));
    std::printf("  %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}

// A real expert, the pack's loader path: expert_layout_load(<pack>) reads native_experts.txt v5 + hadamard.txt (from
// tools/iq_pack.py), and the layer's NativeFmt is what the engine's CPU pool would use.  (The signs come from the
// pack: strata::GgufFile keeps only the first 64 items of a metadata array, so it cannot read sign_values.)
int real(const char* pack, const char* path, int layer, int expert) {
    std::string err;
    if (!expert_layout_load(pack, 48, 512, err)) { std::fprintf(stderr, "expert_layout_load: %s\n", err.c_str()); return 1; }
    const ExpertLayout& lay = expert_layout();
    const HadamardSpec& hs = hadamard_spec();
    if (!lay.native || !hs.any() || lay.version != 5) { std::fprintf(stderr, "%s: not a v5 (Hadamard) native pack\n", pack); return 1; }
    const NativeFmt& lf = lay.fmt[(size_t) layer];
    if (lf.had_block != B || !lf.had_gu || !lf.had_d) { std::fprintf(stderr, "layer %d is not folded in the pack\n", layer); return 1; }
    strata::GgufFile g(path);
    Expert e;
    if (const float* sh = hadamard_signs(H)) e.s_h.assign(sh, sh + H);
    if (const float* sf = hadamard_signs(FF)) e.s_ff.assign(sf, sf + FF);
    auto slice = [&](const char* role, std::vector<uint8_t>& dst, int& type) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
        const strata::TensorInfo* t = g.find(name);
        if (!t || t->shape.size() != 3) { std::fprintf(stderr, "missing %s\n", name.c_str()); std::exit(1); }
        type = (int) t->type;
        const size_t per = ggml_row_size((ggml_type) type, (int64_t) t->shape[0]) * t->shape[1];
        const uint8_t* p = g.tensor_data(*t) + per * (size_t) expert;
        dst.assign(p, p + per);
    };
    int dt = 0, ut = 0;
    slice("gate", e.gate, e.gu_type);
    slice("up", e.up, ut);
    slice("down", e.down, dt);
    e.d_type = dt;
    if (e.gu_type != lf.gu_type || e.d_type != lf.d_type) { std::fprintf(stderr, "the pack and the GGUF disagree\n"); return 1; }
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(H);
    for (auto& v : x) v = nd(rng);
    x[17] = 40.f;
    const auto ours = strata_expert(e, x, true, &lf);
    const auto llama = llama_expert(e, x);
    const auto unrot = strata_expert(e, x, false);
    const double e_ll = rel(ours, llama), e_un = rel(unrot, llama);
    std::printf("%s layer %d expert %d (%s/%s, signs %s): strata vs llama.cpp graph %.2e | without the rotation %.3f\n",
                path, layer, expert, ggml_type_name((ggml_type) e.gu_type), ggml_type_name((ggml_type) e.d_type),
                e.s_h.empty() ? "identity" : "explicit", e_ll, e_un);
    bool ok = e_ll < 2e-3 && e_un > 0.3;
    std::printf("  %s\n", ok ? "ok" : "FAILED");

    // the engine's CPU entry point: ExpertPool::run_split_multi_native with the pack's NativeFmt, three tokens
    // routed to this expert (x quantized by native_quant_act, as expert_source.cpp does for the doorbell copy)
    {
        constexpr int NT = 3;
        ExpertPool pool(4, false, false);
        std::vector<uint8_t> blob(lf.bytes);
        std::memcpy(blob.data(), e.gate.data(), e.gate.size());
        std::memcpy(blob.data() + lf.up_off, e.up.data(), e.up.size());
        std::memcpy(blob.data() + lf.down_off, e.down.data(), e.down.size());
        std::vector<std::vector<float>> xs(NT, std::vector<float>(H)), outs(NT, std::vector<float>(H));
        std::vector<std::vector<uint8_t>> acts(NT, std::vector<uint8_t>(kNativeActBytes));
        ExpertJobMulti job;
        job.blob = blob.data();
        job.nt = NT;
        for (int t = 0; t < NT; ++t) {
            for (auto& v : xs[t]) v = nd(rng);
            xs[t][(size_t) (101 * t + 5)] = 30.f;
            native_quant_act(lf, xs[t].data(), acts[t].data());
            job.nact[t] = acts[t].data();
            job.out[t] = outs[t].data();
        }
        pool.run_split_multi_native(lf, &job, 1);
        double worst = 0;
        for (int t = 0; t < NT; ++t) worst = std::max(worst, rel(outs[t], llama_expert(e, xs[t])));
        const bool pok = worst < 2e-3;
        std::printf("  ExpertPool::run_split_multi_native, %d tokens: worst vs llama.cpp graph %.2e  %s\n", NT, worst,
                    pok ? "ok" : "FAILED");
        ok = ok && pok;
    }
    return ok ? 0 : 1;
}

// the FWHT against the dense matrix llama.cpp builds, on one block
int fwht_check() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(B), y(B), s(B);
    for (auto& v : x) v = nd(rng);
    for (int i = 0; i < B; ++i) s[i] = (i * 37 % 5) < 2 ? -1.f : 1.f;
    y = x;
    hadamard_rotate(y.data(), B, B, s.data());
    double worst = 0;
    for (int r = 0; r < B; ++r) {
        double acc = 0;
        for (int c = 0; c < B; ++c) acc += ((__builtin_popcount(r & c) & 1) ? -1.0 : 1.0) * s[c] * x[c];
        worst = std::max(worst, std::fabs(acc / std::sqrt((double) B) - y[r]));
    }
    std::printf("FWHT vs dense normalized Sylvester matrix: max abs diff %.2e  %s\n", worst, worst < 1e-5 ? "ok" : "FAILED");
    return worst < 1e-5 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (!native_experts_available()) { std::fprintf(stderr, "built without native experts\n"); return 1; }
    int rc = fwht_check();
    if (argc >= 5 && std::strcmp(argv[1], "--pack") == 0 && std::strcmp(argv[3], "--gguf") == 0) {
        for (int i = 5; i + 1 < argc || i == 5; i += 2) {   // [layer expert]...
            const int layer = argc > i ? std::atoi(argv[i]) : 0, expert = argc > i + 1 ? std::atoi(argv[i + 1]) : 0;
            rc |= real(argv[2], argv[4], layer, expert);
        }
        return rc;
    }
    // ggml type ids of the agention fork: 144 TQ2_T, 145 TQK6, 146 TQK7
    if (GGML_TYPE_COUNT <= 146) { std::fprintf(stderr, "this ggml has no trellis types (build with STRATA_GGML_DIR = the agention fork): skipped\n"); return 77; }   // 77: ctest SKIP_RETURN_CODE (#3358)
    rc |= synthetic(8, 8);       // Q8_0: the fold's convention, against the unrotated float expert
    rc |= synthetic(145, 146);   // Gyro-S
    rc |= synthetic(144, 144);   // Gyro-M
    return rc;
}
