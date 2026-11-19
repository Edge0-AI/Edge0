// metal/tests/latent_routed_parity.cpp - the latent routed FFN of the decode windows (latent_routed.metal) against a
// double-precision CPU reference.
//
//   latent_routed_parity [--experts q3k|fp4] [--du q4|q8] [--latent1792] [--ntok 1..8] [--seed S]
//
// q3k: column-scaled Q3_K gate/up with IQ4_NL down.  fp4: NVFP4 for all three, no column scale, {s_gate, s_up,
// s_down, 0} (float32) after an expert's down rows.  D/U is Q4_0 or Q8_0; the latent width is 1280 or 1792.
//
// Synthetic: random valid blocks for 64 experts and for U, a random group per expert (the last group has two experts,
// so some tokens leave it unused), random projected rows x ~ U(-1,1) (the D projection's output: gemv_parity covers
// D itself), random distinct top-10 per token with weights summing to 1.  Every stage is checked against double, and
// each reference takes the GPU's output of the stage before it, so a stage's numbers are that kernel's own:
// the column scale, h = silu(gate) * up, down, the rows pooled by group, U per group, the sum with the shared expert,
// and U per entry (the ungrouped path).  The engine's three lane tilings run one after the other: the default
// window, the one-token window and the verify window.  Reported as relative L2 and max |err| / max |ref|.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {
constexpr int HIDDEN = 2560, N_FF = 640, TOPK = 10, GROUPS = 8, NE = 64;
struct LatentRoutedArgs {
    uint32_t n_tok, groups, hidden, latent, ff, topk, plan_token_entry;
    uint32_t u_row, gu_row, down_row, up_off, down_off, expert_bytes, tail_off;
};
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
    bool ok() const { return rel_l2() < 1e-5 && rel_max() < 1e-4; }
};
std::string lanes(const std::string& kernel, int l) { return l == 32 ? kernel : kernel + "_" + std::to_string(l); }
}  // namespace

int main(int argc, char** argv) {
    bool nv = false, du8 = false;
    int latent = 1280, ntok = 4, seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--experts" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v != "q3k" && v != "fp4") { std::fprintf(stderr, "--experts q3k|fp4\n"); return 2; }
            nv = v == "fp4";
        } else if (a == "--du" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v != "q4" && v != "q8") { std::fprintf(stderr, "--du q4|q8\n"); return 2; }
            du8 = v == "q8";
        } else if (a == "--latent1792") latent = 1792;
        else if (a == "--ntok" && i + 1 < argc) ntok = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (ntok < 1 || ntok > 8) { std::fprintf(stderr, "--ntok 1..8\n"); return 2; }
    const int gu = nv ? ref::NVFP4 : ref::Q3_K, dn = nv ? ref::NVFP4 : ref::IQ4_NL, du = du8 ? ref::Q8_0 : ref::Q4_0;
    const size_t gu_row = ref::row_bytes(gu, latent), down_row = ref::row_bytes(dn, N_FF), u_row = ref::row_bytes(du, latent);
    const size_t up_off = gu_row * N_FF, down_off = 2 * up_off, tail_off = down_off + down_row * latent;
    const size_t blob = tail_off + (nv ? 4 * sizeof(float) : 0);
    const int entries = ntok * TOPK, slots = ntok * GROUPS;

    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "latent_routed.metal"});
    std::printf("device %s; %s experts, %s D/U, latent %d, %d tokens\n", cx.device_name().c_str(),
                nv ? "NVFP4" : "Q3_K/IQ4_NL", ref::type_name(du), latent, ntok);

    mtl::Buffer W = cx.alloc(blob * NE), U = cx.alloc(u_row * GROUPS * HIDDEN), labels = cx.alloc(4 * NE),
                inverse = cx.alloc(4 * latent), ids = cx.alloc(4 * entries), wts = cx.alloc(4 * entries),
                plan = cx.alloc(4 * entries), projected = cx.alloc(4 * (size_t)slots * latent),
                middle = cx.alloc(4 * (size_t)entries * N_FF), down = cx.alloc(4 * (size_t)entries * latent),
                active = cx.alloc(4 * slots), grouped = cx.alloc(4 * (size_t)slots * HIDDEN),
                per_entry = cx.alloc(4 * (size_t)entries * HIDDEN), shared = cx.alloc(4 * ntok * HIDDEN),
                shared_gate = cx.alloc(4 * ntok), out = cx.alloc(4 * ntok * HIDDEN);
    ref::Rng r(977 + seed);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = W.as<uint8_t>() + e * blob;
        ref::random_blocks(gu, b, 2 * N_FF, latent, 100 * seed + e);
        ref::random_blocks(dn, b + down_off, latent, N_FF, 100 * seed + NE + e);
        if (nv) {
            const float tail[4] = {0.5f + r.uni(), 0.5f + r.uni(), 0.5f + r.uni(), 0.0f};
            std::memcpy(b + tail_off, tail, sizeof(tail));
        }
        labels.as<int>()[e] = e < 2 ? GROUPS - 1 : (int)(r.next() % (GROUPS - 1));
    }
    ref::random_blocks(du, U.as<uint8_t>(), GROUPS * HIDDEN, latent, 100 * seed + 3 * NE);
    for (int k = 0; k < latent; ++k) inverse.as<float>()[k] = 0.5f + r.uni();
    std::vector<float> x0((size_t)slots * latent);
    for (auto& v : x0) v = r.sym();
    for (int t = 0; t < ntok; ++t) {
        std::set<int> used;
        float sum = 0;
        for (int k = 0; k < TOPK; ++k) {
            int e;
            do e = (int)(r.next() % NE); while (used.count(e));
            used.insert(e);
            ids.as<int>()[t * TOPK + k] = e;
            sum += (wts.as<float>()[t * TOPK + k] = 0.2f + r.uni());
        }
        for (int k = 0; k < TOPK; ++k) wts.as<float>()[t * TOPK + k] /= sum;
        shared_gate.as<float>()[t] = r.sym() * 2.0f;
    }
    for (int i = 0; i < entries; ++i) plan.as<int>()[i] = i;   // the ungrouped U writes entry i to slot i
    for (int i = 0; i < ntok * HIDDEN; ++i) shared.as<float>()[i] = r.sym();

    LatentRoutedArgs args{(uint32_t)ntok, GROUPS, HIDDEN, (uint32_t)latent, N_FF, TOPK, 0, (uint32_t)u_row, (uint32_t)gu_row,
                          (uint32_t)down_row, (uint32_t)up_off, (uint32_t)down_off, (uint32_t)blob, (uint32_t)(nv ? tail_off : 0)};
    auto run = [&](const std::string& name, std::vector<mtl::Arg> buffers, uint32_t grid, uint32_t rows, uint32_t threads = 128) {
        mtl::Pipeline p = cx.pipeline(lib, name);
        mtl::Dispatch d;
        d.p = &p;
        d.args = std::move(buffers);
        d.args.push_back(mtl::Arg::V(args));
        d.grid[0] = grid;
        d.grid[1] = rows;
        d.tg[0] = threads;
        cx.run({d});
    };

    // the weights the tokens use, as the reference dequantizers give them
    std::vector<std::vector<float>> wg(NE), wu(NE), wd(NE);
    std::vector<float> uw((size_t)GROUPS * HIDDEN * latent);
    ref::dequant(du, U.as<uint8_t>(), uw.data(), uw.size());
    for (int i = 0; i < entries; ++i) {
        const int e = ids.as<int>()[i];
        if (!wg[e].empty()) continue;
        const uint8_t* b = W.as<uint8_t>() + e * blob;
        wg[e].resize((size_t)N_FF * latent); wu[e].resize(wg[e].size()); wd[e].resize((size_t)latent * N_FF);
        ref::dequant(gu, b, wg[e].data(), wg[e].size());
        ref::dequant(gu, b + up_off, wu[e].data(), wu[e].size());
        ref::dequant(dn, b + down_off, wd[e].data(), wd[e].size());
    }
    auto tail_of = [&](int e, int k) {
        float v = 1.0f;
        if (nv) std::memcpy(&v, W.as<uint8_t>() + e * blob + tail_off + 4 * k, 4);
        return (double)v;
    };

    struct Mode { const char* name; int gu, dn, u; };
    const Mode modes[3] = {{"default", 32, 32, 32}, {"one token", 8, 4, 8}, {"verify", nv ? 8 : 4, 4, 4}};
    bool all_ok = true;
    for (const Mode& m : modes) {
        // ---- the column scale (q3k), on a fresh copy of x
        std::memcpy(projected.ptr, x0.data(), x0.size() * 4);
        bool scale_ok = true;
        if (!nv) {
            run("latent_colscale", {mtl::Arg::B(projected), mtl::Arg::B(inverse)}, ((uint32_t)slots * latent + 127) / 128, 1);
            for (size_t i = 0; i < x0.size(); ++i) {
                const float want = x0[i] * inverse.as<float>()[i % latent];
                scale_ok = scale_ok && std::memcmp(&want, projected.as<float>() + i, 4) == 0;
            }
        }
        const std::vector<float> x(projected.as<float>(), projected.as<float>() + x0.size());
        // ---- gate/up and the SwiGLU
        run(lanes(nv ? "latent_gate_up_nv" : "latent_gate_up", m.gu),
            {mtl::Arg::B(W), mtl::Arg::B(labels), mtl::Arg::B(ids), mtl::Arg::B(projected), mtl::Arg::B(middle)},
            N_FF * m.gu / 128, entries);
        Err eh, ed, ep, eu, ey, ee;
        for (int i = 0; i < entries; ++i) {
            const int e = ids.as<int>()[i], t = i / TOPK, g = labels.as<int>()[e];
            const float* xt = x.data() + ((size_t)t * GROUPS + g) * latent;
            for (int f = 0; f < N_FF; ++f) {
                double gate = 0, up = 0;
                for (int k = 0; k < latent; ++k) {
                    gate += (double)wg[e][(size_t)f * latent + k] * xt[k];
                    up += (double)wu[e][(size_t)f * latent + k] * xt[k];
                }
                gate *= tail_of(e, 0);
                up *= tail_of(e, 1);
                eh.add(middle.as<float>()[(size_t)i * N_FF + f], gate / (1.0 + std::exp(-gate)) * up);
            }
        }
        // ---- down, from the GPU's h
        run(lanes(nv ? "latent_down_nv" : "latent_down", m.dn), {mtl::Arg::B(W), mtl::Arg::B(ids), mtl::Arg::B(middle), mtl::Arg::B(down)},
            (uint32_t)latent * m.dn / 128, entries);
        for (int i = 0; i < entries; ++i) {
            const int e = ids.as<int>()[i];
            const float* h = middle.as<float>() + (size_t)i * N_FF;
            for (int k = 0; k < latent; ++k) {
                double s = 0;
                for (int f = 0; f < N_FF; ++f) s += (double)wd[e][(size_t)k * N_FF + f] * h[f];
                ed.add(down.as<float>()[(size_t)i * latent + k], s * tail_of(e, 2));
            }
        }
        // ---- the rows pooled by group (into the projected rows' buffer, as the engine does), from the GPU's down
        run("latent_group_reduce", {mtl::Arg::B(labels), mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::B(down), mtl::Arg::B(projected),
                                    mtl::Arg::B(active)}, ((uint32_t)slots * latent + 127) / 128, 1);
        bool active_ok = true;
        int unused = 0;
        for (int s = 0; s < slots; ++s) {
            const int t = s / GROUPS, g = s % GROUPS;
            std::vector<double> pooled(latent, 0.0);
            bool used = false;
            for (int k = 0; k < TOPK; ++k) {
                const int i = t * TOPK + k;
                if (labels.as<int>()[ids.as<int>()[i]] != g) continue;
                used = true;
                for (int c = 0; c < latent; ++c) pooled[c] += (double)wts.as<float>()[i] * down.as<float>()[(size_t)i * latent + c];
            }
            unused += !used;
            active_ok = active_ok && (active.as<uint32_t>()[s] != 0) == used;
            for (int c = 0; c < latent; ++c) ep.add(projected.as<float>()[(size_t)s * latent + c], pooled[c]);
        }
        // ---- U per group, from the GPU's pooled rows; an unused group gives zeros
        run(lanes(du8 ? "latent_group_u_q8" : "latent_group_u", m.u), {mtl::Arg::B(U), mtl::Arg::B(projected), mtl::Arg::B(active), mtl::Arg::B(grouped)},
            HIDDEN * m.u / 128, slots);
        for (int s = 0; s < slots; ++s) {
            const int g = s % GROUPS;
            const float* p = projected.as<float>() + (size_t)s * latent;
            for (int row = 0; row < HIDDEN; ++row) {
                double y = 0;
                if (active.as<uint32_t>()[s])
                    for (int c = 0; c < latent; ++c) y += (double)uw[((size_t)g * HIDDEN + row) * latent + c] * p[c];
                eu.add(grouped.as<float>()[(size_t)s * HIDDEN + row], y);
            }
        }
        // ---- the groups and the shared expert, from the GPU's U
        run("latent_group_combine", {mtl::Arg::B(grouped), mtl::Arg::B(shared), mtl::Arg::B(shared_gate), mtl::Arg::B(out)},
            ((uint32_t)ntok * HIDDEN + 255) / 256, 1, 256);
        for (int t = 0; t < ntok; ++t)
            for (int row = 0; row < HIDDEN; ++row) {
                double y = (double)shared.as<float>()[t * HIDDEN + row] / (1.0 + std::exp(-(double)shared_gate.as<float>()[t]));
                for (int g = 0; g < GROUPS; ++g) y += grouped.as<float>()[((size_t)t * GROUPS + g) * HIDDEN + row];
                ey.add(out.as<float>()[t * HIDDEN + row], y);
            }
        // ---- U per entry (the ungrouped path), from the GPU's down
        run(du8 ? "latent_u_q8" : "latent_u", {mtl::Arg::B(U), mtl::Arg::B(labels), mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::B(plan),
                                             mtl::Arg::B(down), mtl::Arg::B(per_entry)}, (HIDDEN + 3) / 4, entries);
        for (int i = 0; i < entries; ++i) {
            const int g = labels.as<int>()[ids.as<int>()[i]];
            const float* d = down.as<float>() + (size_t)i * latent;
            for (int row = 0; row < HIDDEN; ++row) {
                double y = 0;
                for (int c = 0; c < latent; ++c) y += (double)uw[((size_t)g * HIDDEN + row) * latent + c] * d[c];
                ee.add(per_entry.as<float>()[(size_t)i * HIDDEN + row], y * wts.as<float>()[i]);
            }
        }
        const bool ok = scale_ok && active_ok && eh.ok() && ed.ok() && ep.ok() && eu.ok() && ey.ok() && ee.ok();
        std::printf("parity %s window (lanes gu %d dn %d u %d): scale %s, groups %s (%d of %d unused), rel L2 / max: h %.1e %.1e, "
                    "down %.1e %.1e, pooled %.1e %.1e, U %.1e %.1e, sum %.1e %.1e, U per entry %.1e %.1e  %s\n",
                    m.name, m.gu, m.dn, m.u, nv ? "none" : scale_ok ? "ok" : "WRONG", active_ok ? "ok" : "WRONG", unused, slots,
                    eh.rel_l2(), eh.rel_max(), ed.rel_l2(), ed.rel_max(), ep.rel_l2(), ep.rel_max(), eu.rel_l2(), eu.rel_max(),
                    ey.rel_l2(), ey.rel_max(), ee.rel_l2(), ee.rel_max(), ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    return all_ok ? 0 : 1;
}
