// src/core/dflash2_parity.cpp - the engine's DFlash2 drafter against the torch port, on a synthetic case (GPU, no model).
//
//     dflash2_parity <pack> <case.bin> [rounds]
//
// case.bin comes from tools/dflash2/mk_parity_case.py: n_ctx random tap rows, an anchor, a random Q8_0 head over a draft-vocabulary subset, the
// target embedding rows of the anchor and the mask token, and the torch answer (tokens, probabilities, the 8 final normalized rows).  The engine's
// drafter reads the same pack (Q8_0 weights, Q8_1 activations), so the rows agree to quantization noise, not bitwise: the test checks the rows'
// cosine similarity, the walk's tokens and the time of a round.
#include "strata/core/dflash2.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::Dflash2Drafter;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: dflash2_parity <pack> <case.bin> [rounds]\n"); return 2; }
    const int rounds = argc > 3 ? std::atoi(argv[3]) : 20;
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    int32_t hdr[4];
    if (std::fread(hdr, 4, 4, f) != 4) return 2;
    const int n_ctx = hdr[0], anchor = hdr[1], vp = hdr[2];
    const int H = 2560, tapn = 5 * H;
    std::vector<float> taps((size_t) n_ctx * tapn), emb(2 * H), fn_ref(8 * H);
    std::vector<int32_t> dvocab(vp), tok_ref(7);
    std::vector<float> prob_ref(7);
    const size_t row_bytes = (size_t) (H / 32) * 34;
    std::vector<uint8_t> head((size_t) vp * row_bytes);
    bool ok = std::fread(taps.data(), 4, taps.size(), f) == taps.size() && std::fread(emb.data(), 4, emb.size(), f) == emb.size() &&
              std::fread(dvocab.data(), 4, dvocab.size(), f) == dvocab.size() && std::fread(head.data(), 1, head.size(), f) == head.size() &&
              std::fread(tok_ref.data(), 4, 7, f) == 7 && std::fread(prob_ref.data(), 4, 7, f) == 7 && std::fread(fn_ref.data(), 4, fn_ref.size(), f) == fn_ref.size();
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "case file is cut short\n"); return 2; }

    std::string err;
    Dflash2Drafter d;
    if (!d.load(argv[1], 0, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    float *d_taps = nullptr, *d_emb = nullptr;
    uint8_t* d_head = nullptr;
    cudaMalloc((void**) &d_taps, taps.size() * 4);
    cudaMalloc((void**) &d_emb, emb.size() * 4);
    cudaMalloc((void**) &d_head, head.size());
    cudaMemcpy(d_taps, taps.data(), taps.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_emb, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_head, head.data(), head.size(), cudaMemcpyHostToDevice);
    auto embed = [&](const int32_t*, int n, float* out, cudaStream_t cs) {
        cudaMemcpyAsync(out, d_emb, (size_t) H * 4, cudaMemcpyDeviceToDevice, cs);
        for (int i = 1; i < n; ++i) cudaMemcpyAsync(out + (size_t) i * H, d_emb + H, (size_t) H * 4, cudaMemcpyDeviceToDevice, cs);
    };
    if (!d.bind_raw(embed, d_head, 8, vp, dvocab, err)) { std::fprintf(stderr, "bind: %s\n", err.c_str()); return 1; }
    std::printf("drafter VRAM %.0f MiB\n", (double) d.vram_bytes() / 1048576.0);

    // the context but the last row, then one draft round that adds the last row
    if (!d.context(d_taps, n_ctx - 1, 0, err)) { std::fprintf(stderr, "context: %s\n", err.c_str()); return 1; }
    int32_t drafts[7];
    float probs[7];
    const float* last = d_taps + (size_t) (n_ctx - 1) * tapn;
    if (!d.draft(last, 1, n_ctx - 1, anchor, drafts, probs, err)) { std::fprintf(stderr, "draft: %s\n", err.c_str()); return 1; }
    std::vector<float> rows;
    d.debug_rows(rows, err);
    int bad = 0;
    double min_cos = 1.0;
    for (int r = 0; r < 8; ++r) {
        double dot = 0, na = 0, nb = 0;
        for (int c = 0; c < H; ++c) {
            const double a = rows[(size_t) r * H + c], b = fn_ref[(size_t) r * H + c];
            dot += a * b; na += a * a; nb += b * b;
        }
        const double cs = dot / std::sqrt(na * nb + 1e-30);
        min_cos = std::min(min_cos, cs);
        std::printf("row %d: cos %.5f  norm ratio %.4f\n", r, cs, std::sqrt(na / (nb + 1e-30)));
    }
    int same = 0;
    for (int i = 0; i < 7; ++i) {
        same += drafts[i] == tok_ref[i];
        std::printf("draft %d: engine %d (p %.3f)  torch %d (p %.3f)\n", i, drafts[i], probs[i], tok_ref[i], prob_ref[i]);
    }
    if (min_cos < 0.98) { std::printf("FAIL: a final row's cosine similarity %.4f is below 0.98\n", min_cos); ++bad; }
    // timing: the same round again (the graph replays; the ring rewrites the same slot)
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i)
        if (!d.draft(last, 1, n_ctx - 1, anchor, drafts, probs, err)) { std::fprintf(stderr, "draft: %s\n", err.c_str()); return 1; }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / rounds;
    std::printf("tokens equal %d of 7; %.3f ms per round (1 context row + the block), GPU %.3f ms\n", same, ms, d.rounds > 0 ? d.ms_gpu / (double) d.rounds : 0.0);
    std::printf(bad ? "dflash2_parity: FAIL\n" : "dflash2_parity: ok\n");
    return bad ? 1 : 0;
}
