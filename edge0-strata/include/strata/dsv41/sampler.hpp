// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/kernels/sampler.hpp - the sampler chain, host-callable (P2.S2).
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// llama.cpp's default chain (issue #53): ONE penalties stage, first, and temperature AFTER the truncation filters;
// top_p cuts before min_p.  `src/kernels/cuda/sampler.cu` has the details and `sampler_parity` pins them.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::dsv41k {

struct SamplerParams {
    int top_k = 20;              // sampled path: 1..64 as given; 0 (off) or more than 64 keep the widest list, 64
    float top_p = 0.95f;         // 1.0 disables the filter
    float min_p = 0.0f;          // 0 disables; keeps tokens with p >= min_p * p_max (a prefix of the top_k list)
    float temperature = 1.0f;    // <= 0 means greedy
    int min_keep = 1;            // top_p keeps at least this many
    int penalty_last_n = 0;      // 0 disables; the window over `history` to count occurrences in
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;           // drives Philox, which is counter-based on (seed, token index)
    uint64_t counter = 0;        // absolute draw index of row 0; advance across decode calls
    bool greedy = false;
    // a request's logit_bias: a DEVICE array of n_vocab floats on the sampling device, added to every row's logits
    // before the penalties and the filters (-inf bans a token); nullptr = none
    const float* logit_bias = nullptr;
    // the GLM engine's penalty history, on the HOST (never read by a kernel): every token the request's state has been
    // fed so far, oldest first - its prompt, then each token it wrote.  Glm5Model::pick_tokens hands each pick the last
    // penalty_last_n of them (a verify window's row t: these, then the window's drafts 1..t).  nullptr = none
    const std::vector<int32_t>* host_history = nullptr;
};

// logits (n_tokens, n_vocab) -> one sampled token id per row in `out`.
//
// **`logits` AND `out` ARE DEVICE POINTERS.**  This is a CUDA kernel launch, not a host function, and nothing
// in the parameter names or the types says so - `const float* logits` reads exactly like a host buffer.
// Measured: passing host memory for `logits` faults inside the kernel with an ILLEGAL MEMORY ACCESS that is
// reported by whatever synchronising call happens NEXT, which will be somewhere else entirely and will name a
// buffer that has nothing to do with it.  A sticky async fault does not know where it came from.
//
// `history` is (n_tokens, history_len) int32 - ONE ROW PER TOKEN ROW, at a stride of `history_len` - the most
// recent tokens that row's pick follows, with any unused slots set to -1; only the last `p.penalty_last_n` of
// each row are counted, and ids outside [0, n_vocab) are ignored.  Pass nullptr and 0 when no penalties apply.
// A verify window's rows need DIFFERENT histories: row t follows the window's drafts 1..t (`penalty_rows`).
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream);

// SPECULATIVE SAMPLING (Leviathan et al. 2023, Chen et al. 2023) for the pipelined NextN decode - opt-in
// (STRATA_GLM_SPEC_SAMPLE=1).  The draft is DRAWN from the draft block's own distribution q (this chain on its logits,
// a Philox stream of its own) and its kept list is stored; the model's token is then the draft with probability
// min(1, p(d) / q(d)), else a draw from the residual max(0, p - q) renormalised - so it is distributed exactly as the
// plain draw from p, and a draft stands whenever the token is the draft (sum over x of min(p(x), q(x)) of the time
// instead of p(argmax q)).  Same (seed, counter): the same tokens - but not the plain draw's tokens.
struct SpecList {
    int n = 0;          // the kept list's length (1..64)
    int ids[64];        // descending, as the selection produced them
    double p[64];       // q over the kept list (sums to 1)
};

// draft rows: logits (n_tokens, n_vocab) -> out[t] drawn through this chain, q[t] its kept list.  DEVICE pointers,
// like sample_tokens; a sampled request without penalties only (greedy keeps the argmax draft).
void sample_draft(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, SpecList* q, int* out,
                  void* stream);
// verify rows: the model's logits (n_tokens, n_vocab), the draft draft[t] drawn from q[t] (same p.seed / p.counter as
// its sample_draft) -> out[t], distributed as sample_tokens' draw.  A draft q[t] does not hold gets the plain draw.
void sample_verify(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, const SpecList* q,
                   const int* draft, int* out, void* stream);

// The penalty-history rows of a verify window, on the host: row t of `out` (T rows of `h` slots) is the last `h`
// tokens of `tail[0..n_tail)` followed by `window[0..t]`, most recent LAST, -1 in the unused front slots.
// `tail` is what the state consumed before the window, `window[0]` the fed-back token and `window[1..]` the
// drafts: row t is exactly the history plain decode counts when it picks the token after `window[t]`, so drafting
// cannot change which tokens are penalised.  Row 0 alone is the single row the engine staged before 0.1.19.
inline void penalty_rows(const int32_t* tail, int64_t n_tail, const int32_t* window, int T, int h, int32_t* out) {
    for (int t = 0; t < T; ++t) {
        int32_t* row = out + (size_t) t * (size_t) h;
        const int64_t avail = n_tail + t + 1;                      // tail + window[0..t]
        const int take = (int) std::min<int64_t>(h, avail);
        std::fill(row, row + (h - take), -1);
        for (int j = 0; j < take; ++j) {
            const int64_t i = avail - take + j;                     // index into (tail, window)
            row[h - take + j] = i < n_tail ? tail[i] : window[i - n_tail];
        }
    }
}

}  // namespace strata::dsv41k
