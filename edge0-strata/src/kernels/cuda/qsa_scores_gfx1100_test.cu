// Test-only translation unit: production scorer plus the unchanged warp dispatch.
// The production library does not compile this reference wrapper.
#include "qsa_select.cu"

namespace strata::kernels {
void qsa_block_scores_reference(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        // STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
        static const bool early = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
        block_scores_multi_kernel<<<256, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
                                                                                     max_blocks, scores, early);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores multi: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}
