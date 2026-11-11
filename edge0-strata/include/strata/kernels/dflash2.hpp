// include/strata/kernels/dflash2.hpp - the small kernels of the DFlash2 drafter (core/dflash2.hpp) for Qwen3.8-Flash-Next.
//
// The drafter's matrices run through the engine's multi-column MMVQ (native_q8_0_mmvq); what is here is everything between
// them: the grouped dynamic 2-tap convolution, the per-head RMS norm + rotary rotation, the windowed non-causal attention over a
// K/V ring, the top-16 selection over the head's logits and the selector's left-to-right walk.  All launches are asynchronous on
// the given stream, allocate nothing and are graph-capturable.  Layouts are row-major f32 unless noted.
#pragma once

#include <cstdint>

namespace strata::kernels {

constexpr int kDf2Rows = 8;            ///< block rows (the anchor + 7 masks) and the most context rows one update takes
constexpr int kDf2Kv = 8;              ///< K/V heads
constexpr int kDf2Gqa = 4;             ///< query heads per K/V head
constexpr int kDf2Hd = 128;            ///< head dim
constexpr int kDf2Chunk = 128;         ///< keys per attention block
constexpr int kDf2MaxChunks = 17;      ///< ceil((2047 context keys + 8 block keys) / kDf2Chunk)
constexpr int kDf2TopK = 16;
constexpr int kDf2TopBlocks = 64;      ///< partial top-k blocks per row (a block holds at most 4096 logits)

/// out[b][c] = sum_{o < 2, b - o >= 0} (base[o][c] + dyn[b * dyn_stride + o * groups + c / group]) * x[(b - o) * H + c]  (+ resid[b][c] when
/// resid != null; out may equal resid).  x is [rows][H]; base is [2][H]; `dyn` points at this convolution's [2][groups] slice of row 0.
void df2_conv(const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out, int rows, int H,
              int group, void* stream);

/// table[r][0][i], table[r][1][i] = cos, sin of pos[r] * theta^(-2 i / 128) for i < 64 (the angle reduced in double first, so a large
/// position keeps its precision under fast math).  table: [rows][2][64].
void df2_rope_table(const int32_t* pos, int rows, float theta, float* table, void* stream);

/// Per (row, head): out = rope(rmsnorm(in) * gamma).  `in` rows are `in_stride` floats apart (heads * 128 contiguous per row).  The
/// rotation is the neox/rotate_half one over the whole head dim with row r's entry of `rope_table` (null: the norm only).  The result of row r goes to
/// out + slot(r) * heads * 128 with slot(r) = (slot_mod > 0 ? pos[r] % slot_mod : r).  `rope` false: the norm only.
void df2_norm_rope(const float* in, int in_stride, const float* gamma, const int32_t* pos, const float* rope_table, float* out, int slot_mod,
                   int rows, int heads, float eps, void* stream);

/// out[slot(r)][c] = in[r][c], slot as above (the value half of the ring write).
void df2_copy_rows(const float* in, float* out, const int32_t* pos, int slot_mod, int rows, int cols, void* stream);

/// Attention of the block's 8 query rows (q: [8][32][128], already normed and rotated) over the context ring (kring, vring:
/// head-major [8][ring_slots][128]) and the block's own keys (kblk, vblk: [8][8][128]).  Row i at position s + i sees context position j when
/// s + i - j < window (j in [max(0, s - window + 1), s - 1], slot j % ring_slots) and every block key.  `s` is read from *anchor_pos
/// on the device.  Partials in `part` ([kDf2MaxChunks][8][32][130] floats) are combined and the 8 output rows (4096 wide) written planar-quantized to xq / xd.
void df2_attention(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk, const int32_t* anchor_pos,
                   int ring_slots, int window, float* part, void* xq, float* xd, void* stream);

/// The Q8 GEMM the drafter's matrices run on.  A weight matrix is TILED (see the layout note in dflash2.cu; df2_q8_tile makes it from GGUF Q8_0 blocks):
/// the int8 plane at `w` (n_out * n_in bytes) then the fp16 scale plane.  Activations are planar (df2_quant_rows and the fused quantizers below):
/// xq is int8 [col][n_in] (8 columns of n_in bytes), xd float [col][n_in / 32].  Up to 3 matrices that share the input run in one launch; y (matrix i)
/// is [ncols][n_out].  n_in a multiple of 512, n_out of 16, ncols 1..8 (the other columns of xq are read and dropped).
/// Bench only: pick a GEMM kernel variant (0 = the default).
void df2_gemm_variant(int v);
/// Bench only: attention variants (bit 0 skips Q K^T, bit 1 skips P V).
void df2_attention_variant(int v);
struct Df2Mat { const void* w; float* y; int n_out; };
void df2_gemm(const Df2Mat* mats, int n, const void* xq, const float* xd, int n_in, int ncols, void* stream);
/// Planar quantization of rows x[ncols][K].
void df2_quant_rows(const float* x, int K, int ncols, void* xq, float* xd, void* stream);
/// RMS norm (weights gamma) of `rows` rows of N, written to xn (optional) and quantized.
void df2_rms_quant(const float* h, const float* gamma, float* xn, int rows, int N, float eps, void* xq, float* xd, void* stream);
/// df2_conv without residual, then quantized.
void df2_conv_quant(const float* x, const float* dyn, int dyn_stride, const float* base, int rows, int H, int group, void* xq, float* xd, void* stream);
/// silu(gate) * up of `rows` rows of I, quantized.
void df2_swiglu_quant(const float* gate, const float* up, int rows, int I, void* xq, float* xd, void* stream);
/// GGUF Q8_0 blocks (34 bytes) of a matrix with rows of n_in -> the tiled qs plane and the tiled d plane (count = rows * n_in / 32 blocks).
void df2_q8_tile(const void* blocks, void* qs, void* dpl, int n_in, int64_t count, void* stream);

/// h += conv(x) (the finish convolution with the residual), then the RMS norm of the new h (weights gamma) -> xn (optional) and the planar quantization.
/// rows <= 8, H <= 3072 (H a multiple of 32).
void df2_conv_rms_quant(const float* x, const float* dyn, int dyn_stride, const float* base, float* h, const float* gamma, float* xn, int rows, int H, int group,
                        float eps, void* xq, float* xd, void* stream);
/// The block's q (q_heads x 128) and k (k_heads x 128) per row: RMS norm and rope with the table of the block's positions -> qo, ko (row-major).
void df2_norm_rope_qk(const float* q, const float* k, const float* gq, const float* gk, const float* table, float* qo, float* ko, int rows, int q_heads, int k_heads,
                      float eps, void* stream);
/// The context update's write: K (norm + rope) and V of `rows` rows into the head-major rings at slot pos[r] % slots.
void df2_kv_ring(const float* kin, const float* vin, const float* gk, const int32_t* pos, const float* table, float* kring, float* vring, int slots, int rows,
                 int heads, float eps, void* stream);

/// Top-16 per row of logits [rows][n] (value-descending; ties to the lower index).  idx_map (optional) maps a column to a token id.
/// scratch: rows * kDf2TopBlocks * 16 * 2 words.  Outputs val[rows][16] and tok[rows][16].  n <= 4096 * kDf2TopBlocks.
void df2_topk(const float* logits, int n, const int32_t* idx_map, int rows, uint32_t* scratch, float* val, int32_t* tok, void* stream);

/// The candidate selector's walk: for row i = 0..rows-1 the 16 candidates' scores are unary[i][j] + dot(succ[tok[i][j]],
/// pred[prev] * hp[i]) over `rank` values (codebooks are bf16 rows of `rank`), the pick is the argmax, prev = the pick.  Writes
/// out_tok[i] and out_prob[i] (the softmax probability of the pick among the 16).  `prev0` is read from *anchor_tok on the device.
void df2_select(const float* hp, const float* unary, const int32_t* tok, const uint16_t* pred_cb, const uint16_t* succ_cb, int rank, int rows,
                const int32_t* anchor_tok, int32_t* out_tok, float* out_prob, void* stream);

/// The verify window's tap: out[r][d] = mean over the hc streams c of (R[r][c][d] + bo[r][d] * 2 * sigmoid(inject[r][c] / hc)) - the layer output with
/// its pending FFN write (gr_write) applied, averaged over the streams.  R rows are `R_stride` floats apart, bo rows `bo_stride`, inject rows
/// `inj_stride`, out rows `out_stride`.
void df2_tap_gr(const float* R, int64_t R_stride, const float* bo, int64_t bo_stride, const float* inject, int64_t inj_stride, float* out, int64_t out_stride,
                int rows, int64_t n_embd, int hc, void* stream);

/// out[r][:n_embd] = the mean of the hc streams of row r of R ([rows][hc][n_embd]) written to out + r * out_stride (a tap slot).
void df2_tap_rows(const float* R, float* out, int64_t out_stride, int64_t n_embd, int hc, int rows, void* stream);

}  // namespace strata::kernels
