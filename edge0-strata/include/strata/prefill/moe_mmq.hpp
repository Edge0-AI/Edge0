// include/strata/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts through llama.cpp's MMQ
// kernels (ggml-cuda mmq.cuh, MIT): the weights stay quantized and the activations are rounded to q8_1, the
// products run on int8 tensor cores.  The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is gathered into one buffer
// (`gather_*`, one launch per expert as its blob arrives) and multiplied in one launch per product.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// MMQ covers this ggml type (the i-quants and Q2_0 the packs use, Q8_0, NVFP4 unless fp16, and with
/// STRATA_MMQ_KQUANTS the K-quants Q4_K / Q5_K / Q5_1 / Q6_K: Unsloth's UD-Q4_K_XL experts (CUDA), and the dense GGUF
/// projections of the mixed-quant packs through Gemm::native's STRATA_DENSE_MMQ path (HIP); IQ1_M is not covered).
bool supported(int ggml_type);
/// #420: `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of
/// `w_rows` rows that fits the card's shared memory - the same test its tile choice makes, which aborts the process
/// ("J_best=0") when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// NVFP4 prompt precision, STRATA_PREFILL_NVFP4: w4a8 (default) - int8 tensor cores, Q8_1 activations, what decode
/// runs at; w4a4 - Blackwell's FP4 x FP4 MMA, activations rounded to NVFP4 (faster, first-token KL up to 0.03 vs
/// fp16); fp16 - the dequantize + FP16 GEMM path (the reference).
enum class Nvfp4Mode { W4A8, W4A4, FP16 };
Nvfp4Mode nvfp4_mode();
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream, float* yscale = nullptr);
/// NVFP4 on Blackwell in w4a4 mode: the MMQ kernel multiplies FP4 x FP4, so its activations are NVFP4 too, with
/// one float scale per row that `quantize` writes to `yscale` and the product reads as Product::y_scale.  Decided
/// by the mode and a device probe of the same compile.
bool fp4_activations(int ggml_type);
/// NVFP4 expert tails: GU rows [row0, row0 + nrows) of a group of `n` experts (absolute `bounds`, n + 1 of them,
/// on the device) scaled by their expert's {s_gate, s_up, s_down, 0} (`tails`, 4 floats each, on the device): the
/// gate half by s_gate, the up half by s_up.
void scale_gu_rows(float* gu, int64_t ld, int64_t n_ff, const int32_t* bounds, int n, const float* tails,
                   int64_t row0, int64_t nrows, void* stream);
/// The down product's rows [0, nrows) (`ld` floats each, relative `bounds`) times their expert's s_down - on the FP32
/// output, not folded into up, where the hidden would sit near 1e-5 (FP16-subnormal block scales in q8 formats).
void scale_down_rows(float* d, int64_t ld, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream);
/// The rows quantize(x, src, ...) writes for tokens x k_used rows in expert order, written token by token: token t's
/// row of x is quantized once and stored at rows slot[t * k_used + k] (`slot` the inverse of `src`) - the same bytes,
/// a k_used-th of the quantizations (llama.cpp's quantize_scatter_mmq_q8_1_cuda).  STRATA_QUANT_GATHER=1: quantize()
/// over `src`, as before.
void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int ggml_type, int64_t cols,
                      int64_t ld, int64_t tokens, int k_used, void* stream);

constexpr int kGatherGroupMax = 16;
/// STRATA_MMQ_DIRECT: each expert's own weight address (gate|up or down), in place of `w + z * expert_bytes` -
/// expert z < n <= kGatherGroupMax starts at p[z].  The launch (tiles, stream-k split, fixup) is the same as
/// with `w`: only the base address of each expert's matrix changes.
struct WeightTab {
    const void* p[kGatherGroupMax] = {};
};

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
    const float* y_scale = nullptr;     ///< NVFP4 w4a4: the per-row activation scales (fp4_activations)
    /// set: expert z's matrix starts at tab->p[z] (w and expert_bytes are not used for addressing).  CUDA only,
    /// and only where `tab_fits` said so.
    const WeightTab* tab = nullptr;
};

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);

private:
    void* ctx_ = nullptr;
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
/// `tail` (16 bytes, may be null) goes to `tail_dst`; `zero_bytes` zeroed right after gu_dst's and d_dst's copies
/// (the MMQ tail after a group's last expert).
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail = nullptr, void* tail_dst = nullptr,
                   size_t zero_bytes = 0);
/// gather_native for an MMQ group's experts [first, n) in ONE launch: expert q's blob (`blob[q]`; gate at +0, up at
/// +up_off, down at +down_off) to gu_dst + q * gu_stride and d_dst + q * d_stride - the same bytes as one gather_native
/// each.  Every pointer, offset and size 16-byte aligned (false otherwise: nothing launched, gather one at a time).
/// NVFP4: `tail_off` (0 = none) is each blob's 16-byte tail, copied to tail_dst + 16 q; `zero_bytes` are zeroed after
/// the last expert's gu and d slots (the MMQ tail).
struct GatherGroup {
    const uint8_t* blob[kGatherGroupMax] = {};
    int first = 0, n = 0;
};
bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream,
                         size_t tail_off = 0, void* tail_dst = nullptr, size_t zero_bytes = 0);
/// A Strata-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);
/// swiglu (n_ff 640), then quantize() of its rows for a down product of `ggml_type`, in one pass: H never stored,
/// the same bytes in hq.  Only where swiglu_quant_ok (a q8_1 D4 down type; gu 16-byte aligned): CUDA, the
/// STRATA_EXPERIMENTAL_SM60 build.
bool swiglu_quant_ok(int ggml_type);
void swiglu_quant(const float* gu, void* hq, int64_t rows, bool interleaved, void* stream);
/// swiglu of an NVFP4 group's gate/up rows (split halves) with scale_gu_rows' scales applied as they are read.
void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream);
/// sd[r] = the s_down of row r's expert (group-local bounds): the combine applies it as it reads the row.
void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

/// y[i] = float(x[i]) for FP16 bits x.
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream);
/// dst = {0, rows} (the bounds of one matrix; dst on the device).
void set_bounds(int32_t* dst, int32_t rows, void* stream);

// ---- STRATA_MMQ_DIRECT: the prompt MMQ reads resident experts in place (docs/DETAILS.md, "STRATA_MMQ_DIRECT") ----
/// This build has the pointer-table MMQ (CUDA, the build option STRATA_MMQ_TAB on - off by default - and the
/// llama.cpp tree matched the mirrored scheduler's fingerprint).  false under HIP, gfx906, SYCL and builds without MMQ.
bool tab_built();
/// `tab_built`, `fits`, rows a multiple of 128 (no fallback tiles) and a pointer-table instance for the type (the
/// native packs' nine: Q2_0 IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS Q8_0; not Q5_0 or the K-quants).
bool tab_fits(int ggml_type, int64_t w_rows);
/// The bytes past a [rows, cols] matrix that its last row's final K tile reads (the largest over-read of any
/// config on any visible GPU): 72 for IQ4_NL and 136 for Q8_0 at 640 columns, 0 for IQ3_S at 2560 (sm_86).  0 without MMQ.
size_t overread_bytes(int ggml_type, int64_t cols);
/// The room kept after each sized expert slot's blob for that over-read (the guard).
constexpr size_t kSlotGuard = 256;
/// Every launch of `n_experts` experts of [w_rows, *] `ggml_type` weights computes each output tile in one block
/// with no stream-k K split, and every tile width has the same I/K tiles and SRAM layout, on every visible GPU - so an
/// expert's rows get the same bits in any such launch, whatever its group-mates and max_rows (STRATA_MMQ_GROUP_SORT).
/// 160 or 320 tiles per column tile on 82 SMs: yes; 108 or 128 SMs: no (stream-k splits the few-tile launches).
bool whole_tiles(int ggml_type, int64_t w_rows, int n_experts);
/// STRATA_MMQ_DIRECT, read once: 0 (default) today's gather; 1 the guard layout and resident experts read in
/// place; 2 the guard layout with today's gather (the bitwise A/B reference).  0 whenever !tab_built().
int direct_mode();
/// kSlotGuard when direct_mode() != 0, else 0 (no CUDA call).
size_t direct_slot_guard();
/// The guards a group's direct experts' down over-reads land in (each `guard_bytes` long).
struct GuardTab {
    uint8_t* p[kGatherGroupMax] = {};
    int n = 0;
};
/// One launch: zeroes `guard_bytes` at each g.p[i] and `tail_bytes` at tail_gu and tail_d (null: skipped).  It
/// replaces the two tail memsets of a group in direct mode.  Nothing is launched when there is nothing to zero.
void prep_group(const GuardTab& g, size_t guard_bytes, void* tail_gu, void* tail_d, size_t tail_bytes, void* stream);

}  // namespace strata::prefill::mmq
