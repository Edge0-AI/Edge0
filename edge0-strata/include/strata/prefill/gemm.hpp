// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::prefill {

/// STRATA_F16_RANGE: count what the gfx103x FP16 route converts on this device - the BF16 weights to FP16 (slot 0)
/// and the FP16 GEMM outputs widened to FP32 (slot 1): the largest |value| and how many were beyond 65504 or not
/// finite.  gemm_f16_range_read returns them (and resets when asked); false where the route does not exist (CUDA).
void gemm_set_f16_range(bool on);
bool gemm_f16_range_read(float max_abs[2], unsigned long long over[2], bool reset);


class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: BF16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    /// Below sm_80 (no BF16 tensor cores): Volta converts both to FP16 and runs the FP16 tensor-core GEMM (Turing with
    /// STRATA_BF16_TC=1), Pascal widens both to fp32 (cublasSgemm); see gemm.cu.
    /// `x_f16`: X holds, instead of BF16, the FP16 values the Turing tensor-core path would convert it to
    /// (prefill's xn16 with its f16img writers): that path skips the conversion; any other converts back.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f, int64_t ldx = 0, bool x_f16 = false);
    /// true when bf16() runs its products through FP16 tensor cores here (Turing, STRATA_BF16_TC): where X may be FP16
    bool bf16_converts_x() const;

    /// S (STRATA_HCD_EXACT): the HC down projection (N 320, K 10240) by the WMMA kernel that reproduces hipBLASLt's
    /// solution 1176 / 1177 bit for bit; false (nothing launched) unless hipBLASLt would take one of those for this
    /// shape (so the caller falls back to bf16(), with the same ldx).
    bool bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.  `ldx` (> K) is
    /// X's padded row stride, taken only by STRATA_PF_PAD's path (0 = K).
    /// On an MMQ build a beta = 0 product whose type is covered, whose K is a multiple of 256 values and whose matrix fits
    /// the card's shared memory runs through llama.cpp's int8 MMQ instead (opt-in: STRATA_DENSE_MMQ=1).
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f, int64_t ldx = 0);

    /// HIP gfx103x (prompt_f16()): bf16() takes X as the FP16 image the prompt path's kernels write (set_act_f16),
    /// and bf16() / f16() / native() run the GEMM with FP16 out (rocBLAS's tuned kernels there), widened in place in
    /// Y's own rows.  Off (default): every call is what it was.  Set by Prefill::init, together with set_act_f16.
    void set_f16_io(bool on) { f16_io_ = on; }

    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    /// STRATA_DEQUANT_SIDE=1 (gemm_dequant_side): a second dequant scratch of scratch_elems() elements, caller-owned
    /// like the first.  native() then dequantizes on a side stream into the two in turn, so a weight's dequant runs
    /// while the kernels queued ahead of its GEMM do; events order every write and read of each scratch, so the
    /// products are the serial form's, bit for bit.  nullptr: the serial form.  Called after init_external and
    /// after every rebind (the "scratch free" events are re-armed on the compute stream).  CUDA only.
    bool set_dequant_side(uint16_t* scratch2, std::string& err);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    // STRATA_DEQUANT_SIDE (set_dequant_side): the second scratch, the side stream and, per scratch, "the dequant
    // landed" (recorded on the side stream) and "the GEMM read it" (recorded on the compute stream); side_cur_ is
    // the scratch the next native() dequantizes into (0: scratch_, 1: side_scratch_)
    uint16_t* side_scratch_ = nullptr;
    void* side_stream_ = nullptr;
    void* side_ready_[2] = {};
    void* side_free_[2] = {};
    int side_cur_ = 0;
    // STRATA_PF_W16_CACHE_MIB=N (CUDA, opt-in): native() keeps the FP16 dequantization of up to N MiB of weights, so a
    // repeat product skips the dequant kernel; the FP16 values and the product are the uncached form's, bit for bit.
    // A weight is keyed by its device pointer and shape (the engine's dense weights never move).
    struct W16Entry { const void* p; int type; int64_t n, k; uint16_t* d; };
    std::vector<W16Entry> w16_;
    int64_t w16_budget_ = -1;   // bytes; -1: env not read yet, 0: off
    int64_t w16_used_ = 0;
    void* hipblaslt_state_ = nullptr;
    void* rocblas_state_ = nullptr;   // HIP: the rocBLAS solution table of f16_inplace (STRATA_ROCBLAS_TUNING)
    bool f16_io_ = false;
    /// cuBLAS 12 answers the default FP16 algorithm of a few (N, K) shapes with CUBLAS_STATUS_INTERNAL_ERROR once
    /// another CUDA device in the process has initialized its context (issue #1650, Turing + Pascal).  A fixed
    /// algorithm (ALGO2) survives that state, but is markedly slower for the wide shapes, so only the shapes that
    /// fail move, and they are remembered here ((K << 32) | N, up to kF16AlgoSlots).  CUDA only.
    static constexpr int kF16AlgoSlots = 32;
    int64_t f16_algo_shape_[kF16AlgoSlots] = {};
    int f16_algo_n_ = 0;
    // RDNA2 (gfx103x, HIP): the product as an SGEMM on FP32 copies (tc_w_, tc_x_); false: the native call runs (gemm.cu)
    bool rdna2_sgemm(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                     float beta, bool bf16);
    // below sm_80: FP16 (Pascal: fp32) copies of a BF16 product's weight and activation slice (Gemm::bf16);
    // RDNA2: FP32 copies of the weight and an activation slice (Gemm::rdna2_sgemm)
    uint16_t* tc_w_ = nullptr;
    int64_t tc_w_elems_ = 0;
    uint16_t* tc_x_ = nullptr;
    int64_t tc_x_elems_ = 0;
    bool native_mmq(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                    int64_t ldy);
    /// A padded X (ldx > K) that no kernel above took: its T rows copied to a contiguous FP16 buffer of stride K, grown
    /// as needed; nullptr when the allocation or the copy failed (then the caller reports it).  The padded product is
    /// the gfx11 matrix cores' (STRATA_PF_PAD), so this is the path of another card whose activations arrived padded.
    const uint16_t* unpad_x(const uint16_t* X, int64_t T, int64_t K, int64_t ldx);
    uint16_t* unpadded_x_ = nullptr;
    int64_t unpadded_x_elems_ = 0;
    void* mmq_ctx_ = nullptr;
    void* mmq_buf_ = nullptr;
    bool mmq_failed_ = false;
    /// Y = X . W^T with FP16 out, written into Y's own rows and widened there (no buffer): prompt_f16() only.
    void f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy);
};

/// Pascal, opt-in: Gemm::f16's large dense shapes as cublasSgemm on FP32 copies (gemm.cu; bitwise the FP16 call's
/// result at the sizes it takes with the cuBLAS it was measured on).  0 off, 1 on (any card), -1 the default
/// (STRATA_DENSE_F32=1 on compute capability 6.x, else off).  For gemm_f32_parity; a no-op outside the experimental
/// CUDA 12 build.
void gemm_set_dense_f32(int mode);

/// HIP on gfx103x (RDNA2), for the current device: rocBLAS's tuned GEMMs there are FP16 in -> FP16 out only (FP16 or
/// BF16 in -> FP32 out runs ~6x slower), so the prompt path's 16-bit GEMMs run in FP16 (Gemm::set_f16_io,
/// set_act_f16).  Cached per device (a layer split can mix cards).  STRATA_HIP_PROMPT_F16=0/1 overrides.
bool prompt_f16();

/// STRATA_DEQUANT_SIDE=1: the prompt path's dense-weight dequant on a side stream with a second GEMM scratch
/// (Gemm::set_dequant_side; the prompt path takes the scratch from its borrowed region when this is on).  Off by
/// default; always off on HIP (its padded / MMQ paths use the scratch on the compute stream without the events).
bool gemm_dequant_side();



}  // namespace strata::prefill
