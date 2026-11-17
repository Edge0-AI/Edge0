// include/strata/kernels/elementwise.hpp - the small ops a LAYER needs between its GEMVs, P2.S5.
//
// These are not interesting kernels and that is why they are in one file with one parity test: they are the
// glue in the per-layer graph, and glue that is written inline at each call site is glue with several
// slightly different float conventions.  Three of them exist because a projection returns a DIFFERENT
// quantity from the one the next kernel wants:
//
//   * `gdn_gate` turns (alpha, dt, ssm_a) into the recurrence's `gate`.  `ref/gdn.py` L265:
//     `gate = softplus(w_alpha . x + dt) * ssm_a`, with `softplus(x) = log1p(exp(x))` and the large-x branch
//     `ggml_compute_softplus_f32` uses.  `ssm_a` is NEGATIVE, which is what puts `exp(gate)` in (0,1).
//   * `scale_inplace` applies the delta net's `1/sqrt(S)` to q.  It is a separate op because
//     `ref/gdn.py` says the two llama.cpp paths apply it in DIFFERENT places, so the caller has to own the
//     choice rather than have it buried in a kernel.
//   * `f32_to_f16_bulk` is what lets the f32 arithmetic of the mixer feed the fp16-activation GEMVs.
//     `docs/activation-contract.md` is explicit that this is NOT the contract for a quantized weight - it is
//     the bridge between two stages that both already exist, and the contract decision is a separate one.
//
// `silu_inplace` is here because `shared_expert` has its own copy inline and `gdn` needs one for the conv
// output; a third copy would be two too many.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Decode one canonical embedding row already selected by the caller. Codes are
/// packed low bits first (2, 4 or 8 bits); scales/offsets are FP32 per group.
/// The optional offset defaults to +0. Multiplication and addition round separately,
/// matching the former CPU decoder. Launches asynchronously on the supplied stream.
/// The caller validates positive, divisible dimensions and row/plane bounds.
void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream);

/// `y[t][h] = softplus(alpha[t][h] + dt[h]) * ssm_a[h]`, one thread per (t, h).  In place is not possible:
/// `alpha` is read and `gate` is written, and they are different buffers by construction.
///
/// `h_v` is the number of v heads (48) and `n_tokens` the leading dimension; the real call is (1, 48).
void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream);

/// `x[i] *= s`, in place.
void scale_inplace(float* x, int64_t n, float s, void* stream);
/// `dst[i] += src[i]`.  **ADDED FOR R4's HIT/MISS SPLIT, WHERE THE TWO HALVES COME FROM DIFFERENT
/// ENGINES.**  The CPU writes the misses into `parts` and the GPU writes the resident experts into a
/// buffer of their own, so that the GPU's work can run CONCURRENTLY with the CPU's instead of after it -
/// and the two are summed here, on the stream, immediately before `moe_combine` reads the result.
///
/// The alternative - having the hit kernel write straight into `parts` - costs the whole overlap, because
/// the copy of the misses into `parts` cannot happen until the pool has produced them, and the kernel is
/// then ordered behind that copy.  Measured: the drain fell 18.2 -> 10.2 ms and the token did not move.
void add_inplace(float* dst, const float* src, int64_t n, void* stream);

/// `y[i] = f16(x[i])`, round-to-nearest-even, using the shared conversion in `f16_bits.hpp`.
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);

/// `y[i] = bf16(x[i])`, for the weights whose contract is a bf16 activation - the BF16 `hc_*`, `ssm_alpha`,
/// `ssm_beta`, `indexer.*` and `ple_value` tensors.
///
/// A SEPARATE ENTRY POINT FROM `f32_to_f16_bulk` and not a flag on it, because the two conversions are not
/// variants of one idea: fp16 has 5 exponent bits and 10 mantissa bits, bf16 has 8 and 7.  A caller that
/// picked the wrong one would get a plausible tensor at the wrong precision - 2^-11 against 2^-9, a factor of
/// four - and nothing downstream could tell which was intended.
void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);

/// `x[i] = x[i] / (1 + exp(-x[i]))`, in place.
void silu_inplace(float* x, int64_t n, void* stream);

/// `build_norm`: `y[r][c] = x[r][c] / sqrt(MEAN_c(x[r]^2) + eps) * w[c]`, over the LAST axis.
///
/// QSA's norm, used on `attn_q` (24x256), `attn_k` (2x256) and `indexer.q_proj` (4x128). `w` may be null.
///
/// **IT IS NOT `gdn_l2_norm` AND THE DIFFERENCE IS A FACTOR OF sqrt(cols) = 16 HERE.** `gdn_l2_norm` divides by
/// `sqrt(SUM(x^2) + eps)` and applies no weight; this divides by `sqrt(MEAN(x^2) + eps)` and multiplies by `w`.
/// Two kernels whose names both say "norm" and whose bodies differ in one `/ cols` is exactly the pair that
/// gets swapped, and swapping them produces a well-scaled, plausible tensor - so they are separate entry
/// points and the parity test asserts the two readings are OBSERVABLE apart rather than trusting the names.
///
/// There is NO `(1 + w)` folding: `ref/qsa.py::rms_norm` is `y * w`, and `build_norm` in the source is
/// `ggml_rms_norm` followed by a plain multiply. Some norms in this artifact DO fold (the GDN path records
/// which); this one does not, and the same `sqrt(cols)`-sized argument applies to getting it wrong.
void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream);

/// *seq = v in MAPPED PINNED memory - THE DOORBELL.  A KERNEL and not cudaEventRecord, because round 199
/// found that an event record inside a capture is SILENTLY DROPPED while a kernel is captured normally, and
/// round 209 measured the host seeing this one's result 0.050 ms into a 39.8 ms graph.
///
/// It lives here because the caller that needs it - moe_layer - is a HOST translation unit, where
/// __global__ and <<<>>> do not exist.
/// *seq += 1 - AN INCREMENT, NOT A STORE OF A CALLER-SUPPLIED VALUE.  A value passed in is evaluated on the
/// HOST at CAPTURE time, so a captured graph would ring the same number on every replay.  Reading and
/// incrementing in memory is what makes the count advance under cudaGraphLaunch.
void doorbell_ring(uint32_t* d_seq, void* stream);

/// Plan v0.3 P3 token graph: one thread spins until the host-written `*d_flag` equals the ring value `*d_seq`
/// (both mapped pinned memory), then fences so everything the host wrote before the flag is visible to the
/// nodes that follow.  Captured once per layer between `pre[l]` and the parts copy; it is what lets one graph
/// hold the whole token (bench/micro/device_wait.cu: 5.2 us per handoff, no driver call).
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream);

/// Plan v0.3 P3 token graph: copy `n` floats from MAPPED pinned host memory (`src` is its device pointer) into
/// device memory with a kernel, so the handoff stays on the compute queue (a memcpy node is a copy-engine
/// operation, which WDDM submits separately and which measured 67 flushes per token).
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream);
/// `rows` rows of `width` floats from mapped memory, except the rows listed in hit_rows[0, *count) (device),
/// which are written +0.0 instead (a verify window's GPU-computed entries: the pool leaves zeros there).
/// multi-GPU: dst[rows[r] * width ..] = src[r * width ..] for r < n (float4; dst may be mapped host memory).
void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream);
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream);
/// copy_rows_from_mapped after spinning on *flag >= value (every block's thread 0 polls the mapped flag; the
/// wait_flag_ge launch it replaces is one dependent kernel node, ~2.4 us on a gfx1201 graph).
void copy_rows_from_mapped_after(const uint32_t* flag, uint32_t value, float* dst, const float* src, int64_t rows,
                                 int64_t width, const int32_t* hit_rows, const int32_t* count, void* stream);

/// Plan v0.3 P3: the doorbell's payload and its ring in ONE kernel.  Copies `x` (n floats), `ids` and `weights`
/// (k each) into the mapped host regions, fences, and increments the mapped sequence number - replacing three
/// device-to-host memcpy nodes (copy-engine operations in the middle of the layer chain) and the ring kernel.
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream);
/// #649 (HIP, STRATA_DOORBELL_STORE=1): the ring stored as `value` instead of incremented over PCIe.
void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                            float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value,
                            void* stream);
void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                          float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream);
/// SYCL: publishes launched or recorded while `on` store the payload with plain stores instead of uncached ones, for
/// a caller that reads the payload only after the publishing kernel has ended (the stepped verify window).
void doorbell_plain_payload(bool on);

/// The ring can reach the host before its payload on some GPUs: the payload stores and the ring store are not ordered
/// by the fences alone (measured on an Arc A770: 14 of 2,832 rings in one run were stale, and the CPU pool then
/// computed on a partly stale payload). So the publish kernels also write, next to the sequence number, the ring it
/// belongs to and a checksum of the payload (the wrapping sum of every 32-bit word mixed with its position, murmur3's
/// finalizer - a plain sum missed a reordering, [3,7] -> [7,3], of the expert ids) in a per-ring slot modulo 4:
/// seq[1 + 2 (r % 4)] is the tag, seq[2 + 2 (r % 4)] the checksum (a split verify publishes two rings before the host
/// reads the first). The sequence word therefore needs 9 words of mapped host memory.  The two publish shapes
/// differ (doorbell_publish / doorbell_publish_value publish the routing weights, doorbell_publish_res does not),
/// so the wait accepts either.  CUDA/HIP: always true (their ordering holds), so callers may skip the wait there.
bool doorbell_payload_ready(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                            const float* h_weights, int64_t k, uint32_t want);
/// Spins on doorbell_payload_ready for up to `timeout_ms`; false on timeout.
bool doorbell_wait_payload(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                           const float* h_weights, int64_t k, uint32_t want, int timeout_ms = 2000);

/// Plan v0.3 P3: copy `n` int32 from mapped pinned host memory into device memory with a kernel (the QSA
/// per-token step and positions), instead of a host-to-device memcpy node in the middle of a layer.
void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream);
/// knife-0: copy `bytes` from mapped host memory to device memory with a KERNEL (no host CUDA call in a verify
/// window - the host-issued cudaMemcpyAsync there is the issue #31 deadlock).  16-byte words when both sides allow.
void copy_bytes_from_mapped(void* dst, const void* src, int64_t bytes, void* stream);
/// Up to 8 copies from mapped host memory in one launch: `words` 4-byte words each (16 bytes a load where both
/// ends are 16-byte aligned and the count a multiple of 4).  Graph-capturable.
struct MappedCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    int64_t words = 0;
};
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream);

}  // namespace strata::kernels
