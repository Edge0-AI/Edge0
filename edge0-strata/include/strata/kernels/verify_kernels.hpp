// include/strata/kernels/verify_kernels.hpp - plan v0.3 P6: kernels for the speculative VERIFY window, where T
// tokens (the last accepted token and T-1 drafts) go through a layer at once.
//
// The GDN recurrence is the one piece of state that cannot simply be overwritten by the next window, so its
// kernels come in two halves:
//
//   * the verify half reads the conv history and the recurrent state, runs all T tokens in order (bitwise the
//     single-token kernels' arithmetic per token) and writes ONLY the per-token outputs - the state is left
//     untouched;
//   * the commit half, launched once acceptance is known, replays the first `n_keep` tokens from the inputs the
//     verify half stored and writes the state.  Rejected tokens therefore never touch the state and there is no
//     snapshot of the 113 MB recurrent state to keep.
//
// Everything that varies per window (token ids, `n_keep`) is read from DEVICE memory so the kernels can be captured.
#pragma once

#include "strata/kernels/window_limits.hpp"

#include <cstdint>

namespace strata::kernels {

/// The most rows one verify / batch window holds (8; 16 in the CUDA / HIP builds, window_limits.hpp).  A window of up to
/// kVerifyNarrowT rows runs the kernels it always ran.
inline constexpr int kVerifyMaxT = kWindowMaxT;

/// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
/// `history` is NOT written, but with `commit` (one token): it then keeps the token, as gdn_conv_commit.  Bitwise
/// `fused_gdn_conv_l2` per token.
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0, bool commit = false);
/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// alpha/beta for T columns of x (T, n_embd): gate (T, h_v), beta (T, h_v).  Bitwise `fused_gdn_ab` per column.
/// gdn_conv_l2_multi and gdn_ab_multi in one launch (143-amdk)
void gdn_conv_ab_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, int n_tok, int t_begin, bool commit, const float* x, const uint16_t* w_alpha,
                       const uint16_t* w_beta, const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd,
                       int h_v, void* stream);
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// The recurrence + output norm for T tokens (h = (T, conv_channels) as q|k|v, gate/beta (T, h_v), z/y
/// (T, value_dim)).  With `n_keep == nullptr` the state is read and NOT written (verify); otherwise the first
/// *n_keep tokens are run and the state is written (commit; `y` may be scratch).  Bitwise `fused_gdn_step_norm`.
/// Batch windows: up to 8 slot groups in one launch - group i is rows [first[i], first[i] + ntok[i]) of the
/// window's buffers, recurring over its own slot's state[i] (and committing keep[i] tokens when keep is given).
struct GdnGroups {
    float* state[8] = {};
    const int32_t* keep[8] = {};
    int first[8] = {};
    int ntok[8] = {};
    int n = 0;
};
/// gdn_step_norm_multi over several slots' groups at once (the generic kernel, grid.y = group; per group the
/// same arithmetic as one call each: bit-identical).  `commit`: every group keeps *keep[i] tokens, no outputs used.
void gdn_step_norm_groups(const GdnGroups& groups, const float* h, int conv_channels, const float* gate, const float* beta,
                          const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, bool commit,
                          void* stream);
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin = 0, void* xq_out = nullptr);
/// The commit of `n_layers` GDN layers in two launches (engine-latency): layer i's conv history is
/// `state + i * state_stride + conv_off` and its qkv rows `qkv + i * qkv_stride`, as gdn_conv_commit; its recurrence
/// state `state + i * state_stride` over the first *n_keep rows of `h + i * h_stride` / `gate|beta + i * gb_stride`, as
/// gdn_step_norm_multi's commit (n_keep set, no outputs).  Every layer's arithmetic is the per-layer call's, so the result
/// is bitwise the loop of per-layer calls.  Returns false (nothing launched) when the per-layer commit would not take the
/// state-only kernel (STRATA_GDN_SPLIT=1, STRATA_GDN_COMMIT_SPLIT=0, STRATA_DBG_GDN=1): the caller then keeps the loop.
bool gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv, int64_t qkv_stride,
                       const float* h, int64_t h_stride, const float* gate, const float* beta, int64_t gb_stride,
                       int conv_channels, int h_k, int h_v, int n_max, int n_layers, const int32_t* n_keep, void* stream);
/// (xq_out, S26 STRATA_QFUSE: also write the q8_1 image of output rows [t_out_begin, n_tok) there - the bytes
/// native_quantize_q8_1(y + t_out_begin * value_dim, xq_out, value_dim, n_tok - t_out_begin) would write.)
/// Spin until *flag >= value (a mapped host flag).  The value is fixed at capture, so several rings can be
/// outstanding at once (the split verify window keeps two).
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream);
/// wait_flag_ge and copy_i32_from_mapped in ONE launch (143-amdk): thread 0 spins on *flag >= value, the block then
/// copies dst[0, n) = src[0, n) (mapped host memory, volatile reads).  One kernel node fewer per layer: a dependent
/// kernel costs ~2.4 us on a gfx1201 graph whatever it does.  n <= 128 * k for any k (a strided loop).
void wait_copy_i32_from_mapped(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int64_t n,
                               void* stream);

/// Layer split, overlapped hand-off (STRATA_SPLIT_OVERLAP): dst (mapped) = [a (na) | b (nb) | c (nc)] with volatile
/// stores, then the LAST block to finish raises *flag = 1 (mapped) after a system fence, so the next stage's GPU can
/// start (wait_flag_ge(flag, 1)) without the host syncing this stage first.  `counter` is one device word, 0 between
/// calls (the last block resets it).  Volatile because on RDNA a plain store to mapped memory can sit in the L2.
/// `drop` (mapped, may be null): a test hook - nonzero leaves the flag down, as if the publish were lost.
void handoff_publish(float* dst, const float* a, int64_t na, const float* b, int64_t nb, const float* c, int64_t nc,
                     uint32_t* counter, uint32_t* flag, void* stream, const uint32_t* drop = nullptr);
/// The reading stage's hand-off wait, bounded: spin until *flag >= value, or after timeout_ns (0: no bound) set
/// *err = 1 (mapped) and return, so the window's graph completes and the host fails the window cleanly.
void wait_handoff(const uint32_t* flag, uint32_t value, uint32_t* err, unsigned long long timeout_ns, void* stream);
/// wait_flag_ge, also writing the GPU's clock (ns, gpu_stamp's) as the kernel starts into *t_in and as it returns
/// into *t_out (either may be null): the verify window's PCIe-share balance, at no extra launch.
void wait_flag_ge_stamped(const uint32_t* flag, uint32_t value, unsigned long long* t_in, unsigned long long* t_out,
                          void* stream);
/// the GPU's %globaltimer (ns) into buf[i] (a one-thread kernel: the verify window's stage profiler).  Inside a PDL
/// stretch (pdl.hpp) the stamp passes the early launch on, so a profiled window keeps the chain it measures.
void gpu_stamp(unsigned long long* buf, int i, void* stream);
/// an empty one-thread kernel (a graph node that does nothing: STRATA_DUMMY_NODES measures what a node costs in a window)
void gpu_noop(void* stream);
/// dst[r][0, w) = src[r][0, w) for `rows` rows of source stride `src_w` floats (the query half of each q/gate head
/// pair): the strided device-to-device cudaMemcpy2DAsync as a kernel, so the window's chain stays kernel to kernel.
/// w and src_w multiples of 4 floats, both pointers 16-byte aligned.
void copy_rows_strided(float* dst, const float* src, int64_t rows, int64_t w, int64_t src_w, void* stream);

// ---- perf-review E-6: a layer whose routed experts are all in VRAM needs nothing from the host
/// One group's plan, built on the device when every routed expert of its n*k entries is resident: the host pool's
/// layout (counts | start | dst | tok | pad | ptr | ptr2 | start2, `capx` entries) and order (distinct experts in
/// routing order, their entries ascending), no PCIe groups.  *skip = ring when it did, else 0.
/// #871: without `skip` (the all-resident graph) an entry whose expert is not in VRAM cannot be planned: the plan is
/// left empty and *plan_err (mapped host memory) is set to 1, so the host sees it instead of running a stale plan.
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err = nullptr);
/// wait_flag_ge that also returns when *skip == value (device memory).
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream);
/// copy_i32_from_mapped unless *skip == value.
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream);
/// copy_from_mapped, or zeros when *skip == value (n a multiple of 4, 16-byte aligned).
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream);

/// Rows of the S2/S4/S8 embedding for T token ids read from DEVICE memory; out (T, n).  Bitwise `embedding_gather`.
void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream);
/// R[t][c][:] = x[t][:] for the hc streams.
void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// out[:n_embd] = the mean of the hc streams of ONE row R[hc][n_embd] (the DFlash2 drafter's target features; only the
/// STRATA_TAP_FILE hook calls it).
void hc_mean_streams(const float* R, float* out, int64_t n_embd, int hc, void* stream);
/// STRATA_ROUTE_STATS only: acc[r] += the number of ids of row r (k per row, n rows <= 8) that no earlier row of the window holds
/// (duplicates inside a row count once).
void route_marginal_stats(const int32_t* ids, int n, int k, unsigned long long* acc, void* stream);
/// dst[:n] = src[(*index) * stride + :n]  (index read from device memory; a negative index copies nothing).
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream);

/// Plan v0.3 P6: copy *n (device memory) blobs of `blob_bytes` from mapped host memory (src[k], device aliases)
/// into dst + k * blob_bytes with coalesced 16-byte loads - the PCIe share of a layer's missed experts, staged
/// into VRAM before the grouped expert kernel reads them.  Launched for a capacity of `cap` blobs.
void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream);
/// table[pairs[2k]] = pairs[2k + 1] for k < n, `pairs` in mapped host memory: a residency table's changed entries,
/// stream-ordered before the window that reads it, without the copy engine (where the tier's swaps may queue).
void res_patch(int32_t* table, const int32_t* pairs, int n, void* stream);
/// ptr[k] = base + k * blob_bytes for k < *n (the staged copies `fetch_blobs` made).
void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream);

// ---- the MTP draft layer (src/core/mtp.cpp)
/// R[t][c][:] = h[t][c][:] + e[t][:]  (the embedding branch added to every stream).
void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// Every routed expert is resident (slot = expert id): slot[i] = ids[i], dst[i] = i, *count = n.
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
/// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
/// row = *row_dev (device memory).  `out` may be mapped host memory.
void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs = nullptr,
                float* out_p = nullptr);
/// --pipeline-windows, the drafter's chain teacher forced: `*tok = force[j]` when force[j] >= 0 (`force` is mapped
/// host memory, read when the kernel runs), else `*tok` is left as it is.
void force_token(int32_t* tok, const int32_t* force, int j, void* stream);
/// Row *row_dev of a, b and c (a_n, b_n, c_n floats a row) copied to their row 0 (the draft layer's rest runs on
/// row 0).  Graph-capturable: the row is read on the device.
void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream);
/// dst row i = src row ids[i] (row_bytes each, multiple of 16), for n rows.
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream);
/// ids[t] = table[ids[t]] for n entries (a subset index back to a token id).
void map_ids(int32_t* ids, const int32_t* table, int n, void* stream);
/// probs[t] = softmax(logits[t])[ids[t]] for n_rows rows of n_vocab (the probability of each row's argmax).
void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);
/// out[t] = the argmax of row t (n floats a row; ties to the lowest index, NaN never picked, 0 when no value is above
/// -inf): bitwise sample_tokens' greedy pick.  Up to 128 blocks a row scan slices of it, the row's last block merges
/// their picks.  `scratch`: argmax_rows_scratch_bytes(n_rows), zero before the first launch (each launch leaves it
/// so).  Graph-capturable.
uint64_t argmax_rows_scratch_bytes(int n_rows);
void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream);
/// row_top_prob over 8 blocks a row, each computing 4 of its 32 warps' sums, the row's last block adding the 32 in
/// order: bitwise row_top_prob.  `scratch`: row_top_prob_scratch_bytes(n_rows), zero before the first launch.
uint64_t row_top_prob_scratch_bytes(int n_rows);
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream);
/// #879 guard: flags[t * kNonfiniteBlocks + b] = 1 when slice b of row t (n floats a row) holds a NaN or an infinity,
/// else 0, for n_rows rows.  Every word is written on every launch (nothing to clear between windows), so `flags` can
/// be mapped host memory read after the window's own sync.  Graph-capturable; CUDA and HIP.
inline constexpr int kNonfiniteBlocks = 16;
void logits_nonfinite_rows(const float* logits, int n_rows, int n, uint32_t* flags, void* stream);
/// True where row_top_prob_split runs as its own kernel (CUDA; AMD keeps the one-block row_top_prob, so it calls
/// that).  STRATA_MULTI_BLOCK_ARGMAX=0: off.
bool multi_block_head_ops();
/// True where the verify head's and the draft's greedy pick should be argmax_rows: sm_80 to sm_89 (the one-block
/// kernel is slow there; sm_90+ has sample_tokens' cluster kernel, other cards keep sample_tokens).
bool argmax_rows_wanted();
/// Dense-attention step records for `n` cells: [cell, cell+1, (cell+1)/4, cell+1] from cells[i] (device memory).
void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream);
/// A sliding attention window: for `n` step records (kStepCount ints each, n_kv at [1]) the selection becomes the
/// last `window` cells: ids[q * ids_stride + j] = max(0, n_kv - window) + j and the record's width = the count.
void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream);
/// window_ids whose window starts no lower than the cell `*floor_cell` (device-visible memory, read when the kernel
/// runs; KVA: the first draft cell built from a real row).  `window` may be INT_MAX for "every cell".  A floor of 0
/// selects exactly what window_ids selects.  CUDA / HIP only.
void window_ids_floor(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, const int32_t* floor_cell,
                      void* stream);

}  // namespace strata::kernels
