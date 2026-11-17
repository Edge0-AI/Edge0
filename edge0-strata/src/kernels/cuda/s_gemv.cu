// src/kernels/cuda/s_gemv.cu - P2.S2: the S-family GEMV, one thread per output row.
//
// Naive per the phase rule: dequantize on the fly, FP32 accumulation inside the row, no shared memory, no
// vector loads.  The kernel is templated on the code WIDTH (a compile-time property of the unpacking loop) and
// takes the rest of the per-type attributes as run-time arguments, because the manifest supplies them per
// tensor and a 13-way switch inside the inner loop would be the naive-but-wrong kind of naive.
#include "strata/kernels/s_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// `kvalues_iq4nl`: the non-linear codebook, verbatim from ggml-common.h / ggml-quants.c.
//
// **IT WAS IN `__constant__` FOR THE WRONG REASON, AND IT COST 2.12x.**  The old comment justified constant
// memory with "every thread reads it and it is the same 16 values for the whole launch".  That reasoning is
// exactly backwards: constant memory is fast when the access is UNIFORM - every lane reading the SAME address -
// and a NON-LINEAR codebook means every lane reads a DIFFERENT index, which is the pattern it handles worst.
//
// Measured on `attn_qkv` (IQ4_XS, 10240 x 2560), forcing the Affine codebook so that the memory traffic and the
// instruction count stay identical and only the table read is removed:
//
//     codebook probe: IQ4NL 0.2032 ms   Affine 0.0960 ms   ratio 2.12x
//
// `decode_tbl` below reads a SHARED copy instead: 16 bytes loaded once per block, then an index into shared
// memory that costs at most a four-way bank conflict.  The `__constant__` array stays as the source of it.
__constant__ signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                       1,   13,   25,  38,  53,  69,  89,  113};

template <int CODE_BITS>
__device__ __forceinline__ float decode(int code, int bias, int codebook) {
    if (codebook == (int) Codebook::Iq4Nl) return (float) kIq4Nl[code & 0x0F];
    return (float) (code + bias);       // the bias is applied to the CODE, in the integer domain
}

/// The same decode with the codebook in SHARED memory, which is what the hot kernels use - see the measurement
/// above.  A divergent read of a 16-entry table is the one access pattern constant memory is worst at.
template <int CODE_BITS>
__device__ __forceinline__ float decode_tbl(int code, int bias, int codebook, const signed char* tbl) {
    if (codebook == (int) Codebook::Iq4Nl) return (float) tbl[code & 0x0F];
    return (float) (code + bias);
}

// ---- Q8_K ACTIVATIONS ----------------------------------------------------------------------------------
//
// 292 bytes per 256 elements: `{ f32 d ; int8_t qs[256] ; int16_t bsums[16] }`.  `bsums` is not read - it
// exists for ggml's AVX2 dot product - but the STRIDE includes it, so a buffer from `quantize_q8_K` passes
// straight in.
//
// The block's `d` sits at a 4-byte-aligned offset from the buffer base as long as the base is aligned, which
// `cudaMalloc` guarantees.
constexpr int Q8K_BLOCK_BYTES = 292;
constexpr int Q8K_BLOCK_ELEMS = 256;

__device__ __forceinline__ float q8k_at(const uint8_t* __restrict__ x, long long i) {
    const uint8_t* blk = x + (i / Q8K_BLOCK_ELEMS) * Q8K_BLOCK_BYTES;
    const float d = __ldg((const float*) blk);
    const int8_t q = ((const int8_t*) (blk + 4))[i % Q8K_BLOCK_ELEMS];
    return d * (float) q;
}

/// **THE Q8_0 ACTIVATION**, which is what the LEGACY block formats want: `block_q8_0` is `{fp16 d; int8 qs[32]}`,
/// 34 bytes per 32 elements, produced by `quantize_q8_0`.
///
/// THE ENGINE HAD NO GEMV FOR THIS, and it was the last activation-contract gap (LEDGER L54).  It was described
/// as "Q8_K is not implemented", which is not the situation: `ffn_down_shexp` is IQ4_NL/Q4_0/Q5_0/Q8_0 in EVERY
/// layer, whose `vec_dot_type` is Q8_0 - and its `n_in` is 640, which is not a multiple of 256, so **Q8_K is
/// STRUCTURALLY IMPOSSIBLE for it rather than merely absent**.  Every other legacy tensor in the pack is Q2_0,
/// whose attributes happen to be S2's, which is how the S2-only kernel covered them and the hole stayed hidden.
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_0_BLOCK_ELEMS = 32;

__device__ __forceinline__ float q8_0_at(const uint8_t* __restrict__ x, long long i) {
    const uint8_t* blk = x + (i / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    // a `uint16_t` needs 2-byte alignment and the stride is 34, so every block's `d` is aligned
    const float d = __half2float(__ushort_as_half(*(const uint16_t*) blk));
    const int8_t q = ((const int8_t*) (blk + 2))[i % Q8_0_BLOCK_ELEMS];
    return d * (float) q;
}

/// The Q8_K path, with the SAME weight decode as `s_gemv_kernel` and a different activation loader.  The two
/// are separate kernels rather than one templated on the activation kind because the activation POINTER TYPE
/// differs (`uint16_t*` vs `uint8_t*`) and a templated pointer would need a cast at every call site - which is
/// where a wrong stride would hide.
template <int CODE_BITS>
__global__ void s_gemv_q8k_kernel(const uint8_t* __restrict__ x, const uint8_t* __restrict__ codes,
                                  const float* __restrict__ scales, const float* __restrict__ offset,
                                  float* __restrict__ y, long long n_in, long long n_out, int bias,
                                  int codebook, int group_elems, int has_offset) {
    // THE CODEBOOK, IN SHARED MEMORY - 16 bytes, loaded once per block.  See the note on `kIq4Nl`.
    // THE LOAD AND THE BARRIER COME BEFORE THE EARLY RETURN: `o` is per-THREAD in this kernel, so a thread
    // that returned first would leave the others waiting at a barrier it never reaches.
    __shared__ signed char s_iq4nl[16];
    if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = kIq4Nl[threadIdx.x];
    __syncthreads();
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc = 0.0f;
    for (long long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const long long base = g * (long long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            // The offset belongs to the WEIGHT and is applied before the activation multiply - see the note in
            // `s_gemv_kernel`, where writing it the other way rounded differently and only the Q4_K case could
            // tell.
            const float w = decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * d + b;
            acc += w * q8k_at(x, i);
        }
    }
    y[o] = acc;
}

/// One warp per output row, lanes striding the reduction axis: for a fixed `i` consecutive lanes read
/// consecutive activations, which is the coalescing that matters once the output width is small.
///
/// **`group_shift`, NOT `group_elems`, AND IT COSTS A DIVISION PER ELEMENT IF YOU GET IT WRONG.**  The
/// sibling fp16 kernel was given this treatment ("a 64-bit integer division on this hardware is a long
/// instruction sequence - one per element, in the hottest loop in the engine") and THIS one was not: it kept
/// `i / group_elems` with `group_elems` a RUNTIME value, once per element, in the kernel that carries most of
/// the model's weight bytes.  Every group size the format defines is a power of two, so the divisor is passed
/// as its logarithm and the division becomes a shift; the host refuses a non-power-of-two group rather than
/// computing a wrong index.
/// ONE KERNEL FOR BOTH QUANTIZED ACTIVATIONS, templated on which block format `x` holds.  A second copy with a
/// different loader would be a second thing to get wrong, and the only difference IS the loader - the weight
/// decode, the quad loop, the shared codebook and the reduction are identical.
template <int CODE_BITS, bool Q8K>
__global__ void s_gemv_q8_split_kernel(const uint8_t* __restrict__ x, const uint8_t* __restrict__ codes,
                                        const float* __restrict__ scales, const float* __restrict__ offset,
                                        float* __restrict__ y, long long n_in, long long n_out, int bias,
                                        int codebook, int group_shift, int has_offset) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long o = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    // THE CODEBOOK, IN SHARED MEMORY.  16 bytes, loaded once per block by the first 16 threads, instead of a
    // divergent constant read per element - measured at 2.12x of this whole kernel (see the note on `kIq4Nl`).
    //
    // THE LOAD AND THE BARRIER COME BEFORE THE EARLY RETURN, as in `s_gemv_q8k_kernel`.  The return used to
    // sit above them, so whenever `n_out % 8 != 0` the last block's surplus warps left before a barrier the
    // others still waited at - undefined behaviour that every shape in the pack (a multiple of 8) happened to
    // avoid.  `o` is per-WARP, so the return below keeps whole warps together and the shuffle reduction's
    // full mask stays valid.
    __shared__ signed char s_iq4nl[16];
    if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = kIq4Nl[threadIdx.x];
    __syncthreads();
    if (o >= n_out) return;

    const long long n_groups = n_in >> group_shift;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
    // ---- EIGHT CONSECUTIVE ELEMENTS PER LANE, from TWO code loads and ONE scale.
    //
    // **THIS IS THE EXPERIMENT THE `#pragma unroll` NOTE BELOW LEFT OPEN**, and the note said what it would
    // be: "Widening QE to 8 or 16 changes the LOADS THEMSELVES - two or four independent 16-bit words per
    // lane - which an unroll cannot do because the extraction code is written for exactly four codes."  So a
    // lane now takes a double quad: two code words issued before either is consumed, which is what raises the
    // bytes in flight.  The scale and offset are still ONE load, because eight consecutive elements lie in one
    // group whenever `group_elems % 8 == 0` - the launcher asserts that, and every group size this format
    // defines (16, 32, 64) satisfies it.
    //
    // The loop used to take `i = lane, lane+32, ...` - a STRIDE, chosen so consecutive lanes read consecutive
    // activations - and paid a code load, a shift-and-mask, and a scale load FOR EVERY ELEMENT.  But four
    // consecutive elements share one code word (1, 2 or 4 bytes depending on width) and lie in one group,
    // because every group size this format defines is a multiple of four.  So a lane takes a QUAD:
    //
    //     one code load        instead of four
    //     one scale load       instead of four
    //     one shift per element instead of a divide, a modulo and a shift
    //
    // THE BIT ORDER IS THE SAME ONE, verified for all three widths: element `i+k` is bits
    // `[k*CODE_BITS, (k+1)*CODE_BITS)` of the little-endian word at byte `i/PER_BYTE`, which is what the old
    // `(c[e/PER_BYTE] >> ((e%PER_BYTE)*CODE_BITS)) & MASK` computed one element at a time.
    //
    // THE SUMMATION ORDER CHANGES, which is why `s_gemv_parity` and `s_gemv_q8k_parity` carry tolerances and
    // compare against the naive kernel - they are the guard for this, and they are the reason it is safe to do.
    constexpr int QE = 16;
    // BYTES PER FOUR-CODE WORD, not per octet.  The two are easy to confuse and the first version of this
    // did: it branched on the octet's total width and so read a uint32 for S4 (whose four codes are a
    // uint16) and single bytes for S8 (whose four codes are a uint32).  `s_gemv_q8k_parity` failed on
    // exactly that, which is what it is for.
    constexpr int QW = 4 * CODE_BITS / 8;              // 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    long long i = (long long) lane * QE;
    // **`#pragma unroll 4` WAS TRIED HERE AND DID NOTHING.**  Measured: 184.6 -> 183.9 GB/s at n_out 10240,
    // with `s_gemv_parity` and `s_gemv_q8k_parity` both still clean.  The hypothesis it was testing is L93 -
    // that this kernel's flat ~6 ns per output row is a bytes-in-flight limit, one code load and one scale
    // load per lane-iteration with only 20 iterations.  A 4x unroll issues four of each before the first is
    // consumed, so either the compiler was already doing it or **the limit is not instruction scheduling.**
    // Removed rather than kept, because a hint that does not change the timing is a claim that is not true.
    //
    // What it does NOT rule out is the structural version of the same idea: QE is 4, so a lane consumes ONE
    // 2-byte code word per iteration.  Widening QE to 8 or 16 changes the LOADS THEMSELVES - two or four
    // independent 16-bit words per lane - which an unroll cannot do because the extraction code is written for
    // exactly four codes.  That is the experiment that would settle it, and it is not this one.
    for (; i + QE <= n_in; i += 32 * QE) {
        const long long g = i >> group_shift;          // the whole octet is in one group: group_elems % 8 == 0
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const uint8_t* cp = c + i / PER_BYTE;
        unsigned v, v2, v3, v4;
        if (QW == 1) { v = cp[0]; v2 = cp[1]; v3 = cp[2]; v4 = cp[3]; }
        else if (QW == 2) { v = *(const uint16_t*) cp; v2 = *(const uint16_t*) (cp + 2);
                            v3 = *(const uint16_t*) (cp + 4); v4 = *(const uint16_t*) (cp + 6); }
        else { v = *(const uint32_t*) cp; v2 = *(const uint32_t*) (cp + 4);
               v3 = *(const uint32_t*) (cp + 8); v4 = *(const uint32_t*) (cp + 12); }
        // ---- THE ACTIVATION BLOCK IS HOISTED OUT OF THE SIXTEEN ELEMENTS, AND THAT WAS THE KERNEL'S COST.
        //
        // `q8k_at(x, i + k)` computes `(i + k) / 256` and `(i + k) % 256` on EVERY call: a 64-bit integer
        // division AND modulo per element, sixteen times per lane-iteration.  A 64-bit division on a GPU is a
        // software routine of roughly a hundred cycles, and the compiler cannot strength-reduce it because it
        // cannot prove `(i + k) / 256 == i / 256` without knowing `i % 256`.
        //
        // But `QE` is 16, `Q8K_BLOCK_ELEMS` is 256 and `Q8_0_BLOCK_ELEMS` is 32, and the loop advances `i` by
        // `32 * QE`.  So `i % 256` and `i % 32` are ALWAYS multiples of 16 - which means all QE elements of one
        // lane-iteration lie inside ONE block, for either activation kind.  One division and one modulo per
        // iteration replace sixteen of each, and the sixteen activations become sixteen consecutive int8 loads
        // from a pointer computed once.
        //
        // This is exact, not an approximation: `xi + QE <= blk_elems` holds for every reachable `xi`, so no
        // lane-iteration can straddle a block boundary.  The tail loop below still uses the accessors because
        // its elements do not share a block, and it runs at most once per lane.
        const int blk_elems = Q8K ? Q8K_BLOCK_ELEMS : Q8_0_BLOCK_ELEMS;
        const uint8_t* xb = x + (i / blk_elems) * (Q8K ? Q8K_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
        const int xi = (int) (i % blk_elems);
        const float xd = Q8K ? __ldg((const float*) xb)
                             : __half2float(__ushort_as_half(*(const uint16_t*) xb));
        const int8_t* xq = (const int8_t*) (xb + (Q8K ? 4 : 2)) + xi;

        const float w0 = decode_tbl<CODE_BITS>((int) (v & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w1 = decode_tbl<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w2 = decode_tbl<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w3 = decode_tbl<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w4 = decode_tbl<CODE_BITS>((int) (v2 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w5 = decode_tbl<CODE_BITS>((int) ((v2 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w6 = decode_tbl<CODE_BITS>((int) ((v2 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w7 = decode_tbl<CODE_BITS>((int) ((v2 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        acc0 += w0 * (xd * (float) xq[0]);
        acc1 += w1 * (xd * (float) xq[1]);
        acc2 += w2 * (xd * (float) xq[2]);
        acc3 += w3 * (xd * (float) xq[3]);
        acc4 += w4 * (xd * (float) xq[4]);
        acc5 += w5 * (xd * (float) xq[5]);
        acc6 += w6 * (xd * (float) xq[6]);
        acc7 += w7 * (xd * (float) xq[7]);
        const float w8  = decode_tbl<CODE_BITS>((int) (v3 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w9  = decode_tbl<CODE_BITS>((int) ((v3 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w10 = decode_tbl<CODE_BITS>((int) ((v3 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w11 = decode_tbl<CODE_BITS>((int) ((v3 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w12 = decode_tbl<CODE_BITS>((int) (v4 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w13 = decode_tbl<CODE_BITS>((int) ((v4 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w14 = decode_tbl<CODE_BITS>((int) ((v4 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w15 = decode_tbl<CODE_BITS>((int) ((v4 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        acc8  += w8  * (xd * (float) xq[8]);
        acc9  += w9  * (xd * (float) xq[9]);
        acc10 += w10 * (xd * (float) xq[10]);
        acc11 += w11 * (xd * (float) xq[11]);
        acc12 += w12 * (xd * (float) xq[12]);
        acc13 += w13 * (xd * (float) xq[13]);
        acc14 += w14 * (xd * (float) xq[14]);
        acc15 += w15 * (xd * (float) xq[15]);
    }
    // the last, partial quad - at most one per lane
    for (; i < n_in; i += 32 * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long long e = i + k;
            const long long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * s[g] + (off ? off[g] : 0.0f)) * (Q8K ? q8k_at(x, e) : q8_0_at(x, e));
        }
    }
    float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
                (((acc8 + acc9) + (acc10 + acc11)) + ((acc12 + acc13) + (acc14 + acc15)));
    for (int step = 16; step > 0; step >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, step);
    if (lane == 0) y[o] = acc;
}

template <int CODE_BITS>
__global__ void s_gemv_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                              const float* __restrict__ scales, const float* __restrict__ offset,
                              float* __restrict__ y, long long n_in, long long n_out, int bias, int codebook,
                              int group_elems, int has_offset) {
    // THE CODEBOOK, IN SHARED MEMORY - 16 bytes, loaded once per block.  See the note on `kIq4Nl`.
    // THE LOAD AND THE BARRIER COME BEFORE THE EARLY RETURN: `o` is per-THREAD in this kernel, so a thread
    // that returned first would leave the others waiting at a barrier it never reaches.
    __shared__ signed char s_iq4nl[16];
    if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = kIq4Nl[threadIdx.x];
    __syncthreads();
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc = 0.0f;
    for (long long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const long long base = g * (long long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            // The offset belongs to the WEIGHT, so it is applied to the decoded value BEFORE the activation
            // multiply: `w = code*scale + offset; acc += w * x`.  Writing `acc += code*scale*x + offset*x`
            // computes a mathematically equal expression that ROUNDS DIFFERENTLY, because it performs two
            // multiplications and an addition where this performs one of each.  The no-offset types cannot
            // tell the difference - which is why this was wrong until the Q4_K case was added.
            const float w = decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * d + b;
            acc += w * __half2float(__ushort_as_half(x[i]));
        }
    }
    y[o] = acc;
}

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "s_gemv: n_in %lld is not a multiple of group_elems %d\n", (long long) n_in,
                     form.group_elems);
        std::exit(1);
    }
    if (form.has_offset && offset == nullptr) {
        std::fprintf(stderr, "s_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    const int cb = (int) form.codebook;
    switch (form.code_bits) {
        case 2:
            s_gemv_kernel<2><<<(unsigned) blocks, threads>>>(x, codes, scales, offset, y, n_in, n_out,
                                                             form.code_bias, cb, form.group_elems,
                                                             form.has_offset ? 1 : 0);
            break;
        case 4:
            s_gemv_kernel<4><<<(unsigned) blocks, threads>>>(x, codes, scales, offset, y, n_in, n_out,
                                                             form.code_bias, cb, form.group_elems,
                                                             form.has_offset ? 1 : 0);
            break;
        case 8:
            s_gemv_kernel<8><<<(unsigned) blocks, threads>>>(x, codes, scales, offset, y, n_in, n_out,
                                                             form.code_bias, cb, form.group_elems,
                                                             form.has_offset ? 1 : 0);
            break;
        default:
            std::fprintf(stderr, "s_gemv: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s_gemv: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

namespace {

// One BLOCK per output row, `threads_per_row` threads splitting it.  Parallelism becomes
// n_out * threads_per_row instead of n_out, which is the whole point: gate/up is [2560 x 640] and 640 threads
// cannot fill 48 SMs.
//
// The reduction is a fixed tree over shared memory rather than atomics, so the result is DETERMINISTIC for a
// given threads_per_row - a race would make the parity test flaky rather than wrong, which is the worst kind
// of failing test.
// NOTE ON `g = e >> group_shift`.  The group index was `e / group_elems` with `group_elems` a RUNTIME value,
// and a 64-bit integer division on this hardware is a long instruction sequence - one per element, in the
// hottest loop in the engine.  Every group size this format defines is a POWER OF TWO (16, 32, 64), so the
// divisor is passed as its logarithm and the division becomes a shift.  The host wrapper refuses a
// non-power-of-two group rather than silently computing a wrong index.
template <int CODE_BITS>
__global__ void s_gemv_split_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                                    const float* __restrict__ scales, const float* __restrict__ offset,
                                    float* __restrict__ y, long long n_in, long long n_out, int bias,
                                    int codebook, int group_elems, int group_shift, int has_offset,
                                    int threads_per_row) {
    extern __shared__ float partial[];
    // THE CODEBOOK, IN SHARED MEMORY - 16 bytes, loaded once per block.  See the note on `kIq4Nl`.
    // THE LOAD AND THE BARRIER COME BEFORE THE EARLY RETURN: `o` is per-THREAD in this kernel, so a thread
    // that returned first would leave the others waiting at a barrier it never reaches.
    __shared__ signed char s_iq4nl[16];
    if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = kIq4Nl[threadIdx.x];
    __syncthreads();
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const uint8_t* c = codes + o * (n_in / PER_BYTE);
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    // FOUR ACCUMULATORS, NOT ONE.  L2 measured this kernel at 0.4% of FP32 peak and 2.1% of bandwidth, so
    // neither limit is the constraint - the constraint is the DEPENDENCY CHAIN: a single `acc += ...` serialises
    // every FMA behind the previous one's latency (4-6 cycles each, ~1 per 4 cycles achieved), so the SM sits
    // idle.  Four independent accumulators let the scheduler keep four FMAs in flight, and the loop is
    // unrolled by four so the four chains are interleaved.  The summation ORDER changes, which is why the
    // parity test carries a tolerance and why this kernel's output is checked against the naive one.
    // FOUR CONSECUTIVE ELEMENTS PER THREAD, from ONE code load and ONE scale - the same transformation the
    // Q8_K kernel got, for the same reason and with the same guard (`s_gemv_parity`).  See the long note there
    // for the bit order, which is unchanged, and for why the group size is passed as a shift.
    //
    // THIS KERNEL IS THE ONE `attn_output` AND `shared_expert` USE, so the fix has to exist in both or half the
    // model keeps the old instruction count.
    constexpr int QE = 4;
    constexpr int QB = QE * CODE_BITS / 8;             // 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    long long i = (long long) tid * QE;
    for (; i + QE <= n_in; i += (long long) threads_per_row * QE) {
        const long long g = i >> group_shift;
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const uint8_t* cp = c + i / PER_BYTE;
        unsigned v;
        if (QB == 1) v = cp[0];
        else if (QB == 2) v = *(const uint16_t*) cp;
        else v = *(const uint32_t*) cp;
        // four halves in ONE 8-byte load; `i` is a multiple of four, so `x + i` is 8-byte aligned
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(x + i));
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(x + i + 2));
        acc0 += (decode_tbl<CODE_BITS>((int) (v & MASK), bias, codebook, s_iq4nl) * d + b) * f01.x;
        acc1 += (decode_tbl<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b) * f01.y;
        acc2 += (decode_tbl<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b) * f23.x;
        acc3 += (decode_tbl<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b) * f23.y;
    }
    // the last, partial quad - at most one per thread
    for (; i < n_in; i += (long long) threads_per_row * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long long e = i + k;
            const long long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * s[g] + (off ? off[g] : 0.0f)) *
                    __half2float(__ushort_as_half(x[e]));
        }
    }
    partial[tid] = (acc0 + acc1) + (acc2 + acc3);
    __syncthreads();
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) y[o] = partial[0];
}

}  // namespace

static void s_gemv_split_impl(const uint16_t* x, const uint8_t* codes, const float* scales,
                               const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form,
                               int threads_per_row, cudaStream_t stream, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 || threads_per_row > 1024) {
        std::fprintf(stderr, "s_gemv_split: threads_per_row must be a power of two in 1..1024, got %d\n",
                     threads_per_row);
        std::exit(1);
    }
    const int cb = (int) form.codebook;
    // Every group size this format defines is a power of two, which is what lets the per-element group index
    // be a shift.  Refuse anything else rather than compute a wrong index quietly.
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP: four consecutive elements share a scale and the kernel reads
    // exactly one.  Every group size this format defines is 16, 32 or 64, so this cannot fire on a real pack.
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems %d is not a multiple of 4\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    const size_t smem = (size_t) threads_per_row * sizeof(float);
    const unsigned grid = (unsigned) n_out;
    switch (form.code_bits) {
        case 2:
            s_gemv_split_kernel<2><<<grid, threads_per_row, smem, stream>>>(
                x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems,
                group_shift, form.has_offset ? 1 : 0, threads_per_row);
            break;
        case 4:
            s_gemv_split_kernel<4><<<grid, threads_per_row, smem, stream>>>(
                x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems,
                group_shift, form.has_offset ? 1 : 0, threads_per_row);
            break;
        case 8:
            s_gemv_split_kernel<8><<<grid, threads_per_row, smem, stream>>>(
                x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems,
                group_shift, form.has_offset ? 1 : 0, threads_per_row);
            break;
        default:
            std::fprintf(stderr, "s_gemv_split: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    if (sync) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "s_gemv_split: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, (cudaStream_t) 0, true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, (cudaStream_t) stream,
                      false);
}

// ---- the Q8_K entry points -----------------------------------------------------------------------------

namespace {

bool q8k_form_ok(const SForm& form, int64_t n_in, const char* who) {
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of group_elems %d\n", who, (long long) n_in,
                     form.group_elems);
        return false;
    }
    if (n_in % Q8K_BLOCK_ELEMS != 0) {
        // `quantize_q8_K` requires this too, and a partial block would read past the end of the activation.
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of the Q8_K block %d\n", who, (long long) n_in,
                     Q8K_BLOCK_ELEMS);
        return false;
    }
    return true;
}

void finish(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream != nullptr) return;
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", who, cudaGetErrorString(s));
        std::exit(1);
    }
}

}  // namespace

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k")) std::exit(1);
    const int threads = 128;
    const unsigned blocks = (unsigned) ((n_out + threads - 1) / threads);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    switch (form.code_bits) {
        case 4:
            s_gemv_q8k_kernel<4><<<blocks, threads, 0, (cudaStream_t) stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho);
            break;
        case 8:
            s_gemv_q8k_kernel<8><<<blocks, threads, 0, (cudaStream_t) stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho);
            break;
        default:
            // 2-bit S2 never has a Q8_K activation: its `vec_dot_type` is Q8_0.  Refusing is better than
            // running a kernel that would be numerically wrong in a way the caller cannot see.
            std::fprintf(stderr, "s_gemv_q8k: code_bits %d has no Q8_K contract "
                                 "(S2 uses Q8_0; see docs/activation-contract.md)\n", form.code_bits);
            std::exit(1);
    }
    finish("s_gemv_q8k", stream);
}

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k_split")) std::exit(1);
    // THE GROUP SIZE IS PASSED AS ITS LOGARITHM: see the note on the kernel.  A non-power-of-two group would
    // make the shift a WRONG INDEX rather than a slow one, so it is refused here.
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    if ((1 << group_shift) != form.group_elems) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a power of two\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP.  Four consecutive elements share a scale, so a group smaller
    // than four would need two of them and the kernel reads exactly one.  Every group size this format
    // defines is 16, 32 or 64, so this cannot fire on a real pack - it is here because a future one could.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    const int threads = 256;
    const int warps = threads / 32;
    const unsigned blocks = (unsigned) ((n_out + warps - 1) / warps);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    switch (form.code_bits) {
        case 4:
            s_gemv_q8_split_kernel<4, true><<<blocks, threads, 0, (cudaStream_t) stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, group_shift, ho);
            break;
        case 8:
            s_gemv_q8_split_kernel<8, true><<<blocks, threads, 0, (cudaStream_t) stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, group_shift, ho);
            break;
        default:
            std::fprintf(stderr, "s_gemv_q8k_split: code_bits %d has no Q8_K contract\n", form.code_bits);
            std::exit(1);
    }
    finish("s_gemv_q8k_split", stream);
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // THE ACTIVATION IS `block_q8_0`, 32 elements per block, so `n_in` must be a multiple of 32 - which is a
    // WEAKER requirement than Q8_K's 256 and is exactly why this kernel has to exist: `ffn_down_shexp` has
    // n_in = 640, a multiple of 32 that is NOT a multiple of 256.
    if (n_in % Q8_0_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: n_in %lld is not a multiple of %d\n", (long long) n_in,
                     Q8_0_BLOCK_ELEMS);
        std::exit(1);
    }
    // the same two checks `s_gemv_q8k_split` makes, for the same reasons - see the notes there
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems must be a power of two, got %d\n",
                     form.group_elems);
        std::exit(1);
    }
    // SIXTEEN, NOT FOUR: a lane-iteration takes QE = 16 consecutive elements under ONE scale (see the kernel),
    // so a group of 4 or 8 would read the wrong scale for most of them.  The Q8_K launcher already said 16.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;

    const int threads = 256;
    const int warps = threads / 32;
    const unsigned blocks = (unsigned) ((n_out + warps - 1) / warps);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    switch (form.code_bits) {
        case 4:
            s_gemv_q8_split_kernel<4, false><<<blocks, threads, 0, (cudaStream_t) stream>>>(
                x_q8_0, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, group_shift, ho);
            break;
        case 8:
            s_gemv_q8_split_kernel<8, false><<<blocks, threads, 0, (cudaStream_t) stream>>>(
                x_q8_0, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, group_shift, ho);
            break;
        default:
            std::fprintf(stderr, "s_gemv_q8_0_split: code_bits %d has no Q8_0 contract\n", form.code_bits);
            std::exit(1);
    }
    finish("s_gemv_q8_0_split", stream);
}

}  // namespace strata::kernels
