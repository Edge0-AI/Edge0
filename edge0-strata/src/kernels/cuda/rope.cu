// src/kernels/cuda/rope.cu - P2.S2: NEOX partial RoPE.
//
// Semantics from `ref/qsa.py` (`rope_freqs` + `rope_neox`), which `bench/micro/rope_xcheck` validates against
// the oracle:
//
//     inv[i] = theta ** (-2i / n_rot)          i in [0, n_rot/2)
//     ang    = pos * inv
//     cos, sin = cos(ang), sin(ang)
//     out[:half]     = a*cos - b*sin            a = x[:half], b = x[half:n_rot]
//     out[half:n_rot] = a*sin + b*cos
//
// THE PAIRING IS NEOX - (i, i + n_rot/2), NOT the adjacent pair (2i, 2i+1).  `ggml.h` writes NEOX
// n_dims=8 as [ccccssss]; the adjacent-pair ("NORMAL") ordering is the other common convention and choosing
// wrongly rotates the wrong dimensions together, which produces correctly-shaped output with scrambled
// content.  It is also PARTIAL: only the first n_rot of head_dim are touched, here 64 of 256.
//
// WHY THE TABLE IS BUILT ON THE HOST.  The reference computes the frequencies in FLOAT64 (`theta ** (...)`
// on a float64 arange, cos/sin in float64).  Reproducing that on device means double-precision pow and cos,
// which are slow on a consumer GPU and - more to the point - not guaranteed to agree with the host's libm.
// Building the table once on the host and passing it in makes the TABLE exactly the reference's values and
// leaves the kernel a pure rotation, so each half can be checked tightly instead of both halves sharing one
// loose tolerance.  The engine wants a cached table anyway: it is per (n_rot, theta, position), not per token.
//
// THE SCALED TABLE IS THE SAME IDEA WITH MORE MATH IN THE HOST LOOP.  Linear rescales the angle by
// `freq_scale`; YaRN blends extrapolation and interpolation along the pairs (ggml's `rope_yarn`) and folds
// the mscale magnitude correction into the same cos/sin values.  All of it float64, in the reference's
// order, so `rope_parity` can hold every variant to a bit-exact float64 transcription of the same spec.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
// The process's rope config (rope_scaling.hpp).  One writer - the engine's startup thread, before
// session_init builds any table or captures any graph - and readers after it.
RopeScaling g_rope_scaling;
}  // namespace

void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);   // the original loop, verbatim
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;   // ggml: the correction rides on ext_factor, not the type
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;    // the trained angle, ggml's theta_extrap
            const double interp = fs * extrap;         // ggml's theta_interp
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

namespace {

// ONE THREAD PER ROW, not per (row, pair).  The first version had one thread per pair and did the tail copy
// with 	hreadIdx.x strided loops - so up to 32 threads wrote into the SAME row, with no barrier between the
// copy and the rotation, and a thread's copy of orow[i] could land AFTER another thread had rotated it.  A
// race that silently writes the unrotated value is worse than a wrong answer: it is intermittent.
//
// A row is head_dim floats (256 here) and there are batch x heads of them, so one thread per row is a
// handful of threads for a token and removes the ordering question entirely rather than synchronising it.
__global__ void rope_neox_kernel(const float* __restrict__ x, float* __restrict__ out, long long rows,
                                 int head_dim, int n_rot, const float* __restrict__ cos_tab,
                                 const float* __restrict__ sin_tab, const int* __restrict__ pos,
                                 const int32_t* __restrict__ mtab) {
    const long long r = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int half = n_rot / 2;
    const float* xr = x + r * head_dim;
    float* orow = out + r * head_dim;

    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];      // the PARTIAL rotation's untouched tail
    for (int i = 0; i < half; ++i) {
        const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;   // the image path's per-pair position
        // THE PAIRING AND THE SIGNS COME FROM `rope_neox_pair`, shared with the indexer's pooling kernel.
        rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
    }
}

}  // namespace

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    const int threads = 128;
    const unsigned grid = (unsigned) ((rows + threads - 1) / threads);
    rope_neox_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, out, rows, head_dim, n_rot, cos_tab,
                                                                 sin_tab, pos, mrope_table());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rope_neox_apply launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
