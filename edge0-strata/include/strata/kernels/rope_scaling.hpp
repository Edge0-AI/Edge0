// include/strata/kernels/rope_scaling.hpp - rope scaling: extending the context past the trained one.
//
// WHAT IT IS.  The artifact's RoPE is trained over `orig_ctx` positions (the artifact ships no
// `rope.scaling` keys, so this is a pure runtime feature).  Scaling re-uses positions past that by
// rescaling the rotation angles.  Three types, the llama.cpp set:
//
//   none    freq_scale 1, no correction                      - today's behaviour, exactly
//   linear  Position Interpolation: angles shrunk by 1/factor - every position fits again, all
//           frequencies compressed uniformly
//   yarn    YaRN: high-frequency pairs keep their trained angles (extrapolate), low-frequency
//           pairs are interpolated, and the pairs in between blend - plus a magnitude correction
//           `mscale` that keeps the attention temperature where training put it
//
// THE MATH IS GGML'S, TRANSCRIBED, NOT RE-DERIVED.  `third_party/llama.cpp` is vendored and is the
// numerical contract (the same standing `native_rope.cu` declares): ramp, correction dims, the
// interpolation mix and the mscale formula below are line-for-line that source, with the MIT
// attribution at each helper.  The one deliberate deviation is bookkeeping: ggml's device kernel
// speaks in even dim indices (`i0`) and halves them inside the ramp, while every Strata rope kernel
// speaks in PAIR indices - so `rope_yarn_ramp` here takes the pair index and compares it against
// `corr_dims` directly.  Same numbers, one less division per call site.
//
// HOW IT REACHES THE ROPE.  Two forms, one config:
//
//   - the TABLE path (the default decode path, the indexer's pooled key) never changes its kernels:
//     scaling is baked into the cos/sin contents by `build_rope_table`, which the parity test checks
//     bit-exactly against a float64 reference;
//   - the ANALYTIC paths (`--native-rope`, the native indexer, prefill) get the resolved constants
//     as kernel arguments.
//
// **PROCESS CONSTANTS, SET ONCE BEFORE CAPTURE.**  A CUDA graph bakes kernel ARGUMENTS at capture
// time, so the scaling must be fixed before `session_init` captures the layers - which is exactly
// the constraint the per-token values already respect by living in device memory (`st.pos_dev`).
// `rope_scaling_set()` is called once from the engine's startup, after CLI and GGUF defaults meet,
// before any graph exists.  There is deliberately no way to change it mid-process: the K already in
// the cache is stored POST-RoPE, so a mid-run change would mix two scalings in one cache.
#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_ROPE_SCALING_HD __host__ __device__
#else
#define STRATA_ROPE_SCALING_HD
#endif

namespace strata::kernels {

enum class RopeScalingType { None, Linear, YaRN };

/// The float32 constants the ANALYTIC rotation kernels take (`rope_scaled_angle`'s arguments).  The defaults are
/// the identity: freq_scale 1, no correction, magnitude 1 - with them the helper computes `cosf(theta) * 1.0f`
/// on `1.0f * theta`, which is exactly today's unscaled `cosf(theta)` bit for bit.
struct RopeKernelArgs {
    float freq_scale = 1.0f;
    float corr_low = 0.0f;
    float corr_high = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
};

/// The one rope configuration, as the CLI flag family and the (future) GGUF rope keys resolve to.
///
/// Default-constructed this is EXACTLY today's engine: freq_scale 1, mscale 1, no correction.  That
/// identity is a regression gate, not a coincidence - the parity test compares the default's table
/// against the pre-scaling builder bit for bit.
struct RopeScaling {
    RopeScalingType type = RopeScalingType::None;
    double freq_base = 1e7;    ///< `rope.freq_base` (the value pinned by `qsa_freq_base()`)
    double factor = 1.0;       ///< `--rope-scale`: how far past the trained context the run goes
    double freq_scale_in = 0;  ///< `--rope-freq-scale`: the raw ggml knob; 0 = 1/factor
    double orig_ctx = 262144;  ///< `--yarn-orig-ctx`: the context the model was trained over
    double ext_factor = 0;     ///< `--yarn-ext-factor`: correction depth; 0 = off, 1 = full YaRN
    double attn_factor = 1.0;  ///< `--yarn-attn-factor`: extra magnitude scale (ggml's attn_factor)
    double beta_fast = 32.0;   ///< `--yarn-beta-fast`
    double beta_slow = 1.0;    ///< `--yarn-beta-slow`

    /// The angle shrink.  ggml's `freq_scale`: 1 means "the trained angles", 1/factor squeezes
    /// `factor` positions into the trained range.
    double freq_scale() const {
        if (type == RopeScalingType::None) return 1.0;
        return freq_scale_in > 0 ? freq_scale_in : 1.0 / factor;
    }

    /// The magnitude correction.  ggml's `rope_yarn` multiplies cos and sin by this; the log term
    /// rides on `ext_factor != 0` (not on the type), exactly as that source has it.  With the
    /// default knobs YaRN factor F gives `1 + 0.1*ln(F)` - YaRN's temperature correction.
    double mscale() const {
        if (ext_factor == 0) return attn_factor;
        return attn_factor * (1.0 + 0.1 * std::log(1.0 / freq_scale()));
    }

    /// The correction window, in PAIR indices: below `out[0]` pairs extrapolate (keep the trained
    /// angles), above `out[1]` they interpolate, in between they blend.  Transcribed from ggml's
    /// `ggml_rope_yarn_corr_dims` (see the derivation comment at `rope_yarn_corr_dim` below).
    void corr_dims(int n_rot, double out[2]) const {
        const double start = std::floor(rope_yarn_corr_dim(n_rot, orig_ctx, beta_fast, freq_base));
        const double end = std::ceil(rope_yarn_corr_dim(n_rot, orig_ctx, beta_slow, freq_base));
        out[0] = start < 0 ? 0 : start;
        out[1] = end > (double) (n_rot - 1) ? (double) (n_rot - 1) : end;
    }

    /// The analytic kernels' constants.  `none` returns the identity constants WHATEVER the other knobs hold -
    /// the table builder ignores them for `none` too, so the two rotation forms cannot disagree there, and an
    /// unscaled run computes exactly what it computed before scaling existed.  Linear/YaRN: ggml's `rope_yarn`
    /// arguments (`attn_factor` is the raw knob; the helper applies the log term when correcting).
    RopeKernelArgs kernel_args(int n_rot) const {
        RopeKernelArgs a;
        if (type == RopeScalingType::None) return a;
        double cd[2];
        corr_dims(n_rot, cd);
        a.freq_scale = (float) freq_scale();
        a.corr_low = (float) cd[0];
        a.corr_high = (float) cd[1];
        a.ext_factor = (float) ext_factor;
        a.attn_factor = (float) attn_factor;
        return a;
    }

private:
    /// ggml/src/ggml.c: "Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)`
    /// for x, we get `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`".
    /// float64 rather than ggml's float32: the table path computes in float64 and is held to a
    /// bit-exact float64 reference, so its inputs are float64 too.
    static double rope_yarn_corr_dim(int n_dims, double n_ctx_orig, double n_rot, double base) {
        return (double) n_dims * std::log(n_ctx_orig / (n_rot * 2.0 * 3.14159265358979323846)) /
               (2.0 * std::log(base));
    }
};

/// The ramp between extrapolation (1) and interpolation (0).
///
/// ggml/src/ggml-cuda/rope.cu `rope_yarn_ramp`: `1 - clamp(y, 0, 1)` with
/// `y = (i0 / 2 - low) / max(0.001, high - low)`.  ggml passes the even dim index `i0`; this takes
/// the pair index (ggml's `i0/2`) so Strata's pair-indexed kernels don't re-derive the mapping.  The
/// 0.001 floor on the denominator is ggml's, kept verbatim: `corr_dims` can produce low == high.
STRATA_ROPE_SCALING_HD inline float rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

/// The scaled angle for one pair, float32, the ANALYTIC paths' contract (pinned ggml arithmetic).
///
/// Takes `theta_extrap` - the position times that pair's trained `inv_freq` - and returns the angle
/// to take cos/sin of, with ggml's mscale already folded in.  `mscale_in` must be `attn_factor`
/// (ggml's kernels receive the raw knob and apply the log term here, where it is per-pair constant;
/// this helper keeps that shape so the native path reads against ggml line by line).
STRATA_ROPE_SCALING_HD inline void rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low,
                                                     float corr_high, float ext_factor, float mscale_in,
                                                     int pair, float& cos_out, float& sin_out) {
    // The interpolation, always applied; the correction mixed in on top (ggml-cuda/rope.cu rope_yarn).
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        // "Get n-d magnitude scaling corrected for interpolation" - the log term only when correcting.
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_out = cosf(theta) * mscale;
    sin_out = sinf(theta) * mscale;
}

/// The ONE validity rule for a resolved configuration, shared by the engine's startup check and every rotation
/// site that takes the struct, so a bad knob cannot pass one site and trip (or silently NaN) another.  Returns
/// nullptr when the configuration is usable, else what is wrong.  Every value is tested with `std::isfinite`
/// FIRST: a NaN passes every `<`/`<=` range comparison, which is how a `--rope-scale nan` would otherwise slip
/// through.  `none` passes with any knob values - the knobs are inert there (the resolution resets them).
inline const char* rope_scaling_invalid(const RopeScaling& s) {
    if (!std::isfinite(s.freq_base) || s.freq_base <= 1.0) return "the frequency base must be a finite value above 1";
    if (s.type == RopeScalingType::None) return nullptr;
    if (!std::isfinite(s.factor) || s.factor < 1.0) return "the scaling factor must be a finite value >= 1";
    if (!std::isfinite(s.freq_scale_in) || s.freq_scale_in < 0.0)
        return "the frequency scale must be finite and positive (0 = 1/factor)";
    const double fs = s.freq_scale();
    if (!std::isfinite(fs) || fs <= 0.0) return "the resolved frequency scale must be a finite positive value";
    if (!std::isfinite(s.orig_ctx) || s.orig_ctx < 1.0) return "the original (trained) context must be finite and > 0";
    if (!std::isfinite(s.ext_factor) || s.ext_factor < 0.0) return "the YaRN ext factor must be finite and >= 0";
    if (!std::isfinite(s.attn_factor) || s.attn_factor <= 0.0) return "the attention factor must be finite and > 0";
    if (!std::isfinite(s.beta_fast) || s.beta_fast <= 0.0 || !std::isfinite(s.beta_slow) || s.beta_slow <= 0.0)
        return "the YaRN betas must be finite and > 0";
    if (!std::isfinite(s.mscale()) || s.mscale() <= 0.0) return "the resolved magnitude correction must be finite and > 0";
    return nullptr;
}

/// The process's one rope configuration.  Set it ONCE at startup, after the CLI and the model file
/// have agreed on the values and BEFORE `session_init` (which builds the rope table and captures the
/// graphs); read it at table build time and when launching the analytic kernels.  Plain storage, no
/// atomics: one writer on the startup thread, every reader after it.
void rope_scaling_set(const RopeScaling& scaling);
const RopeScaling& rope_scaling();

}  // namespace strata::kernels
