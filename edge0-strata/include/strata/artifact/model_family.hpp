// include/strata/artifact/model_family.hpp - which model family a GGUF holds, from its general.architecture.
//
// Strata was written for one model (Qwen3.8-Flash-Next, `qwen4exp`).  Qwen3.6-35B-A3B (`qwen35moe`) is a close
// cousin (same tokenizer, the same Gated-DeltaNet + gated-attention 3:1 layer pattern, the same head dims), so the
// reader, the geometry and the layout tables know both.  The ENGINE runs only the families `model_family_runs`
// names; a recognised family it does not run yet is refused with `model_family_not_supported`, not mis-run.
//
// No dependencies: the GGUF reader (the architecture guard) and the core geometry both include this.
#pragma once

#include <string>

namespace strata {

enum class ModelFamily {
    Unknown = 0,
    Qwen4Exp,    ///< Qwen3.8-Flash-Next and its fine-tunes / pruned variants (Swift 1.5, the Coder)
    Qwen35Moe,   ///< Qwen3.6-35B-A3B (Unsloth's MTP GGUFs): plain pre-norm residual, dense attention, nextn MTP block
};

inline ModelFamily model_family_from_arch(const std::string& arch) {
    if (arch == "qwen4exp" || arch == "edge0") return ModelFamily::Qwen4Exp;
    if (arch == "qwen35moe") return ModelFamily::Qwen35Moe;
    return ModelFamily::Unknown;
}

/// The GGUF architecture string (also the metadata key prefix: "<arch>.block_count").
inline const char* model_family_arch(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "qwen4exp";
    case ModelFamily::Qwen35Moe:
        return "qwen35moe";
    default:
        return "";
    }
}

/// The model's name for messages.
inline const char* model_family_title(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "Qwen3.8-Flash-Next";
    case ModelFamily::Qwen35Moe:
        return "Qwen3.6-35B-A3B";
    default:
        return "an unknown model";
    }
}

/// Whether Flash-Next's engine path runs the family.  Qwen3.6 has an engine path of its own (src/qwen35), which the
/// CUDA and HIP builds of `strata` switch to at the top of main(); the Intel Arc (SYCL) build has none yet, so there
/// (and on Flash-Next's path) the family is refused with `model_family_not_supported`.
inline bool model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen4Exp; }

/// The refusal for a recognised family the engine does not run (empty when it runs it).
inline std::string model_family_not_supported(ModelFamily f) {
    if (model_family_runs(f)) return {};
    if (f == ModelFamily::Unknown) return "this is not a model family Strata knows";
    return std::string(model_family_title(f)) + " (architecture '" + model_family_arch(f) +
           "') runs on the NVIDIA (CUDA) and AMD (HIP) builds of Strata; this build does not run it yet (the Intel "
           "Arc / SYCL port is in development). Use a CUDA or HIP build, or a Qwen3.8-Flash-Next model file";
}

}  // namespace strata
