#pragma once
// The qwen35 engine's host sampler settings and the repeat / frequency / presence penalties, header-only so a unit test
// (src/qwen35/sampling_test.cpp) can run them without a GPU.
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace strata::qwen35 {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// The penalties on the logit row `lg` for the tokens of the last `s.last_n` (all when < 0) history tokens.  A row can be
/// shorter than the vocabulary (the MTP draft head picks among the first N token ids only, --draft-vocab N): a history
/// token that is not in the row (id < 0 or >= lg.size()) is not in it and takes no penalty - it used to index past the
/// row (#3911).
inline void apply_penalties(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            if (id < 0 || (size_t) id >= lg.size()) continue;
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
}

}  // namespace strata::qwen35
