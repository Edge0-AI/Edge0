#pragma once
#include "edge0/generation.h"
#include <functional>
#include <random>
#include <unordered_set>

namespace edge0::app {
class SamplingRandom {
public:
    explicit SamplingRandom(std::uint64_t seed) : engine_(seed) {}
    double uniform() { ++draws_; return static_cast<double>(engine_() >> 11) * 0x1p-53; }
    std::size_t draws() const { return draws_; }
private:
    std::mt19937_64 engine_;
    std::size_t draws_ = 0;
};
using SamplingHistory = std::unordered_set<std::int32_t>;
using GeneratedCounts = std::map<std::int32_t,std::size_t>;
Result<std::int32_t> sample_logits(const std::vector<float>&, const GenerationConfiguration&,
    const SamplingHistory&, const std::function<double()>& uniform, const GeneratedCounts& = {});
TokenLogprob raw_logprob(const std::vector<float>&, std::int32_t, std::size_t);
}
