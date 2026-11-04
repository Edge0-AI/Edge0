#pragma once
#include "edge0/engine.h"
#include "edge0/weight_activity.h"
#include <map>
#include <array>
#include <cstdint>
namespace edge0 {
struct ForwardCheckOptions {
    std::string model_directory, device;
    bool observe = false, split_last = false;
    std::vector<std::int32_t> input, generated;
    std::map<std::string,std::string> model_options;
    std::string route_trace = {};
    std::size_t observe_decode_steps = 2;
};
struct ForwardCheckTensor {
    std::string name;
    int step = -1;
    std::int64_t position = 0;
    int type = 0;
    std::array<std::int64_t,4> shape{};
    std::vector<std::uint8_t> data;
};
struct ForwardCheckReport {
    WeightActivity weight_activity;
    std::vector<float> logits;
    std::size_t rows = 0, vocabulary = 0;
    std::vector<ForwardCheckTensor> layers;
    RuntimeBudget budget;
    std::size_t budget_peak = 0, budget_current = 0;
    std::vector<std::size_t> weight_registrations;
};
Result<ForwardCheckReport> forward_check(const RuntimeReference& runtime, const ForwardCheckOptions& options);
}
