#include "GenerationOptions.h"
#include <cassert>

void sampling_defaults(const edge0::GenerationOverrides &options) {
    assert(options.temperature == 0.2 && !options.top_p && !options.repetition_penalty);
    assert(!options.top_k && !options.seed && !options.stop);
    assert(!options.presence_penalty && !options.frequency_penalty && !options.logit_bias);
    assert(!options.top_logprobs);
}
int main() {
    for (bool thinking : {false, true}) {
        auto normal = example_chat::generation_overrides(thinking, std::nullopt, false);
        assert(normal.thinking && *normal.thinking == thinking);
        assert(!normal.max_tokens && !normal.logprobs);
        sampling_defaults(normal);
    }
    for (double temperature : {0.2, 0.7, 1.0}) assert(example_chat::generation_overrides(false, std::nullopt, false, temperature).temperature == temperature);
    auto real_smoke = example_chat::generation_overrides(false, 96, false);
    assert(real_smoke.max_tokens == 96 && !real_smoke.logprobs);
    sampling_defaults(real_smoke);
    auto tiny_smoke = example_chat::generation_overrides(true, 16, true);
    assert(tiny_smoke.max_tokens == 16 && tiny_smoke.logprobs == true);
    sampling_defaults(tiny_smoke);
}
