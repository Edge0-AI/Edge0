#include "tools/cli/generation_json.h"
#include "tests/support/check.h"
#include <nlohmann/json.hpp>
#include <limits>
#include <iostream>

int main() {
    using Json = nlohmann::ordered_json;
    edge0::GenerationConfiguration configuration;
    const auto defaults = Json::parse(cli::generation_json(configuration));
    const Json expected = {{"temperature", 0.0}, {"top_k", 0}, {"top_p", 1.0},
        {"repetition_penalty", 1.0}, {"max_tokens", 256}, {"thinking", false},
        {"presence_penalty", 0.0}, {"frequency_penalty", 0.0}, {"seed", nullptr},
        {"stop", Json::array()}, {"logit_bias", Json::object()}, {"logprobs", false},
        {"top_logprobs", 0}, {"sources", Json::object()}};
    CHECK(defaults == expected);

    configuration.temperature = 0.8;
    configuration.top_k = 20;
    configuration.top_p = 0.95;
    configuration.repetition_penalty = 1.1;
    configuration.max_tokens = 111;
    configuration.thinking = true;
    configuration.presence_penalty = 0.5;
    configuration.frequency_penalty = -0.25;
    configuration.seed = std::numeric_limits<std::uint64_t>::max();
    configuration.stop = {"quote\"", "line\nnext", "back\\slash"};
    configuration.logit_bias = {{4, -2.5}, {17, 3.0}};
    configuration.logprobs = true;
    configuration.top_logprobs = 5;
    configuration.sources = {{"temperature", edge0::ConfigurationSource::request},
        {"top_p", edge0::ConfigurationSource::model},
        {"max_tokens", edge0::ConfigurationSource::application},
        {"thinking", edge0::ConfigurationSource::platform}};
    const auto measured = Json::parse(cli::generation_json(configuration));
    const Json sampled = {{"temperature", 0.8}, {"top_k", 20}, {"top_p", 0.95},
        {"repetition_penalty", 1.1}, {"max_tokens", 111}, {"thinking", true},
        {"presence_penalty", 0.5}, {"frequency_penalty", -0.25},
        {"seed", std::numeric_limits<std::uint64_t>::max()},
        {"stop", Json::array({"quote\"", "line\nnext", "back\\slash"})},
        {"logit_bias", {{"4", -2.5}, {"17", 3.0}}}, {"logprobs", true}, {"top_logprobs", 5},
        {"sources", {{"max_tokens", "application"}, {"temperature", "request"},
                     {"thinking", "platform"}, {"top_p", "model"}}}};
    CHECK(measured == sampled);
    CHECK(measured["seed"].get<std::uint64_t>() == *configuration.seed);
    CHECK(Json::parse(measured.dump()) == measured);
    CHECK(configuration.temperature == 0.8 && configuration.stop.size() == 3);
    configuration.seed.reset();
    CHECK(Json::parse(cli::generation_json(configuration))["seed"].is_null());
    std::cout << "effective generation JSON defaults, sampling, sources, escaping and full-width seed passed\n";
}
