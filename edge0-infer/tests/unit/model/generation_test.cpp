#include "model/asset_index.h"
#include "tests/support/check.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
using namespace edge0;
int main(int argc, char**) {
    if (argc > 1) {
        const auto* reference = std::getenv("MOE_TINY");
        if (!reference) return 77;
        auto result = model::read_generation_defaults((std::filesystem::path(reference) / "tiny-a").string());
        CHECK(result);
        CHECK(result.value().temperature == 1.0 && result.value().top_k == 20 && result.value().top_p == 0.95);
        CHECK(!result.value().repetition_penalty);
        return 0;
    }
    const auto root = std::filesystem::temp_directory_path() / ("moe-generation-defaults-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    auto missing = model::read_generation_defaults(root.string());
    CHECK(missing && !missing.value().temperature);
    for (const auto& text : {"{}", "{\"ignored\":[1,null,{}]}",
        "{\"temperature\":0,\"top_k\":0,\"top_p\":1,\"repetition_penalty\":1.2}"}) {
        { std::ofstream out(root / "generation_config.json"); out << text; }
        CHECK(model::read_generation_defaults(root.string()));
    }
    // Unknown provider fields must not constrain the supported generation defaults.
    for (const auto& text : {
        "{\"seed\":18446744073709551615,\"temperature\":0.7}",
        "{\"ignored\":[-9999999999999999999999999999999999,1e999],\"top_k\":20}",
        "{\"provider\":{\"counter\":9999999999999999999999999999999999},\"top_p\":0.9}"}) {
        { std::ofstream out(root / "generation_config.json"); out << text; }
        CHECK(model::read_generation_defaults(root.string()));
    }
    for (const auto& text : {"{\"temperature\":-1}", "{\"temperature\":true}", "{\"temperature\":\"1\"}",
        "{\"top_k\":-1}", "{\"top_k\":1.5}", "{\"top_k\":false}", "{\"top_p\":0}",
        "{\"top_p\":1.5}", "{\"top_p\":null}", "{\"repetition_penalty\":0}",
        "{\"repetition_penalty\":[]}", "{\"temperature\":1e999}", "{\"temperature\":1.}",
        "{\"top_k\":18446744073709551615}", "{\"ignored\":1e+}",
        "{\"ignored\":9999999999999999999999999999999999x}"}) {
        { std::ofstream out(root / "generation_config.json"); out << text; }
        auto result = model::read_generation_defaults(root.string());
        CHECK(!result && result.error().category == ErrorCategory::asset_invalid);
    }
    std::filesystem::remove_all(root);
}
