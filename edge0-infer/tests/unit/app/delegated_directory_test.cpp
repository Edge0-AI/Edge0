// The native API refuses a directory whose model.json names an architecture this
// library does not implement (such as an Edge0 145B directory, which runs on
// edge0-strata) with an unsupported error that points to `edge0-infer serve`,
// before reading assets.tsv or any other file in it.
#include "edge0/assets.h"
#include "edge0/engine.h"
#include "edge0/generation.h"
#include "tests/support/check.h"
#include <filesystem>
#include <fstream>
#include <random>

namespace fs = std::filesystem;
int main() {
    std::random_device random;
    const auto root = fs::temp_directory_path() / ("delegated-directory-" + std::to_string(random()) + std::to_string(random()));
    fs::create_directories(root / "mtp");
    std::ofstream(root / "model.json") << R"({"architecture": "edge0_145b", "format_version": 1})";
    // Unreadable files: any attempt to open one would fail differently (asset_invalid).
    for (const auto* name : {"assets.tsv", "experts.bin", "labels.bin", "latent_dense_q8.bin", "model.gguf", "mtp/dense.bin"}) {
        std::ofstream(root / name) << "x";
        fs::permissions(root / name, fs::perms::none, fs::perm_options::replace);
    }
    auto refused = [](const edge0::Error& error) {
        return error.category == edge0::ErrorCategory::unsupported && error.message.find("edge0-infer serve") != std::string::npos &&
               error.message.find("edge0_145b") != std::string::npos && error.message.find("edge0_35b") != std::string::npos;
    };
    const auto inspected = edge0::inspect_assets({root.string(), false});
    CHECK(!inspected && refused(inspected.error()));
    const auto digested = edge0::inspect_assets({root.string(), true});
    CHECK(!digested && refused(digested.error()));
    auto runtime = edge0::open_runtime({{"CPU"}, {}});
    CHECK(runtime);
    const auto resolved = edge0::resolve_load_configuration(runtime.value(), root.string());
    CHECK(!resolved && refused(resolved.error()));
    edge0::LoadConfiguration configuration;
    configuration.directory = root.string();
    configuration.device = "CPU";
    const auto loaded = edge0::load_model(runtime.value(), configuration);
    CHECK(!loaded && refused(loaded.error()));
    // A service configuration is rejected before any other file is read.
    std::ofstream(root / "model.json") << R"({"model_name":"edge0-120b-fp4","args":["--pack","ignored"],
        "env":{"STRATA_LATENT_A":"ignored","nested":{"array":[1,2]}}})";
    auto external = [](const edge0::Error& error) {
        return error.category == edge0::ErrorCategory::unsupported &&
               error.message.find("edge0-infer serve") != std::string::npos;
    };
    const auto service_inspected = edge0::inspect_assets({root.string(), false});
    CHECK(!service_inspected && external(service_inspected.error()));
    const auto service_digested = edge0::inspect_assets({root.string(), true});
    CHECK(!service_digested && external(service_digested.error()));
    const auto service_resolved = edge0::resolve_load_configuration(runtime.value(), root.string());
    CHECK(!service_resolved && external(service_resolved.error()));
    const auto service_loaded = edge0::load_model(runtime.value(), configuration);
    CHECK(!service_loaded && external(service_loaded.error()));
    for (const auto* malformed : {R"({"architecture":"edge0_35b"})", R"({"format_version":1})",
                                  R"({"format_version":null,"architecture":null})"}) {
        std::ofstream(root / "model.json") << malformed;
        const auto invalid = edge0::inspect_assets({root.string(), false});
        CHECK(!invalid && invalid.error().category == edge0::ErrorCategory::asset_invalid);
    }
    // Without model.json the error is what it was before delegated models existed.
    fs::remove(root / "model.json");
    const auto missing = edge0::inspect_assets({root.string(), false});
    CHECK(!missing && missing.error().category == edge0::ErrorCategory::asset_invalid);
    CHECK(missing.error().message.find("model.json: cannot read") != std::string::npos);
    CHECK(edge0::finish_runtime_use(runtime.value()));
    std::error_code error;
    for (const auto& entry : fs::recursive_directory_iterator(root, error))
        fs::permissions(entry.path(), fs::perms::owner_all, fs::perm_options::replace, error);
    fs::remove_all(root, error);
    std::cout << "PASS delegated directory: inspect_assets, resolve_load_configuration and load_model refuse without reading other files\n";
}
