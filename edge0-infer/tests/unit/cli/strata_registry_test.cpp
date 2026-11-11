// Delegate mode, pure parts: registry matching, options, template expansion,
// contract check, response accounting and /status assembly.
#include "tools/cli/serve_delegate.h"
#include "tools/cli/serve_delegate_status.h"
#include "tests/support/check.h"
#include <fstream>
#include <cstdlib>
#include <random>
#include <sstream>

namespace fs = std::filesystem;
namespace {
struct Temp {
    fs::path path;
    Temp() {
        std::random_device random;
        path = fs::temp_directory_path() / ("strata-registry-" + std::to_string(random()) + std::to_string(random()));
        fs::create_directories(path);
    }
    ~Temp() { std::error_code error; fs::remove_all(path, error); }
};
void write(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}
template <class F> bool throws_config(F f, const std::string& part = "") {
    try { f(); } catch (const delegate::ConfigError& error) { return std::string(error.what()).find(part) != std::string::npos; }
    return false;
}
const char* placeholder = R"({"model_name":"edge0-145b-q3k","exe":"ignored","args":["ignored"],"env":{"ignored":"value"}})";
void complete(const fs::path& model) {
    write(model / "model.json", placeholder);
    for (const auto* name : {"latent_dense_q8.bin", "latent_dense_q4.bin",
                             "model-skeleton.gguf", "ple-fp8.gguf", "profile.bin"})
        write(model / name, name);
    for (const auto* name : {"experts.bin", "labels.bin", "dense.bin", "index.txt", "native_experts.txt"})
        write(model / name, name);
    write(model / "scales/layer.bin", "scale");
    write(model / "mtp" / "dense.bin", "1234567");
    write(model / "tokenizer" / "tokenizer.json", "{}");
}
}

int main() {
    using namespace delegate;
    const auto& registration = registry().at(0);
    CHECK(registry().size() == 2 && registration.id == "edge0_145b" && registration.engine == "edge0-strata");
    CHECK(registration.thinking && registration.thinking_default);

    // Matching: only a model.json naming a registered model_name is delegated; the native library
    // handles everything else (no model.json, other architectures, malformed files).
    Temp t;
    const auto model = t.path / "model";
    fs::create_directories(model);
    CHECK(!detect(model));
    CHECK(!detect(t.path / "absent"));
    complete(model);
    auto match = detect(model);
    CHECK(match && match->registration == &registration && match->complete());
    CHECK(match->missing(resolve_options(registration, {})).empty());
    for (const auto* other : {R"({"architecture": "edge0_35b", "format_version": 1})", R"({"architecture": "edge0-145b"})", "{}",
                              "[]", R"({"architecture": 145})", "not json",
                              R"({"model_name":"edge0-145b-q3k","architecture":"edge0_35b","format_version":1})",
                              R"({"model_name":"edge0-145b-q3k","architecture":null})",
                              R"({"architecture":"edge0_145b","format_version":1})",
                              R"({"model_name":12})", R"({"model_name":"unknown"})"}) {
        write(model / "model.json", other);
        CHECK(!detect(model));
    }
    fs::remove(model / "model.json");
    CHECK(!detect(model));  // every other file present, still native
    write(model / "model.json", placeholder);
    fs::remove_all(model);
    write(model / "model.json", placeholder);
    match = detect(model);  // model.json alone is enough to be delegated (and to report what is missing)
    CHECK(match && !match->complete() && match->missing(resolve_options(registration, {})).size() == 12);
    complete(model);
    fs::remove(model / "labels.bin");
    fs::remove_all(model / "mtp");
    match = detect(model);
    CHECK(match && !match->complete());
    CHECK((match->missing(resolve_options(registration, {})) == std::vector<std::string>{"labels.bin", "mtp/"}));
    write(model / "labels.bin", "x");
    write(model / "mtp" / "dense.bin", "1234567");
    fs::remove(model / "latent_dense_q8.bin");
    match = detect(model);
    CHECK(!match->complete());  // inspect uses the same default D/U as serve
    CHECK(match->complete({{"du", "q4"}}));
    CHECK((match->missing(resolve_options(registration, {})) == std::vector<std::string>{"latent_dense_q8.bin"}));
    CHECK(match->missing(resolve_options(registration, {{"du", "q4"}})).empty());
    fs::remove(model / "latent_dense_q4.bin");
    CHECK(!detect(model)->complete());
    write(model / "latent_dense_q8.bin", "latent_dense_q8.bin");
    write(model / "latent_dense_q4.bin", "latent_dense_q4.bin");

    // inspect --json
    const auto report = inspect_json(*detect(model), model);
    CHECK(report["adapter"] == "edge0_145b" && report["ok"] == true && report["delegated"] == true);
    CHECK(report["files"].size() == registration.files.size());
    for (const auto& file : report["files"]) {
        CHECK(file["exists"] == true);
        if (file["filename"] == "mtp/") CHECK(file["bytes"] == 7);
        if (file["filename"] == "experts.bin") CHECK(file["bytes"] == 11);
        if (file["filename"] == "latent_dense_q4.bin") CHECK(file["required"] == "du=q4");
        if (file["filename"] == "model-skeleton.gguf") CHECK(file["required"] == "always");
        if (file["filename"] == "model.json") CHECK(file["required"] == "always" && file["bytes"] == std::string(placeholder).size());
    }

    // Devices and model options.
    CHECK(parse_device("cuda") == 0 && parse_device("cuda:0") == 0 && parse_device("cuda:1") == 1 && parse_device("cuda:12") == 12);
    for (const auto* bad : {"cpu", "metal", "vulkan", "cuda:", "cuda:x", "cuda:-1", "CUDA", "cuda:1 "})
        CHECK(throws_config([&] { parse_device(bad); }, "CUDA GPU"));
    const auto defaults = resolve_options(registration, {});
    CHECK((defaults == std::map<std::string, std::string>{{"du", "q8"}, {"ple-io", "direct"}}));
    CHECK((resolve_options(registration, {{"du", "q4"}, {"ple-io", "ram"}}) == std::map<std::string, std::string>{{"du", "q4"}, {"ple-io", "ram"}}));
    CHECK(throws_config([&] { resolve_options(registration, {{"du", "q5"}}); }, "q8|q4"));
    CHECK(throws_config([&] { resolve_options(registration, {{"ple-io", "mmap"}}); }, "direct|ram"));
    CHECK(throws_config([&] { resolve_options(registration, {{"experts", "resident"}}); }, "unknown model option experts"));
    CommonArgs args;
    CHECK(common_argument("--model-option", "du=q4", args) && args.model_options.at("du") == "q4");
    CHECK(throws_config([&] { common_argument("--model-option", "du=q8", args); }, "more than once"));
    CHECK(throws_config([&] { common_argument("--context", "0", args); }));
    CHECK(!common_argument("--prefill-chunk", "64", args));
    CHECK(throws_config([] { unsupported_option("--prefill-chunk"); }, "--prefill-chunk"));

    // Template expansion.
    const auto config_template = Json::parse(R"({"exe":"${STRATA_DIR}/build/strata","cwd":"${STRATA_DIR}",
        "args":["--pack","${MODEL_DIR}","--ple-io","${PLE_IO}","--max-context","131072"],
        "env":{"STRATA_LATENT_DENSE":"${DU_FILE}","STRATA_LATENT_DENSE_FMT":"${DU_FMT}","STRATA_LATENT_A":"${MODEL_DIR}/experts.bin","STRATA_LAT_FUSED_Q3K":"${LAT_FUSED_Q3K}"},"model_name":"edge0-145b","lib_dirs":[]})");
    LaunchOptions options;
    options.strata_dir = "/opt/edge0-strata";
    options.model = "/models/edge0-145b";
    options.model_options = defaults;
    options.experts = "q3k";
    auto config = expand_template(config_template, registration, options, Platform::posix);
    CHECK(config["exe"] == "/opt/edge0-strata/build/strata" && config["cwd"] == "/opt/edge0-strata");
    CHECK(config["args"] == Json::array({"--pack", options.model.string(), "--ple-io", "direct", "--max-context", "131072"}));
    CHECK(config["env"]["STRATA_LATENT_DENSE"] == (fs::path("/models/edge0-145b") / "latent_dense_q8.bin").string());
    CHECK(config["env"]["STRATA_LATENT_DENSE_FMT"] == "8" && config["gpu"] == 0 && config["model_name"] == "edge0_145b");
    options.model_options = resolve_options(registration, {{"du", "q4"}, {"ple-io", "ram"}});
    options.gpu = 1;
    options.alias = "big";
    options.context = 8192;
    config = expand_template(config_template, registration, options, Platform::posix);
    CHECK(config["env"]["STRATA_LATENT_DENSE_FMT"] == "4" && config["args"][3] == "ram" && config["args"][5] == "8192");
    CHECK(config["env"]["STRATA_LATENT_DENSE"] == (fs::path("/models/edge0-145b") / "latent_dense_q4.bin").string());
    CHECK(config["gpu"] == 1 && config["model_name"] == "big");
    CHECK(throws_config([&] { expand_template(Json::parse(R"({"exe":"${NOPE}","args":[]})"), registration, options, Platform::posix); }, "${NOPE}"));

    // --strata-arg: engine words appended in order after the template's args (and after --context's rewrite);
    // options the template already sets, and --max-context, are refused. The template mirrors the published ones.
    {
        const auto published = Json::parse(R"({"exe":"${STRATA_DIR}/build/strata","args":["--pack","${MODEL_DIR}",
            "--native","${MODEL_DIR}/model-skeleton.gguf","--ple-io","${PLE_IO}","--expert-cache","auto","--prefill","auto",
            "--spec","4","--mtp","${MODEL_DIR}/mtp","--max-context","131072","--kv","int8"]})");
        LaunchOptions launch;
        launch.strata_dir = "/opt/edge0-strata";
        launch.model = "/models/edge0-145b";
        launch.model_options = defaults;
        launch.context = 8192;
        const auto before = expand_template(published, registration, launch, Platform::posix)["args"];
        CHECK(before.size() == 18 && before[15] == "8192" && before.back() == "int8");
        // "auto" equals a template value and "-x" is not an option ("--"): both are passed on unchecked.
        const std::vector<std::string> tuning = {"--pool-workers", "24", "--pcie-mode", "dma", "--pcie-frac", "0.1",
                                                 "--vram-reserve-mib", "700", "--kv-resident", "32768", "-x", "auto"};
        launch.strata_args = tuning;
        const auto after = expand_template(published, registration, launch, Platform::posix)["args"];
        CHECK(after.size() == before.size() + tuning.size());
        for (std::size_t i = 0; i < before.size(); ++i) CHECK(after[i] == before[i]);  // template part unchanged, --context applied
        for (std::size_t i = 0; i < tuning.size(); ++i) CHECK(after[before.size() + i] == tuning[i]);
        launch.context.reset();
        launch.strata_args.clear();
        CHECK(expand_template(published, registration, launch, Platform::posix)["args"][15] == "131072");
        for (const auto* option : {"--kv", "--pack", "--spec", "--expert-cache"}) {
            launch.strata_args = {"--pool-workers", "24", option, "x"};
            CHECK(throws_config([&] { expand_template(published, registration, launch, Platform::posix); },
                                std::string("invalid_argument: --strata-arg ") + option + ": set by the launch template"));
        }
        for (const auto& context : {std::optional<std::int64_t>(), std::optional<std::int64_t>(4096)}) {
            launch.context = context;
            launch.strata_args = {"--max-context", "65536"};
            CHECK(throws_config([&] { expand_template(published, registration, launch, Platform::posix); },
                                "invalid_argument: --strata-arg --max-context: use --context"));
        }
        // A template without --max-context: --context appends it, and the given words still come last.
        const auto short_template = Json::parse(R"({"exe":"x","args":["--kv","int8"]})");
        launch.context = 4096;
        launch.strata_args = {"--pool-workers", "24"};
        CHECK((expand_template(short_template, registration, launch, Platform::posix)["args"] ==
               Json::array({"--kv", "int8", "--max-context", "4096", "--pool-workers", "24"})));
        // --max-context is refused also when a template does not set it.
        launch.context.reset();
        launch.strata_args = {"--max-context"};
        CHECK(throws_config([&] { expand_template(short_template, registration, launch, Platform::posix); }, "use --context"));
        CommonArgs given;
        CHECK(common_argument("--strata-arg", "--pool-workers", given) && common_argument("--strata-arg", "24", given));
        CHECK((given.strata_args == std::vector<std::string>{"--pool-workers", "24"}));
        CHECK(throws_config([&] { common_argument("--strata-arg", "", given); }, "invalid_argument: --strata-arg needs a value"));
        CHECK(given.strata_args.size() == 2);
    }

    // The release templates: every model/format resolves all model files and D/U choices.
    CHECK(registry()[1].id == "edge0_120b");
    CHECK(registry()[1].config_template == "edge0/models/edge0-120b.json");
    const auto release_template = Json::parse(R"({
        "exe":"${STRATA_DIR}/build/strata${EXE_SUFFIX}",
        "args":["--pack","${MODEL_DIR}","--native","${MODEL_DIR}/model-skeleton.gguf",
                "--native-dense-gguf","${MODEL_DIR}/model-skeleton.gguf","--ple-gguf","${MODEL_DIR}/ple-fp8.gguf",
                "--ple-io","${PLE_IO}","--expert-profile","${MODEL_DIR}/profile.bin",
                "--expert-cache","auto","--prefill","auto","--spec","4","--spec-min-p","0.5",
                "--mtp","${MODEL_DIR}/mtp","--max-context","131072","--kv","int8"],
        "cwd":"${STRATA_DIR}","tokenizer":"${MODEL_DIR}/tokenizer","model_name":"ignored",
        "env":{"STRATA_LATENT_A":"${MODEL_DIR}/experts.bin","STRATA_LATENT_LABELS":"${MODEL_DIR}/labels.bin",
               "STRATA_LATENT_DENSE":"${DU_FILE}","STRATA_LATENT_DENSE_FMT":"${DU_FMT}",
               "STRATA_EMBED_VRAM":"1","STRATA_LATENT_FUSED":"1","STRATA_LATENT_FUSED_DU":"1",
               "STRATA_LATENT_U_GPU":"1","STRATA_LAT_FUSED_Q3K":"${LAT_FUSED_Q3K}"},
        "gpus_asked":true,"lib_dirs":["${CUDA_LIB_DIR}"]})");
    for (const auto& entry : registry()) {
        const std::string default_du = entry.id == "edge0_145b" ? "q8" : "q4";
        CHECK(resolve_options(entry, {}).at("du") == default_du);
        for (const auto& named : entry.names) {
            complete(model);
            write(model / "model.json", Json({{"model_name", named.first}, {"exe", "do-not-run"},
                  {"args", Json::array({"--pack", "wrong"})}, {"env", {{"STRATA_LAT_FUSED_Q3K", "wrong"}}}}).dump());
            if (named.second == "fp4") fs::remove_all(model / "scales");
            match = detect(model);
            CHECK(match && match->registration == &entry && match->experts == named.second && match->complete());
            const auto inspected = inspect_json(*match, model);
            CHECK(inspected["files"].size() == entry.files.size() - (named.second == "fp4" ? 1 : 0));
            for (const auto& f : inspected["files"])
                if (named.second == "fp4") CHECK(f["filename"] != "scales/");
            CHECK(throws_config([&] { resolve_options(entry, {{"experts", named.second}}); }, "unknown model option experts"));
            for (const auto& du : {default_du, std::string(default_du == "q8" ? "q4" : "q8")}) {
                LaunchOptions launch;
                launch.model = model;
                launch.strata_dir = t.path / "engine";
                launch.experts = match->experts;
                launch.model_options = resolve_options(entry, {{"du", du}});
                const auto expanded = expand_template(release_template, entry, launch, Platform::posix);
                const auto& argv = expanded["args"];
                auto argument = [&](const std::string& key) {
                    for (std::size_t i = 0; i + 1 < argv.size(); ++i) if (argv[i] == key) return argv[i + 1].get<std::string>();
                    return std::string();
                };
                // Templates join with a forward slash, while a Windows model path uses backslashes: compare as paths.
                auto same = [](const std::string& value, const fs::path& expected) { return fs::path(value) == expected; };
                auto env = [&](const char* key) { return expanded["env"][key].get<std::string>(); };
                CHECK(argument("--pack") == model.string());
                CHECK(same(argument("--native"), model / "model-skeleton.gguf"));
                CHECK(same(argument("--native-dense-gguf"), model / "model-skeleton.gguf"));
                CHECK(same(argument("--expert-profile"), model / "profile.bin"));
                CHECK(same(env("STRATA_LATENT_A"), model / "experts.bin"));
                CHECK(same(env("STRATA_LATENT_LABELS"), model / "labels.bin"));
                CHECK(expanded["env"]["STRATA_LAT_FUSED_Q3K"] == (named.second == "q3k" ? "1" : "0"));
                CHECK(same(env("STRATA_LATENT_DENSE"), model / ("latent_dense_" + du + ".bin")));
                CHECK(expanded["env"]["STRATA_LATENT_DENSE_FMT"] == du.substr(1));
                CHECK(expanded["model_name"] == entry.id);
            }
            fs::remove(model / "latent_dense_q4.bin");
            match = detect(model);
            const auto q4 = resolve_options(entry, {{"du", "q4"}});
            CHECK((match->missing(q4) == std::vector<std::string>{"latent_dense_q4.bin"}));
            CHECK((match->alternatives(q4) == std::vector<std::string>{"du=q8"}));
            fs::remove_all(model / "scales");
            match = detect(model);
            if (named.second == "q3k") {
                CHECK(!match->complete({{"du", "q8"}}));
                CHECK(match->alternatives(q4).empty());
            } else CHECK(match->complete({{"du", "q8"}}));
            fs::remove_all(model / "tokenizer");
            CHECK(detect(model)->alternatives(q4).empty());
        }
    }
    complete(model);
    CHECK(throws_config([&] {
        expand_template(Json::parse(R"({"exe":"${EXPERTS_DIR}","args":[]})"), registration, options, Platform::posix);
    }, "unknown placeholder"));

    // Engine directory and contract version.
    const auto engine = t.path / "edge0-strata";
    CHECK(throws_config([&] { locate_engine(engine.string(), ""); }, "not a directory"));
    fs::create_directories(engine / "edge0");
    CHECK(throws_config([&] { locate_engine(engine.string(), ""); }, "edge0/CONTRACT"));
    write(engine / "edge0" / "CONTRACT", "4\n");
    CHECK(locate_engine(engine.string(), "") == fs::absolute(engine));
    check_contract(engine);
    write(engine / "edge0" / "CONTRACT", "3\n");
    CHECK(throws_config([&] { check_contract(engine); }, "version 3 is not supported; this edge0-infer supports version 4"));
    write(engine / "edge0" / "CONTRACT", "one\n");
    CHECK(throws_config([&] { check_contract(engine); }, "unreadable"));

    // Platform rules are exercised on every host, without changing the process environment.
    write(engine / "edge0" / "CONTRACT", "4\n");
    const auto installed_exe = t.path / "installation" / "infer" / "build" / "bin" / "edge0-infer.exe";
    const auto adjacent = t.path / "installation" / "edge0-strata";
    write(adjacent / "edge0" / "CONTRACT", "4\n");
    // An explicit directory overrides discovery; the caller supplies the executable's actual path.
    CHECK(locate_engine(engine.string(), installed_exe) == fs::absolute(engine));
    const char* inherited_engine = std::getenv("EDGE0_STRATA_DIR");
    const auto saved_engine = inherited_engine ? std::optional<std::string>(inherited_engine) : std::nullopt;
#ifdef _WIN32
    CHECK(_putenv_s("EDGE0_STRATA_DIR", "") == 0);
#else
    CHECK(unsetenv("EDGE0_STRATA_DIR") == 0);
#endif
    CHECK(locate_engine(std::nullopt, installed_exe) == fs::absolute(adjacent));
#ifdef _WIN32
    CHECK(_putenv_s("EDGE0_STRATA_DIR", saved_engine ? saved_engine->c_str() : "") == 0);
#else
    if (saved_engine) CHECK(setenv("EDGE0_STRATA_DIR", saved_engine->c_str(), 1) == 0);
#endif
    CommonArgs platform_args;
    platform_args.model = model.string();
    platform_args.strata_dir = engine.string();
    const auto cuda = t.path / "CUDA toolkit";
    const auto platform_template = Json::parse(R"({"exe":"${STRATA_DIR}/build/strata${EXE_SUFFIX}","args":[],"lib_dirs":["${CUDA_LIB_DIR}"]})");
    for (const auto platform : {Platform::posix, Platform::windows}) {
        fs::remove_all(cuda);
        for (int layout = 0; layout < 4; ++layout) {
            if (layout == 2) fs::create_directories(cuda / "bin");
            if (layout == 3) write(cuda / "bin" / "x64" / "cublas64_13.dll", "fixture");
            const auto launch = resolve(platform_args, *detect(model), installed_exe, platform,
                                        layout == 0 ? "" : cuda.string());
            const bool windows = platform == Platform::windows;
            CHECK(launch.python == engine / ".venv" / (windows ? fs::path("Scripts") / "python.exe" : fs::path("bin") / "python"));
            const auto expanded = expand_template(platform_template, registration, launch, platform);
            CHECK(fs::path(expanded["exe"].get<std::string>()) == engine / "build" / (windows ? "strata.exe" : "strata"));
            CHECK(launch.lib_dirs.size() == (windows ? (layout == 3 ? 2U : layout == 2 ? 1U : 0U) : 0U));
            CHECK(expanded["lib_dirs"].size() == launch.lib_dirs.size());
            if (windows && layout == 3) {
                CHECK(launch.lib_dirs[0] == cuda / "bin" / "x64");
                CHECK(launch.lib_dirs[1] == cuda / "bin");
            }
            CHECK(launch.warnings.empty() == (!windows || layout == 3));
            if (!launch.warnings.empty()) {
                const auto& warning = launch.warnings[0];
                CHECK(warning.find("cublas64_13.dll") != std::string::npos);
                CHECK(warning.find("0xC0000135") != std::string::npos);
                CHECK(warning.find(layout == 0 ? "CUDA_PATH is not set" : (cuda / "bin" / "x64").string()) != std::string::npos);
                if (layout) CHECK(warning.find((cuda / "bin").string()) != std::string::npos);
            }
            for (const auto* bad : {R"({"exe":"${CUDA_LIB_DIR}","args":[]})",
                                    R"({"exe":"x","args":["${CUDA_LIB_DIR}"]})",
                                    R"({"exe":"x","args":[],"lib_dirs":["prefix${CUDA_LIB_DIR}"]})",
                                    R"({"exe":"x","args":[],"nested":{"lib_dirs":["${CUDA_LIB_DIR}"]}})",
                                    R"({"exe":"x","args":[],"${CUDA_LIB_DIR}":"x"})"})
                CHECK(throws_config([&] { expand_template(Json::parse(bad), registration, launch, platform); }, "${CUDA_LIB_DIR}"));
        }
        platform_args.python = (t.path / "custom python").string();
        CHECK(resolve(platform_args, *detect(model), installed_exe, platform, "").python == *platform_args.python);
        platform_args.python.reset();
        CHECK(resolve(platform_args, *detect(model), installed_exe, platform, "").strata_args.empty());
        platform_args.strata_args = {"--pool-workers", "24"};
        CHECK((resolve(platform_args, *detect(model), installed_exe, platform, "").strata_args == platform_args.strata_args));
        platform_args.strata_args.clear();
        for (const auto* version : {"0", "1", "2", "3"}) {
            write(engine / "edge0" / "CONTRACT", version);
            CHECK(throws_config([&] { check_contract(engine); }, "supports version 4"));
        }
        write(engine / "edge0" / "CONTRACT", "4\n");
        check_contract(engine);
    }

    // Response accounting: plain JSON, SSE split at arbitrary points, other dialects.
    Accounting plain;
    account(plain, Json::parse(R"({"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":13,"completion_tokens":14},
        "timings":{"cache_n":3,"prompt_n":10,"prompt_ms":163.3,"prompt_per_second":61.2,"predicted_n":14,"predicted_ms":222.5,"predicted_per_second":62.9}})"));
    CHECK(plain.input_tokens == 13 && plain.generated_tokens == 14 && plain.reused_tokens == 3 && plain.prompt_tokens_read == 10);
    CHECK(plain.prompt_seconds && std::abs(*plain.prompt_seconds - 0.1633) < 1e-9 && plain.decode_tokens_per_second == 62.9);
    CHECK(stop_reason(plain.finish) == "length");
    const std::string sse = "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n"
                            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2},"
                            "\"timings\":{\"cache_n\":1,\"prompt_n\":4,\"prompt_ms\":10,\"predicted_n\":2,\"predicted_ms\":20,\"predicted_per_second\":100}}\n\n"
                            "data: [DONE]\n\n";
    for (std::size_t split : {std::size_t(1), std::size_t(7), std::size_t(64), sse.size()}) {
        Accounting streamed;
        std::string pending;
        for (std::size_t i = 0; i < sse.size(); i += split) account_sse(streamed, pending, sse.data() + i, std::min(split, sse.size() - i));
        CHECK(pending.empty() && streamed.finish == "stop" && streamed.input_tokens == 5 && streamed.reused_tokens == 1);
        CHECK(streamed.decode_seconds == 0.02 && streamed.predicted_tokens == 2);
    }
    Accounting anthropic;
    account(anthropic, Json::parse(R"({"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":9}})"));
    CHECK(stop_reason(anthropic.finish) == "length" && anthropic.generated_tokens == 9);
    Accounting responses;
    account(responses, Json::parse(R"({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":6}}})"));
    CHECK(stop_reason(responses.finish) == "end_of_generation" && responses.input_tokens == 4 && responses.generated_tokens == 6);
    Accounting responses_plain;  // non-streaming Responses: the finish state is the top-level status
    account(responses_plain, Json::parse(R"({"object":"response","status":"completed","usage":{"input_tokens":63,"output_tokens":61}})"));
    CHECK(stop_reason(responses_plain.finish) == "end_of_generation" && responses_plain.input_tokens == 63);
    Accounting messages_plain;  // Anthropic input_tokens excludes the cached part; the input is all three
    account(messages_plain, Json::parse(R"({"type":"message","stop_reason":"end_turn",
        "usage":{"input_tokens":1,"output_tokens":50,"cache_creation_input_tokens":67,"cache_read_input_tokens":2}})"));
    CHECK(messages_plain.input_tokens == 70 && messages_plain.generated_tokens == 50 && stop_reason(messages_plain.finish) == "end_of_generation");
    Accounting messages_stream;  // message_start holds the input; message_delta does not replace it
    account(messages_stream, Json::parse(R"({"type":"message_start","message":{"type":"message","stop_reason":null,
        "usage":{"input_tokens":62,"output_tokens":0}}})"));
    account(messages_stream, Json::parse(R"({"type":"message_delta","delta":{"stop_reason":"end_turn"},
        "usage":{"input_tokens":1,"cache_creation_input_tokens":5,"cache_read_input_tokens":0,"output_tokens":42}})"));
    CHECK(messages_stream.input_tokens == 62 && messages_stream.generated_tokens == 42 && stop_reason(messages_stream.finish) == "end_of_generation");
    // Fields of the wrong type count as missing; nothing throws.
    for (const char* text : {R"({"object":null,"status":"completed"})", R"({"object":5,"status":"completed"})",
                             R"({"object":"response","status":null})", R"({"object":"response","status":5})",
                             R"({"type":null,"stop_reason":7})", R"({"usage":null})", R"({"usage":"many"})",
                             R"({"type":"message","usage":{"input_tokens":"3","cache_read_input_tokens":null,"output_tokens":[1]}})",
                             R"({"response":null,"message":7,"delta":{"stop_reason":null},"choices":[null,{"finish_reason":3}]})"}) {
        Accounting odd;
        account(odd, Json::parse(text));
        CHECK(odd.finish.empty() && !odd.input_tokens && !odd.generated_tokens);
    }
    Accounting typed;  // a non-string type is no type: the usage is read the plain way
    account(typed, Json::parse(R"({"type":7,"usage":{"input_tokens":3,"cache_creation_input_tokens":9}})"));
    CHECK(typed.input_tokens == 3);
    Accounting unfinished;  // a Responses stream that ended before its terminal event: the finish is unknown
    account(unfinished, Json::parse(R"({"type":"response.created","response":{"object":"response","status":"in_progress"}})"));
    account(unfinished, Json::parse(R"({"type":"response.queued","response":{"object":"response","status":"queued"}})"));
    CHECK(stop_reason(unfinished.finish) == "unknown");
    Accounting both;  // Chat usage carrying both names: prompt_tokens stands
    account(both, Json::parse(R"({"usage":{"prompt_tokens":13,"input_tokens":4,"completion_tokens":2}})"));
    CHECK(both.input_tokens == 13);
    // A malformed event in a stream is skipped; the events around it still count.
    Accounting skipping;
    std::string tail;
    const std::string events = "data: {\"usage\":{\"prompt_tokens\":5}}\n\ndata: {\"object\":null}\n\ndata: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n";
    account_sse(skipping, tail, events.data(), events.size());
    CHECK(skipping.input_tokens == 5 && skipping.finish == "stop" && tail.empty());
    CHECK(stop_reason("tool_use") == "tool_calls" && stop_reason("") == "unknown");

    // Live progress: text-carrying fragments of the three stream formats, one token each, in the shapes
    // edge0-strata sends. Each stream has exactly `fragments` counted events; the others carry no text.
    struct LiveCase {
        const char* name;
        std::string sse;
        std::int64_t fragments;
    };
    auto data = [](const std::string& json) { return "data: " + json + "\n\n"; };
    auto named = [&](const char* event, const std::string& json) { return std::string("event: ") + event + "\n" + data(json); };
    const std::vector<LiveCase> live_cases{
        {"chat",
         data(R"({"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]})") +
             data(R"({"choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]})") +
             data(R"({"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]})") +
             data(R"({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]})") +
             data(R"({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}}]},"finish_reason":null}]})") +
             data(R"({"choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]})") +
             data(R"({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":3},)"
                  R"("timings":{"cache_n":2,"prompt_n":7,"prompt_ms":50,"prompt_per_second":140,"predicted_n":3,"predicted_ms":30,"predicted_per_second":100}})") +
             "data: [DONE]\n\n",
         3},
        {"responses",
         named("response.created", R"({"type":"response.created","response":{"object":"response","status":"in_progress"}})") +
             named("response.reasoning_text.delta", R"({"type":"response.reasoning_text.delta","delta":"hm"})") +
             named("response.output_text.delta", R"({"type":"response.output_text.delta","delta":"Hi"})") +
             named("response.output_text.delta", R"({"type":"response.output_text.delta","delta":""})") +
             named("response.output_text.done", R"({"type":"response.output_text.done","text":"Hi"})") +
             named("response.function_call_arguments.delta", R"({"type":"response.function_call_arguments.delta","delta":"{}"})") +
             named("response.custom_tool_call_input.delta", R"({"type":"response.custom_tool_call_input.delta","delta":"x"})") +
             named("response.completed", R"({"type":"response.completed","response":{"object":"response","status":"completed",)"
                                         R"("usage":{"input_tokens":12,"output_tokens":4}}})"),
         4},
        {"messages",
         named("message_start", R"({"type":"message_start","message":{"type":"message","stop_reason":null,"usage":{"input_tokens":21,"output_tokens":0}}})") +
             named("content_block_start", R"({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}})") +
             named("content_block_delta", R"({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"so"}})") +
             named("content_block_delta", R"({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}})") +
             named("content_block_delta", R"({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":""}})") +
             named("content_block_delta", R"({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Yes"}})") +
             named("content_block_delta", R"({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\""}})") +
             named("content_block_stop", R"({"type":"content_block_stop","index":2})") +
             named("message_delta", R"({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}})") +
             named("message_stop", R"({"type":"message_stop"})"),
         3}};
    // The accounting a stream produces without live counting: every data object folded by account() alone.
    auto record_only = [](const std::string& sse) {
        Accounting a;
        std::istringstream lines(sse);
        for (std::string line; std::getline(lines, line);)
            if (line.rfind("data: {", 0) == 0) account(a, Json::parse(line.substr(6)));
        return a;
    };
    auto record_of = [](const Accounting& a) {
        ProxyRecord r;
        r.entered = true; r.http_status = 200; r.stream = true; r.time = "t"; r.accounting = a;
        r.stop_reason = stop_reason(a.finish);
        ProxyHistory h;
        h.add(r);
        const auto& t = h.totals;
        return record_json(r).dump() + "|" + std::to_string(t.input_tokens) + "," + std::to_string(t.reused_tokens) + "," +
               std::to_string(t.generated_tokens) + "," + std::to_string(t.prompt_tokens_read) + "," +
               std::to_string(t.predicted_tokens) + "," + std::to_string(t.first_seconds) + "," + std::to_string(t.decode_seconds);
    };
    for (const auto& c : live_cases) {
        std::int64_t counted = 0;
        std::size_t first_line_end = std::string::npos;  // the newline that completes the first counted event
        {
            std::istringstream lines(c.sse);
            std::size_t offset = 0;
            for (std::string line; std::getline(lines, line); offset += line.size() + 1)
                if (line.rfind("data: {", 0) == 0) {
                    const auto n = content_fragments(Json::parse(line.substr(6)));
                    if (n && first_line_end == std::string::npos) first_line_end = offset + line.size();
                    counted += n;
                }
        }
        CHECK(counted == c.fragments);
        const auto expected_record = record_of(record_only(c.sse));
        // Cut at every byte: two chunks, the first arriving at 1 s and the second at 2 s.
        for (std::size_t cut = 0; cut <= c.sse.size(); ++cut) {
            Accounting a;
            std::string pending;
            account_sse(a, pending, c.sse.data(), cut, 1.0);
            account_sse(a, pending, c.sse.data() + cut, c.sse.size() - cut, 2.0);
            CHECK(pending.empty() && a.saw_content && a.streamed_tokens == c.fragments);
            CHECK(a.first_token_seconds == (cut > first_line_end ? 1.0 : 2.0));
            CHECK(record_of(a) == expected_record);
        }
        // Fixed chunk sizes; the n-th chunk arrives at n s.
        for (std::size_t split : {std::size_t(1), std::size_t(7), std::size_t(64), c.sse.size()}) {
            Accounting a;
            std::string pending;
            for (std::size_t i = 0; i < c.sse.size(); i += split)
                account_sse(a, pending, c.sse.data() + i, std::min(split, c.sse.size() - i), double(i / split + 1));
            CHECK(pending.empty() && a.streamed_tokens == c.fragments);
            CHECK(a.first_token_seconds == double(first_line_end / split + 1));
            CHECK(record_of(a) == expected_record);
        }
    }
    // Events whose text field is empty, missing or of the wrong type carry nothing; nothing throws.
    for (const char* text : {R"({"choices":[{"delta":{"content":5,"reasoning_content":null}}]})",
                             R"({"choices":[{"delta":{"role":"assistant"}}],"usage":{"completion_tokens":4}})",
                             R"({"choices":[{"delta":{},"finish_reason":"stop"}]})",
                             R"({"choices":[{"delta":{"tool_calls":[{"function":{"name":"f"}},{"function":{"arguments":7}},null]}}]})",
                             R"({"choices":[null,{"delta":"text"}]})",
                             R"({"type":"response.output_text.delta","delta":null})",
                             R"({"type":"response.output_text.done","delta":"x"})",
                             R"({"type":"content_block_delta","delta":{"type":"text_delta","text":null}})",
                             R"({"type":"content_block_delta","delta":{"type":"thinking_delta","text":"x"}})",
                             R"({"type":"content_block_start","content_block":{"type":"text","text":"x"}})",
                             R"({"type":"message_start","message":{"content":[{"type":"text","text":"x"}]}})", R"([1])", R"("x")"})
        CHECK(content_fragments(Json::parse(text)) == 0);
    // Two choices that both carry text are two fragments.
    CHECK(content_fragments(Json::parse(R"({"choices":[{"delta":{"content":"a"}},{"delta":{"reasoning_content":"b"}}]})")) == 2);
    // A non-streaming body carries its whole answer at once: it is accounted, but produces no fragments.
    Accounting whole;
    account(whole, Json::parse(R"({"choices":[{"message":{"role":"assistant","content":"all of it"},"finish_reason":"stop"}],)"
                               R"("usage":{"prompt_tokens":5,"completion_tokens":3}})"));
    CHECK(!whole.saw_content && !whole.streamed_tokens && !whole.first_token_seconds && whole.generated_tokens == 3);

    // Totals: rejected requests are counted apart; rates use only requests that carry both numbers.
    ProxyHistory history;
    ProxyRecord entered;
    entered.entered = true; entered.http_status = 200; entered.accounting = plain; entered.time = "t1";
    history.add(entered);
    ProxyRecord bare;
    bare.entered = true; bare.http_status = 200; bare.time = "t2";
    history.add(bare);
    for (int status : {401, 429, 400}) { ProxyRecord r; r.http_status = status; history.add(r); }
    const auto& totals = history.totals;
    CHECK(totals.requests == 2 && totals.unauthorized == 1 && totals.rejected_busy == 1 && totals.rejected_invalid == 1);
    CHECK(totals.input_tokens == 13 && totals.generated_tokens == 14 && totals.reused_tokens == 3);
    CHECK(history.recent.size() == 5 && history.last->time == "t2");

    // /status assembly: mapped, recorded and not-applicable fields.
    DelegateStatusInput input;
    input.instance = {{"id", "1-2"}};
    input.listen = {{"host", "127.0.0.1"}};
    input.system = Json{{"memory", nullptr}, {"thermal", {{"state", nullptr}}}};
    input.alias = "edge0_145b";
    input.directory = "/m";
    input.device = "cuda:1";
    input.context = 131072;
    input.external = Json::parse(R"({"engine":"0.1.55","machine":{"gpu":{"name":"RTX","used_mib":1024,"total_mib":4096},"ram":{"used_gib":2,"total_gib":8}}})");
    input.history = history;
    input.now = std::chrono::steady_clock::now();
    input.active = ActiveRequest{"/v1/messages", "127.0.0.1", input.now - std::chrono::seconds(2), Accounting{}};
    auto status = status_json(input);
    CHECK(status["engine"]["version"] == "0.1.55" && status["engine"]["ggml_version"].is_null() && status["engine"]["backends"].is_null());
    CHECK(status["engine"]["devices"][0]["name"] == "cuda:1" && status["engine"]["devices"][0]["memory_total_bytes"] == 4096LL * 1048576);
    CHECK(status["engine"]["devices"][0]["memory_free_bytes"] == 3072LL * 1048576 && status["engine"]["devices"][0]["description"] == "RTX");
    CHECK(status["model"]["context_length"] == 131072 && status["model"]["device"] == "cuda:1" && status["model"]["prefill_device"] == "cuda:1");
    CHECK(status["memory"]["ledger"].is_null() && status["memory"]["resident"].is_null() && status["memory"]["physical"].is_null());
    CHECK(status["system"]["memory"]["total"]["bytes"] == 8LL * 1073741824 && status["system"]["memory"]["available"]["bytes"] == 6LL * 1073741824);
    CHECK(status["system"]["memory"]["total"]["source"] == "edge0-strata" && status["system"]["thermal"]["state"].is_null());
    CHECK(status["current"]["endpoint"] == "/v1/messages" && status["current"]["elapsed_seconds"] >= 2.0 && status["current"]["phase"].is_null());
    CHECK(status["totals"]["requests"] == 2 && status["totals"]["computed_tokens"].is_null() && status["totals"]["continued_turns"].is_null());
    CHECK(std::abs(status["totals"]["prefill_tokens_per_second"].get<double>() - 10 / 0.1633) < 1e-9);
    CHECK(std::abs(status["totals"]["decode_tokens_per_second"].get<double>() - 14 / 0.2225) < 1e-9);
    CHECK(status["recent_requests"].size() == 5 && status["last"]["ended_at"] == "t2" && status["last"]["input_tokens"].is_null() && status["last"]["error_stage"].is_null() && status["last"]["checkpoint_restored"].is_null());
    const auto first = status["recent_requests"][4];
    CHECK(first["input_tokens"] == 13 && first["reused_tokens"] == 3 && !first.contains("computed_tokens") && !first.contains("decode_forwards"));
    CHECK(first["first_token_seconds"] == plain.prompt_seconds.value() && !first.contains("reuse_method") && !first.contains("error_stage"));
    CHECK(!status["recent_requests"][0].contains("stop_reason"));  // a rejected request has no generation fields
    CHECK(status["current"]["generated_tokens"].is_null() && !status["current"].contains("first_token_seconds"));
    // `current` while a request runs. Non-streaming: phase and count unknown even once the body has been read.
    const auto last_before = status["last"].dump(), recent_before = status["recent_requests"].dump(), totals_before = status["totals"].dump();
    input.active->accounting = whole;
    status = status_json(input);
    CHECK(status["current"]["phase"].is_null() && status["current"]["generated_tokens"].is_null());
    CHECK(!status["current"].contains("first_token_seconds") && status["current"]["input_tokens"] == 5);
    // A stream before its first text fragment: prefill; Messages already carries the input count.
    Accounting opening;
    opening.streaming = true;
    std::string opening_pending;
    const auto& messages_sse = live_cases[2].sse;
    const auto first_event = messages_sse.substr(0, messages_sse.find("\n\n") + 2);
    account_sse(opening, opening_pending, first_event.data(), first_event.size(), 0.5);
    input.active->accounting = opening;
    status = status_json(input);
    CHECK(status["current"]["phase"] == "prefill" && status["current"]["generated_tokens"].is_null());
    CHECK(!status["current"].contains("first_token_seconds") && status["current"]["input_tokens"] == 21 && status["current"]["reused_tokens"].is_null());
    // After the first fragment: decode, the fragment count and the first fragment's arrival.
    account_sse(opening, opening_pending, messages_sse.data() + first_event.size(), messages_sse.size() - first_event.size(), 1.25);
    input.active->accounting = opening;
    status = status_json(input);
    CHECK(status["current"]["phase"] == "decode" && status["current"]["generated_tokens"] == 3);
    CHECK(status["current"]["first_token_seconds"] == 1.25 && status["current"]["input_tokens"] == 21);
    CHECK(status["current"]["computed_tokens"].is_null() && status["current"]["pending_tokens"].is_null() && !status["current"].contains("decode_elapsed_seconds"));
    // The live counts never reach the records: last, recent_requests and totals are as before.
    CHECK(status["last"].dump() == last_before && status["recent_requests"].dump() == recent_before && status["totals"].dump() == totals_before);
    input.external = nullptr;
    input.active.reset();
    status = status_json(input);
    CHECK(status["engine"]["version"].is_null() && status["engine"]["devices"].is_null() && status["current"].is_null());
    CHECK(status["system"]["memory"]["total"]["bytes"].is_null() && !status["system"]["memory"]["total"]["reason"].get<std::string>().empty());
    std::cout << "PASS strata registry: matching, options, devices, template, contract, accounting, status assembly\n";
}
