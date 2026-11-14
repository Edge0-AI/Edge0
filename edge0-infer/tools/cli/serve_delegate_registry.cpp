#include "serve_delegate.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

namespace delegate {
const std::vector<Registration>& registry() {
    // The files each model directory must hold. The user documentation lists the same files, and an L0 check keeps the two equal.
    static const std::vector<Registration> entries = {
        {"edge0_145b", "edge0-strata", "edge0/models/edge0-145b.json",
         {{"model.json", false, ""},  // release identifier
          {"experts.bin", false, ""},  // latent experts, 48 layers x 512
          {"labels.bin", false, ""},  // D/U group of each expert, 48 x 512 int32
          {"dense.bin", false, ""},  // dense portion of the expert pack
          {"index.txt", false, ""},  // expert pack index
          {"native_experts.txt", false, ""},  // native expert mapping
          {"scales", true, "experts=q3k"},  // per-layer column scales
          {"latent_dense_q8.bin", false, "du=q8"},  // D/U projections, Q8
          {"latent_dense_q4.bin", false, "du=q4"},  // D/U projections, Q4
          {"model-skeleton.gguf", false, ""},  // backbone: attention, routing, shared experts, embeddings, output head
          {"ple-fp8.gguf", false, ""},  // per-layer embedding table (FP8)
          {"profile.bin", false, ""},  // expert ranking that decides which experts are cached up front
          {"mtp", true, ""},  // original MTP draft layer (speculative decoding)
          {"tokenizer", true, ""}},  // tokenizer
         {{"du", {"q8", "q4"}}, {"ple-io", {"direct", "ram"}}},
         true,
         true,
         {{"edge0-145b-q3k", "q3k"}, {"edge0-145b-fp4", "fp4"}},
         {{"metal", {{"du", {"q4"}}, {"ple-io", {"direct"}}}}}},
        {"edge0_120b", "edge0-strata", "edge0/models/edge0-120b.json",
         {{"model.json", false, ""},  // release identifier
          {"experts.bin", false, ""},  // latent experts, 48 layers x 512
          {"labels.bin", false, ""},  // D/U group of each expert, 48 x 512 int32
          {"dense.bin", false, ""},  // dense portion of the expert pack
          {"index.txt", false, ""},  // expert pack index
          {"native_experts.txt", false, ""},  // native expert mapping
          {"scales", true, "experts=q3k"},  // per-layer column scales
          {"latent_dense_q8.bin", false, "du=q8"},  // D/U projections, Q8
          {"latent_dense_q4.bin", false, "du=q4"},  // D/U projections, Q4
          {"model-skeleton.gguf", false, ""},  // backbone: attention, routing, shared experts, embeddings, output head
          {"ple-fp8.gguf", false, ""},  // per-layer embedding table (FP8)
          {"profile.bin", false, ""},  // expert ranking that decides which experts are cached up front
          {"mtp", true, ""},  // original MTP draft layer (speculative decoding)
          {"tokenizer", true, ""}},  // tokenizer
         {{"du", {"q4", "q8"}}, {"ple-io", {"direct", "ram"}}},
         true,
         true,
         {{"edge0-120b-q3k", "q3k"}, {"edge0-120b-fp4", "fp4"}},
         {{"metal", {{"du", {"q4"}}, {"ple-io", {"direct"}}}}}},
    };
    return entries;
}

namespace {
namespace fs = std::filesystem;
std::uintmax_t size_of(const fs::path& path, bool directory) {
    std::error_code error;
    if (!directory) {
        const auto bytes = fs::file_size(path, error);
        return error ? 0 : bytes;
    }
    std::uintmax_t total = 0;
    for (fs::recursive_directory_iterator it(path, fs::directory_options::follow_directory_symlink, error), end;
         !error && it != end; it.increment(error)) {
        std::error_code inner;
        if (it->is_regular_file(inner)) total += it->file_size(inner);
    }
    return total;
}
const std::vector<std::pair<std::string, std::vector<std::string>>>& options_for(const Registration& registration, const std::string& device) {
    const auto found = registration.device_options.find(device);
    return found == registration.device_options.end() ? registration.options : found->second;
}
bool condition_holds(const std::string& condition, const std::map<std::string, std::string>& options) {
    if (condition.empty()) return true;
    const auto equals = condition.find('=');
    const auto found = options.find(condition.substr(0, equals));
    return found != options.end() && found->second == condition.substr(equals + 1);
}
std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}
}

std::vector<std::string> Match::missing(const std::map<std::string, std::string>& options) const {
    auto conditions = options;
    conditions["experts"] = experts;
    std::vector<std::string> result;
    for (const auto& state : files)
        if (!state.exists && condition_holds(state.file->condition, conditions))
            result.push_back(state.file->path + (state.file->directory ? "/" : ""));
    return result;
}

bool Match::complete(const std::map<std::string, std::string>& selected, const std::string& device) const {
    if (!metal_rejection(*this, device, selected).empty()) return false;
    return missing(resolve_options(*registration, selected, device)).empty();
}

std::vector<std::string> Match::alternatives(const std::map<std::string, std::string>& options, const std::string& device) const {
    auto conditions = options;
    conditions["experts"] = experts;
    std::vector<std::string> involved;
    for (const auto& state : files) {
        if (state.exists || !condition_holds(state.file->condition, conditions)) continue;
        if (state.file->condition.empty() || state.file->condition.rfind("experts=", 0) == 0) return {};
        const auto name = state.file->condition.substr(0, state.file->condition.find('='));
        if (std::find(involved.begin(), involved.end(), name) == involved.end()) involved.push_back(name);
    }
    std::vector<std::string> result;
    for (const auto& option : options_for(*registration, device)) {
        if (std::find(involved.begin(), involved.end(), option.first) == involved.end()) continue;
        const auto value = std::find_if(option.second.begin(), option.second.end(), [&](const std::string& value) {
            const auto condition = option.first + "=" + value;
            return value != options.at(option.first) && std::all_of(files.begin(), files.end(), [&](const auto& state) {
                       return state.file->condition != condition || state.exists;
                   });
        });
        if (value == option.second.end()) return {};
        result.push_back(option.first + "=" + *value);
    }
    return result;
}

static std::optional<std::string> model_name_of(const fs::path& directory) {
    std::error_code error;
    if (!fs::is_regular_file(directory / "model.json", error)) return std::nullopt;
    std::ifstream input(directory / "model.json", std::ios::binary);
    if (!input) return std::nullopt;
    const auto value = Json::parse(input, nullptr, false);
    if (value.is_discarded() || !value.is_object() || value.contains("format_version") || !value.contains("model_name") || !value["model_name"].is_string())
        return std::nullopt;
    return value["model_name"].get<std::string>();
}

std::optional<Match> detect(const fs::path& directory) {
    // The native library reports every problem with model.json itself; only a registered
    // model_name makes a directory delegated.
    const auto name = model_name_of(directory);
    if (!name) return std::nullopt;
    for (const auto& registration : registry()) {
        const auto named = registration.names.find(*name);
        if (named == registration.names.end()) continue;
        Match match;
        match.registration = &registration;
        match.model_name = *name;
        match.experts = named->second;
        for (const auto& file : registration.files) {
            if (file.condition.rfind("experts=", 0) == 0 && file.condition != "experts=" + match.experts) continue;
            FileState state;
            state.file = &file;
            const auto path = directory / file.path;
            std::error_code inner;
            state.exists = file.directory ? fs::is_directory(path, inner) : fs::is_regular_file(path, inner);
            if (state.exists) state.bytes = size_of(path, file.directory);
            match.files.push_back(state);
        }
        return match;
    }
    return std::nullopt;
}

std::optional<std::string> model_argument(int argc, char** argv) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--model") return std::string(argv[i + 1]);
    return std::nullopt;
}

Json inspect_json(const Match& match, const fs::path& directory, const std::map<std::string, std::string>& selected,
                  const std::string& device) {
    Json files = Json::array();
    for (const auto& state : match.files)
        files.push_back({{"filename", state.file->path + (state.file->directory ? "/" : "")},
                         {"required", state.file->condition.empty() ? "always" : state.file->condition},
                         {"exists", state.exists},
                         {"bytes", state.bytes}});
    return {{"adapter", match.registration->id},
            {"delegated", true},
            {"engine", match.registration->engine},
            {"directory", directory.string()},
            {"ok", match.complete(selected, device)},
            {"files", files}};
}

void reject_options_illegal_everywhere(const Registration& registration, const std::map<std::string, std::string>& given,
                                       const std::string& device);

InspectResult inspect_report(const Match& match, const fs::path& directory, const std::map<std::string, std::string>& given,
                             const std::string& device) {
    reject_options_illegal_everywhere(*match.registration, given, device);
    const auto reason = metal_rejection(match, device, given);
    const auto selected = reason.empty() ? resolve_options(*match.registration, given, device) : given;
    auto report = inspect_json(match, directory, selected, device);
    if (report["ok"].get<bool>()) return {std::move(report), {}, 0};
    if (!reason.empty()) return {std::move(report), reason, 1};
    std::string message = "asset_invalid: " + match.registration->id + " directory is incomplete; missing: ";
    const auto missing = match.missing(selected);
    for (std::size_t i = 0; i < missing.size(); ++i) message += (i ? ", " : "") + missing[i];
    message += '.';
    const auto alternatives = match.alternatives(selected, device);
    if (!alternatives.empty()) {
        // "instead": the user may already have given these options explicitly, and repeating one is an error.
        message += " To use another installed format, use";
        for (const auto& choice : alternatives) message += " --model-option " + choice;
        message += " instead.";
    }
    return {std::move(report), message, 1};
}

std::string metal_rejection(const Match& match, const std::string& device, const std::map<std::string, std::string>& given) {
    if (device != "metal") return {};
    if (match.experts != "q3k")
        return "unsupported on Metal: " + match.model_name + " uses fp4 experts; Metal supports q3k";
    const auto rejected = [](const std::map<std::string, std::string>& given, const char* name, const char* only) -> std::string {
        const auto found = given.find(name);
        if (found == given.end() || found->second == only) return {};
        // A value that is not allowed on any device keeps the generic "must be" error from resolve_options.
        return "unsupported on Metal: " + std::string(name) + "=" + found->second + "; Metal only supports " + name + "=" + only;
    };
    if (const auto du = rejected(given, "du", "q4"); !du.empty() && given.at("du") == "q8") return du;
    if (const auto ple = rejected(given, "ple-io", "direct"); !ple.empty() && given.at("ple-io") == "ram") return ple;
    return {};
}

ParsedDevice parse_device(const std::string& value, Platform platform) {
    if (platform == Platform::macos) {
        if (value == "metal") return {"metal", 0};
        throw ConfigError("invalid_argument: on macOS this model runs on Metal through edge0-strata; "
                          "--device must be metal (got " + value + ")");
    }
    const std::string message = "invalid_argument: this model runs on a CUDA GPU through edge0-strata; "
                                "--device must be cuda or cuda:<n> (got " + value + ")";
    if (value == "cuda") return {"cuda:0", 0};
    if (value.size() <= 5 || value.compare(0, 5, "cuda:") != 0) throw ConfigError(message);
    const auto digits = value.substr(5);
    if (digits.size() > 3 || digits.find_first_not_of("0123456789") != std::string::npos) throw ConfigError(message);
    const int gpu = std::stoi(digits);
    return {"cuda:" + std::to_string(gpu), gpu};
}

std::string default_device(Platform platform) {
    return platform == Platform::macos ? "metal" : "cuda:0";
}

std::map<std::string, std::string> resolve_options(const Registration& registration, const std::map<std::string, std::string>& given,
                                                   const std::string& device) {
    const auto& options = options_for(registration, device);
    std::map<std::string, std::string> result;
    for (const auto& item : given) {
        const auto option = std::find_if(options.begin(), options.end(),
                                         [&](const auto& candidate) { return candidate.first == item.first; });
        if (option == options.end()) {
            std::string names;
            for (const auto& candidate : options) names += (names.empty() ? "" : ", ") + candidate.first;
            throw ConfigError("invalid_argument: unknown model option " + item.first + " for " + registration.id + " (supported: " + names + ")");
        }
        if (std::find(option->second.begin(), option->second.end(), item.second) == option->second.end()) {
            std::string values;
            for (const auto& value : option->second) values += (values.empty() ? "" : "|") + value;
            throw ConfigError("invalid_argument: model option " + item.first + " must be " + values + " (got " + item.second + ")");
        }
        result[item.first] = item.second;
    }
    for (const auto& option : options) result.emplace(option.first, option.second.front());
    return result;
}

void reject_options_illegal_everywhere(const Registration& registration, const std::map<std::string, std::string>& given,
                                       const std::string& device) {
    // A name or value no device accepts is a configuration error, ahead of the Metal fp4 report.
    auto accepted = [](const std::vector<std::pair<std::string, std::vector<std::string>>>& options, const std::string& name,
                       const std::string& value) {
        for (const auto& option : options)
            if (option.first == name && std::find(option.second.begin(), option.second.end(), value) != option.second.end())
                return true;
        return false;
    };
    for (const auto& item : given) {
        bool any = accepted(registration.options, item.first, item.second);
        for (const auto& entry : registration.device_options)
            any = any || accepted(entry.second, item.first, item.second);
        if (!any) resolve_options(registration, {{item.first, item.second}}, device);
    }
}

fs::path locate_engine(const std::optional<std::string>& given, const fs::path& executable) {
    auto valid = [](const fs::path& path) {
        std::error_code error;
        return fs::is_regular_file(path / "edge0" / "CONTRACT", error);
    };
    std::string source;
    fs::path candidate;
    if (given) {
        candidate = *given;
        source = "--strata-dir";
    } else if (const char* value = std::getenv("EDGE0_STRATA_DIR"); value && *value) {
        candidate = value;
        source = "EDGE0_STRATA_DIR";
    } else {
        for (auto folder = executable.parent_path(); !folder.empty(); folder = folder.parent_path()) {
            if (valid(folder / "edge0-strata")) return fs::absolute(folder / "edge0-strata");
            if (folder == folder.parent_path()) break;
        }
        throw ConfigError("edge0-strata not found: pass --strata-dir <directory> or set EDGE0_STRATA_DIR "
                          "(no edge0-strata/ directory beside the edge0-infer installation)");
    }
    std::error_code error;
    if (!fs::is_directory(candidate, error))
        throw ConfigError("edge0-strata not found: " + candidate.string() + " (from " + source + ") is not a directory");
    if (!valid(candidate))
        throw ConfigError("edge0-strata not found: " + candidate.string() + " (from " + source + ") has no edge0/CONTRACT");
    return fs::absolute(candidate);
}

void check_contract(const fs::path& engine) {
    std::ifstream input(engine / "edge0" / "CONTRACT", std::ios::binary);
    std::ostringstream text;
    text << input.rdbuf();
    const auto value = trim(text.str());
    if (!input || value.empty() || value.find_first_not_of("0123456789") != std::string::npos || value.size() > 6)
        throw ConfigError("edge0-strata launch contract is unreadable: " + (engine / "edge0" / "CONTRACT").string());
    if (std::stoi(value) != supported_contract)
        throw ConfigError("edge0-strata launch contract version " + value + " is not supported; this edge0-infer supports version " +
                          std::to_string(supported_contract));
}

Json load_template(const fs::path& engine, const Registration& registration, const std::string& device) {
    fs::path relative = registration.config_template;
    if (device == "metal")
        relative = relative.parent_path() / (relative.stem().string() + "-metal" + relative.extension().string());
    const auto path = engine / relative;
    std::ifstream input(path, std::ios::binary);
    if (!input) throw ConfigError("edge0-strata has no configuration template for " + registration.id + ": " + path.string());
    const auto value = Json::parse(input, nullptr, false);
    if (value.is_discarded() || !value.is_object() || !value.contains("exe") || !value.contains("args") || !value["args"].is_array())
        throw ConfigError("edge0-strata configuration template is invalid: " + path.string());
    return value;
}

Json expand_template(const Json& config_template, const Registration& registration, const LaunchOptions& options, Platform platform,
                     std::vector<PrepareStep>* prepare) {
    const auto du = options.model_options.at("du");
    const std::map<std::string, std::string> values = {
        {"STRATA_DIR", options.strata_dir.string()},
        {"EXE_SUFFIX", platform == Platform::windows ? ".exe" : ""},
        {"MODEL_DIR", options.model.string()},
        {"LAT_FUSED_Q3K", options.experts == "q3k" ? "1" : "0"},
        {"DU_FILE", (options.model / ("latent_dense_" + du + ".bin")).string()},
        {"DU_FMT", du == "q4" ? "4" : "8"},
        {"PLE_IO", options.model_options.at("ple-io")},
    };
    std::function<Json(const Json&)> expand = [&](const Json& value) -> Json {
        if (value.is_string()) {
            std::string text = value.get<std::string>(), out;
            for (std::size_t i = 0; i < text.size();) {
                if (text.compare(i, 2, "${") == 0) {
                    const auto end = text.find('}', i);
                    const auto name = end == std::string::npos ? std::string() : text.substr(i + 2, end - i - 2);
                    const auto found = values.find(name);
                    if (found == values.end())
                        throw ConfigError("edge0-strata configuration template uses an unknown placeholder: " + text.substr(i));
                    out += found->second;
                    i = end + 1;
                } else out += text[i++];
            }
            return out;
        }
        if (value.is_array()) {
            Json result = Json::array();
            for (const auto& item : value) result.push_back(expand(item));
            return result;
        }
        if (value.is_object()) {
            Json result = Json::object();
            for (const auto& item : value.items()) {
                if (item.key().find("${CUDA_LIB_DIR}") != std::string::npos)
                    throw ConfigError("${CUDA_LIB_DIR} must be a whole item of lib_dirs");
                result[item.key()] = expand(item.value());
            }
            return result;
        }
        return value;
    };
    Json prepared = config_template;
    if (prepared.contains("lib_dirs") && prepared["lib_dirs"].is_array()) {
        Json directories = Json::array();
        for (const auto& entry : prepared["lib_dirs"]) {
            if (entry == "${CUDA_LIB_DIR}") {
                if (platform == Platform::windows)
                    for (const auto& path : options.lib_dirs) directories.push_back(path.string());
            } else directories.push_back(entry);
        }
        prepared["lib_dirs"] = directories;
    }
    Json config = expand(prepared);
    config["model_name"] = options.alias.empty() ? registration.id : options.alias;
    config["gpu"] = options.gpu;
    if (options.context) {
        auto& args = config["args"];
        bool replaced = false;
        for (std::size_t i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--max-context") { args[i + 1] = std::to_string(*options.context); replaced = true; }
        if (!replaced) { args.push_back("--max-context"); args.push_back(std::to_string(*options.context)); }
    }
    // Machine tuning for the engine, given word by word. Only options ("--...") are compared, so a value that
    // equals one of the template's values is passed on; the template's model settings cannot be overridden.
    auto& args = config["args"];
    const auto template_args = args;
    for (const auto& word : options.strata_args) {
        if (word == "--max-context") throw ConfigError("invalid_argument: --strata-arg --max-context: use --context");
        if (word.rfind("--", 0) == 0 && std::find(template_args.begin(), template_args.end(), word) != template_args.end())
            throw ConfigError("invalid_argument: --strata-arg " + word + ": set by the launch template");
        args.push_back(word);
    }
    // `prepare` is carried out by edge0-infer before the server starts. It is not part of the server's config.
    std::vector<PrepareStep> steps;
    if (config.contains("prepare")) {
        if (!config["prepare"].is_array())
            throw ConfigError("edge0-strata configuration template prepare must be an array");
        for (const auto& item : config["prepare"]) {
            if (!item.is_object() || !item.contains("output") || !item["output"].is_string() || item["output"].get<std::string>().empty() ||
                !item.contains("script") || !item["script"].is_string() || item["script"].get<std::string>().empty() ||
                !item.contains("args") || !item["args"].is_array())
                throw ConfigError("edge0-strata configuration template prepare entry needs output, script and args");
            PrepareStep step;
            step.output = item["output"].get<std::string>();
            fs::path script = item["script"].get<std::string>();
            if (script.is_relative()) script = options.strata_dir / script;
            step.script = std::move(script);
            for (const auto& arg : item["args"]) {
                if (!arg.is_string()) throw ConfigError("edge0-strata configuration template prepare args must be strings");
                step.args.push_back(arg.get<std::string>());
            }
            steps.push_back(std::move(step));
        }
        config.erase("prepare");
    }
    if (prepare) *prepare = std::move(steps);
    return config;
}

}

namespace delegate {
bool common_argument(const std::string& key, const std::string& value, CommonArgs& args) {
    if (key == "--model") args.model = value;
    else if (key == "--device") args.device = value;
    else if (key == "--prefill-device") args.prefill_device = value;
    else if (key == "--alias") args.alias = value;
    else if (key == "--strata-dir") args.strata_dir = value;
    else if (key == "--strata-python") args.python = value;
    else if (key == "--strata-arg") {
        if (value.empty()) throw ConfigError("invalid_argument: --strata-arg needs a value");
        args.strata_args.push_back(value);
    }
    else if (key == "--model-option") {
        const auto equals = value.find('=');
        if (equals == std::string::npos || !equals || equals + 1 == value.size()) throw ConfigError("invalid_argument: expected model option name=value");
        if (!args.model_options.emplace(value.substr(0, equals), value.substr(equals + 1)).second)
            throw ConfigError("invalid_argument: model option given more than once: " + value.substr(0, equals));
    } else if (key == "--context") {
        if (value.empty() || value.size() > 9 || value.find_first_not_of("0123456789") != std::string::npos || std::stoll(value) == 0)
            throw ConfigError("invalid_argument: --context must be a positive integer");
        args.context = std::stoll(value);
    } else if (key == "--strata-ready-timeout") {
        if (value.empty() || value.size() > 6 || value.find_first_not_of("0123456789") != std::string::npos || std::stoll(value) == 0)
            throw ConfigError("invalid_argument: --strata-ready-timeout must be a positive number of seconds");
        args.ready_timeout = static_cast<double>(std::stoll(value));
    } else return false;
    return true;
}

void unsupported_option(const std::string& key) {
    throw ConfigError("invalid_argument: " + key + " is not available for a delegated model: edge0-strata runs it with its own "
                      "fixed settings (send sampling fields with each request instead)");
}

LaunchOptions resolve(const CommonArgs& args, const Match& match, const fs::path& executable, Platform platform,
                      const std::string& cuda_path) {
    LaunchOptions options;
    const auto parsed = parse_device(args.device.value_or(default_device(platform)), platform);
    options.device = parsed.name;
    options.gpu = parsed.gpu;
    if (args.prefill_device) {
        if (parsed.name == "metal") {
            if (parse_device(*args.prefill_device, platform).name != parsed.name)
                throw ConfigError("invalid_argument: --prefill-device must equal --device for a delegated model (edge0-strata uses one GPU)");
        } else if (*args.prefill_device != args.device.value_or(default_device(platform))) {
            // Same string comparison as before Metal: "cuda" and "cuda:0" are not the same text, and
            // "cuda:01" is not "cuda:0".
            throw ConfigError("invalid_argument: --prefill-device must equal --device for a delegated model (edge0-strata uses one GPU)");
        }
    }
    reject_options_illegal_everywhere(*match.registration, args.model_options, parsed.name);
    if (const auto reason = metal_rejection(match, parsed.name, args.model_options); !reason.empty()) throw ConfigError(reason);
    options.model_options = resolve_options(*match.registration, args.model_options, parsed.name);
    const auto missing = match.missing(options.model_options);
    if (!missing.empty()) {
        std::string list;
        for (const auto& name : missing) list += (list.empty() ? "" : ", ") + name;
        throw ConfigError("asset_invalid: " + match.registration->id + " directory is incomplete; missing: " + list);
    }
    options.model = fs::absolute(args.model);
    options.experts = match.experts;
    options.alias = args.alias.empty() ? match.registration->id : args.alias;
    options.context = args.context;
    options.strata_args = args.strata_args;
    options.ready_timeout = args.ready_timeout;
    options.strata_dir = locate_engine(args.strata_dir, executable);
    options.platform = platform;
    options.python = fs::absolute(args.python ? fs::path(*args.python) : options.strata_dir / ".venv" /
        (platform == Platform::windows ? fs::path("Scripts") / "python.exe" : fs::path("bin") / "python"));
    if (platform == Platform::windows) {
        bool found = false;
        std::string checked = "CUDA_PATH is not set (no directories checked)";
        if (!cuda_path.empty()) {
            checked.clear();
            for (const auto& path : {fs::path(cuda_path) / "bin" / "x64", fs::path(cuda_path) / "bin"}) {
                checked += (checked.empty() ? "" : ", ") + path.string();
                std::error_code error;
                if (fs::is_directory(path, error)) options.lib_dirs.push_back(path);
                if (fs::is_regular_file(path / "cublas64_13.dll", error)) found = true;
            }
        }
        if (!found) options.warnings.push_back("cublas64_13.dll not found; checked: " + checked +
            "; edge0-strata may exit with 0xC0000135. Continuing with the inherited PATH.");
    }
    return options;
}
}
