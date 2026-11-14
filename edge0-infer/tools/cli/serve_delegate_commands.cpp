#include "serve_delegate.h"
#include <iostream>

// Command dispatch for delegated model directories (serve and chat live in their own files).
namespace delegate {
int refuse(const std::string& command, const Match& match) {
    std::cerr << "configuration_error: " << match.registration->id << " is a delegated model that runs on "
              << match.registration->engine << "; `" << command << "` cannot use it. Use `edge0-infer serve` or `edge0-infer chat`.\n";
    return 2;
}

int inspect(int argc, char** argv, const Match& match, Platform platform) try {
    bool as_json = false;
    std::string directory;
    CommonArgs options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc && directory.empty()) directory = argv[++i];
        else if (arg == "--json" && !as_json) as_json = true;
        else if (arg == "--model-option" && i + 1 < argc) common_argument(arg, argv[++i], options);
        else if (arg == "--digest") {
            std::cerr << "configuration_error: --digest is not available for delegated models (no checksums are registered)\n";
            return 2;
        } else {
            std::cerr << "Usage: edge0-infer inspect --model <directory> [--model-option name=value] [--json]\n";
            return 2;
        }
    }
    const auto result = inspect_report(match, directory, options.model_options, default_device(platform));
    if (as_json) std::cout << result.report.dump() << '\n';
    else {
        std::cout << "adapter: " << match.registration->id << "\ndelegated: " << match.registration->engine << "\nok: "
                  << (result.report["ok"].get<bool>() ? "true" : "false") << '\n';
        for (const auto& file : result.report["files"])
            std::cout << "file." << file["filename"].get<std::string>() << ": " << (file["exists"].get<bool>() ? "present" : "missing")
                      << " bytes=" << file["bytes"].get<std::uintmax_t>() << " required=" << file["required"].get<std::string>() << '\n';
    }
    if (result.code != 0) {
        std::cerr << result.message << '\n';
        return result.code;
    }
    return 0;
} catch (const ConfigError& error) {
    std::cerr << "configuration_error: " << error.what() << '\n';
    return 2;
}

std::optional<int> dispatch(const std::string& command, int argc, char** argv) {
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--help") return std::nullopt;
    const auto model = model_argument(argc, argv);
    if (!model) return std::nullopt;
    const auto match = detect(*model);
    if (!match) return std::nullopt;
    if (command == "serve") return serve(argc, argv, *match);
    if (command == "chat") return chat(argc, argv, *match);
    if (command == "inspect") return inspect(argc, argv, *match);
    if (command == "info") return std::nullopt;
    return refuse(command, *match);
}
}
