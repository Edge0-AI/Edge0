#include "edge0/text.h"
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
namespace {
void tokens(const std::vector<std::int32_t> &ids) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << ids[i];
    }
    std::cout << '\n';
}
std::string escape(const std::string &bytes) {
    std::ostringstream out;
    for (unsigned char c : bytes) {
        if (c == '\\') out << "\\\\";
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32 || c >= 127)
            out << "\\x" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
        else out << static_cast<char>(c);
    }
    return out.str();
}
int failure(const edge0::Error &e) {
    const char *name = e.category == edge0::ErrorCategory::asset_invalid ? "asset_invalid"
        : e.category == edge0::ErrorCategory::unsupported ? "unsupported"
        : e.category == edge0::ErrorCategory::invalid_argument ? "invalid_argument" : "incompatible";
    std::cerr << name << ": " << e.message << '\n';
    return 1;
}
int usage() {
    std::cerr << "Usage: edge0-infer tokenize --model <directory> [--parse-special] [--pieces] <text>\n"
                 "       edge0-infer prompt --model <directory> [--system <text>] --user <text> [--thinking on|off] [--tokens]\n";
    return 2;
}
}
int text_command(int argc, char **argv) {
    bool prompt = std::string(argv[1]) == "prompt";
    std::string directory;
    std::optional<std::string> input, system;
    bool parse_special = false, pieces = false, show_tokens = false;
    edge0::PromptOptions options;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc && directory.empty()) directory = argv[++i];
        else if (!prompt && arg == "--parse-special" && !parse_special) parse_special = true;
        else if (!prompt && arg == "--pieces" && !pieces) pieces = true;
        else if (prompt && arg == "--tokens" && !show_tokens) show_tokens = true;
        else if(prompt && arg=="--thinking" && i+1<argc) {
            const std::string value=argv[++i];if(value!="on" && value!="off") return usage();
            options.thinking=value=="on";
        }
        else if (prompt && arg == "--system" && i + 1 < argc && !system) system = argv[++i];
        else if (prompt && arg == "--user" && i + 1 < argc && !input) input = argv[++i];
        else if (!prompt && arg.rfind("--", 0) != 0 && !input) input = arg;
        else return usage();
    }
    if (directory.empty() || !input) return usage();
    try {
        if (prompt) {
            std::vector<edge0::TextMessage> messages;
            if (system) messages.push_back({"system", *system, {}});
            messages.push_back({"user", *input, {}});
            auto r = edge0::render_prompt(directory, messages,options);
            if (!r) return failure(r.error());
            std::cout << r.value().text;
            if (show_tokens) { std::cout << '\n'; tokens(r.value().tokens); }
        } else {
            auto r = edge0::tokenize(directory, *input, parse_special);
            if (!r) return failure(r.error());
            if (pieces) {
                for (std::size_t i = 0; i < r.value().tokens.size(); ++i)
                    std::cout << r.value().tokens[i] << '\t' << escape(r.value().pieces[i]) << '\n';
            } else tokens(r.value().tokens);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "asset_invalid: " << e.what() << '\n';
        return 1;
    }
}
