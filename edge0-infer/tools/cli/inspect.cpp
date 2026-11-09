#include "edge0/assets.h"
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
namespace {
std::string quote(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 32)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
            else
                out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}
template <class T> void numbers(const T &values) {
    std::cout << '[';
    bool first = true;
    for (auto v : values) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << v;
    }
    std::cout << ']';
}
template <class T> void dictionary(const T &values) {
    std::cout << '{';
    bool first = true;
    for (const auto &v : values) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << quote(v.first) << ':' << v.second;
    }
    std::cout << '}';
}
void json(const edge0::AssetReport &r) {
    std::cout << "{\"architecture\":" << quote(r.architecture) << ",\"adapter\":" << quote(r.adapter)
              << ",\"gguf_architecture\":" << quote(r.gguf_architecture)
              << ",\"integrity\":" << quote(r.integrity) << ",\"parameters\":";
    dictionary(r.parameters);
    std::cout << ",\"numeric_parameters\":";
    dictionary(r.numeric_parameters);
    std::cout << ",\"layer_groups\":{";
    bool first_group = true;
    for (const auto& group : r.layer_groups) {
        if (!first_group) std::cout << ',';
        first_group = false;
        std::cout << quote(group.first) << ':';
        numbers(group.second);
    }
    std::cout << '}';
    std::cout << ",\"tensor_counts\":";
    dictionary(r.tensor_counts);
    std::cout << ",\"unused_roles\":[";
    bool first = true;
    for (const auto &s : r.unused_roles) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << quote(s);
    }
    std::cout << "],\"files\":[";
    first = true;
    for (const auto &f : r.files) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << "{\"role\":" << quote(f.role) << ",\"filename\":" << quote(f.filename)
                  << ",\"bytes\":" << f.bytes
                  << ",\"digest_verified\":" << (f.digest_verified ? "true" : "false") << '}';
    }
    std::cout << "],\"tensor_requirements\":[";
    first = true;
    for (const auto &t : r.tensor_requirements) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << "{\"role\":" << quote(t.role) << ",\"name\":" << quote(t.name)
                  << ",\"type\":" << quote(t.type) << ",\"shape\":";
        numbers(t.shape);
        std::cout << '}';
    }
    std::cout << "],\"expert_slices\":[";
    first = true;
    for (const auto &s : r.expert_slices) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << "{\"layer\":" << s.layer << ",\"expert\":" << s.expert
                  << ",\"projection\":" << quote(s.projection) << ",\"file\":" << quote(s.file)
                  << ",\"type\":" << quote(s.type) << ",\"offset\":" << s.offset << ",\"length\":" << s.length
                  << '}';
    }
    std::cout << "]}\n";
}
void text(const edge0::AssetReport &r) {
    std::cout << "architecture: " << r.architecture << "\nadapter: " << r.adapter
              << "\ngguf.architecture: " << r.gguf_architecture << "\nintegrity: " << r.integrity << '\n';
    for (const auto &f : r.files)
        std::cout << "file." << f.role << ": " << f.filename << " bytes=" << f.bytes
                  << " digest=" << (f.digest_verified ? "verified" : "not_verified") << '\n';
    std::cout << "unused_roles:";
    for (const auto &s : r.unused_roles)
        std::cout << ' ' << s;
    std::cout << "\nmodel.layers: " << r.layers;
    for (const auto& group : r.layer_groups) {
        std::cout << "\nmodel.layers." << group.first << ':';
        for (auto n : group.second) std::cout << ' ' << n;
    }
    std::cout << "\nmodel.experts: " << r.experts << " used=" << r.experts_per_token
              << '\n';
    for (const auto &p : r.parameters)
        std::cout << "model." << p.first << ": " << p.second << '\n';
    for (const auto &p : r.tensor_counts)
        std::cout << "tensors." << p.first << ": " << p.second << '\n';
    for (const auto &s : r.expert_slices)
        if (s.expert == 0 || s.expert + 1 == r.experts)
            std::cout << "expert." << s.layer << '.' << s.expert << '.' << s.projection << ": file=" << s.file
                      << " offset=" << s.offset << " length=" << s.length << " type=" << s.type << '\n';
}
} // namespace
int inspect_command(int argc, char **argv) {
    edge0::InspectOptions options;
    bool as_json = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc && options.directory.empty())
            options.directory = argv[++i];
        else if (arg == "--digest" && !options.digest)
            options.digest = true;
        else if (arg == "--json" && !as_json)
            as_json = true;
        else {
            std::cerr << "Usage: edge0-infer inspect --model <directory> [--digest] [--json]\n";
            return 2;
        }
    }
    if (options.directory.empty()) {
        std::cerr << "Usage: edge0-infer inspect --model <directory> [--digest] [--json]\n";
        return 2;
    }
    try {
        auto result = edge0::inspect_assets(options);
        if (!result) {
            const auto &e = result.error();
            const char *category = e.category == edge0::ErrorCategory::asset_invalid ? "asset_invalid"
                                   : e.category == edge0::ErrorCategory::unsupported ? "unsupported"
                                                                                   : "incompatible";
            std::cerr << category << ": " << e.message << '\n';
            return 1;
        }
        if (as_json)
            json(result.value());
        else
            text(result.value());
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "asset_invalid: " << e.what() << '\n';
        return 1;
    }
}
