// metal/tools/gguf_list.cpp - tensor directory and scalar metadata of a GGUF (name type shape offset), for the
// Metal window's wiring.   clang++ -std=c++20 -I include metal/tools/gguf_list.cpp -o gguf_list
#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: gguf_list <file.gguf> [substring]\n"); return 2; }
    strata::GgufFile g(argv[1]);
    const std::string f = argc > 2 ? argv[2] : "";
    for (const auto& [k, v] : g.metadata()) {
        if (!f.empty() && k.find(f) == std::string::npos) continue;
        if (v.is_num()) std::printf("META %s = %g\n", k.c_str(), v.num());
        else if (v.type == strata::MetaType::STRING) std::printf("META %s = %s\n", k.c_str(), v.s.substr(0, 80).c_str());
        else std::printf("META %s = [%llu items]\n", k.c_str(), (unsigned long long) v.count);
    }
    for (const auto& t : g.tensors()) {
        if (!f.empty() && t.name.find(f) == std::string::npos) continue;
        std::printf("%s %s [", t.name.c_str(), t.type_name());
        for (size_t i = 0; i < t.shape.size(); ++i) std::printf(i ? ",%llu" : "%llu", (unsigned long long) t.shape[i]);
        std::printf("] off %llu\n", (unsigned long long) t.offset);
    }
    return 0;
}
