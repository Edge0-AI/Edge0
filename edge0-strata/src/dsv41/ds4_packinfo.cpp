// Strata. `strata-dsv41 --pack-info`: the pack as a description, before anything is loaded onto a GPU - the type map
// (layer x role), the bytes an expert and a token, the tier sizes, and the tensors whose type has no device kernel (they
// would go to the CPU lane, or stop the load).  Host only, no device, no weights read: tensors.txt and ds4.txt are the
// pack's whole description (tools/dsv41/pack_gguf.py).  The same report is the first line of every benchmark log.
#include "ds4_packinfo.hpp"
#include "ds4_iq.cuh"      // iq::supported / iq::name: the device-type list below is derived from the kernels, not copied

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <cstring>
#define strtok_r strtok_s
#endif
namespace strata::ds4 {
namespace {

// The ggml block types the device kernels take (ds4_iq.cu's Ty<> / DS4_IQ_SWITCH + ds4_gguf.cu's MXFP4).  A name here
// and not in the pack's tensor list is fine; a tensor whose type is not here is the CPU lane's (ggml-cpu's row dots).
// It was a copy of the supported() switch and fell behind when IQ1_S, IQ1_M and IQ4_NL got kernels (2026-10-23), so it is
// now the switch itself, by type name.

bool device_type(const std::string& t) {
    static const std::set<std::string> ok = [] {
        std::set<std::string> s = {"MXFP4", "TQK6", "TQK7", "TQK5X", "TQK4X", "TQK5T", "TQK4T"};   // ds4_gguf.cu / trellis
        for (int ty = 0; ty < 64; ++ty) {
            const char* n = gguf::iq::name(ty);
            if (n && std::strcmp(n, "?") != 0 && gguf::iq::supported(ty)) s.insert(n);
        }
        return s;
    }();
    return ok.count(t) != 0;
}
// The types the dense path reads as they are (ds4_gguf.cu: BF16 / F32 / MXFP8 rows, Q8_0 rows; ds4_iq's GEMV for the
// K-quants is the same device support as the experts').
bool dense_type(const std::string& t) {
    return t == "BF16" || t == "F32" || t == "F16" || t == "MXFP8" || t == "Q8_0" || device_type(t);
}

struct Tensor {
    std::string dtype;
    double bytes = 0;
    std::vector<long long> shape;
};

double gb(double b) { return b / 1e9; }

}  // namespace

int pack_info(const std::string& dir, std::FILE* out) {
    std::map<std::string, std::string> kv;
    {
        std::ifstream f(dir + "/ds4.txt");
        if (!f) {
            std::fprintf(out, "pack-info: %s/ds4.txt: not found\n", dir.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
    }
    auto num = [&](const char* k, long long d) {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::atoll(it->second.c_str());
    };
    const long long nl = num("num_hidden_layers", 0), ne = num("n_routed_experts", 0), topk = num("num_experts_per_tok", 0);
    const long long dim = num("hidden_size", 0), ff = num("moe_intermediate_size", 0), vocab = num("vocab_size", 0);
    std::map<std::string, Tensor> t;
    std::vector<std::string> files;
    std::map<int, double> need;   // bytes the pack needs from each file (its largest tensor end)
    {
        std::FILE* f = std::fopen((dir + "/tensors.txt").c_str(), "r");
        if (f == nullptr) {
            std::fprintf(out, "pack-info: %s/tensors.txt: not found\n", dir.c_str());
            return 1;
        }
        std::vector<char> buf(1 << 16);
        while (std::fgets(buf.data(), (int) buf.size(), f) != nullptr) {
            char* s = buf.data();
            if (s[0] == 'f') {   // file <i> <name>
                char name[4096];
                int i = 0;
                if (std::sscanf(s, "file %d %4095s", &i, name) == 2) {
                    if ((int) files.size() <= i) files.resize((size_t) i + 1);
                    files[(size_t) i] = name;
                }
                continue;
            }
            if (s[0] != 't') continue;
            char* save = nullptr;
            ::strtok_r(s, " \n", &save);
            const char* name = ::strtok_r(nullptr, " \n", &save);
            const char* dt = ::strtok_r(nullptr, " \n", &save);
            const char* fi = ::strtok_r(nullptr, " \n", &save);
            const char* off = ::strtok_r(nullptr, " \n", &save);
            const char* nb = ::strtok_r(nullptr, " \n", &save);
            if (name == nullptr || dt == nullptr || fi == nullptr || off == nullptr || nb == nullptr) continue;
            Tensor x;
            x.dtype = dt;
            x.bytes = (double) std::strtoull(nb, nullptr, 10);
            const int idx = std::atoi(fi);
            need[idx] = std::max(need[idx], (double) std::strtoull(off, nullptr, 10) + x.bytes);
            while (const char* d = ::strtok_r(nullptr, " \n", &save)) x.shape.push_back(std::atoll(d));
            t[name] = std::move(x);
        }
        std::fclose(f);
    }
    if (t.empty()) {
        std::fprintf(out, "pack-info: %s/tensors.txt: no tensors\n", dir.c_str());
        return 1;
    }
    // the files the pack points at: they must be there, and at least as long as the largest tensor end says (the weights
    // live elsewhere; a pack whose model_dir was cleaned up still describes itself perfectly - this is the check)
    const std::string md = kv.count("model_dir") ? kv["model_dir"] : "";
    int missing = 0;
    for (const auto& [i, n] : need) {
        const std::string name = (size_t) i < files.size() ? files[(size_t) i] : "";
        if (name.empty()) {
            std::fprintf(out, "  file %d: no name in tensors.txt\n", i);
            ++missing;
            continue;
        }
        const std::string path = md + "/" + name;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(out, "  MISSING %s (%.2f GB wanted)\n", path.c_str(), n / 1e9);
            ++missing;
            continue;
        }
#if defined(_WIN32)
        _fseeki64(f, 0, SEEK_END);   // (long is 32 bits there: the weights are far past 2 GB)
        const long long sz = _ftelli64(f);
#else
        std::fseek(f, 0, SEEK_END);
        const long long sz = std::ftell(f);
#endif
        std::fclose(f);
        if ((double) sz + 0.5 < n) {
            std::fprintf(out, "  SHORT %s: %.2f GB there, %.2f GB wanted\n", path.c_str(), (double) sz / 1e9, n / 1e9);
            ++missing;
        }
    }

    // ---- the expert map: layer -> role types and bytes an expert
    struct Role {
        std::string type[3];
        double bytes[3] = {0, 0, 0};
        bool mixed = false;
        int n = 0;
    };
    std::vector<Role> lay((size_t) nl);
    std::map<std::string, long long> type_count;   // tensor count a type (the experts' 3 roles)
    double experts_bytes = 0;
    std::set<std::string> no_kernel;
    for (long long il = 0; il < nl; ++il) {
        Role& r = lay[(size_t) il];
        const std::string pre = "layers." + std::to_string(il) + ".ffn.experts.";
        for (long long e = 0; e < ne; ++e) {
            const char* role[3] = {"w1", "w3", "w2"};
            for (int k = 0; k < 3; ++k) {
                auto it = t.find(pre + std::to_string(e) + "." + role[k] + ".weight");
                if (it == t.end()) {
                    std::fprintf(out, "pack-info: %s%s%lld.%s.weight: missing\n", pre.c_str(), "", e, role[k]);
                    return 1;
                }
                if (e == 0) {
                    r.type[k] = it->second.dtype;
                    r.bytes[k] = it->second.bytes;
                } else if (it->second.dtype != r.type[k] || it->second.bytes != r.bytes[k]) {
                    r.mixed = true;
                }
                if (e == 0) {
                    ++type_count[it->second.dtype];
                    if (!device_type(it->second.dtype)) no_kernel.insert("layer " + std::to_string(il) + " " + role[k] + " " + it->second.dtype);
                }
            }
        }
        r.n = (int) ne;
        experts_bytes += (double) ne * (r.bytes[0] + r.bytes[1] + r.bytes[2]);
    }
    const double per_expert = nl > 0 ? experts_bytes / ((double) nl * (double) ne) : 0;
    // (a token picks `topk` experts a layer: sum over the layers of topk x one expert of that layer)
    double token_bytes = 0;
    for (long long il = 0; il < nl; ++il) token_bytes += (double) topk * (lay[(size_t) il].bytes[0] + lay[(size_t) il].bytes[1] + lay[(size_t) il].bytes[2]);
    const double expert_params = (double) nl * (double) ne * 3.0 * (double) ff * (double) dim;

    // ---- the dense side: everything that is not an expert tensor
    double dense = 0, dense_embed = 0, dense_head = 0, dense_mtp = 0, dense_vision = 0, dense_engram = 0;
    std::set<std::string> dense_types;
    for (const auto& [name, x] : t) {
        if (name.find(".ffn.experts.") != std::string::npos && name.rfind("layers.", 0) == 0) continue;
        dense += x.bytes;
        dense_types.insert(x.dtype);
        if (name == "embed.weight") dense_embed += x.bytes;
        else if (name == "head.weight") dense_head += x.bytes;
        else if (name.rfind("mtp.", 0) == 0) dense_mtp += x.bytes;
        else if (name.find(".engram.") != std::string::npos) dense_engram += x.bytes;
        else if (name.rfind("vision.", 0) == 0 || name.rfind("aligner.", 0) == 0) dense_vision += x.bytes;
        if (!dense_type(x.dtype)) no_kernel.insert("dense " + name + " " + x.dtype);
    }

    // ---- report
    std::fprintf(out, "pack-info %s\n", dir.c_str());
    std::fprintf(out, "  model_dir %s   format %s   %lld layers x %lld experts, %lld used; dim %lld, ff %lld, vocab %lld\n",
                 kv.count("model_dir") ? kv["model_dir"].c_str() : "?", kv.count("format") ? kv["format"].c_str() : "?", nl, ne, topk,
                 dim, ff, vocab);
    std::fprintf(out, "  layer  gate+up   down      bytes/expert   MB/token\n");
    for (long long il = 0; il < nl; ++il) {
        const Role& r = lay[(size_t) il];
        const double eb = r.bytes[0] + r.bytes[1] + r.bytes[2];
        std::fprintf(out, "  %5lld  %-9s %-9s %13.0f %10.2f%s\n", il, r.type[0].c_str(), r.type[2].c_str(), eb,
                     (double) topk * eb / 1e6, r.mixed ? "  (mixed)" : "");
    }
    std::fprintf(out, "  expert tensors by type:");
    for (const auto& [ty, n] : type_count) std::fprintf(out, " %s %lld", ty.c_str(), n);
    std::fprintf(out, "\n");
    std::fprintf(out, "  an expert %.2f MB, %lld picks a token: %.3f GB a token\n", per_expert / 1e6, topk, token_bytes / 1e9);
    std::fprintf(out, "  all experts %.1f GB at %.2f bpw (%.0f G parameters), dense %.2f GB (embed %.2f, head %.2f, mtp %.2f, vision %.2f, engram %.2f)\n",
                 gb(experts_bytes), expert_params > 0 ? experts_bytes * 8.0 / expert_params : 0.0, expert_params / 1e9, gb(dense),
                 gb(dense_embed), gb(dense_head), gb(dense_mtp), gb(dense_vision), gb(dense_engram));
    std::fprintf(out, "  dense types:");
    for (const auto& d : dense_types) std::fprintf(out, " %s", d.c_str());
    std::fprintf(out, "\n");
    // the geometry the engine needs at run time but that is easy to lose in a pack built by another tool: DSpark's block
    // size / noise token / markov rank (0 there means it drafts one token a run, i.e. it is off) and the Engram's layers
    {
        auto kvn = [&](const char* k) {
            auto it = kv.find(k);
            return it == kv.end() ? 0 : std::atoll(it->second.c_str());
        };
        const int n_mtp = (int) std::count_if(t.begin(), t.end(), [](const auto& p) { return p.first.rfind("mtp.", 0) == 0; });
        const long long nl_eg = num("engram_layer_ids", -1);
        std::fprintf(out, "  dspark: block_size %lld, noise_token_id %lld, markov_rank %lld, routed experts %lld of %lld%s\n",
                     num("dspark_block_size", 0), num("dspark_noise_token_id", 0), num("dspark_markov_rank", 0),
                     num("dspark_n_routed_experts", 0), num("dspark_num_experts_per_tok", 0),
                     num("dspark_block_size", 0) == 0 ? "  <- DSpark cannot draft (block size 0)" : "");
        std::fprintf(out, "  mtp tensors: %d%s\n", n_mtp, n_mtp == 0 ? "  <- no DSpark drafter in the pack" : "");
        auto it = kv.find("engram_layer_ids");
        std::fprintf(out, "  engram layers: %s%s\n", it == kv.end() ? "(none)" : it->second.c_str(),
                     it == kv.end() ? "  <- the Engram tables are not used" : "");
    }
    if (no_kernel.empty()) {
        std::fprintf(out, "  no GPU kernel: none (every tensor is a device type)\n");
    } else {
        std::fprintf(out, "  no GPU kernel (%zu): the CPU lane takes them, they cannot go in the VRAM tier\n", no_kernel.size());
        int shown = 0;
        for (const std::string& s : no_kernel) {
            std::fprintf(out, "    %s\n", s.c_str());
            if (++shown >= 40) {
                std::fprintf(out, "    ... %zu more\n", no_kernel.size() - (size_t) shown);
                break;
            }
        }
    }
    if (missing > 0) {
        std::fprintf(out, "  %d of %zu files missing or short: this pack cannot load (build it again, or point model_dir at "
                          "the weights)\n", missing, need.size());
        return 2;
    }
    std::fprintf(out, "  files: %zu, all present and long enough\n", need.size());
    return 0;
}

}  // namespace strata::ds4
