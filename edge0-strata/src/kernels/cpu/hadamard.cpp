// src/kernels/cpu/hadamard.cpp - see include/strata/kernels/cpu/hadamard.hpp.
#include "strata/kernels/cpu/hadamard.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
HadamardSpec g_spec;
}

const HadamardSpec& hadamard_spec() { return g_spec; }
void hadamard_set(HadamardSpec h) { g_spec = std::move(h); }

const float* hadamard_signs(int width) {
    const auto it = g_spec.signs.find(width);
    return it == g_spec.signs.end() ? nullptr : it->second.data();
}

bool hadamard_parse(const std::string& path, int64_t n_layers, HadamardSpec& h, std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot read " + path; return false; }
    h = HadamardSpec{};
    h.roles.assign((size_t) n_layers, 0);
    std::vector<uint8_t> seen((size_t) n_layers, 0);
    std::string line;
    bool versioned = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            static const char tag[] = "# strata hadamard v";
            if (line.compare(0, sizeof tag - 1, tag) == 0) {
                if (std::atoi(line.c_str() + sizeof tag - 1) != 1) {
                    err = path + ": an unknown hadamard.txt version (this engine reads v1)";
                    return false;
                }
                versioned = true;
            }
            continue;
        }
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "block") {
            long long b = 0;
            if (!(ss >> b) || b < 2 || b > 4096 || (b & (b - 1)) != 0 || h.block) {
                err = path + ": a bad or repeated block line: " + line;
                return false;
            }
            h.block = (int) b;
        } else if (key == "signs") {
            long long w = 0;
            if (!(ss >> w) || w <= 0 || w > (1 << 20) || h.signs.count((int) w)) {
                err = path + ": a bad or repeated signs line";
                return false;
            }
            std::vector<float> v((size_t) w);
            for (auto& x : v) {
                int s = 0;
                if (!(ss >> s) || (s != 1 && s != -1)) { err = path + ": sign values must be +1 or -1"; return false; }
                x = (float) s;
            }
            std::string extra;
            if (ss >> extra) { err = path + ": a signs line longer than its width"; return false; }
            h.signs.emplace((int) w, std::move(v));
        } else if (key == "layer") {
            long long l = -1;
            int g = -1, u = -1, d = -1;
            if (!(ss >> l >> g >> u >> d) || l < 0 || l >= n_layers || (g | u | d) & ~1 || seen[(size_t) l]) {
                err = path + ": a bad or repeated layer line: " + line;
                return false;
            }
            if (g != u) {   // gate and up read one activation: a transform for one of them only is not runnable
                err = path + ": layer " + std::to_string(l) + " folds only one of gate/up";
                return false;
            }
            seen[(size_t) l] = 1;
            h.roles[(size_t) l] = (uint8_t) (g | (u << 1) | (d << 2));
        } else {
            err = path + ": an unknown line: " + line;
            return false;
        }
    }
    if (!versioned || h.block == 0) { err = path + ": no version header or no block line"; return false; }
    for (const auto& [w, v] : h.signs)
        if (w % h.block) { err = path + ": sign width " + std::to_string(w) + " is not whole blocks"; return false; }
    return true;
}

void hadamard_rotate(float* x, int n, int block, const float* signs) {
    if (signs)
        for (int i = 0; i < n; ++i) x[i] *= signs[i];
    const float scale = 1.0f / std::sqrt((float) block);
    for (int b0 = 0; b0 < n; b0 += block) {
        float* v = x + b0;
        // the in-place fast Walsh-Hadamard transform in natural (Sylvester) order: entry (-1)^popcount(r & c)
        for (int len = 1; len < block; len <<= 1)
            for (int i = 0; i < block; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const float a = v[j], c = v[j + len];
                    v[j] = a + c;
                    v[j + len] = a - c;
                }
        for (int i = 0; i < block; ++i) v[i] *= scale;
    }
}

}  // namespace strata::kernels::cpu
