// A token's native expert result must not depend on how many other tokens share its group.
// Compare the invariant entry points with the production singleton dispatch, including SwiGLU.
// Synthetic packed blocks only: no GPU, model download or private fixtures.
#include "ggml.h"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <utility>

namespace cpu = strata::kernels::cpu;
namespace {
constexpr int width = 8, hidden = 512;
size_t checked = 0;
bool equal(const std::vector<float> &actual, const std::vector<float> &expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        ++checked;
        if (!std::isfinite(actual[i]) || std::memcmp(&actual[i], &expected[i], sizeof(float)))
            return false;
    }
    return true;
}

bool check_format(int gu, int down, int ff, bool folded = false) {
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(gu, down, hidden, ff, fmt, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    std::vector<float> gu_signs(hidden), down_signs(ff);
    if (folded) {
        for (int i = 0; i < hidden; ++i) gu_signs[i] = i % 3 ? 1.f : -1.f;
        for (int i = 0; i < ff; ++i) down_signs[i] = i % 5 ? 1.f : -1.f;
        fmt.had_block = 128;
        fmt.had_gu = fmt.had_d = true;
        fmt.gu_signs = gu_signs.data();
        fmt.d_signs = down_signs.data();
    }
    std::mt19937 rng(20261016 + gu * 100 + down + ff);
    std::vector<uint8_t> blob(fmt.bytes);
    for (auto &byte : blob)
        byte = static_cast<uint8_t>(rng());
    // Use finite block scales and random packed values. NVFP4 has four UE4M3 sub-block scales.
    const uint16_t scale = 0x2400; // 2^-6; exclude NaN/Inf scales from the random bytes.
    auto scales = [&](size_t first, size_t last, int type) {
        const size_t block = cpu::tq_supported(type) ? cpu::tq_block_bytes(type) : ggml_type_size(static_cast<ggml_type>(type));
        for (size_t at = first; at < last; at += block) {
            if (type == cpu::kNvfp4Type) std::memset(blob.data() + at, 0x28, 4); // finite UE4M3 sub-block scales
            else {
                std::memcpy(blob.data() + at, &scale, sizeof(scale));
                if (type == 12 || type == 7) std::memcpy(blob.data() + at + 2, &scale, sizeof(scale));
            }
        }
    };
    scales(0, fmt.down_off, gu);
    scales(fmt.down_off, fmt.tail_off ? fmt.tail_off : blob.size(), down);
    if (fmt.tail_off) {
        const float tail[4] = {gu == cpu::kNvfp4Type ? .5f : 1.f, gu == cpu::kNvfp4Type ? .75f : 1.f,
                               down == cpu::kNvfp4Type ? 1.25f : 1.f, 0.f};
        std::memcpy(blob.data() + fmt.tail_off, tail, sizeof(tail));
    }

    std::vector<std::vector<uint8_t>> act(width, std::vector<uint8_t>(fmt.act_bytes));
    std::vector<std::vector<uint8_t>> hq(width, std::vector<uint8_t>(fmt.h_bytes));
    std::vector<std::vector<float>> gates(width, std::vector<float>(ff)), downs(width, std::vector<float>(hidden));
    auto solo_gates = gates, solo_downs = downs;
    const void *ap[width], *hp[width];
    float *gp[width], *dp[width], *sg[width], *sd[width];
    for (int t = 0; t < width; ++t) {
        ap[t] = act[t].data();
        hp[t] = hq[t].data();
        gp[t] = gates[t].data();
        dp[t] = downs[t].data();
        sg[t] = solo_gates[t].data();
        sd[t] = solo_downs[t].data();
    }
    for (float amplitude : {0.f, .01f, .3f, 3.f}) {
        std::vector<float> x(hidden), h(ff);
        for (int t = 0; t < width; ++t) {
            for (auto &v : x)
                v = (static_cast<int>(rng() % 2049) - 1024) * amplitude / 1024.f;
            for (auto &v : h)
                v = (static_cast<int>(rng() % 2049) - 1024) * amplitude / 1024.f;
            cpu::native_quant_act(fmt, x.data(), act[t].data());
            cpu::native_quant_h(fmt, h.data(), hq[t].data());
            cpu::native_gu_rows(fmt, blob.data(), ap + t, 1, sg + t, 0, ff);
            cpu::native_down_rows(fmt, blob.data(), hp + t, 1, sd + t, 0, hidden);
        }
        for (int nt = 1; nt <= width; ++nt) {
            cpu::native_gu_rows_invariant(fmt, blob.data(), ap, nt, gp, 0, ff);
            cpu::native_down_rows_invariant(fmt, blob.data(), hp, nt, dp, 0, hidden);
            for (int t = 0; t < nt; ++t) {
                if (!equal(gates[t], solo_gates[t]) || !equal(downs[t], solo_downs[t])) {
                    std::fprintf(stderr, "native arithmetic mismatch: gu=%d down=%d ff=%d nt=%d token=%d scale=%g\n",
                                 gu, down, ff, nt, t, amplitude);
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace

int main() {
    // 16/17 and IQ4_XS exercise the per-token legacy fallback as well as optimized 18/21/22.
    for (int gu : {16, 17, 18, 21, 22, 23}) {
        if (!check_format(gu, 20, 64) || !check_format(gu, 20, 96) || !check_format(gu, 42, 64))
            return 1;
    }
    // NVFP4's kernels already preserve singleton arithmetic. Mixed formats also carry scale tails.
    for (auto pair : {std::pair{40, 40}, {40, 20}, {40, 42}, {22, 40}, {23, 40}})
        if (!check_format(pair.first, pair.second, 64)) return 1;
    // STRATA_KQ256 selects grouped Q4_K/Q5_1/Q8_0 kernels only when their target arithmetic matches.
    if (!check_format(12, 7, 64) || !check_format(12, 8, 64)) return 1;
    // 0.2 adds Trellis formats and folded expert inputs. Exercise both their
    // production singleton dispatch and every grouped width through the new API.
    for (int gu : {144, 145, 146})
        for (int down : {20, 144, 145, 146})
            for (bool folded : {false, true})
                if (!check_format(gu, down, 256, folded)) return 1;
    if (!check_format(22, 20, 256, true)) return 1;
    std::printf("native_arithmetic_parity: %zu finite bitwise comparisons passed\n", checked);
}
