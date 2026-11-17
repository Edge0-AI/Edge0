// The qwen35 host sampler's penalties (include/strata/qwen35/sampling.hpp), CPU only, no model.
// #3911: a history token outside a short logit row (--draft-vocab N) must not write past the row.
#include "strata/qwen35/sampling.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

using strata::qwen35::Sampling;
using strata::qwen35::apply_penalties;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// The loop as it was before the guard (origin/main of the time), for the in-range comparison.
void old_penalties(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}  // namespace

int main() {
    Sampling s;
    s.repeat = 1.2f; s.freq = 0.1f; s.present = 0.05f; s.last_n = -1;

    // a short row (a 64-token draft vocabulary) and a history with tokens inside it, past it and negative.  The row's
    // storage past its size is filled with a marker first: a write past the row would change the marker.
    {
        std::vector<float> row(256, 7.f);
        row.resize(64);
        for (int i = 0; i < 64; ++i) row[(size_t) i] = (float) i - 20.f;
        const std::vector<float> before(row.begin(), row.end());
        const std::vector<int32_t> hist = {3, 100, 3, 63, 64, 65, 255, 100000, -1, 7};
        apply_penalties(row, s, hist);
        for (int i = 64; i < 256; ++i) check(row.data()[i] == 7.f, "nothing is written past the row");
        check(row.size() == 64, "the row keeps its size");
        // tokens 3 (twice), 63 and 7 are in the row and are penalised; the others are skipped
        std::vector<float> want = before;
        for (int id : {3, 7, 63}) {
            const int c = id == 3 ? 2 : 1;
            float& v = want[(size_t) id];
            v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
        check(same_bits(row, want), "the in-row tokens take the penalty, the rest of the row is untouched");
    }

    // history tokens that are all outside the row: the row does not change
    {
        std::vector<float> row(16, 1.5f);
        const std::vector<float> before = row;
        apply_penalties(row, s, {16, 17, 1000, -5});
        check(same_bits(row, before), "only out-of-row tokens: no change");
    }

    // a row of the whole vocabulary behaves exactly as before the guard, bit for bit, for every window and penalty mix
    {
        const int V = 512;
        std::vector<float> base((size_t) V);
        for (int i = 0; i < V; ++i) base[(size_t) i] = ((float) ((i * 37) % 101) - 50.f) / 7.f;
        std::vector<int32_t> hist;
        for (int i = 0; i < 300; ++i) hist.push_back((i * 53 + 11) % V);
        const float repeats[] = {1.f, 1.1f, 1.5f};
        const float freqs[] = {0.f, 0.2f};
        const float presents[] = {0.f, 0.3f};
        const int lasts[] = {0, -1, 1, 64, 300, 5000};
        for (float r : repeats) for (float f : freqs) for (float p : presents) for (int n : lasts) {
            Sampling t;
            t.repeat = r; t.freq = f; t.present = p; t.last_n = n;
            std::vector<float> a = base, b = base;
            apply_penalties(a, t, hist);
            old_penalties(b, t, hist);
            check(same_bits(a, b), "full-vocabulary row: identical to the unguarded loop");
        }
    }

    // no penalty configured: untouched even with a history
    {
        Sampling t;
        std::vector<float> row(8, 2.f);
        const std::vector<float> before = row;
        apply_penalties(row, t, {1, 2, 99});
        check(same_bits(row, before), "no penalty: no change");
    }

    // an empty row with a history does not crash
    {
        std::vector<float> row;
        apply_penalties(row, s, {0, 1, 2});
        check(row.empty(), "empty row");
    }

    std::printf("qwen35_sampling_test: %d checks passed\n", checks);
    return 0;
}
