// src/spec/suffix_drafter_test.cpp - plan v0.3 P6: suffix-lookup drafter tests and an offline simulation.
//
//   suffix_drafter_test                      unit tests
//   suffix_drafter_test --simulate [--k K] FILE.ids ...
//
// The simulation replays text as if a model had produced it: each step drafts up to K tokens, accepts the
// prefix that matches the true next tokens, and commits that prefix plus one model token (the verify pass's
// bonus). tokens/step is the lookup drafter's ceiling speedup on that text when a K+1-token verify costs the same
// as a 1-token step. Two regimes per file:
//   continue  history = first half of the prompt; generate its second half (natural continuation)
//   copy      history = the whole prompt; generate its middle third again (output quoting input: edits)
// No model runs; this measures the drafter on text, not on the model's own outputs.
#include "strata/spec/suffix_drafter.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using strata::spec::SuffixDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

struct Sim { long steps = 0, tokens = 0, proposals = 0, accepted = 0; };

Sim simulate(const std::vector<int32_t>& history, const std::vector<int32_t>& target, int K) {
    SuffixDrafter d(3, 32, history.size() + target.size() + 16);
    d.append(history.data(), history.size());
    Sim s;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < target.size()) {
        const int n = d.propose(K, draft.data());
        int a = 0;
        while (a < n && i + a < target.size() && draft[a] == target[i + a]) ++a;
        if (n > 0) { ++s.proposals; s.accepted += a; }
        const size_t commit = std::min(target.size() - i, (size_t) a + 1);
        d.append(&target[i], commit);
        i += commit;
        s.tokens += (long) commit;
        ++s.steps;
    }
    return s;
}

void unit_tests() {
    {   // a repeated passage is proposed verbatim
        SuffixDrafter d;
        const std::vector<int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        d.append(doc.data(), doc.size());
        const int32_t again[] = {10, 11, 12};
        d.append(again, 3);
        int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    {   // no earlier occurrence: nothing proposed
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "no match proposes nothing");
    }
    {   // two occurrences of the trigram: the longer match wins even though it is older
        SuffixDrafter d;
        const std::vector<int32_t> doc = {7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    {   // a match shorter than min_match is ignored (only a bigram in common)
        SuffixDrafter d(4);
        const std::vector<int32_t> doc = {1, 2, 3, 9, 5, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    {   // periodic text: the continuation may run into the current suffix
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    {   // bounded memory: appending beyond the nominal capacity does not crash and still works
        SuffixDrafter d(3, 32, 1024);
        std::vector<int32_t> doc;
        for (int i = 0; i < 5000; ++i) doc.push_back(i % 997);
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflow of nominal capacity");
    }
    {
        // --lookup-chain: the match continues from the drafts appended to the history, which stays unchanged
        SuffixDrafter d(3, 32, 256);
        const int32_t hist[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 1, 2};
        d.append(hist, sizeof(hist) / sizeof(hist[0]));
        const int32_t drafts[] = {3, 4};
        int32_t out[8] = {};
        const int k = d.propose_after(drafts, 2, 3, out);
        check(k == 3 && out[0] == 5 && out[1] == 6 && out[2] == 7, "propose_after continues after the drafts");
        check(d.last_match() >= 4, "propose_after match length counts the drafts");
        check(d.size() == sizeof(hist) / sizeof(hist[0]), "propose_after leaves the history unchanged");
        const int32_t miss[] = {99, 98, 97};
        check(d.propose_after(miss, 3, 3, out) == 0, "propose_after without a match proposes nothing");
        // the plain proposal is unaffected: the history's own suffix (52, 1, 2) has no earlier occurrence
        check(d.propose(3, out) == 0, "propose after propose_after");
        check(d.propose_after(nullptr, 0, 3, out) == d.propose(3, out), "propose_after with no drafts == propose");
    }
    {   // randomized soundness (run under ASan/UBSan in CI-like builds): small vocabularies repeat a lot, tiny tables fill up,
        // propose_after is interleaved; every proposal must be a real earlier continuation of the matched suffix
        uint64_t rng = 0x243F6A8885A308D3ull;
        auto next = [&rng] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
        long proposals = 0, bad = 0;
        for (int round = 0; round < 300; ++round) {
            const int vocab = 2 + (int) (next() % 6), min_m = 3 + (int) (next() % 3);
            const size_t cap = 8 + (size_t) (next() % 200);   // often far below the history: the table fills
            SuffixDrafter d(min_m, 4 + (int) (next() % 40), cap);
            std::vector<int32_t> hist;
            for (int step = 0; step < 400; ++step) {
                const int burst = 1 + (int) (next() % 3);
                for (int b = 0; b < burst; ++b) {
                    const int32_t t = hist.size() > 8 && (next() % 3) == 0 ? hist[hist.size() - 1 - (size_t) (next() % 8)]
                                                                           : (int32_t) (next() % (uint64_t) vocab);
                    hist.push_back(t);
                    d.append(t);
                }
                const int max_k = 1 + (int) (next() % 8);
                int32_t out[16] = {};
                int32_t extra[3] = {};
                const int n_extra = (int) (next() % 4);
                for (int e = 0; e < n_extra; ++e) extra[e] = (int32_t) (next() % (uint64_t) vocab);
                const int k = n_extra > 0 && (next() % 2) ? d.propose_after(extra, n_extra, max_k, out) : d.propose(max_k, out);
                if (k < 0 || k > max_k) { ++bad; continue; }
                if (d.size() != hist.size()) { ++bad; continue; }
                if (k == 0) continue;
                ++proposals;
                const int m = d.last_match();
                if (m < min_m) { ++bad; continue; }
                // the same query through propose() on the history (no extra tokens) must agree with the invariant too
                bool found = false;
                for (size_t p = 0; p + 1 < hist.size() + (size_t) n_extra && !found; ++p) {
                    // candidate continuation hist[p+1 ..] after a suffix of length m that ends at p
                    bool match = true;
                    for (int t = 0; t < k && match; ++t) {
                        const size_t at = p + 1 + (size_t) t;
                        const int32_t v = at < hist.size() ? hist[at] : (at - hist.size() < (size_t) n_extra ? extra[at - hist.size()] : -1);
                        match = v == out[t];
                    }
                    found = match;
                }
                if (!found) ++bad;
            }
        }
        check(bad == 0, "randomized proposals are real continuations within bounds");
        check(proposals > 1000, "randomized test exercised proposals");
    }
    std::printf("suffix_drafter unit tests: %s\n", g_fail ? "FAILED" : "OK");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int K = 8, first = 2;
        if (argc > 3 && std::strcmp(argv[2], "--k") == 0) { K = std::atoi(argv[3]); first = 4; }
        std::printf("%-28s %8s | %-44s | %-44s\n", "prompt", "tokens", "continue: tok/step  proposals  acc/proposal",
                    "copy: tok/step  proposals  acc/proposal");
        for (int f = first; f < argc; ++f) {
            const std::vector<int32_t> ids = read_ids(argv[f]);
            if (ids.size() < 64) continue;
            const size_t half = ids.size() / 2;
            const Sim c = simulate({ids.begin(), ids.begin() + half}, {ids.begin() + half, ids.end()}, K);
            const size_t a = ids.size() / 3, b = 2 * ids.size() / 3;
            const Sim p = simulate(ids, {ids.begin() + a, ids.begin() + b}, K);
            const char* name = std::strrchr(argv[f], '/') ? std::strrchr(argv[f], '/') + 1 : argv[f];
            std::printf("%-28s %8zu | %8.2f %10.1f%% %10.2f              | %8.2f %10.1f%% %10.2f\n", name, ids.size(),
                        (double) c.tokens / c.steps, 100.0 * c.proposals / c.steps,
                        c.proposals ? (double) c.accepted / c.proposals : 0.0, (double) p.tokens / p.steps,
                        100.0 * p.proposals / p.steps, p.proposals ? (double) p.accepted / p.proposals : 0.0);
        }
        return 0;
    }
    unit_tests();
    return g_fail ? 1 : 0;
}
