// src/kernels/resident_plan_parity.cpp - the verify window's device-built expert plan (#783 PR-e: the parallel
// grouping and the prefix scan that replaced thread 0's serial pass) against a host replay of its contract:
// distinct experts in routing order, each with its entries ascending, `ptr[g]` = the expert's slot, counts / start /
// dst / tok / start2 in the host pool's layout.
//
// n = 1..80 entries (a window of up to kVerifyMaxT tokens x 10 experts), ids drawn from pools of 3..512 experts so
// duplicates run from "every entry the same expert" to "all distinct", with and without a slot table (`slot_off`).
// Both modes are run: with a `skip` word (ring when the plan was built, 0 when an expert was not resident - and
// then the plan is untouched) and without one (the all-resident graph: a non-resident expert gives an empty plan and
// sets *plan_err).
// Every word of the plan the host pool would read is compared with memcmp. GPU, synthetic, no model.
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
}  // namespace

int main() {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(783);
    const long long blob = 1337;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced

    int32_t *d_ids, *d_res, *d_plan;
    unsigned long long* d_off;
    uint32_t* d_skip;
    ck(cudaMalloc(&d_ids, 128 * 4), "ids");
    ck(cudaMalloc(&d_res, NE * 4), "res");
    ck(cudaMalloc(&d_off, 1024 * 8), "off");
    ck(cudaMalloc(&d_plan, plan_words * 4), "plan");
    ck(cudaMalloc(&d_skip, 4), "skip");
    uint32_t* d_err = nullptr;
    ck(cudaMalloc(&d_err, 4), "err");

    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[e] = (e * 37 + 11) % 1000;      // distinct slots, not the identity
    std::vector<unsigned long long> off(1024);
    for (int s = 0; s < 1024; ++s) off[s] = (unsigned long long) s * 4096 + (s % 7) * 64;
    ck(cudaMemcpy(d_res, res.data(), NE * 4, cudaMemcpyHostToDevice), "res");
    ck(cudaMemcpy(d_off, off.data(), 1024 * 8, cudaMemcpyHostToDevice), "off");

    int cases = 0, bad = 0;
    for (int n = 1; n <= 80; ++n) {
        for (int variant = 0; variant < 4; ++variant) {
            const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = (int32_t) ((rng() % pool) * (512 / pool));
            const bool use_off = (n + variant) % 2 == 1;
            ck(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice), "ids");

            // host replay
            std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
            std::vector<unsigned long long> ptr;
            std::vector<int> first_seen;     // group -> expert
            for (int i = 0; i < n; ++i) {
                int g = -1;
                for (size_t q = 0; q < first_seen.size(); ++q)
                    if (first_seen[q] == ids[i]) g = (int) q;
                if (g < 0) { first_seen.push_back(ids[i]); g = (int) first_seen.size() - 1; }
            }
            const int groups = (int) first_seen.size();
            counts[0] = groups;
            start.assign(groups + 1, 0);
            int pos = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = pos;
                ptr.push_back((unsigned long long) (uintptr_t) cache_base +
                              (use_off ? off[res[first_seen[g]]] : (unsigned long long) res[first_seen[g]] * (unsigned long long) blob));
                for (int i = 0; i < n; ++i)
                    if (ids[i] == first_seen[g]) { dst[pos] = i; tok[pos] = i / K; ++pos; }
            }
            start[groups] = n;

            for (int mode = 0; mode < 2; ++mode) {   // 0: general kernel with a skip word, 1: all resident
                ck(cudaMemset(d_plan, 0xAB, plan_words * 4), "fill");
                ck(cudaMemset(d_skip, 0, 4), "skip0");
                ck(cudaMemset(d_err, 0, 4), "err0");
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx,
                                 mode == 0 ? d_skip : nullptr, 7u, nullptr, d_err);
                ck(cudaDeviceSynchronize(), "sync");
                std::vector<int32_t> plan(plan_words);
                ck(cudaMemcpy(plan.data(), d_plan, plan_words * 4, cudaMemcpyDeviceToHost), "plan");
                uint32_t skip = 0;
                ck(cudaMemcpy(&skip, d_skip, 4, cudaMemcpyDeviceToHost), "skipr");
                bool ok = mode == 1 || skip == 7u;
                ok = ok && std::memcmp(plan.data(), counts.data(), 3 * 4) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t) groups * 8) == 0;
                ok = ok && plan[(size_t) (ptr_off + 4 * capx)] == n;
                ++cases;
                if (!ok) {
                    std::printf("    *** n=%d pool=%d %s %s: the plan differs from the host replay ***\n", n, pool,
                                use_off ? "slot_off" : "blob", mode ? "all_resident" : "general");
                    ++bad;
                }
            }
        }
    }

    // a routed expert that is not resident: the general kernel leaves the plan alone and zeroes *skip
    {
        std::vector<int32_t> res2 = res;
        res2[5] = -1;
        ck(cudaMemcpy(d_res, res2.data(), NE * 4, cudaMemcpyHostToDevice), "res2");
        const int32_t ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        ck(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemset(d_plan, 0xAB, plan_words * 4), "fill");
        const uint32_t seven = 7;
        ck(cudaMemcpy(d_skip, &seven, 4, cudaMemcpyHostToDevice), "skip7");
        k::resident_plan(d_ids, 10, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, nullptr, d_err);
        ck(cudaDeviceSynchronize(), "sync");
        std::vector<int32_t> plan(plan_words);
        ck(cudaMemcpy(plan.data(), d_plan, plan_words * 4, cudaMemcpyDeviceToHost), "plan");
        uint32_t skip = 99;
        ck(cudaMemcpy(&skip, d_skip, 4, cudaMemcpyDeviceToHost), "skipr");
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t) w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched) {
            std::printf("    *** a non-resident expert: skip=%u untouched=%d (want 0 / 1) ***\n", skip, (int) untouched);
            ++bad;
        }
    }

    std::printf("resident_plan: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("resident_plan_parity OK\n");
    return 0;
}
