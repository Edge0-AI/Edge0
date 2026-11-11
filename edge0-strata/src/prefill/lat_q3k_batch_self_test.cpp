// lat_q3k_batch_self_test.cpp - L2a of the q3k acceptance sheet: the batch / row / blob assembly.
//
// Same inputs, two ways of cutting them: every routed expert in one WideBatch per kMaxBatch (the engine's walk,
// resident blobs) versus one WideBatch per expert.  The outputs must be bit-identical: if they are not, the
// tables, the placement rows (off[]) or the blob-by-expert-id mapping is wrong, and no amount of comparing the
// engine's rows against the native path can say which.
//
// The inputs are the engine's own: g-cnt/g-off/g-tok and d-xlq dumped from a real layer 0, and the pack's layer 0
// blobs.  Nothing here runs the engine.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_q3k.hpp"

namespace fused = strata::prefill::fused;
namespace fused_q3k = strata::prefill::fused_q3k;

namespace {
constexpr int64_t kEll = 1280, kIff = 640;
constexpr size_t kStride = 1164800;            // one q3k expert's blob (gu 640*550 + up 352000 + down 640*360)
const char* kDump = "/home/user/lat-s26/q3k/dumps4k";            // g-cnt.bin g-off.bin g-tok.bin d-xlq.bin
const char* kPack = "/home/user/models/v7_l1280_q3k_pack/latent_a_pack/experts.bin";
constexpr int kLayer = 0;                      // native_experts.txt's first line: offset 0

std::vector<uint8_t> slurp(const std::string& p) {
    std::vector<uint8_t> v;
    if (std::FILE* f = std::fopen(p.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        v.resize((size_t) std::ftell(f));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
        std::fclose(f);
    }
    return v;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
}  // namespace

int main() {
    auto cnt_b = slurp(std::string(kDump) + "/g-cnt.bin");
    auto off_b = slurp(std::string(kDump) + "/g-off.bin");
    auto tok_b = slurp(std::string(kDump) + "/g-tok.bin");
    auto xlq_b = slurp(std::string(kDump) + "/d-xlq.bin");
    if (cnt_b.empty() || off_b.empty() || tok_b.empty() || xlq_b.empty()) {
        std::fprintf(stderr, "FAIL: the layer 0 dumps are missing under %s\n", kDump);
        return 2;
    }
    const int E = (int) (cnt_b.size() / 4);
    const int32_t* cnt = (const int32_t*) cnt_b.data();
    const int32_t* off = (const int32_t*) off_b.data();
    const int32_t* tok = (const int32_t*) tok_b.data();
    const int64_t n = (int64_t) (tok_b.size() / 4);       // T * K entries
    const size_t xrow = (size_t) kEll / 32 * 36;
    std::printf("L2a: %d experts, %lld entries, xlq %zu rows of %zu B, off[E] %d\n", E, (long long) n,
                xlq_b.size() / xrow, xrow, off[E]);
    if (off[E] != n) {
        std::fprintf(stderr, "FAIL: off[E] %d != T*K %lld (the placement does not cover the entries)\n", off[E],
                     (long long) n);
        return 2;
    }

    // the pack's layer 0: whole experts, one stride each
    std::vector<uint8_t> blobs((size_t) E * kStride);
    {
        std::FILE* f = std::fopen(kPack, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "FAIL: cannot open %s\n", kPack);
            return 2;
        }
        std::fseek(f, (long) kLayer * 512 * (long) kStride, SEEK_SET);
        const size_t got = std::fread(blobs.data(), 1, blobs.size(), f);
        std::fclose(f);
        if (got != blobs.size()) {
            std::fprintf(stderr, "FAIL: the pack gave %zu of %zu B\n", got, blobs.size());
            return 2;
        }
    }

    // the device side: the tables, the H rows, the xl rows, the blobs and two sets of output rows
    const size_t tables_b = fused::latent_group_bytes(n, E);
    const size_t scratch_b = fused::latent_scratch_bytes(n, E, kIff);
    const size_t lat_h = fused::latent_h_off(n, E);
    std::vector<char> tables(tables_b);
    fused::group_latent(cnt, off, E, tables.data(), tables.size());
    uint8_t* scratch_d = nullptr;
    uint8_t* xlq_d = nullptr;
    uint8_t* blobs_d = nullptr;
    int32_t* tok_d = nullptr;
    float *yA = nullptr, *yB = nullptr;
    ck(cudaMalloc(&scratch_d, scratch_b), "scratch");
    ck(cudaMalloc(&xlq_d, xlq_b.size()), "xlq");
    ck(cudaMalloc(&blobs_d, blobs.size()), "blobs");
    ck(cudaMalloc(&tok_d, (size_t) n * 4), "tok");
    ck(cudaMalloc(&yA, (size_t) n * kEll * 4), "yA");
    ck(cudaMalloc(&yB, (size_t) n * kEll * 4), "yB");
    ck(cudaMemcpy(scratch_d, tables.data(), tables_b, cudaMemcpyHostToDevice), "tables");
    ck(cudaMemcpy(xlq_d, xlq_b.data(), xlq_b.size(), cudaMemcpyHostToDevice), "xlq");
    ck(cudaMemcpy(blobs_d, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "blobs");
    ck(cudaMemcpy(tok_d, tok, (size_t) n * 4, cudaMemcpyHostToDevice), "tok");
    ck(cudaMemset(yA, 0, (size_t) n * kEll * 4), "yA zero");
    ck(cudaMemset(yB, 0, (size_t) n * kEll * 4), "yB zero");
    const fused::LatentGeom lg{kEll, kIff, 550, 360, 352000, 704000, 0};

    // A: the engine's walk - kMaxBatch experts a launch, every blob of the layer present
    std::printf("L2a: A = batches of %d experts, B = one expert a launch\n", (int) fused::kMaxBatch);
    for (int e0 = 0; e0 < E; e0 += (int) fused::kMaxBatch) {
        fused::WideBatch fb;
        fb.e0 = e0;
        fb.e1 = std::min(E, e0 + (int) fused::kMaxBatch);
        for (int e = fb.e0; e < fb.e1; ++e) fb.blob[e - fb.e0] = blobs_d + (size_t) e * kStride;
        fused_q3k::experts_latent_q3k(fb, lg, E, n, scratch_d, xlq_d, tok_d, scratch_d + lat_h, yA, nullptr);
    }
    ck(cudaDeviceSynchronize(), "A sync");

    // B: the same experts, one launch each
    for (int e = 0; e < E; ++e) {
        if (cnt[e] <= 0) continue;
        fused::WideBatch fb;
        fb.e0 = e;
        fb.e1 = e + 1;
        fb.blob[0] = blobs_d + (size_t) e * kStride;
        fused_q3k::experts_latent_q3k(fb, lg, E, n, scratch_d, xlq_d, tok_d, scratch_d + lat_h, yB, nullptr);
    }
    ck(cudaDeviceSynchronize(), "B sync");

    // compare row for row
    std::vector<float> hA((size_t) n * kEll), hB((size_t) n * kEll);
    ck(cudaMemcpy(hA.data(), yA, hA.size() * 4, cudaMemcpyDeviceToHost), "yA back");
    ck(cudaMemcpy(hB.data(), yB, hB.size() * 4, cudaMemcpyDeviceToHost), "yB back");
    size_t bad_rows = 0, bad_vals = 0, first = (size_t) -1;
    double worst = 0;
    for (int64_t r = 0; r < n; ++r) {
        bool bad = false;
        for (int64_t c = 0; c < kEll; ++c) {
            const size_t i = (size_t) r * kEll + (size_t) c;
            if (hA[i] != hB[i]) {
                ++bad_vals;
                bad = true;
                const double d = std::fabs((double) hA[i] - (double) hB[i]);
                if (d > worst) worst = d;
            }
        }
        if (bad) {
            ++bad_rows;
            if (first == (size_t) -1) first = (size_t) r;
        }
    }
    std::printf("L2a rows %lld: differing rows %zu, differing values %zu, first row %lld, max|d| %.3e\n",
                (long long) n, bad_rows, bad_vals, (long long) first, worst);
    if (bad_rows == 0) {
        std::printf("L2a PASS: the batch and the single-expert launches agree bit for bit\n");
    } else {
        std::printf("L2a FAIL: the batch assembly changes the result (%zu rows)\n", bad_rows);
    }
    cudaFree(scratch_d); cudaFree(xlq_d); cudaFree(blobs_d); cudaFree(tok_d); cudaFree(yA); cudaFree(yB);
    return bad_rows == 0 ? 0 : 1;
}
