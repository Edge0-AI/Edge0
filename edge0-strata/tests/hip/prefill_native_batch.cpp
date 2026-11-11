// Focused parity for the opt-in native QSA append and native embedding gather
// selectively ported from q8atnight/Strata PR #108
// (acd487233c0bbe2217a6881c5bb43f8a283b0de5).
#include <cuda_runtime.h>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { \
    const cudaError_t e = (call); \
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return false; } \
} while (0)

namespace {
constexpr int IDX = 128;
constexpr int BLOCK = 4;
constexpr int EMB = 2560;
constexpr int IQ4_NL = 20;
constexpr float EPS = 1e-6f;
// the indexer's rope: plain (no scaling) over the base this test was written for
const strata::kernels::RopeScaling ROPE = [] { strata::kernels::RopeScaling r; r.freq_base = 10000.0; return r; }();

struct IndexState {
    float *tail = nullptr, *dead = nullptr, *pooled = nullptr;
    int32_t* block_pos = nullptr;
};

bool alloc_state(IndexState& s, int64_t max_cells) {
    CHECK(cudaMalloc((void**) &s.tail, (BLOCK - 1) * IDX * sizeof(float)));
    CHECK(cudaMalloc((void**) &s.dead, IDX * sizeof(float)));
    CHECK(cudaMalloc((void**) &s.pooled, (size_t) (max_cells / BLOCK + 1) * IDX * sizeof(float)));
    CHECK(cudaMalloc((void**) &s.block_pos, sizeof(int32_t)));
    return true;
}
void free_state(IndexState& s) {
    if (s.tail) cudaFree(s.tail);
    if (s.dead) cudaFree(s.dead);
    if (s.pooled) cudaFree(s.pooled);
    if (s.block_pos) cudaFree(s.block_pos);
    s = {};
}
bool upload_state(const IndexState& s, int64_t max_cells) {
    std::vector<float> tail((BLOCK - 1) * IDX, 0.0f), dead(IDX, 0.0f);
    std::vector<float> pooled((size_t) (max_cells / BLOCK + 1) * IDX);
    for (size_t i = 0; i < pooled.size(); ++i) pooled[i] = -0.125f * (float) (i + 1);
    const int32_t block_pos = -1;
    CHECK(cudaMemcpy(s.tail, tail.data(), tail.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.dead, dead.data(), dead.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.pooled, pooled.data(), pooled.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.block_pos, &block_pos, sizeof(block_pos), cudaMemcpyHostToDevice));
    return true;
}

bool qsa_case(cudaStream_t stream, int64_t max_cells, int64_t pos0, int64_t T, std::mt19937& rng) {
    const size_t raw_n = (size_t) (pos0 + T) * IDX;
    std::vector<float> raw(raw_n), gamma(IDX);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : raw) x = dist(rng);
    for (float& x : gamma) x = 0.5f + 0.5f * std::fabs(dist(rng));

    float *raw_d = nullptr, *gamma_d = nullptr;
    int32_t* pos_d = nullptr;
    CHECK(cudaMalloc((void**) &raw_d, raw.size() * sizeof(float)));
    CHECK(cudaMalloc((void**) &gamma_d, gamma.size() * sizeof(float)));
    CHECK(cudaMalloc((void**) &pos_d, sizeof(int32_t)));
    CHECK(cudaMemcpyAsync(raw_d, raw.data(), raw.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gamma_d, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    IndexState serial, batched;
    if (!alloc_state(serial, max_cells) || !alloc_state(batched, max_cells)) return false;
    if (!upload_state(serial, max_cells) || !upload_state(batched, max_cells)) return false;

    const auto shapes = strata::kernels::qsa_real_shapes();
    const strata::kernels::QsaIndexerBuffers sb{serial.tail, serial.dead, serial.pooled, serial.block_pos};
    const strata::kernels::QsaIndexerBuffers bb{batched.tail, batched.dead, batched.pooled, batched.block_pos};
    std::vector<int32_t> positions((size_t) (pos0 + T));
    for (size_t p = 0; p < positions.size(); ++p) positions[p] = (int32_t) p;

    // The reference is the original per-cell sequence. The candidate uses the
    // same per-cell prefix, then replaces only this chunk with the two-launch API.
    for (int64_t p = 0; p < pos0 + T; ++p) {
        CHECK(cudaMemcpyAsync(pos_d, positions.data() + p, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        strata::kernels::native_qsa_indexer_append(raw_d + p * IDX, pos_d, 0, gamma_d, EPS, sb,
                                                   shapes, max_cells, ROPE, stream);
        if (p < pos0) {
            CHECK(cudaMemcpyAsync(pos_d, positions.data() + p, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            strata::kernels::native_qsa_indexer_append(raw_d + p * IDX, pos_d, 0, gamma_d, EPS, bb,
                                                       shapes, max_cells, ROPE, stream);
        }
    }
    strata::kernels::native_qsa_indexer_append_batch(raw_d + pos0 * IDX, T, pos0, 0, gamma_d, EPS,
                                                      bb, shapes, max_cells, ROPE, stream);
    CHECK(cudaStreamSynchronize(stream));

    const size_t tail_n = (BLOCK - 1) * IDX, dead_n = IDX;
    const size_t pooled_n = (size_t) (max_cells / BLOCK + 1) * IDX;
    std::vector<float> st(tail_n), bt(tail_n), sd(dead_n), bd(dead_n), sp(pooled_n), bp(pooled_n);
    int32_t spos = 0, bpos = 0;
    CHECK(cudaMemcpy(st.data(), serial.tail, tail_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bt.data(), batched.tail, tail_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sd.data(), serial.dead, dead_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bd.data(), batched.dead, dead_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sp.data(), serial.pooled, pooled_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bp.data(), batched.pooled, pooled_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&spos, serial.block_pos, sizeof(spos), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&bpos, batched.block_pos, sizeof(bpos), cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(st.data(), bt.data(), tail_n * sizeof(float)) == 0 &&
                      std::memcmp(sd.data(), bd.data(), dead_n * sizeof(float)) == 0 &&
                      std::memcmp(sp.data(), bp.data(), pooled_n * sizeof(float)) == 0 && spos == bpos;
    if (!same) std::fprintf(stderr, "QSA append parity mismatch max=%lld pos0=%lld T=%lld\n",
                            (long long) max_cells, (long long) pos0, (long long) T);

    free_state(serial); free_state(batched);
    CHECK(cudaFree(raw_d)); CHECK(cudaFree(gamma_d)); CHECK(cudaFree(pos_d));
    return same;
}

bool qsa_chunks_case(cudaStream_t stream, std::mt19937& rng, int64_t max_cells,
                     int64_t first, int64_t second) {
    const int64_t total = first + second;
    std::vector<float> raw((size_t) total * IDX), gamma(IDX);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : raw) x = dist(rng);
    for (float& x : gamma) x = 0.5f + 0.5f * std::fabs(dist(rng));
    float *raw_d = nullptr, *gamma_d = nullptr;
    int32_t* pos_d = nullptr;
    CHECK(cudaMalloc((void**) &raw_d, raw.size() * sizeof(float)));
    CHECK(cudaMalloc((void**) &gamma_d, gamma.size() * sizeof(float)));
    CHECK(cudaMalloc((void**) &pos_d, sizeof(int32_t)));
    CHECK(cudaMemcpyAsync(raw_d, raw.data(), raw.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gamma_d, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    IndexState serial, chunked;
    if (!alloc_state(serial, max_cells) || !alloc_state(chunked, max_cells)) return false;
    if (!upload_state(serial, max_cells) || !upload_state(chunked, max_cells)) return false;
    const auto shapes = strata::kernels::qsa_real_shapes();
    const strata::kernels::QsaIndexerBuffers sb{serial.tail, serial.dead, serial.pooled, serial.block_pos};
    const strata::kernels::QsaIndexerBuffers cb{chunked.tail, chunked.dead, chunked.pooled, chunked.block_pos};
    std::vector<int32_t> positions((size_t) total);
    for (int i = 0; i < total; ++i) positions[(size_t) i] = i;
    for (int p = 0; p < total; ++p) {
        CHECK(cudaMemcpyAsync(pos_d, positions.data() + p, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        strata::kernels::native_qsa_indexer_append(raw_d + (size_t) p * IDX, pos_d, 0, gamma_d, EPS, sb,
                                                   shapes, max_cells, ROPE, stream);
    }
    strata::kernels::native_qsa_indexer_append_batch(raw_d, first, 0, 0, gamma_d, EPS, cb, shapes,
                                                      max_cells, ROPE, stream);
    strata::kernels::native_qsa_indexer_append_batch(raw_d + (size_t) first * IDX, second, first, 0,
                                                      gamma_d, EPS, cb, shapes, max_cells, ROPE, stream);
    CHECK(cudaStreamSynchronize(stream));
    const size_t tail_n = (BLOCK - 1) * IDX, pooled_n = (size_t) (max_cells / BLOCK + 1) * IDX;
    std::vector<float> st(tail_n), ct(tail_n), sd(IDX), cd(IDX), sp(pooled_n), cp(pooled_n);
    int32_t sbp = 0, cbp = 0;
    CHECK(cudaMemcpy(st.data(), serial.tail, tail_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ct.data(), chunked.tail, tail_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sd.data(), serial.dead, IDX * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cd.data(), chunked.dead, IDX * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sp.data(), serial.pooled, pooled_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cp.data(), chunked.pooled, pooled_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&sbp, serial.block_pos, sizeof(sbp), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&cbp, chunked.block_pos, sizeof(cbp), cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(st.data(), ct.data(), tail_n * sizeof(float)) == 0 &&
                      std::memcmp(sd.data(), cd.data(), IDX * sizeof(float)) == 0 &&
                      std::memcmp(sp.data(), cp.data(), pooled_n * sizeof(float)) == 0 && sbp == cbp;
    if (!same) std::fprintf(stderr, "QSA consecutive chunk parity mismatch %lld+%lld, capacity=%lld\n",
                            (long long) first, (long long) second, (long long) max_cells);
    free_state(serial); free_state(chunked);
    CHECK(cudaFree(raw_d)); CHECK(cudaFree(gamma_d)); CHECK(cudaFree(pos_d));
    return same;
}

bool embed_case(cudaStream_t stream, int type, int64_t T) {
    constexpr int vocab = 19;
    const size_t row_bytes = strata::kernels::iq_row_bytes(type, EMB);
    const size_t block_bytes = type == 20 ? 18 : 110;  // IQ4_NL or IQ3_S GGML block layout
    const size_t table_bytes = (size_t) vocab * row_bytes;
    if (row_bytes == 0 || row_bytes % block_bytes != 0) {
        std::fprintf(stderr, "bad test embedding row size for type=%d\n", type);
        return false;
    }
    void* host_table = nullptr;
    CHECK(cudaHostAlloc(&host_table, table_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    auto* table_bytes_u8 = (uint8_t*) host_table;
    std::memset(table_bytes_u8, 0, table_bytes);
    // Both test formats begin with a finite fp16 scale; the remaining packed code bits are all legal values.
    for (size_t r = 0; r < (size_t) vocab; ++r) {
        for (size_t b = 0; b < row_bytes; b += block_bytes) {
            table_bytes_u8[r * row_bytes + b] = 0x00;
            table_bytes_u8[r * row_bytes + b + 1] = 0x3c;
            for (size_t q = b + 2; q < b + block_bytes; ++q)
                table_bytes_u8[r * row_bytes + q] = (uint8_t) ((q * 37 + r * 19) & 0xff);
        }
    }
    void* table_dev = nullptr;
    CHECK(cudaHostGetDevicePointer(&table_dev, host_table, 0));

    std::vector<uint8_t> image_rows((size_t) T, 0);
    std::vector<int64_t> token_values((size_t) T);
    std::vector<int32_t> ids((size_t) T);
    for (int64_t t = 0; t < T; ++t) {
        if (t == 1 || t == 3) {
            image_rows[(size_t) t] = 1;
            token_values[(size_t) t] = t == 1 ? -1 : vocab + 11;  // image placeholders must never be gathered
            ids[(size_t) t] = 0;                                 // safe row, overwritten below
        } else {
            token_values[(size_t) t] = (t % 3 == 0) ? 7 : (t % 5);  // includes duplicate text IDs
            if (token_values[(size_t) t] < 0 || token_values[(size_t) t] >= vocab ||
                token_values[(size_t) t] > INT32_MAX) {
                std::fprintf(stderr, "test text token failed host-side validation\n");
                return false;
            }
            ids[(size_t) t] = (int32_t) token_values[(size_t) t];
        }
    }

    float *batched = nullptr, *serial = nullptr;
    int32_t* ids_d = nullptr;
    CHECK(cudaMalloc((void**) &batched, (size_t) T * EMB * sizeof(float)));
    CHECK(cudaMalloc((void**) &serial, (size_t) T * EMB * sizeof(float)));
    CHECK(cudaMalloc((void**) &ids_d, (size_t) T * sizeof(int32_t)));
    CHECK(cudaMemcpyAsync(ids_d, ids.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    strata::kernels::iq_embed_rows(type, table_dev, row_bytes, ids_d, T, EMB, batched, stream);
    for (int64_t t = 0; t < T; ++t) if (!image_rows[(size_t) t])
        strata::kernels::iq_dequant_f32(type,
            (const uint8_t*) table_dev + (size_t) ids[(size_t) t] * row_bytes, EMB, serial + t * EMB, stream);

    std::vector<float> image(EMB);
    for (int i = 0; i < EMB; ++i) image[(size_t) i] = 0.001f * (float) (i - EMB / 2);
    for (int64_t t = 0; t < T; ++t) if (image_rows[(size_t) t]) {
        CHECK(cudaMemcpyAsync(batched + t * EMB, image.data(), EMB * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(serial + t * EMB, image.data(), EMB * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    constexpr int64_t rows_per_check = 64;
    std::vector<float> a((size_t) rows_per_check * EMB), b(a.size());
    bool same = true;
    for (int64_t t0 = 0; t0 < T; t0 += rows_per_check) {
        const int64_t rows = std::min(rows_per_check, T - t0);
        const size_t count = (size_t) rows * EMB;
        CHECK(cudaMemcpy(a.data(), batched + t0 * EMB, count * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), serial + t0 * EMB, count * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(a.data(), b.data(), count * sizeof(float)) != 0) {
            size_t first = 0;
            while (first < count && std::memcmp(&a[first], &b[first], sizeof(float)) == 0) ++first;
            std::fprintf(stderr, "NativeEmbed batch parity mismatch type=%d T=%lld at element %lld: %.9g vs %.9g\n",
                         type, (long long) T, (long long) (t0 * EMB + first),
                         first < count ? a[first] : 0.0f, first < count ? b[first] : 0.0f);
            same = false;
            break;
        }
    }
    if (T > 3 && (!image_rows[1] || !image_rows[3] || ids[1] != 0 || ids[3] != 0)) same = false;
    if (!same) std::fprintf(stderr, "NativeEmbed image placeholder sanitization mismatch type=%d T=%lld\n",
                            type, (long long) T);
    CHECK(cudaFree(batched)); CHECK(cudaFree(serial)); CHECK(cudaFree(ids_d));
    CHECK(cudaFreeHost(host_table));
    return same;
}

} // namespace

int main() {
    std::mt19937 rng(0x50524631u);
    cudaStream_t stream = nullptr;
    const cudaError_t create = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (create != cudaSuccess) { std::fprintf(stderr, "cudaStreamCreateWithFlags: %s\n", cudaGetErrorString(create)); return 2; }
    bool ok = true;
    // Covers every slot residue, partial and multiple block chunks, and the fixed-capacity tail.
    const std::array<int64_t, 14> starts = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const std::array<int64_t, 5> lengths = {1, 2, 3, 7, 17};
    // (the batched append's contract is p0 + n <= max_cells: the prompt path never appends past the cache)
    for (int64_t pos0 : starts) for (int64_t T : lengths)
        if (pos0 + T <= 13) ok = qsa_case(stream, 13, pos0, T, rng) && ok;
    ok = qsa_chunks_case(stream, rng, 33, 7, 17) && ok;
    // Production-scale, non-block-aligned first chunk with a short continuation.
    ok = qsa_chunks_case(stream, rng, 8210, 8193, 17) && ok;
    for (int type : {20, 21}) for (int64_t T : {1, 2, 7, 17})
        ok = embed_case(stream, type, T) && ok;
    ok = embed_case(stream, 21, 4096) && ok;
    const cudaError_t destroy = cudaStreamDestroy(stream);
    if (destroy != cudaSuccess) { std::fprintf(stderr, "cudaStreamDestroy: %s\n", cudaGetErrorString(destroy)); return 2; }
    std::printf("Native prefill batch parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
