// Standalone diagnostic harness for strata-prefill's optional MMQ path.
// Keep this artifact outside the project source until it is reviewed and wired into a HIP test target.
// It compares GGML's MMQ result (with its intended q8_1 activation rounding) to a CPU FP32 product using
// GGML's own dequantizer for the exact same weight blocks. The tolerances screen layout/stride/bounds errors;
// they do not assert bitwise or FP16 parity.
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::prefill::mmq::Context;
using strata::prefill::mmq::Product;
using strata::prefill::mmq::matrix_bytes;
using strata::prefill::mmq::q8_bytes;
using strata::prefill::mmq::quantize;
using strata::prefill::mmq::supported;

void hip_check(hipError_t e, const char * what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct DeviceBuffer {
    void * p = nullptr;
    explicit DeviceBuffer(size_t n) { hip_check(hipMalloc(&p, n), "hipMalloc"); }
    ~DeviceBuffer() { if (p) hipFree(p); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;
    template<class T> T * as() const { return static_cast<T *>(p); }
};

size_t row_bytes(ggml_type type, int64_t cols) {
    return ggml_row_size(type, cols);
}

std::vector<float> decode_matrix(ggml_type type, const std::vector<uint8_t> & q, int64_t rows, int64_t cols) {
    const auto * tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) throw std::runtime_error("GGML type has no host dequantizer");
    const size_t rb = row_bytes(type, cols);
    if (q.size() != (size_t) rows * rb) throw std::runtime_error("matrix byte count mismatch");
    std::vector<float> out((size_t) rows * (size_t) cols);
    for (int64_t r = 0; r < rows; ++r) {
        tr->to_float(q.data() + (size_t) r * rb, out.data() + (size_t) r * (size_t) cols, cols);
    }
    return out;
}

std::vector<uint8_t> make_q2_matrix(int64_t rows, int64_t cols, int expert, int trial,
                                    ggml_type type = GGML_TYPE_Q2_0) {
    const auto * tr = ggml_get_type_traits(type);
    if (!tr || !tr->from_float_ref) throw std::runtime_error("host quantizer unavailable");
    const size_t rb = row_bytes(type, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    std::vector<float> row((size_t) cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t k = 0; k < cols; ++k) {
            const int code = (int) ((k * 37 + r * 19 + expert * 23 + trial * 11) % 101) - 50;
            row[(size_t) k] = 0.025f * std::sin(0.013f * (float) (k + 1) + 0.17f * (float) r + 0.31f * (float) expert) +
                              0.0015f * (float) code;
        }
        tr->from_float_ref(row.data(), out.data() + (size_t) r * rb, cols);
    }
    return out;
}

std::vector<float> make_activations(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * (size_t) cols);
    for (int r = 0; r < rows; ++r) {
        for (int64_t k = 0; k < cols; ++k) {
            x[(size_t) r * (size_t) cols + (size_t) k] =
                0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
        }
    }
    // Exercise a complete all-zero row; its expected output is exactly zero even though Q8_1 rounding is enabled.
    std::fill(x.end() - cols, x.end(), 0.0f);
    return x;
}

std::vector<int32_t> permutation(int rows, int shift, bool reverse) {
    std::vector<int32_t> p((size_t) rows);
    for (int i = 0; i < rows; ++i) {
        p[(size_t) i] = reverse ? (int32_t) ((rows - 1 - i + shift) % rows)
                                : (int32_t) ((i + shift) % rows);
    }
    return p;
}

std::vector<int32_t> make_bounds(const std::vector<int> & counts) {
    std::vector<int32_t> b(counts.size() + 1, 0);
    for (size_t i = 0; i < counts.size(); ++i) b[i + 1] = b[i] + counts[i];
    return b;
}

struct Metrics { double rel_l2 = 0, max_over_rms = 0, zero_row_max = 0, ref_rms = 0; };

Metrics run_product(Context & ctx, hipStream_t stream, const std::string & name, ggml_type type,
                    int64_t out_rows, int64_t cols, const std::vector<std::vector<uint8_t>> & expert_weights,
                    const std::vector<int> & counts, const std::vector<float> & x,
                    const std::vector<int32_t> & src_ids, const std::vector<int32_t> & dst_ids) {
    const int rows = (int) src_ids.size();
    const int experts = (int) expert_weights.size();
    const std::vector<int32_t> bounds = make_bounds(counts);
    if (!supported((int) type) || experts <= 0 || counts.size() != expert_weights.size() ||
        bounds.back() != rows || dst_ids.size() != (size_t) rows || x.size() != (size_t) rows * (size_t) cols) {
        throw std::runtime_error(name + ": invalid test geometry");
    }
    const size_t expert_bytes = matrix_bytes((int) type, out_rows, cols);
    const size_t tail = 4096; // match the production group's zero guard after its last expert
    std::vector<uint8_t> w((size_t) experts * expert_bytes + tail, 0);
    for (int e = 0; e < experts; ++e) {
        if (expert_weights[(size_t) e].size() != expert_bytes) throw std::runtime_error(name + ": expert size mismatch");
        std::copy(expert_weights[(size_t) e].begin(), expert_weights[(size_t) e].end(),
                  w.begin() + (size_t) e * expert_bytes);
    }

    DeviceBuffer dx(x.size() * sizeof(float));
    DeviceBuffer dsrc(src_ids.size() * sizeof(int32_t));
    DeviceBuffer ddst(dst_ids.size() * sizeof(int32_t));
    DeviceBuffer dbounds(bounds.size() * sizeof(int32_t));
    DeviceBuffer dw(w.size());
    DeviceBuffer dxq(q8_bytes(rows, cols));
    DeviceBuffer dy((size_t) rows * (size_t) out_rows * sizeof(float));
    hip_check(hipMemcpy(dx.p, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice), "copy x");
    hip_check(hipMemcpy(dsrc.p, src_ids.data(), src_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice), "copy src ids");
    hip_check(hipMemcpy(ddst.p, dst_ids.data(), dst_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice), "copy dst ids");
    hip_check(hipMemcpy(dbounds.p, bounds.data(), bounds.size() * sizeof(int32_t), hipMemcpyHostToDevice), "copy bounds");
    hip_check(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "copy weights and zero tail");
    // On the product's stream: a plain hipMemset runs on the null stream, which a non-blocking stream does not wait
    // for - on a Radeon 8060S (gfx1151, Windows) it landed after the MMQ kernel and left only the sentinel
    hip_check(hipMemsetAsync(dy.p, 0xff, (size_t) rows * (size_t) out_rows * sizeof(float), stream),
              "initialize output sentinel");

    quantize(dx.as<float>(), dsrc.as<int32_t>(), dxq.p, (int) type, cols, cols, rows, (void *) stream);
    const int max_rows = *std::max_element(counts.begin(), counts.end());
    Product p;
    p.w = dw.p; p.type = (int) type; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = expert_bytes;
    p.n = experts; p.xq = dxq.p; p.bounds = dbounds.as<int32_t>(); p.ids = ddst.as<int32_t>();
    p.total_rows = rows; p.max_rows = max_rows; p.dst = dy.as<float>(); p.ld_dst = out_rows;
    ctx.run(p, (void *) stream);
    hip_check(hipGetLastError(), "MMQ launch");
    hip_check(hipStreamSynchronize(stream), "MMQ synchronize");
    std::vector<float> got((size_t) rows * (size_t) out_rows);
    hip_check(hipMemcpy(got.data(), dy.p, got.size() * sizeof(float), hipMemcpyDeviceToHost), "copy output");

    std::vector<float> ref(got.size(), 0.0f);
    double err2 = 0, ref2 = 0;
    float max_abs = 0, zero_max = 0;
    for (int e = 0; e < experts; ++e) {
        const std::vector<float> wd = decode_matrix(type, expert_weights[(size_t) e], out_rows, cols);
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            const int src = src_ids[(size_t) r];
            const int dst = dst_ids[(size_t) r];
            const float * xr = x.data() + (size_t) src * (size_t) cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double acc = 0;
                const float * wr = wd.data() + (size_t) o * (size_t) cols;
                for (int64_t k = 0; k < cols; ++k) acc += (double) wr[k] * xr[k];
                ref[(size_t) dst * (size_t) out_rows + (size_t) o] = (float) acc;
            }
        }
    }
    {
        size_t bad = 0, first = got.size();
        uint32_t first_bits = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (!std::isfinite(got[i])) { if (!bad++) { first = i; std::memcpy(&first_bits, &got[i], 4); } }
        if (bad)
            throw std::runtime_error(name + ": " + std::to_string(bad) + " non-finite or unwritten MMQ outputs, first at row " +
                                     std::to_string(first / (size_t) out_rows) + " col " + std::to_string(first % (size_t) out_rows) +
                                     " bits " + std::to_string(first_bits));
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        err2 += d * d; ref2 += (double) ref[i] * ref[i]; max_abs = std::max(max_abs, (float) std::fabs(d));
    }
    const double rms = std::sqrt(ref2 / (double) ref.size());
    for (int r = 0; r < rows; ++r) {
        if (std::all_of(x.begin() + (size_t) src_ids[(size_t) r] * (size_t) cols,
                        x.begin() + ((size_t) src_ids[(size_t) r] + 1) * (size_t) cols,
                        [](float v) { return v == 0.0f; })) {
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max = std::max(zero_max, std::fabs(got[(size_t) dst_ids[(size_t) r] * (size_t) out_rows + (size_t) o]));
        }
    }
    Metrics m;
    m.rel_l2 = rms > 0 ? std::sqrt(err2 / (double) ref.size()) / rms : std::sqrt(err2);
    m.max_over_rms = rms > 0 ? max_abs / rms : max_abs;
    m.zero_row_max = zero_max; m.ref_rms = rms;
    std::cout << name << ": type=" << (int) type << " rows=" << rows << " experts=" << experts
              << " ref_rms=" << m.ref_rms << " rel_l2=" << m.rel_l2
              << " max_abs/ref_rms=" << m.max_over_rms << " zero_row_max=" << m.zero_row_max << '\n';
    // Q8_1 rounding is expected. These deliberately broad limits catch wrong block order, row strides, ids,
    // expert bounds, or stale output while allowing ordinary activation-quantization error.
    if (m.rel_l2 > 0.04 || m.max_over_rms > 0.35 || m.zero_row_max > std::max(1e-5, 1e-4 * rms))
        throw std::runtime_error(name + ": parity screen failed");
    return m;
}

std::vector<std::vector<uint8_t>> synthetic_q2(int experts, int64_t out_rows, int64_t cols, int trial,
                                               ggml_type type = GGML_TYPE_Q2_0) {
    std::vector<std::vector<uint8_t>> w;
    for (int e = 0; e < experts; ++e) w.push_back(make_q2_matrix(out_rows, cols, e, trial, type));
    return w;
}

// A synthetic matrix of any ggml type with a host quantizer (the K-quants: the dense GGUF tensors the
// STRATA_DENSE_MMQ path multiplies on HIP).  K stays a 256-value multiple: the dense path only takes
// such K, and a partial 256-value chunk would read past the synthetic tensor's end.
std::vector<std::vector<uint8_t>> synthetic_of_type(ggml_type type, int experts, int64_t out_rows, int64_t cols,
                                                     int trial) {
    const auto * tr = ggml_get_type_traits(type);
    if (!tr || !tr->from_float_ref) throw std::runtime_error("no host quantizer for the type");
    std::vector<std::vector<uint8_t>> w;
    for (int e = 0; e < experts; ++e) {
        std::vector<uint8_t> blocks((size_t) out_rows * row_bytes(type, cols));
        std::vector<float> row((size_t) cols);
        for (int64_t r = 0; r < out_rows; ++r) {
            for (int64_t k = 0; k < cols; ++k) {
                const int code = (int) ((k * 29 + r * 17 + e * 31 + trial * 13) % 97) - 48;
                row[(size_t) k] = 0.02f * std::sin(0.011f * (float) (k + 2) + 0.29f * (float) r + 0.37f * (float) e) +
                                  0.0009f * (float) code;
            }
            tr->from_float_ref(row.data(), blocks.data() + (size_t) r * row_bytes(type, cols), cols);
        }
        w.push_back(std::move(blocks));
    }
    return w;
}

bool run_real_iq3_first_expert(Context & ctx, hipStream_t stream, const std::string & pack) {
    std::ifstream meta(pack + "/native_experts.txt");
    if (!meta) throw std::runtime_error("cannot open native_experts.txt: " + pack);
    int layer = -1, gt = -1, dt = -1;
    uint64_t offset = 0, blob_bytes = 0;
    std::string line;
    bool found = false;
    while (std::getline(meta, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        if (ss >> layer >> gt >> dt >> offset >> blob_bytes && layer == 0) { found = true; break; }
    }
    if (!found) throw std::runtime_error("native metadata has no layer 0");
    if (gt != GGML_TYPE_IQ3_XXS || !supported(gt) || !supported(dt)) {
        std::cout << "real-IQ3 first expert skipped: layer 0 types are " << gt << "/" << dt << '\n';
        return false;
    }
    std::ifstream file(pack + "/experts.bin", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open experts.bin: " + pack);
    std::vector<uint8_t> blob((size_t) blob_bytes);
    file.seekg((std::streamoff) offset);
    file.read((char *) blob.data(), (std::streamsize) blob.size());
    if (!file) throw std::runtime_error("short read of first native expert");

    constexpr int64_t N = 2560, FF = 640;
    const size_t gu_row = row_bytes((ggml_type) gt, N), d_row = row_bytes((ggml_type) dt, FF);
    const size_t one_gu = (size_t) FF * gu_row, down_off = 2 * one_gu, one_down = (size_t) N * d_row;
    if (down_off + one_down != blob.size()) throw std::runtime_error("native blob shape/offset mismatch");
    std::vector<uint8_t> gu(blob.begin(), blob.begin() + (ptrdiff_t) down_off);
    std::vector<uint8_t> down(blob.begin() + (ptrdiff_t) down_off, blob.end());
    const int rows = 5;
    const std::vector<int> counts{rows};
    const auto src = permutation(rows, 2, false), dst = permutation(rows, 1, true);
    run_product(ctx, stream, "real-first-IQ3_XXS-GU", (ggml_type) gt, 1280, N, {gu}, counts,
                make_activations(rows, N, 71), src, dst);
    run_product(ctx, stream, "real-first-down", (ggml_type) dt, N, FF, {down}, counts,
                make_activations(rows, FF, 73), src, dst);
    std::cout << "real first expert tested: layer=0 expert=0 blob_bytes=" << blob_bytes << '\n';
    return true;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 2) throw std::runtime_error("usage: mmq-parity-test [optional-native-pack-dir]");
        hip_check(hipSetDevice(0), "hipSetDevice");
        hipStream_t stream = nullptr;
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
        int failures = 0;
        {
            Context ctx; // reuse one MMQ context/pool across every pass below
            const std::vector<std::vector<int>> batches{{1, 3, 3}, {4, 1, 2}, {2, 3}};
            int trial = 0;
            for (const auto & counts : batches) {
                const int rows = std::accumulate(counts.begin(), counts.end(), 0);
                const int n = (int) counts.size();
                const auto src = permutation(rows, trial + 1, false);
                const auto dst = permutation(rows, trial + 2, true);
                run_product(ctx, stream, "synthetic-Q2_0-GU-pass" + std::to_string(trial), GGML_TYPE_Q2_0,
                            1280, 2560, synthetic_q2(n, 1280, 2560, trial), counts,
                            make_activations(rows, 2560, trial), src, dst);
                run_product(ctx, stream, "synthetic-Q2_0-down-pass" + std::to_string(trial), GGML_TYPE_Q2_0,
                            2560, 640, synthetic_q2(n, 2560, 640, trial + 9), counts,
                            make_activations(rows, 640, trial + 3), src, dst);
                ++trial;
            }
            // The K-quant instances a STRATA_MMQ_KQUANTS build adds (the dense GGUF projections of the
            // mixed-quant packs through Gemm::native).  Skipped when the build does not have them.
            for (const auto & [name, type] : std::initializer_list<std::pair<const char *, ggml_type>>{
                     {"Q4_K", GGML_TYPE_Q4_K}, {"Q5_K", GGML_TYPE_Q5_K}, {"Q6_K", GGML_TYPE_Q6_K},
                     {"Q5_1", GGML_TYPE_Q5_1}}) {
                if (!supported((int) type)) {
                    std::cout << "synthetic-" << name << " skipped: not built (STRATA_MMQ_KQUANTS off?)\n";
                    continue;
                }
                const int rows = 5;
                const std::vector<int> counts{rows};
                const auto src = permutation(rows, 3, false), dst = permutation(rows, 2, true);
                run_product(ctx, stream, std::string("synthetic-") + name, type,
                            1280, 2560, synthetic_of_type(type, 1, 1280, 2560, 5), counts,
                            make_activations(rows, 2560, 77), src, dst);
            }
            // APR trellis experts (ggml types 144-146), when the build has their MMQ instances (agention fork): gate/up
            // 1280 x 2560 and down 2560 x 640 (5 blocks per row: the last 256-slice reads past the row, against zero
            // activations)
            for (const ggml_type t : {(ggml_type) 145, (ggml_type) 146, (ggml_type) 144}) {
                if (!supported((int) t)) { std::cout << "type " << (int) t << ": no MMQ instance in this build\n"; continue; }
                const std::vector<int> counts{2, 3};
                const auto src = permutation(5, 1, false), dst = permutation(5, 2, true);
                const char* known = ggml_type_name(t);   // null when this ggml has no name for the id (no trellis types in it): #3358
                const std::string tn = known ? known : "type" + std::to_string((int) t);
                const char* rep = std::getenv("MMQ_PARITY_REPEAT");   // repeat each product (flakiness screen)
                const int reps = rep ? std::max(1, std::atoi(rep)) : 1;
                const auto wgu = synthetic_q2(2, 1280, 2560, 5, t), wd = synthetic_q2(2, 2560, 640, 6, t);
                for (int r = 0; r < reps; ++r) {
                    run_product(ctx, stream, "synthetic-" + tn + "-GU", t, 1280, 2560, wgu, counts,
                                make_activations(5, 2560, 5), src, dst);
                    run_product(ctx, stream, "synthetic-" + tn + "-down", t, 2560, 640, wd, counts,
                                make_activations(5, 640, 6), src, dst);
                }
            }
            if (argc == 2) run_real_iq3_first_expert(ctx, stream, argv[1]);
        }
        hip_check(hipStreamSynchronize(stream), "final synchronize");
        hip_check(hipStreamDestroy(stream), "destroy stream");
        std::cout << "MMQ parity screen passed\n";
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::cerr << "MMQ parity screen failed: " << e.what() << '\n';
        return 1;
    }
}
