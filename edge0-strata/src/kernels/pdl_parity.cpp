// src/kernels/pdl_parity.cpp - programmatic dependent launch (pdl.hpp) in a captured graph, tested bitwise.
//
//     build/pdl_parity [--replays N]
//
// WHAT IT CHECKS.  A chain shaped like the verify window's - quantize, multi-column MMVQ (a small-K and a large-K
// format), BF16 multi-row GEMV, the shared Q8_1 buffer rewritten by the next quantization, a memcpy node, a fork onto a
// second stream and a join - is captured twice: with pdl_scope() off (ordinary launches) and on.  Then:
//   1. every output of the PDL graph is bitwise the ordinary graph's, over many replays; before each replay every
//      intermediate is overwritten with NaN bytes, so a kernel that read its input before the kernel writing it had
//      finished would see NaNs and fail;
//   2. every programmatic edge of the PDL graph joins two kernel nodes, and every node with a programmatic in-edge
//      has kernels only before it (the kernel after the memcpy must be an ordinary launch); with STRATA_DF_PDL=2 also
//      no programmatic edge into a node with a second in-edge (the kernel after the join);
//   3. power: where pdl_supported() (sm_90+ with code built for it) the PDL graph has programmatic edges, elsewhere
//      it has none.
// Also copy_rows_strided against the cudaMemcpy2DAsync it replaces in the verify window.  GPU, synthetic, no model.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// The set-up fills and copies run on the legacy default stream, which does not order the non-blocking streams the
// kernels run on, and may return before the device has finished them: each one is completed before it is used.
template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "memset");
    ck(cudaDeviceSynchronize(), "memset sync");
    return p;
}

// a normal fp16 in +-[2^-10, 2^-5) (mmvq_multi_parity's: finite scales keep every output finite)
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// random quantized weights with sane fp16 scales at byte 0 of each block (Q8_0: 34 bytes, IQ4_XS: 136 bytes)
void* quant_weights(int type, int n_in, int n_out, int block_elems, int block_bytes, std::mt19937& rng) {
    const size_t bytes = k::native_mmvq_weight_bytes(type, n_in, n_out);
    std::vector<uint8_t> w(bytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const size_t blocks = (size_t) n_out * (size_t) (n_in / block_elems);
    if (blocks * (size_t) block_bytes != bytes) {
        std::printf("type %d: %zu blocks of %d bytes are not the %zu weight bytes\n", type, blocks, block_bytes, bytes);
        std::exit(2);
    }
    for (size_t i = 0; i < blocks; ++i) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[i * (size_t) block_bytes], &h, 2);
    }
    void* d = nullptr;
    ck(cudaMalloc(&d, bytes), "malloc w");
    ck(cudaMemcpy(d, w.data(), bytes, cudaMemcpyHostToDevice), "copy w");
    ck(cudaDeviceSynchronize(), "copy w sync");
    return d;
}

uint16_t* bf16_weights(int n_in, int n_out, float scale, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<uint16_t> w((size_t) n_in * n_out);
    for (auto& v : w) v = k::bf16_from_f32(nd(rng));
    uint16_t* d = dalloc<uint16_t>(w.size());
    ck(cudaMemcpy(d, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "copy bf16 w");
    ck(cudaDeviceSynchronize(), "copy bf16 w sync");
    return d;
}

constexpr int T = 4, N = 2560, NB = 6144;   // a window of 4 tokens, n_embd, a wide projection

struct Bufs {
    float *x0, *y1, *y2, *y3, *y4, *y5, *y5c, *y6, *y7, *y8;
    uint8_t *q, *q2;
};

// The chain (one capture).  `side`/`ev*`: the fork and join, as the verify window's branches make them.
void record(const Bufs& b, void* w_q8, void* w_iq4, uint16_t* w_b1, uint16_t* w_b2, cudaStream_t s, cudaStream_t side,
            cudaEvent_t ev_fork, cudaEvent_t ev_join) {
    k::native_quantize_q8_1(b.x0, b.q, N, T, s);                          // A
    k::native_mmvq(8, w_q8, b.q, b.y1, N, N, T, s);                       // B  Q8_0, large K
    k::native_quantize_q8_1(b.y1, b.q, N, T, s);                          // C  rewrites the q B read
    k::native_mmvq(23, w_iq4, b.q, b.y2, N, NB, T, s);                    // D  IQ4_XS, small K
    k::bf16_gemv_fp32_mmvf_multi(b.y2, NB, w_b1, b.y3, N, NB, N, T, s);  // E
    ck(cudaMemcpyAsync(b.y4, b.y3, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, s), "memcpy node");   // F
    k::native_quantize_q8_1(b.y4, b.q, N, T, s);                          // G  after a memcpy: ordinary
    k::native_mmvq(8, w_q8, b.q, b.y5, N, N, T, s);                       // H
    ck(cudaEventRecord(ev_fork, s), "fork");
    ck(cudaStreamWaitEvent(side, ev_fork, 0), "fork wait");
    k::copy_rows_strided(b.y5c, b.y5, T, N, N, side);                     // side: an ordinary kernel after H
    k::native_quantize_q8_1(b.y5c, b.q2, N, T, side);
    k::native_mmvq(8, w_q8, b.q2, b.y6, N, N, T, side);
    k::bf16_gemv_fp32_mmvf_multi(b.y5, N, w_b2, b.y7, N, N, N, T, s);    // main: beside the side branch
    ck(cudaEventRecord(ev_join, side), "join");
    ck(cudaStreamWaitEvent(s, ev_join, 0), "join wait");
    k::bf16_gemv_fp32_mmvf_multi(b.y6, N, w_b2, b.y8, N, N, N, T, s);    // after the join: ordinary
}

cudaGraphExec_t capture(bool pdl, const Bufs& b, void* w_q8, void* w_iq4, uint16_t* w_b1, uint16_t* w_b2,
                        cudaStream_t s, cudaStream_t side, cudaEvent_t ef, cudaEvent_t ej, size_t& prog, size_t& bad,
                        size_t& multi, size_t& mixed) {
    k::pdl_scope() = pdl;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
    record(b, w_q8, w_iq4, w_b1, w_b2, s, side, ef, ej);
    cudaGraph_t g = nullptr;
    ck(cudaStreamEndCapture(s, &g), "end capture");
    k::pdl_scope() = false;
    prog = bad = multi = mixed = 0;
#if CUDART_VERSION >= 12030
    size_t ne = 0;
#if CUDART_VERSION >= 13000
    ck(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &ne), "edges");
#else
    ck(cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &ne), "edges");
#endif
    std::vector<cudaGraphNode_t> from(ne), to(ne);
    std::vector<cudaGraphEdgeData> ed(ne);
#if CUDART_VERSION >= 13000
    ck(cudaGraphGetEdges(g, from.data(), to.data(), ed.data(), &ne), "edges");
#else
    ck(cudaGraphGetEdges_v2(g, from.data(), to.data(), ed.data(), &ne), "edges");
#endif
    std::map<cudaGraphNode_t, int> in_edges, prog_in, nonkernel_in;
    for (size_t i = 0; i < ne; ++i) {
        ++in_edges[to[i]];
        cudaGraphNodeType ta;
        ck(cudaGraphNodeGetType(from[i], &ta), "node type");
        if (ta != cudaGraphNodeTypeKernel) ++nonkernel_in[to[i]];
        if (ed[i].type == cudaGraphDependencyTypeProgrammatic) ++prog_in[to[i]];
    }
    for (size_t i = 0; i < ne; ++i) {
        if (ed[i].type != cudaGraphDependencyTypeProgrammatic) continue;
        ++prog;
        cudaGraphNodeType ta, tb;
        ck(cudaGraphNodeGetType(from[i], &ta), "node type");
        ck(cudaGraphNodeGetType(to[i], &tb), "node type");
        if (ta != cudaGraphNodeTypeKernel || tb != cudaGraphNodeTypeKernel) ++bad;
        if (in_edges[to[i]] != 1) ++multi;
    }
    for (const auto& kv : prog_in)
        if (nonkernel_in[kv.first] != 0) ++mixed;
#endif
    cudaGraphExec_t ex = nullptr;
    ck(cudaGraphInstantiate(&ex, g, 0), "instantiate");
    ck(cudaGraphDestroy(g), "destroy graph");
    return ex;
}

std::vector<uint32_t> read(const float* d, size_t n) {
    std::vector<uint32_t> h(n);
    ck(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "read");
    return h;
}

void copy_rows_strided_check(cudaStream_t s) {
    const int64_t rows = 3 * 32, w = 256, src_w = 512;   // a 3-token window's q heads: query half of each q/gate pair
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src((size_t) rows * src_w);
    for (auto& v : src) v = nd(rng);
    float* ds = dalloc<float>(src.size());
    float* da = dalloc<float>((size_t) rows * w);
    float* db = dalloc<float>((size_t) rows * w);
    ck(cudaMemcpy(ds, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "copy src");
    ck(cudaDeviceSynchronize(), "copy src sync");
    k::copy_rows_strided(da, ds, rows, w, src_w, s);
    ck(cudaMemcpy2DAsync(db, (size_t) w * 4, ds, (size_t) src_w * 4, (size_t) w * 4, (size_t) rows,
                         cudaMemcpyDeviceToDevice, s), "memcpy2d");
    ck(cudaStreamSynchronize(s), "sync");
    const auto a = read(da, (size_t) rows * w), b = read(db, (size_t) rows * w);
    const bool ok = a == b;
    std::printf("copy_rows_strided vs cudaMemcpy2DAsync (%lld rows of %lld of %lld): %s\n", (long long) rows,
                (long long) w, (long long) src_w, ok ? "bitwise equal" : "DIFFERENT");
    if (!ok) ++g_fail;
    cudaFree(ds); cudaFree(da); cudaFree(db);
}

}  // namespace

int main(int argc, char** argv) {
    int replays = 200;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--replays") && i + 1 < argc) replays = std::atoi(argv[++i]);
    int dev = 0;
    cudaDeviceProp prop{};
    ck(cudaGetDevice(&dev), "device");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    const bool supported = k::pdl_supported();
    std::printf("pdl_parity: %s (sm_%d%d), pdl_supported() = %d\n", prop.name, prop.major, prop.minor, (int) supported);

    cudaStream_t s = nullptr, side = nullptr;
    cudaEvent_t ef = nullptr, ej = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "stream");
    ck(cudaEventCreateWithFlags(&ef, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&ej, cudaEventDisableTiming), "event");

    copy_rows_strided_check(s);

    std::mt19937 rng(1234);
    void* w_q8 = quant_weights(8, N, N, 32, 34, rng);
    void* w_iq4 = quant_weights(23, N, NB, 256, 136, rng);
    uint16_t* w_b1 = bf16_weights(NB, N, 0.02f, rng);
    uint16_t* w_b2 = bf16_weights(N, N, 0.02f, rng);
    Bufs b{};
    b.x0 = dalloc<float>((size_t) T * N);
    float** outs[] = {&b.y1, &b.y2, &b.y3, &b.y4, &b.y5, &b.y5c, &b.y6, &b.y7, &b.y8};
    constexpr int NOUT = 9;
    const size_t out_n[NOUT] = {(size_t) T * N, (size_t) T * NB, (size_t) T * N, (size_t) T * N, (size_t) T * N,
                                (size_t) T * N, (size_t) T * N, (size_t) T * N, (size_t) T * N};
    for (int i = 0; i < NOUT; ++i) *outs[i] = dalloc<float>(out_n[i]);
    const size_t qbytes = k::native_q8_1_bytes(N, T);
    b.q = dalloc<uint8_t>(qbytes);
    b.q2 = dalloc<uint8_t>(qbytes);
    {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * N);
        for (auto& v : x) v = nd(rng);
        ck(cudaMemcpy(b.x0, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
        ck(cudaDeviceSynchronize(), "copy x sync");
    }

    size_t prog0, bad0, multi0, mixed0, prog1, bad1, multi1, mixed1;
    cudaGraphExec_t plain = capture(false, b, w_q8, w_iq4, w_b1, w_b2, s, side, ef, ej, prog0, bad0, multi0, mixed0);
    cudaGraphExec_t pdl = capture(true, b, w_q8, w_iq4, w_b1, w_b2, s, side, ef, ej, prog1, bad1, multi1, mixed1);
    const char* mode = std::getenv("STRATA_DF_PDL");
    const bool single = mode != nullptr && std::atoi(mode) == 2;
    std::printf("graph edges: ordinary capture %zu programmatic; PDL capture %zu programmatic, %zu not kernel to kernel, "
                "%zu nodes with a programmatic and a non-kernel in-edge, %zu into a node with other in-edges%s\n", prog0,
                prog1, bad1, mixed1, multi1, single ? " (STRATA_DF_PDL=2: must be 0)" : "");
    if (prog0 != 0 || bad1 != 0 || mixed1 != 0 || (single && multi1 != 0)) ++g_fail;
#if CUDART_VERSION >= 12030
    if (supported && prog1 == 0) {
        std::printf("FAIL: PDL is supported here but the PDL capture has no programmatic edge (the test has no power)\n");
        ++g_fail;
    }
    if (!supported && prog1 != 0) {
        std::printf("FAIL: PDL is not supported here but the capture has programmatic edges\n");
        ++g_fail;
    }
#endif

    auto poison = [&]() {   // NaN bytes in every intermediate and output
        for (int i = 0; i < NOUT; ++i) ck(cudaMemsetAsync(*outs[i], 0xff, out_n[i] * 4, s), "poison");
        ck(cudaMemsetAsync(b.q, 0xff, qbytes, s), "poison");
        ck(cudaMemsetAsync(b.q2, 0xff, qbytes, s), "poison");
    };
    poison();
    ck(cudaGraphLaunch(plain, s), "launch plain");
    ck(cudaStreamSynchronize(s), "sync plain");
    std::vector<std::vector<uint32_t>> ref;
    for (int i = 0; i < NOUT; ++i) ref.push_back(read(*outs[i], out_n[i]));
    long long nonfinite = 0;
    for (const auto& r : ref)
        for (uint32_t v : r) {
            float f;
            std::memcpy(&f, &v, 4);
            if (!std::isfinite(f)) ++nonfinite;
        }
    if (nonfinite) {
        std::printf("FAIL: the ordinary graph left %lld non-finite outputs (the data has no power)\n", nonfinite);
        ++g_fail;
    }
    int bad_replays = 0;
    for (int r = 0; r < replays; ++r) {
        poison();
        ck(cudaGraphLaunch(pdl, s), "launch pdl");
        ck(cudaStreamSynchronize(s), "sync pdl");
        bool same = true;
        for (int i = 0; i < NOUT && same; ++i) same = read(*outs[i], out_n[i]) == ref[(size_t) i];
        if (!same) ++bad_replays;
    }
    std::printf("PDL graph vs ordinary graph over %d replays: %d differ\n", replays, bad_replays);
    if (bad_replays) ++g_fail;

    cudaGraphExecDestroy(plain);
    cudaGraphExecDestroy(pdl);
    std::printf("pdl_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
