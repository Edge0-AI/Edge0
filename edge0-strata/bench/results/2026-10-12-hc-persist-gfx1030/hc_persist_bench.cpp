// The engine's STRATA_HC_PERSIST read (public API) against the default read, per token count: the largest relative
// difference from the default read, and the time per read in a graph of 48 cold reads. Run with STRATA_HC_PERSIST=1
#include <hip/hip_runtime.h>
#include "strata/kernels/fused_gr.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace K = strata::kernels;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
static uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int W = argc > 1 ? std::atoi(argv[1]) : 40;
    K::fused_gr_check();
    K::fused_gr_persist_init();
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng); for (auto& x : bo) x = 0.5f * nd(rng); for (auto& x : inj) x = nd(rng); for (auto& x : wn) x = 1.f + 0.3f * nd(rng);
    const size_t WD = (size_t) LR * D, WU = WD, WI = (size_t) HC * D;
    uint16_t *dwd, *dwu, *dwi; CK(hipMalloc(&dwd, WD * 2 * W)); CK(hipMalloc(&dwu, WU * 2 * W)); CK(hipMalloc(&dwi, WI * 2 * W));
    { std::vector<uint16_t> b(WD); for (int s = 0; s < W; ++s) {
        for (auto& x : b) x = bf16(0.02f * nd(rng)); CK(hipMemcpy(dwd + WD * s, b.data(), WD * 2, hipMemcpyHostToDevice));
        for (auto& x : b) x = bf16(0.05f * nd(rng)); CK(hipMemcpy(dwu + WU * s, b.data(), WU * 2, hipMemcpyHostToDevice));
        for (size_t i = 0; i < WI; ++i) b[i] = bf16(0.02f * nd(rng)); CK(hipMemcpy(dwi + WI * s, b.data(), WI * 2, hipMemcpyHostToDevice)); } }
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    CK(hipMalloc(&dR, R.size() * 4)); CK(hipMalloc(&dbo, bo.size() * 4)); CK(hipMalloc(&dinj, inj.size() * 4)); CK(hipMalloc(&dwn, D * 4));
    CK(hipMalloc(&dxn, (size_t) TM * D * 4)); CK(hipMalloc(&dlo, (size_t) TM * LR * 4)); CK(hipMalloc(&drs, TM * HC * 4));
    CK(hipMalloc(&dio, TM * HC * 4)); CK(hipMalloc(&dmix, (size_t) TM * N * 4));
    CK(hipMemcpy(dbo, bo.data(), bo.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dinj, inj.data(), inj.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dwn, wn.data(), D * 4, hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    auto args = [&](K::FusedGrArgs* a, int T, int set) {
        for (int t = 0; t < T; ++t) { a[t] = K::FusedGrArgs{}; a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = true;
            a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + t * HC; a[t].w_norm = dwn; a[t].w_down = dwd + WD * set;
            a[t].w_up = dwu + WU * set; a[t].w_inject = dwi + WI * set; a[t].eps = 1e-6f; a[t].lo = dlo + t * LR; a[t].rs = drs + t * HC;
            a[t].inject_out = dio + t * HC; a[t].mixed = dmix + (size_t) t * N; } };
    for (int T = 1; T <= TM; ++T) {
        K::FusedGrArgs a[TM]; args(a, T, 0);
        std::vector<float> out[2];
        bool ran = true;
        for (int w = 0; w < 2; ++w) {
            CK(hipMemcpy(dR, R.data(), R.size() * 4, hipMemcpyHostToDevice)); CK(hipMemset(dmix, 0xff, (size_t) TM * N * 4)); CK(hipDeviceSynchronize());
            if (w == 0) K::fused_gr_read_multi(a, T, dxn, st); else ran = K::fused_gr_read_persist(a, T, st);
            CK(hipStreamSynchronize(st));
            out[w].resize((size_t) T * N); CK(hipMemcpy(out[w].data(), dmix, out[w].size() * 4, hipMemcpyDeviceToHost));
        }
        if (!ran) { std::printf("T %d: persistent read not used here (falls back)\n", T); continue; }
        double mr = 0, md = 0; bool fin = true;
        for (size_t i = 0; i < out[0].size(); ++i) { fin = fin && std::isfinite(out[1][i]); mr = std::fmax(mr, std::fabs(out[0][i])); md = std::fmax(md, std::fabs(out[1][i] - out[0][i])); }
        double us[2];
        for (int w = 0; w < 2; ++w) {
            double best = 1e30;
            for (int r = 0; r < 3; ++r) {
                hipGraph_t g; hipGraphExec_t ge;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
                for (int i = 0; i < 48; ++i) { args(a, T, (i * 7) % W); if (w == 0) K::fused_gr_read_multi(a, T, dxn, st); else K::fused_gr_read_persist(a, T, st); }
                CK(hipStreamEndCapture(st, &g)); CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 2; ++i) CK(hipGraphLaunch(ge, st));
                CK(hipEventRecord(e0, st)); for (int i = 0; i < 10; ++i) CK(hipGraphLaunch(ge, st)); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; hipEventElapsedTime(&ms, e0, e1); best = std::fmin(best, 1e3 * ms / 480);
                hipGraphExecDestroy(ge); hipGraphDestroy(g);
            }
            us[w] = best;
        }
        std::printf("T %d: default %6.1f us  persistent %6.1f us  (%.2fx)  max rel diff %.2e%s\n", T, us[0], us[1], us[0] / us[1], md / mr, fin ? "" : "  NON-FINITE");
    }
    return 0;
}
