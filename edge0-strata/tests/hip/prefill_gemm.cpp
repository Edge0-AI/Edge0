// BF16/F16 prefill through the actual engine GEMM path against a CPU reference.
//
// Each shape runs twice: on the engine's GEMM as every card runs it (FP32 out), and with Gemm::set_f16_io(true), the
// gfx103x prompt path's FP16-out GEMM widened in place in Y's own rows (bench/results/2026-10-04-rdna2-fp16-prompt).
// The second mode is forced here on any HIP card, so the path is covered wherever the test runs.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include "strata/prefill/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { CHECK(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void) hipFree(p); }
};

// `f16_io`: the Gemm was set_f16_io(true).  bf16() then takes X as the prompt path's FP16 image (W stays BF16), and
// every GEMM with beta = 0 writes its result as FP16 before widening it: those are checked at FP16's rounding
// (2^-11 relative per element, plus the FP32 accumulation) instead of the FP32-out tolerance.  beta = 1 keeps the
// FP32-out GEMM on the f16() path, and bf16() refuses it there (the prompt path never accumulates a BF16 product).
bool run(strata::prefill::Gemm& gemm, hipStream_t stream, bool bf16,
         int t, int n, int k, int ldy, float beta, bool f16_io = false) {
    std::mt19937 rng(51012 + t + n);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<uint16_t> x((size_t)t*k), w((size_t)n*k);
    std::vector<float> xf(x.size()), wf(w.size());
    auto fill = [&](auto& bits, auto& values, bool as_bf16) {
        for (size_t i = 0; i < bits.size(); ++i) {
            if (as_bf16) {
                // the BF16 bits by hand (round to nearest even on the float's bits): hip_bfloat16(float) produced
                // data 0 in this test's build (ROCm 10.0, not traced further), so both BF16 cases passed on all-zero
                // inputs
                const float f = dist(rng);
                uint32_t u; std::memcpy(&u, &f, 4);
                u += 0x7fffu + ((u >> 16) & 1u);
                bits[i] = static_cast<uint16_t>(u >> 16);
                const uint32_t wide = static_cast<uint32_t>(bits[i]) << 16;
                std::memcpy(&values[i], &wide, 4);
            } else {
                const __half v = __float2half_rn(dist(rng));
                std::memcpy(&bits[i], &v, 2);
                values[i] = __half2float(v);
            }
        }
    };
    fill(x, xf, bf16 && !f16_io); fill(w, wf, bf16);
    const bool f16_out = f16_io && (bf16 || beta == 0.0f);
    constexpr int offset = 5;
    std::vector<float> initial(offset + (size_t)t*ldy + 8, -777.25f);
    for (int r = 0; r < t; ++r) for (int c = 0; c < n; ++c)
        initial[offset + (size_t)r*ldy + c] = (r+c)%17 * 0.03125f;
    Buffer dx(x.size()*2), dw(w.size()*2), dy(initial.size()*4);
    CHECK(hipMemcpyAsync(dx.p,x.data(),x.size()*2,hipMemcpyHostToDevice,stream));
    CHECK(hipMemcpyAsync(dw.p,w.data(),w.size()*2,hipMemcpyHostToDevice,stream));
    CHECK(hipMemcpyAsync(dy.p,initial.data(),initial.size()*4,hipMemcpyHostToDevice,stream));
    if (bf16) gemm.bf16((const uint16_t*)dx.p,(const uint16_t*)dw.p,(float*)dy.p+offset,t,n,k,ldy,beta);
    else gemm.f16((const uint16_t*)dx.p,(const uint16_t*)dw.p,(float*)dy.p+offset,t,n,k,ldy,beta);
    CHECK(hipStreamSynchronize(stream));
    std::vector<float> got(initial.size());
    CHECK(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost));
    std::vector<bool> active(got.size());
    double diff2=0, ref2=0, maximum=0;
    bool ok=true;
    for (int r=0;r<t;++r) for (int c=0;c<n;++c) {
        const size_t j=offset+(size_t)r*ldy+c;
        active[j]=true;
        double ref=beta*initial[j];
        for (int i=0;i<k;++i) ref+=(double)xf[(size_t)r*k+i]*wf[(size_t)c*k+i];
        const double d=(double)got[j]-ref;
        ok=ok && std::isfinite(got[j]);
        if (f16_out) ok=ok && std::abs(d) <= 6e-4*std::abs(ref) + 1e-4;
        diff2+=d*d; ref2+=ref*ref; maximum=std::max(maximum,std::abs(d));
    }
    // the padding of every row (ldy > n) and the guards before and after Y: untouched by the GEMM and the widening
    for (size_t j=0;j<got.size();++j) if (!active[j]) ok=ok && got[j]==initial[j];
    const double rel=std::sqrt(diff2/std::max(ref2,1e-300));
    ok=ok && (f16_out ? rel<1e-3 : (rel<1e-4 && maximum<5e-3));
    std::printf("%s %s%s T=%d N=%d K=%d ldy=%d beta=%.1f rel_l2=%.3g max_abs=%.3g\n",
                ok?"PASS":"FAIL",bf16?"BF16":"F16",f16_io?" (f16 io)":"",t,n,k,ldy,beta,rel,maximum);
    return ok;
}

int main() {
    setvbuf(stdout,nullptr,_IONBF,0);
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    bool ok=true;
    {
        strata::prefill::Gemm gemm;
        std::string error;
        if (!gemm.init(stream,0,error)) { std::fprintf(stderr,"%s\n",error.c_str()); return 2; }
        ok=run(gemm,stream,true,16,96,2560,96,0) && ok;
        ok=run(gemm,stream,true,17,48,2560,64,1) && ok;
        ok=run(gemm,stream,false,32,640,2560,648,0) && ok;
        ok=run(gemm,stream,false,64,2560,640,2560,1) && ok;
        // RDNA2 (gfx103x) runs N >= 64 as an SGEMM on FP32 copies, the activations in slices of 32M floats
        // (Gemm::rdna2_sgemm): at K = 10240 a slice is 3,276 rows, so these cross a slice boundary (padded ldy, beta 1).
        // Other cards run them on the native path.
        ok=run(gemm,stream,false,3300,64,10240,72,1) && ok;
        ok=run(gemm,stream,true,3300,64,10240,64,0) && ok;
    }
    {
        // The FP16-io mode.  The scratch holds 64 rows at K = 2560, so bf16()'s W conversion slices N = 96 in two
        // (64 + 32 rows, the second written at Y + 64).  ldy > N and the odd N check widen_rows_f16's in-place
        // walk (FP16 halves at the start of each FP32 row); T = 1 is a single block.
        strata::prefill::Gemm gemm;
        std::string error;
        if (!gemm.init(stream,64*2560,error)) { std::fprintf(stderr,"%s\n",error.c_str()); return 2; }
        gemm.set_f16_io(true);
        ok=run(gemm,stream,true,16,96,2560,96,0,true) && ok;
        ok=run(gemm,stream,true,17,48,2560,64,0,true) && ok;
        ok=run(gemm,stream,false,32,640,2560,648,0,true) && ok;
        ok=run(gemm,stream,false,64,2560,640,2560,0,true) && ok;
        ok=run(gemm,stream,false,33,257,640,264,0,true) && ok;
        ok=run(gemm,stream,false,1,96,2560,96,0,true) && ok;
        ok=run(gemm,stream,false,17,48,2560,64,1,true) && ok;
    }
    CHECK(hipStreamDestroy(stream));
    return ok?0:1;
}
