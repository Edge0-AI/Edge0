// Exact multi-row BF16/F32 GEMV checks against independent single-row calls.
// Run with STRATA_MMVF_ROWS=1; synthetic data, no model or private prompts.
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* alloc(size_t n) {
    T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); return p;
}
int main() {
    const char* flag = std::getenv("STRATA_MMVF_ROWS");
    if (!flag || flag[0] != '1') { std::fprintf(stderr, "set STRATA_MMVF_ROWS=1\n"); return 2; }
    cudaStream_t s; ck(cudaStreamCreate(&s));
    const int shapes[][2] = {{64,64},{192,67},{512,65},{640,128},{2560,48},
                             {2560,128},{2560,512},{2560,2560},{10240,320}};
    size_t cases = 0, values = 0, mismatches = 0;
    for (const auto& shape : shapes) for (int nt = 1; nt <= 8; ++nt)
    for (int pad : {0,2}) for (int range : {0,1,2}) {
        const int ni = shape[0], no = shape[1], ldx = ni + pad, ldy = no + pad + 1;
        std::mt19937 rng(1234 + nt * 17 + pad * 31 + range * 53 + ni + no);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        const float xs = range == 0 ? 1.f : range == 1 ? 1.e-12f : 1.e12f;
        const float ws = range == 0 ? 1.f : range == 1 ? 1.e-8f : 1.e8f;
        std::vector<float> x((size_t) nt * ldx);
        std::vector<uint16_t> w((size_t) (no + 1) * ni);
        for (auto& v : x) v = d(rng) * xs;
        for (auto& v : w) { const float f = d(rng) * ws; uint32_t b; std::memcpy(&b,&f,4); v = uint16_t(b >> 16); }
        float* dx = alloc<float>(x.size()); uint16_t* dw = alloc<uint16_t>(w.size());
        const size_t count = (size_t) nt * ldy;
        float* ref = alloc<float>(count); float* got = alloc<float>(count);
        float* aux = alloc<float>((size_t)nt * 3); float* aref = alloc<float>((size_t)nt * 3);
        ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
        ck(cudaMemset(ref,0xff,count*4)); ck(cudaMemset(got,0xff,count*4));
        ck(cudaMemset(aux,0xff,(size_t)nt*12)); ck(cudaMemset(aref,0xff,(size_t)nt*12));
        for (int t = 0; t < nt; ++t) {
            k::bf16_gemv_fp32_mmvf(dx+(size_t)t*ldx,dw,ref+(size_t)t*ldy,ni,no,s);
            k::bf16_gemv_fp32_mmvf(dx+(size_t)t*ldx,dw+(size_t)no*ni,aref+t*3,ni,1,s);
        }
        ck(cudaStreamSynchronize(s));
        cudaGraph_t graph; cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        k::bf16_gemv_fp32_mmvf_multi(dx,ldx,dw,got,ldy,ni,no,nt,s);
        ck(cudaStreamEndCapture(s,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
        ck(cudaGraphLaunch(exec,s)); ck(cudaStreamSynchronize(s));
        auto compare = [&](float* a, float* b, size_t n) {
            std::vector<uint32_t> aa(n),bb(n);
            ck(cudaMemcpy(aa.data(),a,n*4,cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(bb.data(),b,n*4,cudaMemcpyDeviceToHost));
            size_t bad = 0; for (size_t i=0;i<n;++i) bad += aa[i]!=bb[i];
            values += n; mismatches += bad;
            if (bad) std::printf("FAIL ni=%d no=%d nt=%d pad=%d range=%d different=%zu\n",ni,no,nt,pad,range,bad);
        };
        compare(ref,got,count);
        ck(cudaMemset(got,0xff,count*4));
        const bool fused = k::bf16_gemv_fp32_mmvf_multi_aux(dx,ldx,dw,got,ldy,ni,no,nt,
                                                          dw+(size_t)no*ni,aux,3,s);
        if (fused) { ck(cudaStreamSynchronize(s)); compare(ref,got,count); compare(aref,aux,(size_t)nt*3); }
        if (fused != (nt >= 2 && no >= 64)) { std::printf("FAIL eligibility\n"); ++mismatches; }
        ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
        ck(cudaFree(dx)); ck(cudaFree(dw)); ck(cudaFree(ref)); ck(cudaFree(got)); ck(cudaFree(aux)); ck(cudaFree(aref));
        ++cases;
    }
    ck(cudaStreamDestroy(s));
    std::printf("mmvf_rows_parity: %zu cases, %zu values, %zu mismatches\n",cases,values,mismatches);
    return mismatches ? 1 : 0;
}
