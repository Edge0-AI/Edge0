#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/gemm.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
struct Q8 { sycl::half d; int8_t qs[32]; };
static_assert(sizeof(Q8)==34);
uint16_t bf(float x) { uint32_t b; std::memcpy(&b,&x,4); return uint16_t(b>>16); }
uint16_t half_bits(float x) { sycl::half h=x; uint16_t b;std::memcpy(&b,&h,2);return b; }
int main() {
 auto& q=dpct::get_in_order_queue(); constexpr int T=3,N=3,K=32,L=40,YSTRIDE=5;
 auto* xb=sycl::malloc_shared<uint16_t>(T*L,q);auto* xh=sycl::malloc_shared<uint16_t>(T*L,q);
 auto* w=sycl::malloc_shared<uint16_t>(N*K,q);auto* qw=sycl::malloc_shared<Q8>(N,q);
 auto* y=sycl::malloc_shared<float>(T*YSTRIDE,q);auto* scratch=sycl::malloc_device<uint16_t>(N*K,q);
 int checks=0;
 for(int pad: {0,1}) for(int beta: {0,1}) {
  int stride=pad?L:K;
  for(int i=0;i<T*L;i++){xb[i]=bf(123.f);xh[i]=half_bits(123.f);}
  for(int t=0;t<T;t++)for(int k=0;k<K;k++){float v=float((t+k)%7-3)/8;xb[t*stride+k]=bf(v);xh[t*stride+k]=half_bits(v);}
  for(int n=0;n<N;n++){qw[n].d=sycl::half(0.125f);for(int k=0;k<K;k++){int v=(n+2*k)%7-3;qw[n].qs[k]=int8_t(v);w[n*K+k]=bf(float(v)/8);}}
  auto check=[&](const char* name){q.wait_and_throw();for(int t=0;t<T;t++)for(int n=0;n<YSTRIDE;n++){
   float expected=7.f;
   if(n<N){expected=beta?7.f:0.f;for(int k=0;k<K;k++)expected+=float((t+k)%7-3)/8*float((n+2*k)%7-3)/8;}
   if(std::fabs(y[t*YSTRIDE+n]-expected)>1e-4f){std::fprintf(stderr,"%s stride=%d beta=%d mismatch\n",name,stride,beta);std::exit(1);}
  }checks++;};
  {
   strata::prefill::Gemm g;std::string err;if(!g.init_external(&q,scratch,N*K,nullptr,0,err))return 2;
   for(int i=0;i<T*YSTRIDE;i++)y[i]=7;
   g.bf16(xb,w,y,T,N,K,YSTRIDE,float(beta),pad?stride:0);check("bf16");
   for(int rows: {1,N}){
    g.rebind(scratch,rows*K,nullptr,0);for(int i=0;i<T*YSTRIDE;i++)y[i]=7;
    g.native(xh,8,qw,y,T,N,K,YSTRIDE,float(beta),pad?stride:0);check("native-q8");
   }
  }
 }
 q.wait_and_throw();for(void* p: {static_cast<void*>(xb),static_cast<void*>(xh),static_cast<void*>(w),static_cast<void*>(qw),static_cast<void*>(y),static_cast<void*>(scratch)})sycl::free(p,q);
 std::printf("GEMM stride checks: %d passed (contiguous/padded, beta0/1, full/sliced native)\n",checks);return 0;
}
