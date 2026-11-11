#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#define CU(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} } while(0)
#define BL(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS %d\n",int(e));exit(1);} } while(0)
__global__ void read_all(const float4* p,size_t n,float* out){
 float4 a=make_float4(0,0,0,0);
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i];a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;}
 out[blockIdx.x*blockDim.x+threadIdx.x]=a.x+a.y+a.z+a.w;
}
int main(){
 CU(cudaSetDevice(0));cudaDeviceProp p;CU(cudaGetDeviceProperties(&p,0));
 const size_t bytes=256ull<<20;float4* d;float* sum;void* host;
 CU(cudaMalloc(&d,bytes));CU(cudaMalloc(&sum,p.multiProcessorCount*16*256*sizeof(float)));CU(cudaMallocHost(&host,bytes));
 CU(cudaMemset(d,0,bytes));cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
 for(int kind=0;kind<3;++kind)for(int trial=0;trial<3;++trial){
  CU(cudaEventRecord(a));for(int i=0;i<30;++i){
   if(kind==0)read_all<<<p.multiProcessorCount*16,256>>>(d,bytes/sizeof(float4),sum);
   else CU(cudaMemcpyAsync(kind==1?(void*)d:host,kind==1?host:(void*)d,bytes,kind==1?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost));
  }CU(cudaEventRecord(b));CU(cudaEventSynchronize(b));float ms;CU(cudaEventElapsedTime(&ms,a,b));
  printf("{\"kind\":\"%s\",\"trial\":%d,\"bytes\":%zu,\"iterations\":30,\"ms\":%.6f,\"GB_s\":%.6f}\n",kind==0?"vram_read":kind==1?"h2d":"d2h",trial,bytes,ms,bytes*30.0/(ms*1e6));
 }
 CU(cudaFree(d));CU(cudaFree(sum));CU(cudaFreeHost(host));
 cublasHandle_t h;BL(cublasCreate(&h));float *w,*x,*y;CU(cudaMalloc(&w,2560*2560*sizeof(float)));CU(cudaMalloc(&x,2560*512*sizeof(float)));CU(cudaMalloc(&y,2560*512*sizeof(float)));
 CU(cudaMemset(w,0,2560*2560*sizeof(float)));CU(cudaMemset(x,0,2560*512*sizeof(float)));float alpha=1,beta=0;
 int rows[]={10,20,40,80,160,512};
 for(int shape=0;shape<2;++shape)for(int n:rows){
  int m=shape==0?1280:2560,k=shape==0?2560:640;
  for(int i=0;i<10;++i)BL(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,w,m,x,k,&beta,y,m));
  for(int trial=0;trial<3;++trial){
   CU(cudaEventRecord(a));for(int i=0;i<100;++i)BL(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,w,m,x,k,&beta,y,m));
   CU(cudaEventRecord(b));CU(cudaEventSynchronize(b));float ms;CU(cudaEventElapsedTime(&ms,a,b));
   printf("{\"kind\":\"sgemm\",\"m\":%d,\"n\":%d,\"k\":%d,\"trial\":%d,\"ms_per_call\":%.6f,\"TFLOP_s\":%.6f}\n",m,n,k,trial,ms/100,2.0*m*n*k*100/(ms*1e9));
  }
 }
 BL(cublasDestroy(h));CU(cudaFree(w));CU(cudaFree(x));CU(cudaFree(y));CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
}
