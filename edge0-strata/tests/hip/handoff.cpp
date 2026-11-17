// Real asynchronous CPU/GPU handoff and graph replay, including changing payloads.
#include <hip/hip_runtime.h>
#include "strata/kernels/elementwise.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;}} while(0)
int main(){
 constexpr int N=4096, rounds=100;
 float *host=nullptr,*mapped=nullptr,*device=nullptr;
 uint32_t *flag=nullptr,*seq=nullptr,*dflag=nullptr,*dseq=nullptr;
 CHECK(hipHostMalloc((void**)&host,N*4,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&flag,4,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&seq,4,hipHostMallocMapped));
 CHECK(hipHostGetDevicePointer((void**)&mapped,host,0));
 CHECK(hipHostGetDevicePointer((void**)&dflag,flag,0));
 CHECK(hipHostGetDevicePointer((void**)&dseq,seq,0));
 CHECK(hipMalloc((void**)&device,N*4));
 hipStream_t stream;CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 hipGraph_t graph;hipGraphExec_t exec;
 *flag=0;*seq=1;
 CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
 strata::kernels::doorbell_wait(dflag,dseq,(void*)stream);
 strata::kernels::copy_from_mapped(device,mapped,N,(void*)stream);
 strata::kernels::scale_inplace(device,N,2.0f,(void*)stream);
 CHECK(hipStreamEndCapture(stream,&graph));CHECK(hipGraphInstantiateWithFlags(&exec,graph,0));
 std::vector<float> got(N);
 for(int r=1;r<=rounds;r++){
  *seq=r;
  CHECK(hipGraphLaunch(exec,stream));
  // Producer deliberately arrives after the GPU consumer; a missing wait must fail.
  std::this_thread::sleep_for(std::chrono::microseconds(50+(r%17)*10));
  for(int i=0;i<N;i++)host[i]=(float)(r*10000+i);
  __atomic_store_n(flag,(uint32_t)r,__ATOMIC_RELEASE);
  CHECK(hipStreamSynchronize(stream));
  CHECK(hipMemcpy(got.data(),device,N*4,hipMemcpyDeviceToHost));
  for(int i=0;i<N;i++)if(got[i]!=(float)(r*10000+i)*2.0f){std::fprintf(stderr,"stale payload r=%d i=%d got=%f\n",r,i,got[i]);return 1;}
 }
 CHECK(hipGraphExecDestroy(exec));CHECK(hipGraphDestroy(graph));
 // The production `moe_route` fallback records three device-to-device copies into
 // mapped pinned buffers, then launches `doorbell_ring` as a separate graph node.
 // Prove that seeing the increment publishes the preceding payload to the CPU.
 float *copy_out=nullptr,*mapped_copy_out=nullptr,*copy_weights=nullptr,*mapped_copy_weights=nullptr;
 int32_t *copy_ids=nullptr,*mapped_copy_ids=nullptr,*copy_ids_device=nullptr;
 float *copy_weights_device=nullptr;
 uint32_t *copy_seq=nullptr,*device_copy_seq=nullptr;
 CHECK(hipHostMalloc((void**)&copy_out,N*4,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&copy_ids,10*4,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&copy_weights,10*4,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&copy_seq,4,hipHostMallocMapped));
 CHECK(hipHostGetDevicePointer((void**)&mapped_copy_out,copy_out,0));
 CHECK(hipHostGetDevicePointer((void**)&mapped_copy_ids,copy_ids,0));
 CHECK(hipHostGetDevicePointer((void**)&mapped_copy_weights,copy_weights,0));
 CHECK(hipHostGetDevicePointer((void**)&device_copy_seq,copy_seq,0));
 CHECK(hipMalloc((void**)&copy_ids_device,10*4));
 CHECK(hipMalloc((void**)&copy_weights_device,10*4));
 *copy_seq=0;
 std::vector<int32_t> copy_route_ids(10);std::vector<float> copy_route_weights(10);
 CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
 CHECK(hipMemcpyAsync(mapped_copy_out,device,N*4,hipMemcpyDeviceToDevice,stream));
 CHECK(hipMemcpyAsync(mapped_copy_ids,copy_ids_device,10*4,hipMemcpyDeviceToDevice,stream));
 CHECK(hipMemcpyAsync(mapped_copy_weights,copy_weights_device,10*4,hipMemcpyDeviceToDevice,stream));
 strata::kernels::doorbell_ring(device_copy_seq,(void*)stream);
 CHECK(hipStreamEndCapture(stream,&graph));CHECK(hipGraphInstantiateWithFlags(&exec,graph,0));
 for(int r=1;r<=rounds;r++){
  for(int i=0;i<N;i++)got[i]=(float)(r*20000+i);
  for(int i=0;i<10;i++){copy_route_ids[i]=r*10+i;copy_route_weights[i]=r+i*.125f;}
  CHECK(hipMemcpyAsync(device,got.data(),N*4,hipMemcpyHostToDevice,stream));
  CHECK(hipMemcpyAsync(copy_ids_device,copy_route_ids.data(),10*4,hipMemcpyHostToDevice,stream));
  CHECK(hipMemcpyAsync(copy_weights_device,copy_route_weights.data(),10*4,hipMemcpyHostToDevice,stream));
  CHECK(hipGraphLaunch(exec,stream));
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  // Do not query or synchronize the stream before validating the mapped payload.
  while(__atomic_load_n(copy_seq,__ATOMIC_ACQUIRE)!=(uint32_t)r){
   if(std::chrono::steady_clock::now()>deadline){std::fprintf(stderr,"separate copy/ring timeout\n");return 10;}
   std::this_thread::yield();
  }
  for(int i=0;i<N;i++)if(copy_out[i]!=(float)(r*20000+i)){std::fprintf(stderr,"stale separate-copy payload r=%d i=%d\n",r,i);return 11;}
  for(int i=0;i<10;i++)if(copy_ids[i]!=copy_route_ids[i]||copy_weights[i]!=copy_route_weights[i]){std::fprintf(stderr,"stale separate-copy route payload r=%d i=%d\n",r,i);return 12;}
 }
 CHECK(hipStreamSynchronize(stream));
 CHECK(hipGraphExecDestroy(exec));CHECK(hipGraphDestroy(graph));
 float *published=nullptr,*dpublished=nullptr,*weights=nullptr,*dweights=nullptr,*gw=nullptr;
 int32_t *ids=nullptr,*dids=nullptr,*gi=nullptr;
 CHECK(hipHostMalloc((void**)&published,N*4,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&weights,40,hipHostMallocMapped));
 CHECK(hipHostMalloc((void**)&ids,40,hipHostMallocMapped));
 CHECK(hipHostGetDevicePointer((void**)&dpublished,published,0));
 CHECK(hipHostGetDevicePointer((void**)&dweights,weights,0));
 CHECK(hipHostGetDevicePointer((void**)&dids,ids,0));
 CHECK(hipMalloc((void**)&gw,40));CHECK(hipMalloc((void**)&gi,40));
 std::vector<int32_t> ids_src(10);std::vector<float> ws(10);
 *flag=0;
 CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
 strata::kernels::doorbell_publish(device,gi,gw,N,10,dpublished,dids,dweights,dflag,(void*)stream);
 CHECK(hipStreamEndCapture(stream,&graph));CHECK(hipGraphInstantiateWithFlags(&exec,graph,0));
 for(int r=1;r<=rounds;r++){
  for(int i=0;i<N;i++)got[i]=(float)(r*10000+i);
  for(int i=0;i<10;i++){ids_src[i]=r*10+i;ws[i]=r+i*.125f;}
  CHECK(hipMemcpyAsync(device,got.data(),N*4,hipMemcpyHostToDevice,stream));
  CHECK(hipMemcpyAsync(gi,ids_src.data(),40,hipMemcpyHostToDevice,stream));
  CHECK(hipMemcpyAsync(gw,ws.data(),40,hipMemcpyHostToDevice,stream));
  CHECK(hipGraphLaunch(exec,stream));
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  // No driver query or stream sync may make the payload visible to this observer.
  while(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=(uint32_t)r){
   if(std::chrono::steady_clock::now()>deadline){std::fprintf(stderr,"publish timeout\n");return 7;}
   std::this_thread::yield();
  }
  for(int i=0;i<N;i++)if(published[i]!=got[i]){std::fprintf(stderr,"stale GPU publication\n");return 8;}
  for(int i=0;i<10;i++)if(ids[i]!=ids_src[i] || weights[i]!=ws[i])return 9;
  CHECK(hipStreamSynchronize(stream));
 }
 CHECK(hipGraphExecDestroy(exec));CHECK(hipGraphDestroy(graph));CHECK(hipStreamDestroy(stream));
 CHECK(hipFree(gi));CHECK(hipFree(gw));CHECK(hipHostFree(published));CHECK(hipHostFree(ids));CHECK(hipHostFree(weights));
 CHECK(hipFree(device));CHECK(hipHostFree(host));CHECK(hipHostFree(flag));CHECK(hipHostFree(seq));
 CHECK(hipFree(copy_weights_device));CHECK(hipFree(copy_ids_device));
 CHECK(hipHostFree(copy_out));CHECK(hipHostFree(copy_ids));CHECK(hipHostFree(copy_weights));CHECK(hipHostFree(copy_seq));
 std::puts("PASS asynchronous mapped-memory handoff: 100 CPU-to-GPU waits, 100 separate copy/rings, 100 fused publishes; 1232800 values checked");
}
