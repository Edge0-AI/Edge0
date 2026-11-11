// Byte parity and event ordering for independent pinned-host expert uploads.
#include "strata/core/dma_batch.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#define CK(call) do { auto e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 1; } } while(0)
int main(int argc, char** argv) {
    const bool registered = argc > 1 && !std::strcmp(argv[1], "--registered");
    int runtime=0, driver=0;
    CK(cudaRuntimeGetVersion(&runtime)); CK(cudaDriverGetVersion(&driver));
    std::printf("headers=%d runtime=%d driver=%d registered=%d\n", CUDART_VERSION, runtime, driver, registered);
    constexpr size_t capacity=129u*2u*1024u*1024u;
    uint8_t *host=nullptr,*out=nullptr,*device=nullptr;
#if defined(_WIN32)
    if (registered) {
        host=(uint8_t*)VirtualAlloc(nullptr,capacity,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
        if (!host) return 3;
        CK(cudaHostRegister(host,capacity,cudaHostRegisterPortable));
    } else
#endif
    CK(cudaMallocHost(&host,capacity));
    CK(cudaMallocHost(&out,capacity));
    CK(cudaMalloc(&device,capacity));
    for(size_t i=0;i<capacity;++i) host[i]=(uint8_t)((i*17+(i>>8)*31+(i>>17))&255);
    cudaStream_t copy,consume; cudaEvent_t ready;
    CK(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&consume,cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    int cases=0;
    for(size_t bytes:{size_t(37),size_t(4096),size_t(2*1024*1024)}) {
        for(int n:{0,1,4,17,64,128,129}) for(int mode:{0,1,2}) {
            std::vector<const uint8_t*> src(n);
            for(int i=0;i<n;++i) src[i]=host+(size_t)((i*37)%129)*bytes;
            std::printf("case mode=%d n=%d bytes=%zu\n",mode,n,bytes); std::fflush(stdout);
            CK(cudaMemsetAsync(device,0xa5,std::max<size_t>(bytes*n,1),copy));
            CK(strata::core::copy_expert_blobs(device,src.data(),n,bytes,copy,mode));
            CK(cudaEventRecord(ready,copy));
            CK(cudaStreamWaitEvent(consume,ready,0));
            CK(cudaMemcpyAsync(out,device,std::max<size_t>(bytes*n,1),cudaMemcpyDeviceToHost,consume));
            CK(cudaStreamSynchronize(consume));
            if(!n && out[0]!=0xa5) return 2;
            for(int i=0;i<n;++i) if(std::memcmp(out+(size_t)i*bytes,src[i],bytes)) {
                std::fprintf(stderr,"mismatch mode=%d n=%d bytes=%zu i=%d\n",mode,n,bytes,i);return 2;
            }
            ++cases;
        }
    }
    std::printf("PASS: %d byte-parity/event-order cases, CUDA %d\n",cases,CUDART_VERSION);
    // Interleaved copy-only diagnostic at representative expert sizes/counts.
    // Host enqueue cost and total wall time are separate, not additive.
    for(int n:{4,16,48}) {
        const size_t bytes=2*1024*1024;
        std::vector<const uint8_t*> src(n);
        for(int i=0;i<n;++i) src[i]=host+(size_t)((i*37)%129)*bytes;
        std::vector<double> wall[3],enqueue[3];
#if defined(_WIN32)
        std::vector<double> cycles[3];
#endif
        for(int rep=0;rep<6;++rep) for(int j=0;j<3;++j) {
            int mode=(rep%2)?2-j:j;
            CK(cudaStreamSynchronize(copy));
#if defined(_WIN32)
            ULONG64 cycle_start=0,cycle_end=0;
            if (!QueryThreadCycleTime(GetCurrentThread(), &cycle_start)) return 3;
#endif
            auto start=std::chrono::steady_clock::now();
            for(int k=0;k<16;++k) CK(strata::core::copy_expert_blobs(device,src.data(),n,bytes,copy,mode));
            auto submitted=std::chrono::steady_clock::now();
#if defined(_WIN32)
            if (!QueryThreadCycleTime(GetCurrentThread(), &cycle_end)) return 3;
#endif
            CK(cudaStreamSynchronize(copy));
            auto end=std::chrono::steady_clock::now();
            if(rep) {
                enqueue[mode].push_back(std::chrono::duration<double,std::milli>(submitted-start).count()/16);
                wall[mode].push_back(std::chrono::duration<double,std::milli>(end-start).count()/16);
#if defined(_WIN32)
                cycles[mode].push_back(double(cycle_end-cycle_start)/16);
#endif
            }
        }
        for(int mode=0;mode<3;++mode) {
            std::sort(wall[mode].begin(),wall[mode].end());std::sort(enqueue[mode].begin(),enqueue[mode].end());
            std::printf("n=%d bytes=%zu mode=%d median_submit_ms=%.4f median_total_ms=%.4f",n,bytes,mode,enqueue[mode][2],wall[mode][2]);
#if defined(_WIN32)
            std::sort(cycles[mode].begin(),cycles[mode].end());
            std::printf(" median_thread_cycles=%.0f",cycles[mode][2]);
#endif
            std::printf("\n");
        }
    }
    CK(cudaEventDestroy(ready)); CK(cudaStreamDestroy(copy)); CK(cudaStreamDestroy(consume));
    CK(cudaFree(device));
#if defined(_WIN32)
    if (registered) { CK(cudaHostUnregister(host)); VirtualFree(host,0,MEM_RELEASE); } else
#endif
    CK(cudaFreeHost(host));
    CK(cudaFreeHost(out));
    return 0;
}
