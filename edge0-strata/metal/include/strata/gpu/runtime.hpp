// metal/include/strata/gpu/runtime.hpp - the device runtime of the Metal build (MAC-PLAN Phase 1, compat design).
//
// The shared host files keep calling the CUDA runtime names; on Metal those names come from
// metal/include/metal_compat/cuda_runtime.h (first on the include path, as include/strata/hip_compat is on HIP), which
// forwards every one to the functions declared here.  metal/src/gpu/runtime.cpp implements them on metal-cpp:
//   - a device pointer is the MTLBuffer's gpuAddress (the host never dereferences it; kernels take it from their
//     argument struct); host_alloc returns contents() and host_device_pointer maps it to its gpuAddress; every buffer is
//     in one residency set;
//   - a Stream is a command queue + the open command buffer / compute encoder + a shared-event timeline; an Event is a
//     (shared event, value) pair;
//   - begin_capture / end_capture / graph_launch record the stream's dispatches and re-encode them per launch (the CUDA
//     graph rule "per-step values live in device buffers, not in arguments" holds here too);
//   - caps().unified_memory is true and host_view(device_pointer) is the CPU address of a device allocation.
// The CUDA and HIP builds never include this directory.
#pragma once

#include <cstddef>
#include <cstdint>

// The signatures and the meaning of each call are those of the CUDA runtime call of the same purpose.

namespace strata::gpu {

enum Error : int { kSuccess = 0, kNotReady = 600, kErrorInvalidValue = 1, kErrorMemoryAllocation = 2, kErrorUnsupported = 801,
                   kErrorLaunchFailure = 719 };
struct StreamImpl;
struct EventImpl;
struct GraphImpl;
struct GraphExecImpl;
using Stream = StreamImpl*;   // nullptr = the default (legacy) stream, as on CUDA
using Event = EventImpl*;
using Graph = GraphImpl*;
using GraphExec = GraphExecImpl*;
enum CopyKind : int { kHostToHost = 0, kHostToDevice = 1, kDeviceToHost = 2, kDeviceToDevice = 3, kCopyDefault = 4 };
using HostFn = void (*)(void*);
struct DeviceProp {   // the fields the host code reads
    char name[256] = {};
    char gcnArchName[256] = {};
    int major = 0, minor = 0, warpSize = 32, multiProcessorCount = 0;
    size_t totalGlobalMem = 0;
    int integrated = 0;   // the DeepSeek engine reads it (a GPU on the PC's RAM: Strix Halo); 0 here: its rules for that part were not tuned for Apple GPUs
};

inline constexpr unsigned kHostAllocDefault = 0;
inline constexpr unsigned kHostAllocPortable = 1;
inline constexpr unsigned kHostAllocMapped = 2;
inline constexpr unsigned kHostRegisterPortable = 1;
inline constexpr unsigned kHostRegisterMapped = 2;
inline constexpr unsigned kHostRegisterReadOnly = 8;
inline constexpr unsigned kEventDefault = 0;
inline constexpr unsigned kEventBlockingSync = 1;
inline constexpr unsigned kEventDisableTiming = 2;
inline constexpr unsigned kStreamNonBlocking = 1;

struct Caps {
    bool unified_memory = true;
};
inline Caps caps() { return Caps{}; }
void* host_view(const void* device_pointer);

Error get_last_error();
Error peek_last_error();
const char* error_string(Error e);

Error get_device(int* dev);
Error set_device(int dev);
Error device_count(int* n);
Error device_synchronize();
Error mem_get_info(size_t* free_b, size_t* total_b);
Error get_device_properties(DeviceProp* p, int dev);
Error device_integrated(int* v, int dev);
Error device_sm_count(int* v, int dev);
Error device_clock_khz(int* v, int dev);
Error runtime_version(int* v);
Error device_can_access_peer(int* v, int dev, int peer);
Error device_enable_peer_access(int peer, unsigned flags);
inline constexpr Error kErrorPeerAccessAlreadyEnabled = Error(704);

Error malloc_device_raw(void** p, size_t bytes);
template <class T> inline Error malloc_device(T** p, size_t bytes) { return malloc_device_raw((void**) (void*) p, bytes); }
Error free_device(void* p);
Error host_alloc_raw(void** p, size_t bytes, unsigned flags);
template <class T> inline Error host_alloc(T** p, size_t bytes, unsigned flags) { return host_alloc_raw((void**) (void*) p, bytes, flags); }
Error free_host(void* p);
template <class T> inline Error malloc_host(T** p, size_t bytes) { return host_alloc_raw((void**) (void*) p, bytes, 0); }
Error host_device_pointer_raw(void** dev, void* host);
template <class T> inline Error host_device_pointer(T** dev, void* host, unsigned = 0) {
    return host_device_pointer_raw((void**) (void*) dev, host);
}
Error host_register(void* p, size_t bytes, unsigned flags);
Error host_unregister(void* p);
Error memset_device(void* p, int value, size_t bytes);
Error memset_async(void* p, int value, size_t bytes, Stream s = nullptr);

Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind k);
Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind k, Stream s = nullptr);
Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, CopyKind k,
                      Stream s = nullptr);
Error memcpy_peer_async(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, Stream s = nullptr);

Error stream_create(Stream* s, unsigned flags);
inline Error stream_create(Stream* s) { return stream_create(s, 0); }
Error stream_destroy(Stream s);
Error stream_synchronize(Stream s);
Error stream_query(Stream s);
Error stream_wait_event(Stream s, Event e, unsigned flags = 0);
Error launch_host_func(Stream s, HostFn fn, void* arg);

Error event_create(Event* e, unsigned flags);
inline Error event_create(Event* e) { return event_create(e, 0); }
Error event_destroy(Event e);
Error event_record(Event e, Stream s = nullptr);
Error event_query(Event e);
Error event_synchronize(Event e);
Error event_elapsed_ms(float* ms, Event a, Event b);

Error begin_capture(Stream s);
Error end_capture(Stream s, Graph* g);
Error graph_node_count(Graph g, size_t* n);
Error graph_instantiate(GraphExec* x, Graph g);
inline Error graph_instantiate(GraphExec* x, Graph g, unsigned long long) { return graph_instantiate(x, g); }
Error graph_launch(GraphExec x, Stream s);
Error graph_upload(GraphExec x, Stream s);
Error graph_destroy(Graph g);
Error graph_exec_destroy(GraphExec x);

}  // namespace strata::gpu
