// The conversation copy streams and events are pooled across threads (#3350, #3470): short-lived threads (the layer
// split's later stages run each prompt chunk on a fresh std::async thread) must not leave a stream and an event behind
// each.  Synthetic 4 KiB copies, no model; needs a CUDA device (skips, exit 77, without one).
#include "conversation_copy.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace strata::core::conversation_detail;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// one thread's work on device `dev`: a queued copy, the ordering event, a synchronous copy
bool one_thread(int dev, void* d, void* h, size_t bytes) {
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    if (copy_nonblocking(d, h, bytes, true) != cudaSuccess) return false;
    if (order_after_queued() != cudaSuccess) return false;
    if (copy_nonblocking(h, d, bytes, false) != cudaSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        (void) cudaGetLastError();
        std::printf("conversation_copy_streams_test: no CUDA device, skipped\n");
        return 77;
    }
    constexpr size_t bytes = 4096;
    void* d = nullptr;
    check(cudaMalloc(&d, bytes) == cudaSuccess, "device buffer");
    std::vector<char> h(bytes, 7);

    // 0. #2409: what the first copy stream and event of a card cost, on a context that is warm (the buffer above is
    //    allocated, and a first copy has loaded the copy path's code on a thread that is then discarded)
    {
        bool fresh_ready = true;
        std::thread probe([&] { fresh_ready = copy_stream_ready(0); });
        probe.join();
        check(!fresh_ready, "before any copy no thread has a stream for the card (copy_stream_ready is false)");
        cudaDeviceSynchronize();
        size_t free0 = 0, free1 = 0, tot = 0;
        check(cudaMemGetInfo(&free0, &tot) == cudaSuccess, "free memory before the first stream");
        bool ok = false, ready_after = false;
        std::thread t([&] {
            ok = one_thread(0, d, h.data(), bytes);
            ready_after = copy_stream_ready(0);
        });
        t.join();
        check(ok && ready_after, "the first copy creates the stream, and the thread then has it");
        check(cudaMemGetInfo(&free1, &tot) == cudaSuccess, "free memory after the first stream");
        const long long used = (long long) free0 - (long long) free1;
        std::printf("conversation_copy_streams_test: the first copy stream + event of the card took %lld KiB of device memory\n",
                    used / 1024);
        // the restore's guard (program/conv_cache.hpp kRestoreFirstUseMib = 64) is 4x this bound
        check(used <= (16ll << 20), "a card's first copy stream and event take under 16 MiB");
        bool pooled = false;
        std::thread t2([&] { pooled = copy_stream_ready(0); });
        t2.join();
        check(pooled, "a finished thread's stream is in the pool: the next thread needs no new one");
    }

    // 1. 200 threads, one after the other (a chunk per std::async thread): the handles are reused, not created per thread
    const long s0 = copy_streams_created(), e0 = copy_events_created();
    for (int i = 0; i < 200; ++i) {
        bool ok = false;
        std::thread t([&] { ok = one_thread(0, d, h.data(), bytes); });
        t.join();
        check(ok, "a short-lived thread's copies succeed");
    }
    check(copy_streams_created() - s0 <= 1, "200 sequential threads create at most one stream");
    check(copy_events_created() - e0 <= 1, "200 sequential threads create at most one event");

    // 2. threads at the same time each have their own (the number is bounded by the concurrency)
    const long s1 = copy_streams_created();
    std::atomic<int> fails{0};
    for (int round = 0; round < 10; ++round) {
        std::vector<std::thread> ts;
        for (int k = 0; k < 6; ++k)
            ts.emplace_back([&] { for (int j = 0; j < 5; ++j) if (!one_thread(0, d, h.data(), bytes)) ++fails; });
        for (auto& t : ts) t.join();
    }
    check(fails.load() == 0, "concurrent threads' copies succeed");
    check(copy_streams_created() - s1 <= 6, "ten rounds of 6 concurrent threads create at most 6 more streams");

    // 3. a second device has its own
    if (n_dev > 1) {
        const long s2 = copy_streams_created();
        for (int i = 0; i < 50; ++i) {
            bool ok = false;
            std::thread t([&] {
                ok = cudaSetDevice(1) == cudaSuccess;
                void* d1 = nullptr;
                ok = ok && cudaMalloc(&d1, bytes) == cudaSuccess && one_thread(1, d1, h.data(), bytes);
                if (d1) cudaFree(d1);
            });
            t.join();
            check(ok, "device 1: a short-lived thread's copies succeed");
        }
        check(copy_streams_created() - s2 <= 1, "device 1: 50 sequential threads create at most one stream");
    }
    cudaFree(d);
    std::printf("conversation_copy_streams_test: %d checks passed (%ld streams, %ld events created in all)\n", checks,
                copy_streams_created(), copy_events_created());
    return 0;
}
