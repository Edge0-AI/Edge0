#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <functional>
#include <thread>
#include <utility>

namespace strata::core {

// Instrumented cudaGraphLaunch may block until device work completes. A graph
// which waits for CPU doorbells must therefore launch outside its service thread.
// The caller still services exactly the same graph and observes the real launch
// error before synchronizing or publishing its output.
class HostGraphLaunch {
public:
    HostGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream, int device,
                    bool threaded, std::function<void()> release_waits)
        : stream_(stream), release_waits_(std::move(release_waits)) {
        if (threaded) {
            pending_.store(true);
            worker_ = std::thread([this, graph, device] {
                auto error = cudaSetDevice(device);
                if (error == cudaSuccess) error = cudaGraphLaunch(graph, stream_);
                error_ = error;
                pending_.store(false, std::memory_order_release);
            });
        } else {
            error_ = cudaGraphLaunch(graph, stream_);
        }
    }
    HostGraphLaunch(const HostGraphLaunch&) = delete;
    HostGraphLaunch& operator=(const HostGraphLaunch&) = delete;

    ~HostGraphLaunch() {
        // On an early service failure, release the GPU's mapped waits before
        // joining a launcher that may be blocked behind those waits.
        if (worker_.joinable() && !finished_) release_waits_();
        if (worker_.joinable()) worker_.join();
    }

    bool pending() const { return pending_.load(std::memory_order_acquire); }
    cudaError_t status() const { return pending() ? cudaSuccess : error_; }
    cudaError_t query() const {
        if (pending()) return cudaErrorNotReady;
        if (error_ != cudaSuccess) return error_;
        return cudaStreamQuery(stream_);
    }
    cudaError_t finish() {
        if (worker_.joinable()) worker_.join();
        finished_ = true;
        return error_;
    }

private:
    cudaStream_t stream_;
    std::function<void()> release_waits_;
    std::atomic<bool> pending_{false};
    cudaError_t error_ = cudaSuccess;
    std::thread worker_;
    bool finished_ = false;
};

} // namespace strata::core
