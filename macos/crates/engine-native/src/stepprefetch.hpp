// Async per-step prefetch: submit(step i+1) returns a ticket; IO threads load into
// the hot stack while compute GEMMs step i. turn_reset → cancel_pending + drain.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hotstack.hpp"
#include "provider.hpp"

namespace e0n {

class StepPrefetch {
 public:
  struct Ticket {
    std::mutex mu;
    std::condition_variable cv;
    int pending = 0;
    std::atomic<bool> cancelled{false};  // dequeued worker skips load but still finishes the count
    bool bad = false;
    std::string err;
    bool done() const { return pending == 0; }
    bool wait() {
      std::unique_lock<std::mutex> g(mu);
      cv.wait(g, [&] { return pending == 0; });
      return !bad;
    }
  };

  StepPrefetch(HotStack &stack, int n_threads);
  ~StepPrefetch();  // cancel then join; no in-flight load at destructor is the caller's duty

  std::shared_ptr<Ticket> submit(const Step &plan);

  void cancel_pending();
  void drain();
  size_t queue_depth() const;

 private:
  struct Job {
    std::shared_ptr<Ticket> t;
    int layer;
    int64_t expert;
  };
  void finish(std::shared_ptr<Ticket> t, std::string err);
  void worker_loop();

  HotStack &stack_;
  std::vector<std::thread> workers_;
  mutable std::mutex qmu_;
  std::condition_variable qcv_;
  std::deque<Job> queue_;
  bool stop_ = false;
  int active_ = 0;
};

}  // namespace e0n
