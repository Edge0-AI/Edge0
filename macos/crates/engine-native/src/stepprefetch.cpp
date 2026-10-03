#include "stepprefetch.hpp"

#include <algorithm>
#include <unordered_set>

namespace e0n {

StepPrefetch::StepPrefetch(HotStack &stack, int n_threads) : stack_(stack) {
  n_threads = std::max(1, n_threads);
  for (int i = 0; i < n_threads; i++) workers_.emplace_back([this] { worker_loop(); });
}

StepPrefetch::~StepPrefetch() {
  {
    std::lock_guard<std::mutex> g(qmu_);
    stop_ = true;
  }
  qcv_.notify_all();
  for (auto &t : workers_) t.join();
}

std::shared_ptr<StepPrefetch::Ticket> StepPrefetch::submit(const Step &plan) {
  auto t = std::make_shared<Ticket>();
  std::lock_guard<std::mutex> g(qmu_);
  t->pending = static_cast<int>(plan.experts.size());
  if (plan.experts.empty()) return t;
  for (auto [layer, expert] : plan.experts) queue_.push_back(Job{t, layer, expert});
  qcv_.notify_all();
  return t;
}

void StepPrefetch::cancel_pending() {
  // Do not decrement ticket counts here (in-flight finish would double-decrement).
  // Mark only: a dequeued worker skips the load and still calls finish so counts stay balanced.
  std::lock_guard<std::mutex> g(qmu_);
  std::unordered_set<Ticket *> seen;
  for (auto &j : queue_) {
    if (seen.insert(j.t.get()).second) j.t->cancelled.store(true);
  }
}

void StepPrefetch::drain() {
  std::unique_lock<std::mutex> g(qmu_);
  qcv_.wait(g, [&] { return queue_.empty() && active_ == 0; });
}

size_t StepPrefetch::queue_depth() const {
  std::lock_guard<std::mutex> g(qmu_);
  return queue_.size();
}

void StepPrefetch::finish(std::shared_ptr<Ticket> t, std::string err) {
  std::lock_guard<std::mutex> g(t->mu);
  if (!err.empty()) { t->bad = true; if (t->err.empty()) t->err = std::move(err); }
  if (--t->pending == 0) t->cv.notify_all();
}

void StepPrefetch::worker_loop() {
  for (;;) {
    Job j;
    {
      std::unique_lock<std::mutex> g(qmu_);
      qcv_.wait(g, [&] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty()) return;
      j = queue_.front();
      queue_.pop_front();
      active_++;
    }
    std::string err;
    if (j.t->cancelled.load()) {
      // Cancelled: skip the load but still finish (count balance; see cancel_pending).
      err.clear();
    } else {
      std::string load_err;
      auto lease = stack_.acquire(j.layer, j.expert, load_err);
      err = load_err;
    }
    finish(j.t, std::move(err));
    {
      std::lock_guard<std::mutex> g(qmu_);
      active_--;
      if (queue_.empty() && active_ == 0) qcv_.notify_all();
    }
  }
}

}  // namespace e0n
