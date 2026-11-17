#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace strata::core {

struct BatchWindowSample {
    int64_t windows = 0;
    int64_t processed_rows = 0;  // rows run by the verifier, including pipeline padding and MTP rows
    int64_t active_rows = 0;     // live request slots represented by those windows
    int64_t elapsed_ms = 0;      // first included window's start to the last completion
};

// Count completed decode windows only. The caller emits one small protocol line per second,
// plus a final partial sample when all slots become idle; the default-disabled path emits none.
class BatchWindowSampler {
public:
    using Clock = std::chrono::steady_clock;

    explicit BatchWindowSampler(bool enabled) : enabled_(enabled) {}

    std::optional<BatchWindowSample> complete(int processed_rows, int active_rows, bool idle,
                                              Clock::time_point started, Clock::time_point now) {
        if (!enabled_ || processed_rows < 1 || active_rows < 1 || active_rows > processed_rows || now < started)
            return std::nullopt;
        if (pending_.windows == 0 || started < first_) first_ = started;
        ++pending_.windows;
        pending_.processed_rows += processed_rows;
        pending_.active_rows += active_rows;
        if (!idle && now - first_ < std::chrono::seconds(1)) return std::nullopt;
        BatchWindowSample sample = pending_;
        sample.elapsed_ms = std::max<int64_t>(1,
            std::chrono::duration_cast<std::chrono::milliseconds>(now - first_).count());
        pending_ = {};
        return sample;
    }

private:
    bool enabled_ = false;
    BatchWindowSample pending_{};
    Clock::time_point first_{};
};

struct StageInFlightSample {
    int64_t elapsed_us = 0;
    std::vector<int64_t> in_flight_us;  // host time between a stage launch and its completed poll
};

// A pipeline stage holds at most one group. Measure the time it is reserved by
// engine work, including poll latency, without claiming GPU kernel utilization.
class StageInFlightSampler {
public:
    using Clock = std::chrono::steady_clock;

    StageInFlightSampler(int stages, bool enabled)
        : enabled_(enabled && stages > 1 && stages <= 16),
          active_(enabled_ ? (std::size_t) stages : 0),
          in_flight_us_(enabled_ ? (std::size_t) stages : 0, 0) {}

    void launch(int stage, Clock::time_point now) {
        if (!valid(stage)) return;
        if (!started_) started_ = now;
        active_[(std::size_t) stage] = now;
    }

    void complete(int stage, Clock::time_point now) {
        if (!valid(stage)) return;
        auto& since = active_[(std::size_t) stage];
        if (!since || now < *since) return;
        in_flight_us_[(std::size_t) stage] +=
            std::chrono::duration_cast<std::chrono::microseconds>(now - *since).count();
        since.reset();
    }

    std::optional<StageInFlightSample> sample(Clock::time_point now, bool idle) {
        if (!enabled_ || !started_ || now < *started_ ||
            (!idle && now - *started_ < std::chrono::seconds(1))) return std::nullopt;
        const auto elapsed = std::max<int64_t>(1,
            std::chrono::duration_cast<std::chrono::microseconds>(now - *started_).count());
        for (std::size_t stage = 0; stage < active_.size(); ++stage) {
            if (!active_[stage]) continue;
            in_flight_us_[stage] += std::chrono::duration_cast<std::chrono::microseconds>(
                now - *active_[stage]).count();
            active_[stage] = now;
        }
        StageInFlightSample result{elapsed, in_flight_us_};
        for (auto& value : result.in_flight_us) value = std::clamp(value, (int64_t) 0, elapsed);
        std::fill(in_flight_us_.begin(), in_flight_us_.end(), 0);
        if (idle) {
            started_.reset();
            std::fill(active_.begin(), active_.end(), std::nullopt);
        } else started_ = now;
        return result;
    }

private:
    bool valid(int stage) const { return enabled_ && stage >= 0 && (std::size_t) stage < active_.size(); }

    bool enabled_ = false;
    std::optional<Clock::time_point> started_;
    std::vector<std::optional<Clock::time_point>> active_;
    std::vector<int64_t> in_flight_us_;
};

}  // namespace strata::core
