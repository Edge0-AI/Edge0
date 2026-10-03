// Resident non-expert weights and shared forward ops (safio read → heap → mlx no-op deleter).
#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "safio.hpp"

namespace e0n {

mlx::core::Shape mkshape(std::initializer_list<int64_t> dims);
inline mlx::core::Shape mkshape(const std::vector<int64_t> &dims) {
  mlx::core::Shape s;
  for (auto d : dims) s.push_back(static_cast<int32_t>(d));
  return s;
}

// Quantized dense linear (attention / shared expert / lm_head / gate int8):
// weight [O, I/pack] U32 + scales/biases [O, I/g] BF16; affine; bits/group per module.
struct QuantLin {
  std::vector<uint8_t> wb, sb, bb;  // resident bytes (stable array pointers)
  mlx::core::array w{0.f}, s{0.f}, b{0.f};
  int64_t rows = 0, cols = 0;       // logical [O, I]
  int64_t bits = 4, group = 64;
  bool loaded = false;

  bool load(const ModelFiles &files, const std::string &weight_name, std::string &err);
  std::optional<mlx::core::array> biases() const {
    if (bb.empty()) return std::nullopt;
    return b;
  }
  mlx::core::array apply(const mlx::core::array &x) const;
};

struct RawLin {
  std::vector<uint8_t> wb;
  mlx::core::array w{0.f};  // original dtype [rows, cols]
  int64_t rows = 0, cols = 0;
  bool load(const ModelFiles &files, const std::string &name, std::string &err);
  std::optional<mlx::core::array> biases() const { return std::nullopt; }
  mlx::core::array apply(const mlx::core::array &x) const;  // x @ wᵀ
};

mlx::core::array e0_silu(const mlx::core::array &x);
// Qwen3Next._precise_swiglu: gate/up to f32, silu(gate)*up, cast back to out dtype.
mlx::core::array e0_swiglu_precise(mlx::core::Dtype out_dt, const mlx::core::array &gate,
                                   const mlx::core::array &up);
// Empty weight → unscaled rms_norm (GDN q/k).
mlx::core::array e0_rms_norm(const mlx::core::array &x,
                             const std::optional<mlx::core::array> &weight, double eps);

bool to_host32(const mlx::core::array &a, std::vector<float> &out, std::string &err);
// Host→graph: iterator ctor copies into mlx-owned buffers. no-op-deleter wrap is
// only safe for member weights that outlive the graph (QuantLin::load); wrapping a
// local vector that a lazy graph still holds after eval is a dangling read.
mlx::core::array wrap_f32(const std::vector<float> &v, mlx::core::Shape shape);

inline float silu_f(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }

struct ExecCounters {
  std::atomic<uint64_t> qmm_calls{0};
  std::atomic<uint64_t> qmm_experts{0};
  std::atomic<uint64_t> qmm_weight_bytes{0};
  std::atomic<uint64_t> host_syncs{0};
  std::atomic<uint64_t> host_wraps{0};
  std::atomic<uint64_t> leases{0};
  struct Snap {
    uint64_t qmm_calls, qmm_experts, qmm_weight_bytes, host_syncs, host_wraps, leases;
  };
  Snap snapshot() const {
    return {qmm_calls.load(std::memory_order_acquire), qmm_experts.load(std::memory_order_acquire),
            qmm_weight_bytes.load(std::memory_order_acquire),
            host_syncs.load(std::memory_order_acquire),
            host_wraps.load(std::memory_order_acquire),
            leases.load(std::memory_order_acquire)};
  }
  static Snap diff(const Snap &a, const Snap &b) {  // b − a
    return {b.qmm_calls - a.qmm_calls,       b.qmm_experts - a.qmm_experts,
            b.qmm_weight_bytes - a.qmm_weight_bytes, b.host_syncs - a.host_syncs,
            b.host_wraps - a.host_wraps,     b.leases - a.leases};
  }
  void reset() {
    qmm_calls = 0; qmm_experts = 0; qmm_weight_bytes = 0;
    host_syncs = 0; host_wraps = 0; leases = 0;
  }
};
ExecCounters &exec_counters();

struct TimingCounters {
  std::atomic<uint64_t> eval_ns{0};
  std::atomic<uint64_t> host_sync_ns{0};
  std::atomic<uint64_t> gate_cpu_ns{0};
  std::atomic<uint64_t> staging_ns{0};
  struct Snap { uint64_t eval_ns, host_sync_ns, gate_cpu_ns, staging_ns; };
  Snap snapshot() const {
    return {eval_ns.load(std::memory_order_relaxed), host_sync_ns.load(std::memory_order_relaxed),
            gate_cpu_ns.load(std::memory_order_relaxed), staging_ns.load(std::memory_order_relaxed)};
  }
  static bool enabled();  // EDGE0_TIMING=1 (once per process)
};
TimingCounters &timing_counters();
struct TimeSink {
  std::atomic<uint64_t> &sink;
  uint64_t t0;
  explicit TimeSink(std::atomic<uint64_t> &s)
      : sink(s), t0(TimingCounters::enabled() ? monotonic_ns() : 0) {}
  ~TimeSink() { stop(); }
  void stop() {
    if (t0) {
      sink.fetch_add(monotonic_ns() - t0, std::memory_order_relaxed);
      t0 = 0;
    }
  }
  static uint64_t monotonic_ns();
};
#define E0_TIME_CAT2(a, b) a##b
#define E0_TIME_CAT(a, b) E0_TIME_CAT2(a, b)
#define E0_TIME(field) ::e0n::TimeSink E0_TIME_CAT(_e0_ts_, __LINE__)(field)

enum class ExecMode { Legacy, Batched };
ExecMode exec_mode();  // first call reads env (default batched; unknown → warn + legacy)

// EDGE0_PREFILL=ondemand (default) vs production. production is an env-only hybrid
// (hot fast path is not built); it is not a faithful production-shape replica.
bool prefill_ondemand();

struct Entry;
struct ExpertPtrs {
  const uint32_t *gw = nullptr, *uw = nullptr, *dw = nullptr;
  const uint16_t *gs = nullptr, *gb = nullptr, *us = nullptr, *ub = nullptr, *ds = nullptr,
                 *db = nullptr;
};
bool expert_ptrs(const Entry &e, ExpertPtrs &p, std::string &err);

bool embed_lookup(const QuantLin &emb, const std::vector<int32_t> &ids, int64_t H, int64_t V,
                  std::vector<float> &out, std::string &err);

}  // namespace e0n
