// Expert load/staging: mmap shards, global LRU, prefetch buffer, and staged double-buffer.
// Stack/lazy-graph construction stays on the main thread; pool threads only produce Bundles.
// Prefetch done-callbacks are registered outside the lock (deadlock otherwise).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#include "safio.hpp"

namespace e0n {
namespace mx = mlx::core;

// 9-region bundle. Field order is the moe_math signature: (up,gate,down)×(w,s,b),
// not file key order (gate_proj before up_proj). Shape is the stacked tensor
// without the expert axis [O,K]; elements live in mlx-owned buffers.
struct Bundle {
  Bundle()
      : f{mx::array(0.f), mx::array(0.f), mx::array(0.f), mx::array(0.f), mx::array(0.f),
         mx::array(0.f), mx::array(0.f), mx::array(0.f), mx::array(0.f)} {}
  mx::array f[9];  // 0..2=up w/s/b, 3..5=gate w/s/b, 6..8=down w/s/b
  uint64_t bytes = 0;
  const mx::array &at(int p) const { return f[p]; }
  mx::array &at_mut(int p) { return f[p]; }
};

class SharedExpertCache {
 public:
  explicit SharedExpertCache(int slots) : slots_(slots) {}
  std::shared_ptr<Bundle> get(uint64_t key);
  std::shared_ptr<Bundle> peek(uint64_t key);
  void put(uint64_t key, std::shared_ptr<Bundle> b);
  std::shared_ptr<Bundle> pop(uint64_t key);
  bool contains(uint64_t key);
  size_t size();
  uint64_t evictions() const { return evictions_; }

 private:
  int slots_;
  uint64_t evictions_ = 0;
  std::mutex mu_;
  std::deque<uint64_t> order_;  // front = LRU
  std::map<uint64_t, std::shared_ptr<Bundle>> map_;
};

class PrefetchBuffer {
 public:
  explicit PrefetchBuffer(int cap) : cap_(cap) {}
  bool put(uint64_t key, std::shared_ptr<Bundle> b);     // true = kept, false = overflow drop
  std::shared_ptr<Bundle> pop(uint64_t key);
  bool contains(uint64_t key);
  void set_cap(int c) { cap_ = c; }

 private:
  int cap_;
  std::mutex mu_;
  std::deque<uint64_t> order_;
  std::map<uint64_t, std::shared_ptr<Bundle>> map_;
};

class IoPool {
 public:
  explicit IoPool(int n);
  ~IoPool();
  template <class F>
  auto submit(F &&f) -> std::shared_future<decltype(f())> {
    using R = decltype(f());
    auto pk = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut = pk->get_future().share();
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.emplace_back([pk] { (*pk)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
  bool stop_ = false;
  std::vector<std::thread> th_;
};

class ExpertIo {
 public:
  struct LayerCfg {
    int layer = 0;
    std::string prefix;   // "model.layers.N.mlp.experts" / "...mlp.switch_mlp"
    int64_t E = 0;
    int64_t H = 0, F = 0;
    int group = 64;
    int staged_n = 8;            // prod_k8=8 / staged_k4=4
    bool incr_mode = false;      // 8b prod_k8 false; 35b staged_k4 true
    bool incr_writeback = false; // 35b true
    bool warm_willneed = false;  // 35b true / 8b false
    bool asm_cache = true;
    int prefill_hot = 0;             // 35b staged_k4=32 / 8b prod_k8=0
    bool full_layer_prefill = false; // 35b false / 8b true
    int prefill_full_layers = 0;
    int hot_per_layer = 0;
    int hot_update_interval = 4;
    double hot_decay = 0.75;
  };

  ExpertIo(const ModelFiles *files, int cache_slots, int prefetch_cap, int load_threads,
           int prefetch_threads);
  void add_layer(const LayerCfg &cfg);

  std::shared_ptr<Bundle> build(int li, int64_t expert);
  // LRU → prefetch_buf.pop (promote) → wait inflight (promote) → pool sync + LRU.
  std::map<int64_t, std::shared_ptr<Bundle>> get_bundles(int li,
                                                         const std::vector<int64_t> &experts);
  void prefetch(int li, const std::vector<int64_t> &experts);
  void warm_pages(int li, const std::vector<int64_t> &experts);
  void warm_willneed(int li, const std::vector<int64_t> &experts);
  void shard_advise_whole(const std::string &shard_name);
  void shard_seq_read(const std::string &shard_name);

  struct StagedState {
    std::vector<std::shared_ptr<Bundle>> bundles;
    std::vector<int64_t> exp;  // sorted(set)[:staged_n]
    bool active = false;
  };
  void stage_experts(int li, const std::vector<int64_t> &experts);  // main thread
  void wait_staged(int li);
  const StagedState &staged(int li) const;
  const StagedState &next_staged(int li) const;

  bool incr_mode(int li) const;
  bool incr_ready(int li) const;
  const std::vector<mx::array> &incr_tensors(int li);
  const mx::array &incr_slot_table(int li) const;

  // Non-incr consume path: slot table (set-key cache, cap 8) + 9×[n+1,...] stack (cap 1).
  // Returns a cheap copy (mx::array is a node pointer). asm_cache=false rebuilds each call.
  struct Asm {
    std::vector<mx::array> wargs;  // 9×[n+1,...]
    mx::array slot_of{0.f};        // [E] int32, eval'd GPU-resident table
    std::vector<int64_t> exp;
  };
  Asm asm_for(int li, const StagedState &st);

  void load_hot_layer(int li, int n_hot);
  void materialize_hot(int li);
  void dematerialize_hot(int li);
  void clear_hot_layer(int li);
  void load_full_layer(int li);
  void clear_full_layer(int li);
  bool hot_ready(int li) const;
  bool full_ready(int li) const;
  const std::vector<mx::array> &hot_weights(int li);
  const std::vector<mx::array> &full_weights(int li);
  const std::vector<int64_t> &hot_key(int li) const;
  void note_call(int li);
  void refresh_hot_pins(int li);
  void note_prefill_use(int li, const std::vector<int64_t> &flat, int k, bool staged_mode);
  void prefetch_from_prefill(int li);
  const std::vector<int64_t> &last_used(int li) const;

  void reset_layer(int li);  // only _staged_state/_next; the rest survives

  struct Metrics {
    uint64_t loads = 0, hits = 0, misses = 0, evictions = 0;
    uint64_t prefetch_submitted = 0, prefetch_hits = 0, prefetch_wasted = 0;
    uint64_t staged_used = 0, staged_fallback = 0, stage_builds = 0;
    uint64_t heap_bytes = 0, heap_peak = 0;
    size_t cache_size = 0, inflight = 0;
  };
  Metrics snapshot() const;
  size_t inflight_size() const {
    std::lock_guard<std::mutex> lk(infl_mu_);
    return inflight_.size();
  }

  static uint64_t key_of(int li, int64_t e) {
    return (uint64_t(li) << 32) | uint64_t(e);
  }

 private:
  struct Layer {
    LayerCfg cfg;
    struct Region {
      const TensorInfo *ti = nullptr;
      int64_t rows = 0, cols = 0;
      uint64_t per = 0;
      uint32_t esize = 4;          // element bytes (u32=4 / bf16=2)
    };
    Region reg[9];
    std::unique_ptr<std::mutex> mu = std::make_unique<std::mutex>();
    StagedState state, next;
    std::deque<std::string> slot_keys;
    std::map<std::string, mx::array> slot_cache;
    std::map<std::string, Asm> asm_cache;
    std::vector<mx::array> zero_rows;
    std::vector<mx::array> incr_tensors;
    mx::array incr_table{0.f};
    std::vector<int64_t> occ;
    std::map<int64_t, std::shared_ptr<Bundle>> incr_bundles;
    // Hot expert keys only; shard mmap holds the bytes. shared_ptr snapshot so
    // pool threads can read keys without UAF.
    struct HotStack {
      std::vector<int64_t> key;
      std::vector<int64_t> sorted;
    };
    std::shared_ptr<const HotStack> hot_backing;
    // Separate from L.mu: build is also called from stage_experts while L.mu is held
    // (non-recursive lock). Writers (load/clear_hot_layer) vs pool-thread build readers.
    std::unique_ptr<std::mutex> backing_mu = std::make_unique<std::mutex>();
    std::vector<int64_t> hot_key;
    std::vector<mx::array> hot_weights;
    uint64_t hot_heap = 0;
    std::vector<mx::array> full_w;
    uint64_t full_heap = 0;
    std::vector<int64_t> hc_order;
    std::unordered_map<int64_t, double> hc_val;
    std::vector<int64_t> last_prefill_topk;
    std::vector<int64_t> last_used;
    int64_t calls = 0;
    // New keys append; existing keys update in place (dict insertion order).
    void bump_count(int64_t e, double decay) {
      auto it = hc_val.find(e);
      if (it == hc_val.end()) { hc_order.push_back(e); hc_val.emplace(e, 1.0); }
      else it->second = it->second * decay + 1.0;
    }
  };

  void stage_incr_locked(Layer &L, const std::vector<int64_t> &uniq);
  int layer_idx(int li) const;

  struct HeapAcc {
    std::atomic<uint64_t> live{0}, peak{0};
    void add(uint64_t b) {
      uint64_t v = live.fetch_add(b) + b;
      uint64_t p = peak.load();
      while (v > p && !peak.compare_exchange_weak(p, v)) {
      }
    }
    void sub(uint64_t b) { live.fetch_sub(b); }
  };

  const ModelFiles *files_;
  struct ShardMaps {
    ~ShardMaps();
    const uint8_t *raw(const std::string &shard_path, uint64_t off, uint64_t len);
    void advise_range(const std::string &shard_path, uint64_t off, uint64_t len);
    void advise_whole(const std::string &shard_path);
    void seq_read(const std::string &shard_path);
    std::mutex mu_;
    std::map<std::string, std::pair<uint8_t *, uint64_t>> maps_;
    std::map<std::string, int> fds_;
  } shards_;

  std::vector<Layer> layers_;
  std::map<int, int> li2idx_;
  SharedExpertCache cache_;
  PrefetchBuffer pbuf_;
  IoPool load_pool_, prefetch_pool_;
  mutable std::mutex infl_mu_;
  std::map<uint64_t, std::shared_future<std::shared_ptr<Bundle>>> inflight_;
  std::shared_ptr<HeapAcc> heap_{std::make_shared<HeapAcc>()};
  std::atomic<uint64_t> loads_{0}, hits_{0}, misses_{0}, psub_{0}, phi_{0}, pwaste_{0},
      sused_{0}, sfallback_{0}, sbuilds_{0};
};

}  // namespace e0n
