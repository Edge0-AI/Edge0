#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace e0n {

struct PrefixCacheConfig {
  bool enabled = true;
  uint64_t budget_bytes = 512ull * 1024ull * 1024ull;
  uint32_t min_prefix_tokens = 128;
};

struct PrefixCacheLookup {
  std::shared_ptr<void> snapshot;
  size_t matched_tokens = 0;
  std::string reason;
};

struct PrefixCacheMetrics {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t cached_tokens = 0;
  uint64_t entries = 0;
  uint64_t bytes = 0;
  uint64_t evictions = 0;
  uint64_t budget_bytes = 0;
  uint32_t min_prefix_tokens = 0;
  bool enabled = false;
  size_t last_cached_tokens = 0;
  size_t last_prefill_tokens = 0;
  std::string last_reason = "cold";
};

// Prefix index and LRU tracking are independent from backend-specific KV state.
// Hashes only narrow candidates; the complete token vector is always rechecked.
class PrefixCache {
 public:
  explicit PrefixCache(PrefixCacheConfig cfg = {}) : cfg_(cfg) {}

  PrefixCacheLookup lookup(const std::string &identity,
                          const std::vector<int32_t> &tokens) {
    std::lock_guard<std::mutex> g(mu_);
    PrefixCacheLookup out;
    out.reason = "prefix_miss";
    if (!cfg_.enabled || cfg_.budget_bytes == 0) {
      out.reason = "disabled";
      metrics_.last_cached_tokens = 0;
      metrics_.last_reason = out.reason;
      return out;
    }
    if (tokens.size() < cfg_.min_prefix_tokens) {
      out.reason = "too_short";
      metrics_.last_cached_tokens = 0;
      metrics_.last_reason = out.reason;
      return out;
    }

    uint64_t h1 = kFnvOffset;
    uint64_t h2 = kMixOffset;
    uint64_t best_id = 0;
    size_t best_len = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
      mix_token(tokens[i], h1, h2);
      const size_t len = i + 1;
      if (len < cfg_.min_prefix_tokens) continue;
      Key key{identity, len, h1, h2};
      auto it = index_.find(key);
      if (it == index_.end()) continue;
      for (uint64_t id : it->second) {
        auto ent = entries_.find(id);
        if (ent == entries_.end()) continue;
        if (ent->second.identity != identity || ent->second.tokens.size() != len ||
            !std::equal(ent->second.tokens.begin(), ent->second.tokens.end(), tokens.begin()))
          continue;
        if (len >= best_len) {
          best_len = len;
          best_id = id;
        }
      }
    }
    if (best_id == 0) {
      ++metrics_.misses;
      metrics_.last_cached_tokens = 0;
      metrics_.last_reason = out.reason;
      return out;
    }
    auto it = entries_.find(best_id);
    touch_locked(it->second);
    ++metrics_.hits;
    metrics_.cached_tokens += best_len;
    metrics_.last_cached_tokens = best_len;
    metrics_.last_reason = "hit";
    out.snapshot = it->second.snapshot;
    out.matched_tokens = best_len;
    out.reason = "hit";
    return out;
  }

  bool put(const std::string &identity, const std::vector<int32_t> &tokens,
           std::shared_ptr<void> snapshot, uint64_t bytes, std::string *reason = nullptr) {
    std::lock_guard<std::mutex> g(mu_);
    if (!cfg_.enabled || cfg_.budget_bytes == 0) {
      if (reason) *reason = "disabled";
      return false;
    }
    if (tokens.size() < cfg_.min_prefix_tokens) {
      if (reason) *reason = "too_short";
      return false;
    }
    if (bytes > cfg_.budget_bytes) {
      metrics_.last_reason = "oversize";
      if (reason) *reason = "oversize";
      return false;
    }
    uint64_t h1 = kFnvOffset;
    uint64_t h2 = kMixOffset;
    for (int32_t tok : tokens) mix_token(tok, h1, h2);
    Key key{identity, tokens.size(), h1, h2};
    auto &bucket = index_[key];
    for (uint64_t id : bucket) {
      auto it = entries_.find(id);
      if (it == entries_.end()) continue;
      if (it->second.identity == identity && it->second.tokens == tokens) {
        bytes_ -= it->second.bytes;
        it->second.snapshot = std::move(snapshot);
        it->second.bytes = bytes;
        bytes_ += bytes;
        touch_locked(it->second);
        if (reason) *reason = "updated";
        return true;
      }
    }
    while (bytes_ + bytes > cfg_.budget_bytes && !lru_.empty()) {
      const uint64_t old = lru_.back();
      lru_.pop_back();
      auto it = entries_.find(old);
      if (it == entries_.end()) continue;
      remove_from_index_locked(it->second);
      bytes_ -= it->second.bytes;
      entries_.erase(it);
      ++metrics_.evictions;
    }
    Entry ent;
    ent.id = next_id_++;
    ent.identity = identity;
    ent.tokens = tokens;
    ent.h1 = h1;
    ent.h2 = h2;
    ent.snapshot = std::move(snapshot);
    ent.bytes = bytes;
    lru_.push_front(ent.id);
    ent.lru = lru_.begin();
    bytes_ += bytes;
    bucket.push_back(ent.id);
    entries_.emplace(ent.id, std::move(ent));
    if (reason) *reason = "stored";
    return true;
  }

  void set_last_prefill_tokens(size_t n) {
    std::lock_guard<std::mutex> g(mu_);
    metrics_.last_prefill_tokens = n;
  }

  PrefixCacheMetrics metrics() const {
    std::lock_guard<std::mutex> g(mu_);
    PrefixCacheMetrics m = metrics_;
    m.entries = entries_.size();
    m.bytes = bytes_;
    m.budget_bytes = cfg_.budget_bytes;
    m.min_prefix_tokens = cfg_.min_prefix_tokens;
    m.enabled = cfg_.enabled && cfg_.budget_bytes != 0;
    return m;
  }

  void clear() {
    std::lock_guard<std::mutex> g(mu_);
    index_.clear();
    entries_.clear();
    lru_.clear();
    bytes_ = 0;
    metrics_.last_cached_tokens = 0;
    metrics_.last_reason = "cleared";
  }

 private:
  static constexpr uint64_t kFnvOffset = 1469598103934665603ull;
  static constexpr uint64_t kFnvPrime = 1099511628211ull;
  static constexpr uint64_t kMixOffset = 0x9e3779b97f4a7c15ull;

  struct Key {
    std::string identity;
    size_t length = 0;
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    bool operator==(const Key &o) const {
      return length == o.length && h1 == o.h1 && h2 == o.h2 && identity == o.identity;
    }
  };
  struct KeyHash {
    size_t operator()(const Key &k) const {
      size_t h = std::hash<std::string>{}(k.identity);
      h ^= std::hash<size_t>{}(k.length) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<uint64_t>{}(k.h1) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<uint64_t>{}(k.h2) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };
  struct Entry {
    uint64_t id = 0;
    std::string identity;
    std::vector<int32_t> tokens;
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    std::shared_ptr<void> snapshot;
    uint64_t bytes = 0;
    std::list<uint64_t>::iterator lru;
  };

  static void mix_token(int32_t token, uint64_t &h1, uint64_t &h2) {
    uint32_t v = static_cast<uint32_t>(token);
    for (int i = 0; i < 4; ++i) {
      const uint8_t b = static_cast<uint8_t>(v >> (i * 8));
      h1 ^= b;
      h1 *= kFnvPrime;
      h2 ^= static_cast<uint64_t>(b) + 0x100000001b3ull;
      h2 *= 0x9e3779b185ebca87ull;
      h2 ^= h2 >> 29;
    }
  }

  void touch_locked(Entry &ent) {
    lru_.erase(ent.lru);
    lru_.push_front(ent.id);
    ent.lru = lru_.begin();
  }

  void remove_from_index_locked(const Entry &ent) {
    Key key{ent.identity, ent.tokens.size(), ent.h1, ent.h2};
    auto it = index_.find(key);
    if (it == index_.end()) return;
    auto &v = it->second;
    v.erase(std::remove(v.begin(), v.end(), ent.id), v.end());
    if (v.empty()) index_.erase(it);
  }

  mutable std::mutex mu_;
  PrefixCacheConfig cfg_;
  uint64_t next_id_ = 1;
  uint64_t bytes_ = 0;
  PrefixCacheMetrics metrics_;
  std::list<uint64_t> lru_;
  std::unordered_map<uint64_t, Entry> entries_;
  std::unordered_map<Key, std::vector<uint64_t>, KeyHash> index_;
};

}  // namespace e0n
