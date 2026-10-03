// Cross-layer LRU hot stack: explicit heap buffers + F_NOCACHE pread (eviction is testable).
#pragma once
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "expertmap.hpp"
#include "safio.hpp"

namespace e0n {

struct Entry {
  std::vector<uint8_t> blob;
  std::vector<Region> regions;
  std::vector<uint64_t> blob_offsets;
  uint64_t bytes = 0;
  int layer = 0;
  int64_t expert = -1;
};

struct CacheMetrics {
  uint64_t loaded = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t heap_bytes = 0;
  uint64_t heap_peak = 0;
  uint64_t evictions = 0;
};

class HotStack {
 public:
  using EntryPtr = std::shared_ptr<const Entry>;
  using Lease = EntryPtr;

  HotStack(std::shared_ptr<const ModelFiles> files, std::shared_ptr<const ExpertMap> map,
           uint64_t budget_bytes);
  ~HotStack();

  // Miss → F_NOCACHE cold read on this thread. Failure returns nullptr + err.
  // A live lease keeps the entry allocated (eviction only drops the stack ref).
  Lease acquire(int layer, int64_t expert, std::string &err);

  CacheMetrics snapshot() const;
  uint64_t budget() const { return budget_; }
  size_t size() const;

 private:
  struct Item {
    EntryPtr entry;
    std::list<uint64_t>::iterator lru;
  };
  static uint64_t key_of(int layer, int64_t expert) {
    return (static_cast<uint64_t>(layer) << 32) | static_cast<uint64_t>(expert);
  }
  bool load_entry(int layer, int64_t expert, EntryPtr &out, std::string &err);

  std::shared_ptr<const ModelFiles> files_;
  std::shared_ptr<const ExpertMap> map_;
  uint64_t budget_;
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, Item> items_;
  std::list<uint64_t> lru_;  // front = MRU
  CacheMetrics metrics_;
  std::map<std::string, int> fds_;  // shard → F_NOCACHE fd
};

}  // namespace e0n
