#include "expertio.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#include "fwd_common.hpp"  // mkshape

namespace e0n {
namespace mx = mlx::core;
// ---------------- ShardMaps ----------------

ExpertIo::ShardMaps::~ShardMaps() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto &[p, m] : maps_) munmap(m.first, size_t(m.second));
  for (auto &[p, fd] : fds_) ::close(fd);
}

const uint8_t *ExpertIo::ShardMaps::raw(const std::string &shard_path, uint64_t off,
                                        uint64_t len) {
  uint8_t *base = nullptr;
  uint64_t total = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = maps_.find(shard_path);
    if (it == maps_.end()) {
      int fd = ::open(shard_path.c_str(), O_RDONLY);
      if (fd < 0) return nullptr;
      struct stat sb;
      if (fstat(fd, &sb) != 0) { ::close(fd); return nullptr; }
      void *m = mmap(nullptr, size_t(sb.st_size), PROT_READ, MAP_SHARED, fd, 0);
      if (m == MAP_FAILED) { ::close(fd); return nullptr; }
      int keep = fd;
      it = maps_.emplace(shard_path,
                         std::pair<uint8_t *, uint64_t>{static_cast<uint8_t *>(m),
                                                        uint64_t(sb.st_size)}).first;
      fds_.emplace(shard_path, keep);
    }
    base = it->second.first;
    total = it->second.second;
  }
  if (off + len > total) return nullptr;
  return base + off;
}

void ExpertIo::ShardMaps::advise_range(const std::string &shard_path, uint64_t off,
                                       uint64_t len) {
  if (len == 0) return;
  if (!raw(shard_path, 0, 1)) return;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = maps_.find(shard_path);
  if (it == maps_.end()) return;
  uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
  uint64_t start = off / page * page;
  uint64_t end = (off + len + page - 1) / page * page;
  if (end > it->second.second) end = it->second.second;
  if (end > start) madvise(it->second.first + start, size_t(end - start), MADV_WILLNEED);
}

void ExpertIo::ShardMaps::advise_whole(const std::string &shard_path) {  // mmap.py:43-49
  if (!raw(shard_path, 0, 1)) return;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = maps_.find(shard_path);
  if (it != maps_.end()) madvise(it->second.first, size_t(it->second.second), MADV_WILLNEED);
}

void ExpertIo::ShardMaps::seq_read(const std::string &shard_path) {
  if (!raw(shard_path, 0, 1)) return;
  uint64_t total;
  {
    std::lock_guard<std::mutex> lk(mu_);
    total = maps_[shard_path].second;
  }
  volatile uint64_t sink = 0;
  constexpr uint64_t kChunk = 1ull << 24;
  for (uint64_t off = 0; off < total; off += kChunk) {
    uint64_t n = std::min(kChunk, total - off);
    const uint8_t *p = raw(shard_path, off, n);
    for (uint64_t o = 0; o + 8 <= n; o += 8) {
      uint64_t v;
      std::memcpy(&v, p + o, 8);
      sink += v;
    }
  }
  (void)sink;
}

std::shared_ptr<Bundle> SharedExpertCache::get(uint64_t key) {  // cache.py:24-29
  std::lock_guard<std::mutex> lk(mu_);
  auto it = map_.find(key);
  if (it == map_.end()) return nullptr;
  for (auto o = order_.begin(); o != order_.end(); ++o)
    if (*o == key) { order_.erase(o); order_.push_back(key); break; }
  return it->second;
}
std::shared_ptr<Bundle> SharedExpertCache::peek(uint64_t key) {  // 31-33
  std::lock_guard<std::mutex> lk(mu_);
  auto it = map_.find(key);
  return it == map_.end() ? nullptr : it->second;
}
void SharedExpertCache::put(uint64_t key, std::shared_ptr<Bundle> b) {  // 34-39
  std::lock_guard<std::mutex> lk(mu_);
  if (map_.count(key)) {
    map_[key] = std::move(b);
    for (auto o = order_.begin(); o != order_.end(); ++o)
      if (*o == key) { order_.erase(o); order_.push_back(key); break; }
    return;
  }
  map_[key] = std::move(b);
  order_.push_back(key);
  if (slots_ > 0 && int(map_.size()) > slots_) {  // popitem(last=False)
    uint64_t old = order_.front();
    order_.pop_front();
    map_.erase(old);
    evictions_++;
  }
}
std::shared_ptr<Bundle> SharedExpertCache::pop(uint64_t key) {  // 41-46
  std::lock_guard<std::mutex> lk(mu_);
  auto it = map_.find(key);
  if (it == map_.end()) return nullptr;
  auto b = it->second;
  map_.erase(it);
  for (auto o = order_.begin(); o != order_.end(); ++o)
    if (*o == key) { order_.erase(o); break; }
  return b;
}
size_t SharedExpertCache::size() {
  std::lock_guard<std::mutex> lk(mu_);
  return map_.size();
}
bool SharedExpertCache::contains(uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  return map_.count(key) > 0;
}

bool PrefetchBuffer::put(uint64_t key, std::shared_ptr<Bundle> b) {  // cache.py:66-72
  std::lock_guard<std::mutex> lk(mu_);
  if (map_.count(key)) { map_[key] = std::move(b); return true; }
  map_[key] = std::move(b);
  order_.push_back(key);
  if (cap_ > 0 && int(map_.size()) > cap_) {
    uint64_t old = order_.front();
    order_.pop_front();
    map_.erase(old);
    return false;
  }
  return true;
}
std::shared_ptr<Bundle> PrefetchBuffer::pop(uint64_t key) {  // 74-75
  std::lock_guard<std::mutex> lk(mu_);
  auto it = map_.find(key);
  if (it == map_.end()) return nullptr;
  auto b = it->second;
  map_.erase(it);
  for (auto o = order_.begin(); o != order_.end(); ++o)
    if (*o == key) { order_.erase(o); break; }
  return b;
}
bool PrefetchBuffer::contains(uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  return map_.count(key) > 0;
}

// ---------------- IoPool ----------------

IoPool::IoPool(int n) {
  for (int i = 0; i < n; i++)
    th_.emplace_back([this] {
      for (;;) {
        std::function<void()> job;
        {
          std::unique_lock<std::mutex> lk(mu_);
          cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
          if (stop_ && q_.empty()) return;
          job = std::move(q_.front());
          q_.pop_front();
        }
        job();
      }
    });
}
IoPool::~IoPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &t : th_)
    if (t.joinable()) t.join();
}

// ---------------- ExpertIo ----------------

ExpertIo::ExpertIo(const ModelFiles *files, int cache_slots, int prefetch_cap, int load_threads,
                   int prefetch_threads)
    : files_(files), cache_(cache_slots), pbuf_(prefetch_cap), load_pool_(load_threads),
      prefetch_pool_(prefetch_threads) {}

namespace {
void materialize(const std::shared_ptr<Bundle> &b) {
  std::vector<mx::array> v;
  for (int i = 0; i < 9; i++) v.push_back(b->at(i));
  mx::eval(v);
}

const char *kProjFile[3] = {"up_proj", "gate_proj", "down_proj"};
const char *kPartFile[3] = {"weight", "scales", "biases"};
}  // namespace

void ExpertIo::add_layer(const LayerCfg &cfg) {
  li2idx_[cfg.layer] = int(layers_.size());
  layers_.emplace_back();
  Layer &L = layers_.back();
  L.cfg = cfg;
  for (int p = 0; p < 3; p++)
    for (int q = 0; q < 3; q++) {
      const int i = p * 3 + q;
      const std::string name = cfg.prefix + "." + kProjFile[p] + "." + kPartFile[q];
      const TensorInfo *ti = files_->find(name);
      if (!ti) {
        std::fprintf(stderr, "expertio: missing tensor %s\n", name.c_str());
        continue;
      }
      const int64_t O = (p == 2) ? cfg.H : cfg.F;
      const int64_t K = (q == 0) ? ((p == 2) ? cfg.F / 8 : cfg.H / 8)
                                 : ((p == 2) ? cfg.F / cfg.group : cfg.H / cfg.group);
      L.reg[i] = {ti, O, K, (ti->end - ti->begin) / uint64_t(cfg.E), q == 0 ? 4u : 2u};
      if (ti->dtype == "BF16") L.reg[i].esize = 2;
      else if (ti->dtype == "U32") L.reg[i].esize = 4;
      else std::fprintf(stderr, "expertio: unexpected dtype %s @%s\n", ti->dtype.c_str(), name.c_str());
    }
}

int ExpertIo::layer_idx(int li) const {
  auto it = li2idx_.find(li);
  return it == li2idx_.end() ? -1 : it->second;
}

std::shared_ptr<Bundle> ExpertIo::build(int li, int64_t expert) {
  int idx = layer_idx(li);
  if (idx < 0) return nullptr;
  Layer &L = layers_[size_t(idx)];
  auto b = std::make_shared<Bundle>();
  uint64_t tot = 0;
  for (int i = 0; i < 9; i++) {
      const auto &r = L.reg[i];
      if (!r.ti) return nullptr;
      const uint8_t *src = shards_.raw(files_->shard_path(r.ti->shard),
                                       r.ti->begin + uint64_t(expert) * r.per, r.per);
      if (!src) {
        std::fprintf(stderr, "expertio: raw out of range %s e%lld off=%llu per=%llu\n",
                     files_->shard_path(r.ti->shard).c_str(), (long long)expert,
                     (unsigned long long)(r.ti->begin + uint64_t(expert) * r.per),
                     (unsigned long long)r.per);
        return nullptr;
      }
      if (!r.ti->shape.empty() && r.per != uint64_t(r.rows) * uint64_t(r.cols) * r.esize) {
        std::fprintf(stderr, "expertio: region size mismatch %s per=%llu vs %lldx%lldx%u\n",
                     r.ti->dtype.c_str(), (unsigned long long)r.per, (long long)r.rows,
                     (long long)r.cols, r.esize);
        return nullptr;
      }
      if (r.esize == 4) {
        b->at_mut(i) = mx::array(reinterpret_cast<const uint32_t *>(src),
                                 mkshape({r.rows, r.cols}), mx::uint32);
      } else {
        auto a16 = mx::array(reinterpret_cast<const uint16_t *>(src), mkshape({r.rows, r.cols}),
                             mx::uint16);
        b->at_mut(i) = mx::view(a16, mx::bfloat16);
      }
      tot += r.per;
  }
  b->bytes = tot;
  heap_->add(tot);
  auto heap = heap_;
  auto keep = b;
  return std::shared_ptr<Bundle>(b.get(), [heap, tot, keep](Bundle *p) mutable {
    heap->sub(tot);
    keep.reset();
  });
}

std::map<int64_t, std::shared_ptr<Bundle>> ExpertIo::get_bundles(
    int li, const std::vector<int64_t> &experts) {
  std::map<int64_t, std::shared_ptr<Bundle>> out;
  std::vector<int64_t> missing;
  int lidx = layer_idx(li);
  Layer *L = (lidx >= 0) ? &layers_[size_t(lidx)] : nullptr;
  const double decay = L ? L->cfg.hot_decay : 0.75;

  for (auto e : experts) {  // 430-435
    auto c = cache_.get(key_of(li, e));
    if (c) { out[e] = std::move(c); hits_++; }
    else missing.push_back(e);
  }

  std::vector<int64_t> still;  // 437-449
  for (auto e : missing) {
    auto pb = pbuf_.pop(key_of(li, e));
    if (pb) {
      phi_++; hits_++;
      if (L) L->bump_count(e, decay);
      cache_.put(key_of(li, e), pb);
      out[e] = std::move(pb);
    } else still.push_back(e);
  }

  std::vector<int64_t> cold;  // 451-467 inflight
  for (auto e : still) {
    std::shared_future<std::shared_ptr<Bundle>> fut;
    {
      std::lock_guard<std::mutex> lk(infl_mu_);
      auto it = inflight_.find(key_of(li, e));
      if (it != inflight_.end()) fut = it->second;
    }
    if (fut.valid()) {
        auto b = fut.get();
        {
        std::lock_guard<std::mutex> lk(infl_mu_);
        inflight_.erase(key_of(li, e));
      }
      if (!b) {
        std::fprintf(stderr, "expertio: inflight build failed L%d E%lld\n", li, (long long)e);
        continue;
      }
      phi_++; hits_++;
      if (L) L->bump_count(e, decay);
      cache_.put(key_of(li, e), b);
      out[e] = std::move(b);
    } else cold.push_back(e);
  }
  if (!cold.empty()) {
    std::vector<std::shared_future<std::shared_ptr<Bundle>>> fs;
    for (auto e : cold) fs.push_back(load_pool_.submit([this, li, e] { return build(li, e); }));
    for (size_t i = 0; i < cold.size(); i++) {
      auto b = fs[i].get();
      if (!b) {
        std::fprintf(stderr, "expertio: build failed L%d E%lld (missing region key or map OOB)\n", li,
                     (long long)cold[i]);
        continue;
      }
      loads_++; misses_++;
      if (L) L->bump_count(cold[i], decay);
      cache_.put(key_of(li, cold[i]), b);
      out[cold[i]] = std::move(b);
    }
  }
  for (auto &[e, b] : out) materialize(b);
  return out;
}

void ExpertIo::prefetch(int li, const std::vector<int64_t> &experts) {
  std::vector<int64_t> uniq(experts.begin(), experts.end());
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  if (uniq.empty()) return;
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  std::vector<int64_t> missing;
  for (auto e : uniq) {
    {
      std::lock_guard<std::mutex> lk(infl_mu_);
      if (inflight_.count(key_of(li, e))) continue;
    }
    if (cache_.peek(key_of(li, e)) || pbuf_.contains(key_of(li, e))) continue;
    missing.push_back(e);
  }
  if (missing.empty()) return;
  if (L.cfg.warm_willneed) warm_willneed(li, missing);  // 502-503
  std::vector<std::pair<int64_t, std::shared_future<std::shared_ptr<Bundle>>>> sub;
  for (auto e : missing) {
    auto fut = prefetch_pool_.submit([this, li, e] { return build(li, e); });
    {
      std::lock_guard<std::mutex> lk(infl_mu_);
      inflight_[key_of(li, e)] = fut;
    }
    sub.emplace_back(e, fut);
  }
  psub_ += missing.size();
  for (auto &[e, fut] : sub) {
    uint64_t key = key_of(li, e);
    prefetch_pool_.submit([this, e, key, fut] {
      auto b = fut.get();
      {
        std::lock_guard<std::mutex> lk(infl_mu_);
        auto it = inflight_.find(key);
        if (it != inflight_.end()) inflight_.erase(it);  // 530
      }
      if (cache_.peek(key)) return;                       // 531
      if (!pbuf_.put(key, b)) pwaste_++;                  // 533-535
    });
  }
}

void ExpertIo::warm_willneed(int li, const std::vector<int64_t> &experts) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  for (auto e : experts)
    for (int i = 0; i < 9; i++) {
      if (!L.reg[i].ti) continue;
      shards_.advise_range(files_->shard_path(L.reg[i].ti->shard),
                           L.reg[i].ti->begin + uint64_t(e) * L.reg[i].per, L.reg[i].per);
    }
}

void ExpertIo::warm_pages(int li, const std::vector<int64_t> &experts) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  volatile uint8_t sink = 0;
  uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
  for (auto e : experts)
    for (int i = 0; i < 9; i++) {
      if (!L.reg[i].ti) continue;
      const uint8_t *src = shards_.raw(files_->shard_path(L.reg[i].ti->shard),
                                       L.reg[i].ti->begin + uint64_t(e) * L.reg[i].per,
                                       L.reg[i].per);
      if (!src) continue;
      for (uint64_t o = 0; o < L.reg[i].per; o += page) sink ^= src[o];
    }
  (void)sink;
}

void ExpertIo::shard_advise_whole(const std::string &shard_name) {
  shards_.advise_whole(files_->shard_path(shard_name));
}
void ExpertIo::shard_seq_read(const std::string &shard_name) {
  shards_.seq_read(files_->shard_path(shard_name));
}

void ExpertIo::stage_experts(int li, const std::vector<int64_t> &experts) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  std::vector<int64_t> uniq(experts.begin(), experts.end());
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  if (uniq.empty()) return;  // 721-722
  if (L.cfg.warm_willneed) warm_willneed(li, uniq);  // 723-724
  std::lock_guard<std::mutex> lk(*L.mu);
  if (L.cfg.incr_mode) {
    stage_incr_locked(L, uniq);
    return;
  }
  // 733-738:next = _build_staged_bundles(unique)(539-585).
  const int n = L.cfg.staged_n;
  StagedState st;
  st.exp.assign(uniq.begin(), uniq.begin() + std::min<int>(size_t(n), uniq.size()));
  st.active = true;
  for (auto e : st.exp) {
    std::shared_ptr<Bundle> b;
    for (size_t i = 0; i < L.state.exp.size(); i++)
      if (L.state.exp[i] == e) { b = L.state.bundles[i]; break; }
    if (!b) b = cache_.get(key_of(li, e));
    if (!b) {
      b = build(li, e);
      if (!b) return;
      cache_.put(key_of(li, e), b);
      sbuilds_++;
    }
    st.bundles.push_back(std::move(b));
  }
  L.next = std::move(st);
}

void ExpertIo::wait_staged(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  std::lock_guard<std::mutex> lk(*L.mu);
  if (L.next.active) {
    L.state = L.next;
    L.next = StagedState{};
  }
}
const ExpertIo::StagedState &ExpertIo::staged(int li) const {
  return layers_[size_t(layer_idx(li))].state;
}
const ExpertIo::StagedState &ExpertIo::next_staged(int li) const {
  return layers_[size_t(layer_idx(li))].next;
}

void ExpertIo::stage_incr_locked(Layer &L, const std::vector<int64_t> &uniq) {
  const int li = L.cfg.layer;
  const int n = L.cfg.staged_n;
  std::vector<int64_t> exp(uniq.begin(), uniq.begin() + std::min<int>(size_t(n), uniq.size()));
  std::set<int64_t> exp_set(exp.begin(), exp.end());
  if (L.incr_tensors.empty()) {
    for (int i = 0; i < 9; i++) {
      auto t = mx::zeros(mkshape({int64_t(n) + 1, L.reg[i].rows, L.reg[i].cols}),
                         L.reg[i].esize == 4 ? mx::uint32 : mx::bfloat16);
      mx::eval(t);
      L.incr_tensors.push_back(std::move(t));
    }
    L.incr_table = mx::full(mkshape({L.cfg.E}), int32_t(n), mx::int32);  // core.full(614-616)
    mx::eval(L.incr_table);
    L.occ.assign(n, -1);
  }
  auto &occ = L.occ;
  for (int i = 0; i < n; i++)
    if (occ[i] >= 0 && !exp_set.count(occ[i])) occ[i] = -1;
  std::vector<int> free;
  for (int i = 0; i < n; i++) if (occ[i] < 0) free.push_back(i);
  std::vector<int64_t> changed;  // 633-637
  for (auto e : exp) if (!L.incr_bundles.count(e)) changed.push_back(e);
  std::map<int64_t, std::shared_ptr<Bundle>> bundles;
  for (auto e : exp) if (L.incr_bundles.count(e)) bundles[e] = L.incr_bundles[e];
  struct Write { int slot; std::shared_ptr<Bundle> b; };
  std::vector<Write> writes;
  for (size_t j = 0; j < changed.size(); j++) {  // 646-668
    int64_t e = changed[j];
    if (int(j) >= int(free.size())) continue;
    std::shared_ptr<Bundle> b = cache_.get(key_of(li, e));
    if (!b) {
      b = build(li, e);
      if (!L.cfg.incr_writeback && b) cache_.put(key_of(li, e), b);
      sbuilds_++;
    }
    int slot = free[j];
    materialize(b);
    bundles[e] = b;
    occ[slot] = e;
    writes.push_back({slot, std::move(b)});
  }
  std::vector<int64_t> evicted;  // 669-670
  for (auto &[e, b] : L.incr_bundles)
    if (!exp_set.count(e)) evicted.push_back(e);
  if (L.cfg.incr_writeback)
    for (auto e : evicted) cache_.put(key_of(li, e), L.incr_bundles[e]);
  L.incr_bundles = std::move(bundles);
  for (auto &w : writes)
    for (int i = 0; i < 9; i++) {
      uint64_t row = uint64_t(L.reg[i].rows) * uint64_t(L.reg[i].cols);
      if (L.reg[i].esize == 4)
        std::memcpy(L.incr_tensors[i].data<uint32_t>() + uint64_t(w.slot) * row,
                    w.b->at(i).data<uint32_t>(), row * 4);
      else
        std::memcpy(L.incr_tensors[i].data<uint16_t>() + uint64_t(w.slot) * row,
                    w.b->at(i).data<uint16_t>(), row * 2);
    }
  int32_t *tbl = L.incr_table.data<int32_t>();
  for (auto e : evicted) tbl[e] = int32_t(n);
  for (int i = 0; i < n; i++)
    if (occ[i] >= 0) tbl[occ[i]] = int32_t(i);
  if (!writes.empty() || !evicted.empty() || !changed.empty()) mx::eval(L.incr_table);
  StagedState st;
  st.active = true;
  st.exp = exp;
  for (auto e : exp) st.bundles.push_back(L.incr_bundles[e]);
  L.next = std::move(st);
}

bool ExpertIo::incr_mode(int li) const {
  int idx = layer_idx(li);
  return idx >= 0 && layers_[size_t(idx)].cfg.incr_mode;
}
bool ExpertIo::incr_ready(int li) const {
  int idx = layer_idx(li);
  return idx >= 0 && layers_[size_t(idx)].cfg.incr_mode &&
         !layers_[size_t(idx)].incr_tensors.empty();
}
const std::vector<mx::array> &ExpertIo::incr_tensors(int li) {
  return layers_[size_t(layer_idx(li))].incr_tensors;
}
const mx::array &ExpertIo::incr_slot_table(int li) const {
  return layers_[size_t(layer_idx(li))].incr_table;
}

static std::string exp_key(const std::vector<int64_t> &exp) {
  std::string s;
  for (auto e : exp) { s += std::to_string(e); s += ','; }
  return s;
}

ExpertIo::Asm ExpertIo::asm_for(int li, const StagedState &st) {
  Layer &L = layers_[size_t(layer_idx(li))];
  const std::string key = exp_key(st.exp);
  if (L.cfg.asm_cache) {
    auto it = L.asm_cache.find(key);
    if (it != L.asm_cache.end() && it->second.exp == st.exp) return it->second;
  }
  const int n = L.cfg.staged_n;
  if (L.zero_rows.empty()) {
    for (int i = 0; i < 9; i++)
      L.zero_rows.push_back(mx::zeros(mkshape({L.reg[i].rows, L.reg[i].cols}),
                                      L.reg[i].esize == 4 ? mx::uint32 : mx::bfloat16));
  }
  Asm a;
  a.exp = st.exp;
  auto sit = L.slot_cache.find(key);  // _slot_table_cache(1269-1276)
  if (sit == L.slot_cache.end()) {
    std::vector<int32_t> tab(size_t(L.cfg.E), int32_t(n));  // 557-559
    for (size_t i = 0; i < st.exp.size(); i++) tab[size_t(st.exp[i])] = int32_t(i);
    mx::array tof(tab.begin(), mkshape({L.cfg.E}), mx::int32);  // core.array(list)
    mx::eval(tof);  // 1271-1272 force eval
    L.slot_cache.insert_or_assign(key, tof);
    L.slot_keys.push_back(key);
    if (L.slot_keys.size() > 8) {
      L.slot_cache.erase(L.slot_keys.front());
      L.slot_keys.pop_front();
    }
    a.slot_of = L.slot_cache.at(key);
  } else {
    a.slot_of = sit->second;
  }
  for (int i = 0; i < 9; i++) {
    std::vector<mx::array> rows;
    for (auto &b : st.bundles) rows.push_back(b->at(i));
    while (int(rows.size()) < n + 1) rows.push_back(L.zero_rows[i]);
    a.wargs.push_back(mx::stack(rows));
  }
  sused_ += st.exp.size();
  if (L.cfg.asm_cache) {
    if (!L.asm_cache.empty()) L.asm_cache.clear();  // cap=1
    L.asm_cache.insert_or_assign(key, a);
  }
  return a;
}

void ExpertIo::note_call(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  L.calls++;
  if (L.cfg.hot_update_interval && L.calls % L.cfg.hot_update_interval == 0) {
    (void)L.cfg.hot_per_layer;
  }
}

void ExpertIo::refresh_hot_pins(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  if (layers_[size_t(idx)].cfg.hot_per_layer <= 0) return;
}

void ExpertIo::load_hot_layer(int li, int n_hot) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  std::vector<int64_t> hot;
  if (!L.hc_order.empty()) {
    hot = L.hc_order;
    std::stable_sort(hot.begin(), hot.end(),
                     [&](int64_t a, int64_t b) { return L.hc_val[a] > L.hc_val[b]; });
    if ((int64_t)hot.size() > n_hot) hot.resize(n_hot);  // hot[:n_hot](1007)
  } else {
    for (int64_t e = 0; e < n_hot; e++) hot.push_back(e);  // 1009
  }
  if (L.hot_backing) {
    if (L.hot_key == hot) return;  // 1012-1013
    std::set<int64_t> old_set(L.hot_key.begin(), L.hot_key.end());
    int changed = 0;
    for (auto e : hot)
      if (!old_set.count(e)) changed++;
    if (changed <= std::max(1, n_hot / 8)) return;  // 1016-1017
  }
  auto st = std::make_shared<Layer::HotStack>();
  st->key = hot;
  st->sorted = st->key;
  std::sort(st->sorted.begin(), st->sorted.end());
  {
    std::lock_guard<std::mutex> lk(*L.backing_mu);
    L.hot_backing = st;
  }
  L.hot_key = st->key;
  if (!L.hot_weights.empty()) {
    heap_->sub(L.hot_heap);
    L.hot_heap = 0;
    L.hot_weights.clear();
  }
}

void ExpertIo::materialize_hot(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  if (!L.hot_weights.empty() || !L.hot_backing) return;  // 1053
  std::shared_ptr<const Layer::HotStack> snap;
  {
    std::lock_guard<std::mutex> lk(*L.backing_mu);
    snap = L.hot_backing;
  }
  const int64_t n_e = (int64_t)snap->key.size() + 1;
  L.hot_weights.assign(9, mx::array(0.f));
  for (int i = 0; i < 9; i++) {
    const auto &r = L.reg[i];
    if (!r.ti) return;
    std::vector<uint8_t> rows(static_cast<size_t>(n_e) * r.per, 0);
    const std::string path = files_->shard_path(r.ti->shard);
    for (size_t j = 0; j < snap->key.size(); j++) {
      const uint8_t *src = shards_.raw(path, r.ti->begin + uint64_t(snap->key[j]) * r.per,
                                       r.per);
      if (!src) return;
      std::memcpy(rows.data() + uint64_t(j) * r.per, src, r.per);
    }
    const uint8_t *base = rows.data();
    if (r.esize == 4) {
      L.hot_weights[i] = mx::array(reinterpret_cast<const uint32_t *>(base),
                                   mkshape({n_e, r.rows, r.cols}), mx::uint32);
    } else {
      auto a16 = mx::array(reinterpret_cast<const uint16_t *>(base),
                           mkshape({n_e, r.rows, r.cols}), mx::uint16);
      L.hot_weights[i] = mx::view(a16, mx::bfloat16);
    }
    L.hot_heap += rows.size();
  }
  heap_->add(L.hot_heap);
}

void ExpertIo::dematerialize_hot(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  if (L.hot_weights.empty()) return;
  heap_->sub(L.hot_heap);
  L.hot_heap = 0;
  L.hot_weights.clear();
}

void ExpertIo::clear_hot_layer(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  if (!L.hot_weights.empty()) {
    heap_->sub(L.hot_heap);
    L.hot_heap = 0;
    L.hot_weights.clear();
  }
  std::lock_guard<std::mutex> lk(*L.backing_mu);
  L.hot_backing = nullptr;
  L.hot_key.clear();
}

void ExpertIo::load_full_layer(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  if (!L.full_w.empty()) return;  // 934-935
  const int64_t E = L.cfg.E;
  L.full_w.assign(9, mx::array(0.f));
  uint64_t tot = 0;
  for (int i = 0; i < 9; i++) {
    const auto &r = L.reg[i];
    if (!r.ti) {
      std::fprintf(stderr, "expertio: load_full_layer missing region L%d r%d\n", li, i);
      return;
    }
    const uint64_t span = uint64_t(E) * r.per;
    const uint8_t *raw = shards_.raw(files_->shard_path(r.ti->shard), r.ti->begin, span);
    if (!raw) {
      std::fprintf(stderr, "expertio: load_full_layer raw OOB L%d span=%llu\n", li,
                   (unsigned long long)span);
      return;
    }
    if (r.esize == 4) {  // 974-975 u32_view(raw,shape) → core.array
      L.full_w[i] =
          mx::array(reinterpret_cast<const uint32_t *>(raw), mkshape({E, r.rows, r.cols}),
                    mx::uint32);
    } else {  // 976-978 raw.view(<u2) → core.array → .view(bf16)
      auto a16 = mx::array(reinterpret_cast<const uint16_t *>(raw),
                           mkshape({E, r.rows, r.cols}), mx::uint16);
      L.full_w[i] = mx::view(a16, mx::bfloat16);
    }
    tot += span;
  }
  L.full_heap = tot;
  heap_->add(tot);
}

void ExpertIo::clear_full_layer(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  if (L.full_w.empty()) return;
  heap_->sub(L.full_heap);
  L.full_heap = 0;
  L.full_w.clear();
}

bool ExpertIo::hot_ready(int li) const {
  int idx = layer_idx(li);
  return idx >= 0 && !layers_[size_t(idx)].hot_weights.empty();  // 1121 `_hot_weights is not None`
}
bool ExpertIo::full_ready(int li) const {
  int idx = layer_idx(li);
  return idx >= 0 && !layers_[size_t(idx)].full_w.empty();  // 1231 `_full_weights is not None`
}
const std::vector<mx::array> &ExpertIo::hot_weights(int li) {
  static const std::vector<mx::array> kEmpty;
  int idx = layer_idx(li);
  return idx >= 0 ? layers_[size_t(idx)].hot_weights : kEmpty;
}
const std::vector<mx::array> &ExpertIo::full_weights(int li) {
  static const std::vector<mx::array> kEmpty;
  int idx = layer_idx(li);
  return idx >= 0 ? layers_[size_t(idx)].full_w : kEmpty;
}
const std::vector<int64_t> &ExpertIo::hot_key(int li) const {
  static const std::vector<int64_t> kEmpty;
  int idx = layer_idx(li);
  return idx >= 0 ? layers_[size_t(idx)].hot_key : kEmpty;
}
const std::vector<int64_t> &ExpertIo::last_used(int li) const {
  static const std::vector<int64_t> kEmpty;
  int idx = layer_idx(li);
  return idx >= 0 ? layers_[size_t(idx)].last_used : kEmpty;
}

void ExpertIo::note_prefill_use(int li, const std::vector<int64_t> &flat, int k,
                                bool staged_mode) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  std::vector<int64_t> uniq = flat;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  for (auto e : uniq) L.bump_count(e, L.cfg.hot_decay);  // 1125-1127
  if (staged_mode && (int)flat.size() >= k)
    L.last_prefill_topk.assign(flat.end() - k, flat.end());  // 1128-1130
}

void ExpertIo::prefetch_from_prefill(int li) {
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  if (L.last_prefill_topk.empty()) return;  // 887-888
  std::vector<int64_t> uniq = L.last_prefill_topk;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  if (uniq.empty()) return;  // 890
  L.last_used = uniq;        // 891
  prefetch(li, uniq);        // 892
}

// Cross-request memory plus incremental slot table cache.
void ExpertIo::reset_layer(int li) {  // layer.py:1429-1441
  int idx = layer_idx(li);
  if (idx < 0) return;
  Layer &L = layers_[size_t(idx)];
  std::lock_guard<std::mutex> lk(*L.mu);
  L.state = StagedState{};   // _staged_state = None(1436)
  L.next = StagedState{};    // _staged_state_next = None(1437)
  L.last_prefill_topk.clear();  // 1435
  L.last_used.clear();          // 1440
  if (!L.full_w.empty()) {
    heap_->sub(L.full_heap);
    L.full_heap = 0;
    L.full_w.clear();
  }
}

ExpertIo::Metrics ExpertIo::snapshot() const {
  Metrics m;
  m.loads = loads_.load();
  m.hits = hits_.load();
  m.misses = misses_.load();
  m.evictions = cache_.evictions();
  m.prefetch_submitted = psub_.load();
  m.prefetch_hits = phi_.load();
  m.prefetch_wasted = pwaste_.load();
  m.staged_used = sused_.load();
  m.staged_fallback = sfallback_.load();
  m.stage_builds = sbuilds_.load();
  m.heap_bytes = heap_->live.load();
  m.heap_peak = heap_->peak.load();
  m.cache_size = const_cast<SharedExpertCache &>(cache_).size();
  m.inflight = inflight_size();
  return m;
}

}  // namespace e0n
