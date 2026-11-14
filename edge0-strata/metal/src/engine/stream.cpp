// metal/src/engine/stream.cpp - see stream.h.
#include "stream.h"

#include "ref_quant.h"
#include "arena_budget.h"

#include <sys/sysctl.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace strata::metal_engine {

namespace {
constexpr int NL = 48, NE = 512;
constexpr uint64_t PAGE = 16384;
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int env_int(const char* n, int d) {
    const char* e = std::getenv(n);
    return e && *e ? std::atoi(e) : d;
}
}  // namespace

bool ExpertStream::wanted(const gguf::File& f, const mtl::Context& cx) {
    const char* e = std::getenv("STRATA_METAL_STREAM");
    if (e && std::string(e) == "1") return true;
    if (e && std::string(e) == "0") return false;
    // auto: stream when the mapped weights (experts and dense) and ~12 GiB beside them (the prompt buffers, KV, the
    // draft layer, macOS) do not fit the resident footprint cap: the GPU working set, or on a big Mac the RAM less the
    // 30% macOS and its file cache keep (arena_budget.h: footprint_cap).  M5 Max (115.4 GB working set, 96.2 GB cap):
    // GSQ packs, IQ3_S and UD-Q4_K_XL stay resident, UD-Q5/Q6_K_XL stream; M4 32 GB (26.8 GB): every pack streams.
    uint64_t all = 0;
    for (const auto& kv : f.tensors)
        if (kv.first != "per_layer_token_embd.weight") all += kv.second.bytes;
    uint64_t phys = 0;
    size_t len = sizeof phys;
    sysctlbyname("hw.memsize", &phys, &len, nullptr, 0);
    return !resident_fits((double) phys, (double) cx.recommended_working_set(), (double) all, 12.0 * (1ull << 30));
}

ExpertStream::~ExpertStream() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : io_) t.join();
    if (fd_ >= 0) ::close(fd_);
}

bool ExpertStream::open(const gguf::File& f, std::string& err) {
    fd_ = ::open(f.path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "metal stream: cannot open " + f.path; return false; }
    fcntl(fd_, F_NOCACHE, 1);
    fcntl(fd_, F_RDAHEAD, 0);
    lay_.resize(NL);
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const gguf::Tensor* g = f.find(b + "ffn_gate_exps.weight");
        const gguf::Tensor* u = f.find(b + "ffn_up_exps.weight");
        const gguf::Tensor* d = f.find(b + "ffn_down_exps.weight");
        if (!g || !u || !d) { err = "metal stream: layer " + std::to_string(l) + " has no routed experts"; return false; }
        lay_[(size_t) l] = {g->offset, g->bytes / NE, u->offset, u->bytes / NE, d->offset, d->bytes / NE, -1};
    }
    // the expert-contiguous copy beside the GGUF, when it is this GGUF's (sizes in its header) and its first and
    // last experts match the GGUF's bytes
    if (env_int("STRATA_METAL_CONTIG", 1) != 0) {
        const std::string cp = f.path + ".experts";
        const int cfd = ::open(cp.c_str(), O_RDONLY);
        bool ok = cfd >= 0;
        std::vector<uint64_t> off(NL);
        if (ok) {
            uint8_t h[64];
            std::vector<uint64_t> t((size_t) NL * 4);
            struct stat st {};
            ::fstat(fd_, &st);
            ok = ::pread(cfd, h, 64, 0) == 64 && std::memcmp(h, "STRX", 4) == 0 &&
                 ::pread(cfd, t.data(), t.size() * 8, 64) == (ssize_t) (t.size() * 8);
            uint32_t ver = 0, nl = 0, ne = 0;
            uint64_t src = 0;
            std::memcpy(&ver, h + 4, 4); std::memcpy(&nl, h + 8, 4); std::memcpy(&ne, h + 12, 4); std::memcpy(&src, h + 16, 8);
            ok = ok && ver == 1 && nl == (uint32_t) NL && ne == (uint32_t) NE && src == (uint64_t) st.st_size;
            for (int l = 0; ok && l < NL; ++l) {
                const Lay& x = lay_[(size_t) l];
                ok = t[(size_t) l * 4 + 1] == x.gs && t[(size_t) l * 4 + 2] == x.us && t[(size_t) l * 4 + 3] == x.ds;
                off[(size_t) l] = t[(size_t) l * 4];
            }
            if (ok) {   // spot check: layer 0's first and layer 47's last expert, every slice
                for (const int l : {0, NL - 1}) {
                    const Lay& x = lay_[(size_t) l];
                    const int e = l == 0 ? 0 : NE - 1;
                    std::vector<uint8_t> a(x.gs + x.us + x.ds), b(a.size());
                    ok = ok && ::pread(cfd, a.data(), a.size(), (off_t) (off[(size_t) l] + (uint64_t) e * a.size())) == (ssize_t) a.size() &&
                         ::pread(fd_, b.data(), x.gs, (off_t) (x.g + (uint64_t) e * x.gs)) == (ssize_t) x.gs &&
                         ::pread(fd_, b.data() + x.gs, x.us, (off_t) (x.u + (uint64_t) e * x.us)) == (ssize_t) x.us &&
                         ::pread(fd_, b.data() + x.gs + x.us, x.ds, (off_t) (x.d + (uint64_t) e * x.ds)) == (ssize_t) x.ds &&
                         a == b;
                }
            }
        }
        if (ok) {
            fcntl(cfd, F_NOCACHE, 1);
            fcntl(cfd, F_RDAHEAD, 0);
            cfd_ = cfd;
            coff_ = off;
            std::fprintf(stderr, "strata metal stream: experts read from %s (one request per expert)\n", cp.c_str());
        } else if (cfd >= 0) {
            ::close(cfd);
            std::fprintf(stderr, "strata metal stream: %s is not this GGUF's (sizes or bytes differ): the GGUF is read\n",
                         cp.c_str());
        }
    }
    // size classes: one LRU pool per class.  The layers' slot sizes (9 in the IQ3_XXS mix) are merged into a few
    // classes, each as big as its largest member, so that no member wastes more than STRATA_METAL_CLASS_WASTE (default
    // 0.15) of its slot: every class must hold a whole layer's experts for a prompt chunk, so fewer classes = less
    // memory tied up in those minimums.
    {
        const char* ew = std::getenv("STRATA_METAL_CLASS_WASTE");
        const double waste = ew ? std::atof(ew) : 0.15;
        std::vector<uint64_t> pages_of(NL);
        std::vector<uint64_t> sizes;
        for (int l = 0; l < NL; ++l) {
            const Lay& x = lay_[(size_t) l];
            pages_of[(size_t) l] = (x.gs + x.us + x.ds + PAGE - 1) / PAGE;
            sizes.push_back(pages_of[(size_t) l]);
        }
        std::sort(sizes.begin(), sizes.end());
        sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
        std::vector<uint64_t> tops;   // each class's slot pages, largest first
        for (auto it = sizes.rbegin(); it != sizes.rend(); ++it)
            if (tops.empty() || (double) *it < (1.0 - waste) * (double) tops.back()) tops.push_back(*it);
        for (uint64_t t : tops) { Cls c; c.pages = t; cls_.push_back(c); }
        for (int l = 0; l < NL; ++l) {
            int best = 0;   // the smallest class that holds the layer's slot
            for (int c = 0; c < (int) cls_.size(); ++c)
                if (cls_[(size_t) c].pages >= pages_of[(size_t) l]) best = c;
            lay_[(size_t) l].cls = best;
        }
    }
    nio_ = std::max(1, env_int("STRATA_METAL_IO_THREADS", 6));
    split_ = std::max(1, std::min(8, env_int("STRATA_METAL_READ_SPLIT", 1)));
    // The read-ahead's depth and where its experts wait in the LRU, measured on the M4 (notes-mac-ssd.md, 3 interleaved
    // pairs each): Q2_0 (1.38 MB experts) top 10 at the LRU front 12.2 tok/s vs top 6 at the tail 11.3; the IQ3_XXS
    // mix (1.75 MB, the SSD is the limit) top 6 at the tail 8.06 vs top 10 at the front 7.43.
    uint64_t eb = 0;
    for (const auto& x : lay_) eb += x.gs + x.us + x.ds;
    const bool big = (double) eb / NL > 1.5e6;
    pre_k_ = env_int("STRATA_METAL_PREFETCH", big ? 6 : 10);
    pre_k2_ = env_int("STRATA_METAL_PREFETCH2", 0);
    prefetch_ = pre_k_ > 0;
    pre_mru_ = env_int("STRATA_METAL_PREFETCH_MRU", big ? 0 : 1) != 0;
    // timing only (WRONG answers): every missing expert is pointed at slot 0, nothing is read
    fake_ = env_int("STRATA_METAL_STREAM_FAKE", 0) != 0;
    return true;
}

bool ExpertStream::read(void* dst, uint64_t len, uint64_t off) const {
    uint8_t* p = (uint8_t*) dst;
    while (len > 0) {
        const ssize_t r = ::pread(fd_, p, (size_t) std::min<uint64_t>(len, 64ull << 20), (off_t) off);
        if (r <= 0) return false;
        p += r;
        off += (uint64_t) r;
        len -= (uint64_t) r;
    }
    return true;
}

bool ExpertStream::load_dense(gguf::File& f, mtl::Context& cx, const std::vector<std::string>& skip, mtl::Buffer& out,
                              std::string& err) {
    const double t0 = now_ms();
    std::vector<gguf::Tensor*> ts;
    uint64_t tot = 0;
    for (auto& kv : f.tensors) {
        gguf::Tensor& t = kv.second;
        if (t.name.find("_exps") != std::string::npos) continue;
        // the n-gram table is read by the host from the SSD, a few rows at a time (src/ngram/ple_reader.cpp).  It is its own shard in
        // the ISTA / Unsloth files; in agentionai's Gyro files it is the first 28.8 GB of the one file, and must not be read in here
        if (t.name == "per_layer_token_embd.weight") continue;
        if (std::find(skip.begin(), skip.end(), t.name) != skip.end()) { t.offset = 0; continue; }
        ts.push_back(&t);
        tot += (t.bytes + 255) / 256 * 256;
    }
    out = cx.alloc((tot + PAGE - 1) / PAGE * PAGE);
    std::vector<uint64_t> src(ts.size()), dst(ts.size());
    uint64_t o = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        src[i] = ts[i]->offset;
        dst[i] = o;
        o += (ts[i]->bytes + 255) / 256 * 256;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> ok{true};
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
        th.emplace_back([&] {
            for (;;) {
                const size_t i = next++;
                if (i >= ts.size()) break;
                if (!read(out.as<uint8_t>() + dst[i], ts[i]->bytes, src[i])) ok = false;
            }
        });
    for (auto& x : th) x.join();
    if (!ok) { err = "metal stream: reading the dense tensors failed"; return false; }
    for (size_t i = 0; i < ts.size(); ++i) ts[i]->offset = dst[i];
    std::fprintf(stderr, "strata metal stream: %zu dense tensors read into memory (%.2f GB, %.1f s; %zu left out)\n", ts.size(),
                 tot / 1e9, (now_ms() - t0) / 1e3, skip.size());
    return true;
}

int ExpertStream::cls_of(int s) const {   // the class ranges are contiguous
    int ci = 0;
    while (!(s >= cls_[(size_t) ci].base && s < cls_[(size_t) ci].base + cls_[(size_t) ci].n)) ++ci;
    return ci;
}

void ExpertStream::lru_unlink(int s) {
    Slot& x = slots_[(size_t) s];
    Cls& k = cls_[(size_t) cls_of(s)];
    if (x.prev >= 0) slots_[(size_t) x.prev].next = x.next; else k.head = x.next;
    if (x.next >= 0) slots_[(size_t) x.next].prev = x.prev; else k.tail = x.prev;
    x.prev = x.next = -1;
}

void ExpertStream::lru_back(int s) {
    lru_unlink(s);
    Cls& k = cls_[(size_t) cls_of(s)];
    Slot& x = slots_[(size_t) s];
    x.prev = k.tail;
    x.next = -1;
    if (k.tail >= 0) slots_[(size_t) k.tail].next = s;
    k.tail = s;
    if (k.head < 0) k.head = s;
}

void ExpertStream::lru_front(int s) {
    lru_unlink(s);
    Cls& k = cls_[(size_t) cls_of(s)];
    Slot& x = slots_[(size_t) s];
    x.next = k.head;
    x.prev = -1;
    if (k.head >= 0) slots_[(size_t) k.head].prev = s;
    k.head = s;
    if (k.tail < 0) k.tail = s;
}

int ExpertStream::victim(int ci) {
    Cls& k = cls_[(size_t) ci];
    for (int s = k.tail; s >= 0; s = slots_[(size_t) s].prev)
        if (slots_[(size_t) s].state.load() == 0 && slots_[(size_t) s].pin != stamp_) {
            Slot& x = slots_[(size_t) s];
            if (x.layer >= 0) {
                const int key = x.layer * NE + x.e;
                where_[(size_t) key] = -1;
                smap_.as<uint32_t>()[key] = 0xFFFFFFFFu;
            }
            x.layer = x.e = -1;
            return s;
        }
    return -1;
}

// One piece of an expert: slice part / split_ (gate, up, down), piece part % split_ of it.  The pieces of one expert
// go to several I/O threads, so a single miss is read with several requests in flight (its latency, not only the
// SSD's throughput, is what a layer waits for).
void ExpertStream::read_part(int s, int l, int e, int part) {
    const Lay& x = lay_[(size_t) l];
    if (cfd_ >= 0) {   // the contiguous copy: the whole expert, piece part of split_ (the slot keeps gate | up | down)
        const uint64_t len = x.gs + x.us + x.ds, src = coff_[(size_t) l] + (uint64_t) e * len;
        const uint64_t a = len * (uint64_t) part / (uint64_t) split_ / 4096 * 4096;
        const uint64_t b = part + 1 == split_ ? len : len * (uint64_t) (part + 1) / (uint64_t) split_ / 4096 * 4096;
        uint8_t* p = host_[(size_t) s] + a;
        uint64_t n = b - a, o = src + a;
        while (n > 0) {
            const ssize_t r = ::pread(cfd_, p, (size_t) n, (off_t) o);
            if (r <= 0) {
                std::fprintf(stderr, "strata metal stream: reading expert %d of layer %d failed\n", e, l);
                std::exit(6);
            }
            p += r; o += (uint64_t) r; n -= (uint64_t) r;
        }
        bytes_read_ += b - a;
        return;
    }
    const int k = part / split_, p = part % split_;
    const uint64_t len = k == 0 ? x.gs : (k == 1 ? x.us : x.ds);
    const uint64_t src = (k == 0 ? x.g : (k == 1 ? x.u : x.d)) + (uint64_t) e * len;
    const uint64_t dst = k == 0 ? 0 : (k == 1 ? x.gs : x.gs + x.us);
    const uint64_t a = len * (uint64_t) p / (uint64_t) split_ / 4096 * 4096;
    const uint64_t b = p + 1 == split_ ? len : len * (uint64_t) (p + 1) / (uint64_t) split_ / 4096 * 4096;
    if (b > a && !read(host_[(size_t) s] + dst + a, b - a, src + a)) {
        std::fprintf(stderr, "strata metal stream: reading expert %d of layer %d failed\n", e, l);
        std::exit(6);
    }
    bytes_read_ += b - a;
}

void ExpertStream::io_main() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> g(mu_);
            cv_.wait(g, [&] { return stop_ || !urgent_q_.empty() || !pre_q_.empty(); });
            if (stop_) return;
            if (!urgent_q_.empty()) { j = urgent_q_.front(); urgent_q_.pop_front(); }
            else { j = pre_q_.front(); pre_q_.pop_front(); }
            slots_[(size_t) j.slot].state = 2;
        }
        read_part(j.slot, j.layer, j.e, j.part);
        bool done = false;
        {
            std::lock_guard<std::mutex> g(mu_);
            Slot& x = slots_[(size_t) j.slot];
            if (--x.left == 0) {
                smap_.as<uint32_t>()[j.layer * NE + j.e] = code_[(size_t) j.slot];
                x.state = 0;
                --outstanding_;
                ++reads_;
                done = true;
            }
        }
        if (done) done_cv_.notify_all();
    }
}

void ExpertStream::submit(int s, int l, int e, bool urgent) {
    Slot& x = slots_[(size_t) s];
    x.layer = (int16_t) l;
    x.e = (int16_t) e;
    x.state = 1;
    x.pre = !urgent;
    const int parts = cfd_ >= 0 ? split_ : 3 * split_;
    x.left = parts;
    ++outstanding_;
    where_[(size_t) (l * NE + e)] = s;
    for (int p = 0; p < parts; ++p) (urgent ? urgent_q_ : pre_q_).push_back({s, l, e, p});
}

// a queued read-ahead nobody wants: its pieces dropped, the slot freed (only before any piece has started)
void ExpertStream::cancel_slot(int s) {
    Slot& x = slots_[(size_t) s];
    for (auto it = pre_q_.begin(); it != pre_q_.end();) it = it->slot == s ? pre_q_.erase(it) : it + 1;
    where_[(size_t) (x.layer * NE + x.e)] = -1;
    x.layer = x.e = -1;
    x.state = 0;
    x.pre = false;
    x.left = 0;
    --outstanding_;
    ++pre_cancel_;
    lru_back(s);
}

// a read-ahead that is wanted now: its queued pieces go first
void ExpertStream::promote_slot(int s) {
    std::vector<Job> mine;
    for (auto it = pre_q_.begin(); it != pre_q_.end();)
        if (it->slot == s) { mine.push_back(*it); it = pre_q_.erase(it); }
        else ++it;
    for (auto it = mine.rbegin(); it != mine.rend(); ++it) urgent_q_.push_front(*it);
}

bool ExpertStream::make_arena(mtl::Context& cx, uint64_t budget, const std::string& profile, int min_slots, std::string& err,
                              const std::function<uint64_t()>& recheck, const std::function<bool(uint64_t)>& grow) {
    const double t0 = now_ms();
    shrink_to_ = 0;
    for (auto& k : cls_) k.n = 0;   // a second try (after a shrink) lays the arena out again
    // the ranking: the profile's pairs, then every pair it misses, layer-interleaved
    std::vector<std::pair<int, int>> rank;
    std::vector<uint8_t> seen((size_t) NL * NE, 0);
    if (FILE* f = std::fopen(profile.c_str(), "rb")) {
        char magic[4];
        uint32_t h[5];
        if (std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "STRP", 4) == 0 && std::fread(h, 4, 5, f) == 5 &&
            h[1] == (uint32_t) NL && h[2] == (uint32_t) NE) {
            for (uint32_t i = 0; i < h[4]; ++i) {
                uint16_t p[2];
                if (std::fread(p, 2, 2, f) != 2) break;
                if (p[0] < NL && p[1] < NE && !seen[(size_t) p[0] * NE + p[1]]) {
                    seen[(size_t) p[0] * NE + p[1]] = 1;
                    rank.push_back({p[0], p[1]});
                }
            }
        }
        std::fclose(f);
    } else {
        std::fprintf(stderr, "strata metal stream: no expert profile at %s: the arena starts layer-interleaved\n", profile.c_str());
    }
    const size_t from_profile = rank.size();
    for (int e = 0; e < NE; ++e)
        for (int l = 0; l < NL; ++l)
            if (!seen[(size_t) l * NE + e]) rank.push_back({l, e});
    // the pool of each size class: at least min_slots (a prompt chunk's layer needs up to 512 at once), then the
    // top-ranked pairs that fit in the budget
    uint64_t used = 0;
    for (const auto& k : cls_) used += (uint64_t) min_slots * k.pages * PAGE;
    std::vector<std::pair<int, int>> fill;
    for (const auto& p : rank) {
        Cls& k = cls_[(size_t) lay_[(size_t) p.first].cls];
        const uint64_t cost = k.n < min_slots ? 0 : k.pages * PAGE;
        if (used + cost > budget) break;
        used += cost;
        ++k.n;
        fill.push_back(p);
    }
    int base = 0;
    for (auto& k : cls_) {
        k.n = std::max(k.n, min_slots);
        k.base = base;
        base += k.n;
    }
    // the slots placed in chunks of at most maxBufferLength (and 8 GB), up to 4 (the kernels' buffers 0, 7, 8, 9)
    // (8 GB each unless the arena needs bigger ones to fit 4: the M4 refused one 20.6 GB buffer, the M5 Max takes an
    // 89 GB arena for UD-Q5_K_XL in 4 of ~23 GB; STRATA_METAL_ARENA_CHUNK_GB sets it)
    uint64_t total = 0, biggest = 0;
    for (const auto& k : cls_) {
        total += (uint64_t) k.n * k.pages * PAGE;
        biggest = std::max<uint64_t>(biggest, k.pages * PAGE);
    }
    uint64_t want = std::max<uint64_t>(8ull << 30, total / 4 + 4 * biggest);
    if (const char* e = std::getenv("STRATA_METAL_ARENA_CHUNK_GB")) want = (uint64_t) (std::atof(e) * (1ull << 30));
    const uint64_t cmax = std::min<uint64_t>(cx.max_buffer_length(), want) / PAGE * PAGE;
    code_.assign((size_t) base, 0);
    host_.assign((size_t) base, nullptr);
    std::vector<uint64_t> csize;
    std::vector<std::pair<int, uint64_t>> at((size_t) base);
    uint64_t cur = 0;
    for (const auto& k : cls_)
        for (int s = k.base; s < k.base + k.n; ++s) {
            if (csize.empty() || cur + k.pages * PAGE > cmax) {
                if (!csize.empty()) csize.back() = cur;
                csize.push_back(0);
                cur = 0;
            }
            at[(size_t) s] = {(int) csize.size() - 1, cur / PAGE};
            cur += k.pages * PAGE;
        }
    if (!csize.empty()) csize.back() = cur;
    if (csize.size() > 4) { err = "metal stream: the arena needs more than 4 buffers (lower STRATA_METAL_ARENA_GB)"; return false; }
    if (recheck) {   // the memory macOS can give, asked again just before the arena is committed
        uint64_t total = 0;
        for (uint64_t b : csize) total += b;
        const uint64_t allowed = recheck();
        if (allowed + (64ull << 20) < total) {
            shrink_to_ = allowed;
            err = "metal stream: less memory than planned is free now: the arena shrinks";
            return false;
        }
    }
    for (uint64_t b : csize) {   // anonymous memory wrapped no-copy: a prompt's buffers can alias a region of it
        void* m = mmap(nullptr, (size_t) b, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (m == MAP_FAILED) { err = "metal stream: cannot map the arena"; return false; }
        chunks_.push_back(cx.wrap(m, (size_t) b));
        arena_bytes_ += b;
    }
    for (int s = 0; s < base; ++s) {
        code_[(size_t) s] = ((uint32_t) at[(size_t) s].first << 28) | (uint32_t) at[(size_t) s].second;
        host_[(size_t) s] = chunks_[(size_t) at[(size_t) s].first].as<uint8_t>() + at[(size_t) s].second * PAGE;
    }
    smap_ = cx.alloc((size_t) NL * NE * 4);
    std::memset(smap_.ptr, 0xFF, (size_t) NL * NE * 4);
    slots_ = std::vector<Slot>((size_t) base);
    retired_.assign((size_t) base, 0);
    where_.assign((size_t) NL * NE, -1);
    for (auto& k : cls_) {
        for (int s = k.base; s < k.base + k.n; ++s) {
            slots_[(size_t) s].prev = s == k.base ? -1 : s - 1;
            slots_[(size_t) s].next = s + 1 == k.base + k.n ? -1 : s + 1;
        }
        k.head = k.n ? k.base : -1;
        k.tail = k.n ? k.base + k.n - 1 : -1;
    }
    for (int i = 0; i < nio_; ++i) io_.emplace_back([this] { io_main(); });
    // the fill: the ranked pairs in steps (best first; each step in file order) on the I/O threads; before each step
    // grow() asks whether macOS still has the memory for it (arena_grow); the best-ranked ends at the LRU front
    uint64_t step = 1536ull << 20;
    if (const char* e = std::getenv("STRATA_METAL_ARENA_STEP_GB")) step = std::max<uint64_t>(64ull << 20, (uint64_t) (std::atof(e) * 1e9));
    std::vector<int> next_free(cls_.size());
    for (size_t c = 0; c < cls_.size(); ++c) next_free[c] = cls_[c].base;
    size_t done = 0;
    bool stopped = false;
    uint64_t filled = 0;
    while (done < fill.size()) {
        size_t j = done;
        uint64_t sb = 0;
        while (j < fill.size() && (sb < step || j == done)) sb += cls_[(size_t) lay_[(size_t) fill[j++].first].cls].pages * PAGE;
        if (grow && !grow(sb)) { stopped = true; break; }
        std::vector<std::pair<int, int>> order(fill.begin() + (long) done, fill.begin() + (long) j);
        std::sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
            return lay_[(size_t) a.first].g + (uint64_t) a.second * lay_[(size_t) a.first].gs <
                   lay_[(size_t) b.first].g + (uint64_t) b.second * lay_[(size_t) b.first].gs;
        });
        {
            std::lock_guard<std::mutex> g(mu_);
            for (const auto& p : order) {
                const int c = lay_[(size_t) p.first].cls;
                submit(next_free[(size_t) c]++, p.first, p.second, true);
            }
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> g(mu_);
            done_cv_.wait(g, [&] { return outstanding_ == 0; });
        }
        filled += sb;
        done = j;
    }
    uint64_t retired_bytes = 0;
    size_t nretired = 0;
    {
        std::unique_lock<std::mutex> g(mu_);
        if (stopped)   // the slots not filled stay unused (their pages never touched), all but min_slots of each class
            for (size_t c = 0; c < cls_.size(); ++c) {
                const Cls& k = cls_[c];
                for (int s = std::max(next_free[c], k.base + std::min(min_slots, k.n)); s < k.base + k.n; ++s) {
                    lru_unlink(s);
                    retired_[(size_t) s] = 1;
                    retired_bytes += k.pages * PAGE;
                    ++nretired;
                }
            }
        // the LRU order = the rank order (best at the front)
        for (size_t i = done; i-- > 0;) lru_front(where_[(size_t) (fill[i].first * NE + fill[i].second)]);
    }
    if (stopped)
        std::fprintf(stderr, "strata metal stream: the arena's fill stopped after %.2f GB: macOS's free memory reached what it "
                             "keeps; %.2f GB of the arena (%zu slots) stay unused\n", filled / 1e9, retired_bytes / 1e9, nretired);
    arena_live_ = arena_bytes_ - retired_bytes;
    fill_s_ = (now_ms() - t0) / 1e3;
    fill_bytes_ = bytes_read_.load();
    uint64_t pairs = 0;
    for (const auto& k : cls_) pairs += (uint64_t) k.n;
    std::fprintf(stderr, "strata metal stream: expert arena %.2f GB, %llu slots in %zu size classes, %zu experts read from the "
                         "profile's top (%zu ranked) in %.1f s (%.2f GB/s); %d I/O threads, read-ahead top %d; %zu buffers\n",
                 (arena_bytes_ - retired_bytes) / 1e9, (unsigned long long) (pairs - nretired), cls_.size(), done, from_profile, fill_s_,
                 fill_bytes_ / 1e9 / std::max(1e-9, fill_s_), nio_, prefetch_ ? pre_k_ : 0, chunks_.size());
    // Wire the filled arena (mlock).  Unwired, the arena is ordinary anonymous memory whenever the GPU is not running a command buffer
    // (the GPU's own wiring ends a few seconds after the last one; a Metal residency set did not hold it either, measured), and
    // macOS compresses it when anything else wants memory: a 20 GB file written beside the idle engine took tens of GB of the arena
    // into the compressor in seconds and froze the 128 GB M5 Max (watchdog panic, 2026-10-13).  Wired, a file write can only
    // shrink the file cache.  STRATA_METAL_WIRE_ARENA=0/1; the default is on for a Mac of 64 GB or more (the M4 32 GB was measured
    // unwired).  Not when the fill stopped early for lack of memory (the unused slots would be wired too).
    {
        uint64_t phys = 0;
        size_t plen = sizeof phys;
        sysctlbyname("hw.memsize", &phys, &plen, nullptr, 0);
        const char* we = std::getenv("STRATA_METAL_WIRE_ARENA");
        const bool want = we && *we ? std::atoi(we) != 0 : (double) phys >= kBigMac;
        if (want && !stopped) {
            const double tp = now_ms();
            uint64_t pinned = 0;
            int failed = 0;
            for (const auto& c : chunks_) {
                if (mlock(c.ptr, c.bytes) == 0) pinned += c.bytes;
                else ++failed;
            }
            std::fprintf(stderr, "strata metal stream: %.1f GB of the arena wired (mlock) in %.0f ms%s; macOS cannot compress it while "
                                 "the engine idles (STRATA_METAL_WIRE_ARENA=0 turns it off)\n", pinned / 1e9, now_ms() - tp,
                         failed ? "; SOME CHUNKS COULD NOT BE WIRED" : "");
        }
    }
    (void) err;
    return true;
}

// the experts a predicted routing names, most likely first (by the best logit over the window's tokens)
void ExpertStream::ahead(int ln, int T, const float* pred, int K) {
    K = std::min(K, 32);
    std::pair<float, int> pw[8 * 32];
    int np = 0;
    for (int t = 0; t < T; ++t) {
        const float* lg = pred + (size_t) t * NE;
        int top[32];
        int nt = 0;
        for (int e = 0; e < NE; ++e) {
            if (nt < K) { top[nt++] = e; continue; }
            int m = 0;
            for (int j = 1; j < K; ++j) if (lg[top[j]] < lg[top[m]]) m = j;
            if (lg[e] > lg[top[m]]) top[m] = e;
        }
        for (int j = 0; j < nt; ++j) {
            int q = 0;
            while (q < np && pw[q].second != top[j]) ++q;
            if (q == np) pw[np++] = {lg[top[j]], top[j]};
            else pw[q].first = std::max(pw[q].first, lg[top[j]]);
        }
    }
    std::sort(pw, pw + np, [](const auto& a, const auto& b) { return a.first > b.first; });
    bool any = false;
    {
        std::lock_guard<std::mutex> g(mu_);
        for (int j = 0; j < np; ++j) {
            const int key = ln * NE + pw[j].second;
            const int s0 = where_[(size_t) key];
            if (s0 >= 0) { lru_front(s0); continue; }
            const int s = victim(lay_[(size_t) ln].cls);
            if (s < 0) break;
            submit(s, ln, pw[j].second, false);
            slots_[(size_t) s].pin = stamp_;
            if (pre_mru_) lru_front(s);
            else lru_back(s);
            ++pre_issued_;
            any = true;
        }
    }
    if (any) cv_.notify_all();
}

void ExpertStream::at_layer(int l, int T, int topk, const int32_t* ids, const float* pred, const float* pred2) {
    const double t0 = now_ms();
    const bool prompt = T > 8;   // a prompt chunk (the GEMM path): nearly every expert of the layer is used
    if (l == 0) { if (prompt) ++p_chunks_; else ++windows_; }
    ++stamp_;                    // the slots this layer uses are pinned until the next handshake
    bool used[NE] = {};
    int want[NE];
    int nw = 0;
    for (int i = 0; i < T * topk; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= NE || used[e]) continue;
        used[e] = true;
        want[nw++] = e;
    }
    if (fake_) {
        for (int j = 0; j < nw; ++j)
            if (where_[(size_t) (l * NE + want[j])] < 0) smap_.as<uint32_t>()[l * NE + want[j]] = code_[0];
        host_ms_ += now_ms() - t0;
        return;
    }
    std::vector<int> wait;
    {
        std::lock_guard<std::mutex> g(mu_);
        // read-aheads of this layer (or older) that were not started and are not wanted: dropped (the slot goes free)
        {
            std::vector<int> drop;
            for (const Job& j : pre_q_) {
                if (j.layer > l || slots_[(size_t) j.slot].state.load() != 1) continue;
                if (!(j.layer == l && used[j.e]) && std::find(drop.begin(), drop.end(), j.slot) == drop.end()) drop.push_back(j.slot);
            }
            for (int s : drop) cancel_slot(s);
        }
        for (int j = 0; j < nw; ++j)   // pass 1: pin what is resident (or on its way), so no miss below evicts it
            if (where_[(size_t) (l * NE + want[j])] >= 0) slots_[(size_t) where_[(size_t) (l * NE + want[j])]].pin = stamp_;
        for (int j = 0; j < nw; ++j) {
            const int e = want[j], key = l * NE + e;
            uint64_t& lk = prompt ? p_lookups_ : lookups_;
            ++lk;
            int s = where_[(size_t) key];
            if (s >= 0) {
                Slot& x = slots_[(size_t) s];
                x.pin = stamp_;
                const int st = x.state.load();
                if (st == 0) { if (x.pre) ++(prompt ? p_pre_ : pre_hits_); else ++(prompt ? p_hits_ : hits_); }
                else {
                    ++(prompt ? p_pre_ : pre_hits_);
                    if (!prompt) ++late_;
                    promote_slot(s);   // a queued read-ahead becomes urgent
                    wait.push_back(s);
                }
                if (!(prompt && x.pre)) lru_front(s);   // a prompt's streamed experts stay at the LRU tail
                x.pre = false;
                continue;
            }
            ++(prompt ? p_misses_ : misses_);
            s = victim(lay_[(size_t) l].cls);
            if (s < 0) { std::fprintf(stderr, "strata metal stream: no free slot (layer %d)\n", l); std::exit(6); }
            submit(s, l, e, true);
            slots_[(size_t) s].pin = stamp_;
            if (!prompt) lru_front(s);
            wait.push_back(s);
        }
    }
    if (!wait.empty()) {
        cv_.notify_all();
        const double tw = now_ms();
        std::unique_lock<std::mutex> g(mu_);
        done_cv_.wait(g, [&] {
            for (int s : wait) if (slots_[(size_t) s].state.load() != 0) return false;
            return true;
        });
        (prompt ? p_wait_ms_ : wait_ms_) += now_ms() - tw;
    }
    if (prompt) {
        // a chunk uses nearly all of the next layer's experts: every one not in the arena is streamed in behind this
        // layer's compute, into the LRU tail (the decode-hot experts stay)
        if (l + 1 < NL) {
            bool any = false;
            std::lock_guard<std::mutex> g(mu_);
            const int ln = l + 1;
            for (int e = 0; e < NE; ++e) {
                if (where_[(size_t) (ln * NE + e)] >= 0) continue;
                const int s = victim(lay_[(size_t) ln].cls);
                if (s < 0) break;
                submit(s, ln, e, false);
                slots_[(size_t) s].pin = stamp_;
                ++p_streamed_;
                any = true;
            }
            if (any) cv_.notify_all();
        }
        (prompt ? p_host_ms_ : host_ms_) += now_ms() - t0;
        return;
    }
    // read-ahead: layer l + 1 (each token's top pre_k_ of the predicted router logits), then layer l + 2 (top pre_k2_)
    if (prefetch_ && pred && l + 1 < NL) ahead(l + 1, T, pred, pre_k_);
    if (pre_k2_ > 0 && pred2 && l + 2 < NL) ahead(l + 2, T, pred2, pre_k2_);
    host_ms_ += now_ms() - t0;
}

// A region of `bytes` at the end of one arena buffer, for a prompt's scratch buffers: its slots are emptied and held
// (state 3) while a prompt runs (hold_region), and are ordinary slots again for decoding.
uint8_t* ExpertStream::region(uint64_t bytes) {
    for (int c = (int) chunks_.size() - 1; c >= 0; --c) {
        if (chunks_[(size_t) c].bytes < bytes + (64ull << 20)) continue;
        uint8_t* end = chunks_[(size_t) c].as<uint8_t>() + chunks_[(size_t) c].bytes;
        uint8_t* lo = end - (bytes + PAGE - 1) / PAGE * PAGE;
        region_slots_.clear();
        uint8_t* first = end;
        for (int s = 0; s < (int) slots_.size(); ++s) {
            const uint8_t* h = host_[(size_t) s];
            const uint64_t sz = cls_[(size_t) cls_of(s)].pages * PAGE;
            if (h + sz > lo && h < end) { region_slots_.push_back(s); first = std::min(first, (uint8_t*) h); }
        }
        {   // a class left with fewer slots than a prompt chunk's layer can use (NE) would stop with "no free slot": a small
            // arena (a Mac with little free memory at start) gets the prompt's buffers as their own allocation instead
            std::vector<int> stolen(cls_.size(), 0);
            for (int s : region_slots_) ++stolen[(size_t) cls_of(s)];
            bool tight = false;
            for (size_t i = 0; i < cls_.size(); ++i) tight = tight || (cls_[i].n > 0 && cls_[i].n - stolen[i] < NE);
            if (tight) return standalone_region(bytes);
        }
        region_ = lo;
        region_bytes_ = (uint64_t) (end - lo);
        std::fprintf(stderr, "strata metal stream: %.2f GB of the arena (%zu slots) hold a prompt's buffers while it runs\n",
                     bytes / 1e9, region_slots_.size());
        return lo;
    }
    // no chunk is big enough for the prompt's buffers (an arena that shrank at the start, a 24 GB Mac): the fallback
    // inside the loop above is only reached when a chunk fits, so the buffers get their own allocation here too
    return standalone_region(bytes);
}

// The prompt's buffers as their own anonymous allocation (fresh zero pages), when no region of the arena's slots
// takes them: no slots are held while a prompt runs, and the arena keeps decoding with all of its slots.
uint8_t* ExpertStream::standalone_region(uint64_t bytes) {
    const uint64_t sz = (bytes + PAGE - 1) / PAGE * PAGE;
    void* m = mmap(nullptr, (size_t) sz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    region_slots_.clear();
    region_ = (uint8_t*) m;
    region_bytes_ = 0;   // fresh zero pages; nothing in the arena to empty
    std::fprintf(stderr, "strata metal stream: the arena is small: the prompt's %.2f GB of buffers are their own "
                         "allocation, not a region of its slots\n", bytes / 1e9);
    return region_;
}

void ExpertStream::hold_region(bool hold) {
    std::unique_lock<std::mutex> g(mu_);
    if (hold) {
        // nothing may be reading into those slots: the queued read-aheads are dropped, the running ones waited for
        std::vector<int> drop;
        for (const Job& j : pre_q_)
            if (slots_[(size_t) j.slot].state.load() == 1 && std::find(drop.begin(), drop.end(), j.slot) == drop.end())
                drop.push_back(j.slot);
        for (int s : drop) cancel_slot(s);
        done_cv_.wait(g, [&] { return outstanding_ == 0; });
        for (int s : region_slots_) {
            Slot& x = slots_[(size_t) s];
            if (x.layer >= 0) {
                const int key = x.layer * NE + x.e;
                where_[(size_t) key] = -1;
                smap_.as<uint32_t>()[key] = 0xFFFFFFFFu;
            }
            x.layer = x.e = -1;
            x.pre = false;
            x.state = 3;
        }
        if (region_) std::memset(region_, 0, (size_t) region_bytes_);
    } else {
        for (int s : region_slots_) {
            slots_[(size_t) s].state = 0;
            if (!retired_[(size_t) s]) lru_back(s);   // a retired slot's memory serves the prompt only
        }
    }
}

void ExpertStream::reset_stats() {
    lookups_ = hits_ = pre_hits_ = late_ = misses_ = pre_issued_ = pre_cancel_ = windows_ = 0;
    p_lookups_ = p_hits_ = p_pre_ = p_misses_ = p_streamed_ = p_chunks_ = 0;
    p_wait_ms_ = p_host_ms_ = 0;
    bytes_read_ = 0;
    reads_ = 0;
    wait_ms_ = host_ms_ = 0;
}

void ExpertStream::report(FILE* f) const {
    if (!ready()) return;
    if (p_chunks_)
        std::fprintf(f, "expert stream prompts    %llu chunks: %llu distinct lookups (%.1f per layer), in the arena %.4f, streamed "
                        "ahead %.4f, missed %.4f; %llu streamed in; host %.1f ms per chunk, of it waiting %.1f ms\n",
                     (unsigned long long) p_chunks_, (unsigned long long) p_lookups_, p_lookups_ / 48.0 / p_chunks_,
                     p_hits_ / std::max(1.0, (double) p_lookups_), p_pre_ / std::max(1.0, (double) p_lookups_),
                     p_misses_ / std::max(1.0, (double) p_lookups_), (unsigned long long) p_streamed_, p_host_ms_ / p_chunks_,
                     p_wait_ms_ / p_chunks_);
    const double w = std::max<uint64_t>(1, windows_);
    std::fprintf(f,
                 "expert stream            %llu windows: %llu distinct lookups (%.1f per window), hits %.4f, read-ahead hits "
                 "%.4f (%llu of them late), misses %.4f (%.1f per window); read-aheads %llu (%.1f per window, %llu dropped); %.1f MB "
                 "read per window (%llu reads); host %.2f ms per window, of it waiting %.2f ms\n",
                 (unsigned long long) windows_, (unsigned long long) lookups_, lookups_ / w,
                 hits_ / std::max(1.0, (double) lookups_), pre_hits_ / std::max(1.0, (double) lookups_),
                 (unsigned long long) late_, misses_ / std::max(1.0, (double) lookups_), misses_ / w,
                 (unsigned long long) pre_issued_, pre_issued_ / w, (unsigned long long) pre_cancel_,
                 (bytes_read_.load() - 0) / 1e6 / w, (unsigned long long) reads_.load(), host_ms_ / w, wait_ms_ / w);
}

}  // namespace strata::metal_engine
