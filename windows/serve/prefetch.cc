// serve/prefetch.cc — advisory expert-prefetch router + self-managed L1 pool + accounting.
// Depends only on llama public/near-public APIs (declared inline, linked from llama.lib)
// and parses GGUF/safetensors headers by hand; llama internal headers are not touched.
// The real routing gate is never modified: predictions only warm the page cache. A wrong
// prediction costs latency, never quality.
// Three-way overlap accounting (printed as [pref-trace]):
//   head   = pred@t-1 ∩ teacher@t        teacher = host-side recompute of gate·m_in (gate is F32 in our GGUF)
//   sticky = teacher@t-1 ∩ teacher@t     (the free win the OS page-cache LRU already provides)
//   random = uniform sample of K ∩ teacher@t   (popularity floor)
#include "prefetch.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Near-public staging API (src/llama-ext.h). NOTE: that header has **no extern "C"
// wrapper** — its symbols are C++-decorated, so you must include the real header rather
// than hand-write declarations. This unit is globbed into the llama target by the patch
// band (patches/llama.cpp/common #3), which puts src/ on the PRIVATE include path;
// llama.h is reachable via ../include PUBLIC.
#include "llama.h"
#include "llama-ext.h"


namespace {

// Pool final telemetry (defined in the pool section below) — fwd-declared for ~Pref.
static void e0_pool_trace_final();

// ── Shapes (pinned for the 35B GGUF: E=256, H=2048, feat=2560, K=4, owners 6..38 → consumers 7..39) ──
constexpr int   EK = 256, HH = 2048, FE = 2560, KDIM = 4, NH = 33;
constexpr int   OWNER0 = 6;                       // first owner
std::vector<int> consumers;                        // 7..39

// ── GGUF/safetensors header parsing ──
std::string rd_str(FILE * f) {
    uint64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) return {};
    std::string s((size_t) n, '\0');
    if (n && fread(s.data(), 1, (size_t) n, f) != n) return {};
    return s;
}
struct GGUFType { uint32_t t; };
size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
// GGUF KV value types: 0u8 1i8 2u16 3i16 4u32 5i32 6f32 7bool 8str 9arr 10u64 11i64 12f64
size_t scalar_size(uint32_t t) { static const size_t s[] = {1,1,2,2,4,4,4,1,0,0,8,8,8}; return t <= 12 ? s[t] : 0; }
void skip_val(FILE * f, uint32_t t) {
    if (t == 8) { rd_str(f); return; }
    if (t == 9) {
        uint32_t et; uint64_t n;
        if (fread(&et, 4, 1, f) != 1 || fread(&n, 8, 1, f) != 1) return;
        for (uint64_t i = 0; i < n; ++i) skip_val(f, et);
        return;
    }
    size_t s = scalar_size(t);
    if (s) fseek(f, (long) s, SEEK_CUR);
}
// Tensor dtype uses the GGML_TYPE enum: 0=F32 1=F16 2=BF16 3=Q4_1 (NOT the GGUF KV enum — don't mix).
bool   is_q4_1(uint32_t t) { return t == 3; }
size_t plain_esz(uint32_t t) { return t == 0 ? 4 : (t == 1 || t == 2) ? 2 : 0; }

struct TT {
    std::string name;
    uint32_t type = 0;
    uint64_t dim[4] = {1, 1, 1, 1};
    uint64_t off = 0;
};
size_t tt_bytes(const TT & x) {
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) x.dim[i];
    if (is_q4_1(x.type)) return (size_t) x.dim[1] * x.dim[2] * x.dim[3] * ((size_t) x.dim[0] / 32) * 20;
    return n * plain_esz(x.type);
}

// IEEE half → float (subnormals included)
float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else {                                        // subnormal: renormalize
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        bits = sign | 0x7F800000u | (m << 13);
    } else {
        bits = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

// safetensors: { "<key>": {"dtype":"F16","shape":[..],"data_offsets":[a,b]} }
struct SEnt { long long a = -1, b = -1; std::vector<long long> shape; };
bool st_load(const char * path, std::map<std::string, SEnt> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    uint64_t hn = 0;
    if (fread(&hn, 8, 1, f) != 1) { fclose(f); return false; }
    std::string hdr((size_t) hn, '\0');
    bool ok = fread(hdr.data(), 1, (size_t) hn, f) == hn;
    fclose(f);
    if (!ok) return false;
    size_t i = 0;
    while ((i = hdr.find("\"layers.", i)) != std::string::npos) {
        size_t ke = hdr.find("\":", i + 1);
        size_t vo = hdr.find('{', ke);
        size_t vc = ke == std::string::npos || vo == std::string::npos ? std::string::npos : hdr.find('}', vo);
        if (vc == std::string::npos) break;
        std::string key = hdr.substr(i + 1, ke - i - 1);
        std::string o = hdr.substr(vo, vc - vo + 1);
        SEnt e;
        size_t doff = o.find("\"data_offsets\"");
        size_t lb = doff == std::string::npos ? std::string::npos : o.find('[', doff);
        size_t rb = lb == std::string::npos ? std::string::npos : o.find(']', lb);
        if (rb == std::string::npos) { i = vc; continue; }
        e.a = strtoll(o.c_str() + lb + 1, nullptr, 10);
        const char * p = o.c_str() + lb + 1;
        while (*p && *p != ',') p++;
        if (*p == ',') e.b = strtoll(p + 1, nullptr, 10);
        size_t sp = o.find("\"shape\""), slb = sp == std::string::npos ? std::string::npos : o.find('[', sp),
               srb = slb == std::string::npos ? std::string::npos : o.find(']', slb);
        if (srb != std::string::npos) {
            std::string nums = o.substr(slb + 1, srb - slb - 1);
            const char * q = nums.c_str();
            while (*q) {
                char * r; long long v = strtoll(q, &r, 10);
                if (r != q) e.shape.push_back(v);
                q = r; while (*q == ',' || *q == ' ') q++;
            }
        }
        out[key] = e;
        i = vc;
    }
    return !out.empty();
}

bool grab_f16(FILE * f, uint64_t data_base, const SEnt & e, std::vector<float> & dst) {
    size_t n = 1;
    for (auto d : e.shape) n *= (size_t) d;
    dst.resize(n);
    if (_fseeki64(f, (long long) (data_base + e.a), SEEK_SET) != 0) return false;
    std::vector<uint16_t> raw(n);
    if (fread(raw.data(), 2, n, f) != n) return false;
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(raw[i]);
    return true;
}

// ── Runtime state ──
struct Head { std::vector<float> fc1, fc2, lin; };   // [512][2560] / [256][512] / [256][2560]
struct Hist { std::vector<float> cur, prev; };        // owner-layer two-step onehot (E)

struct Pref {
    bool active = false;
    std::map<int, Head> heads;                        // owner → head triple
    std::map<int, std::vector<float>> gate;           // consumer → [E][H]
    struct EXP { std::vector<TT> t; };               // consumer → up/gate/down exps metadata
    std::map<int, EXP> exps;
    std::map<int, Hist> hist;                         // layers 6..39 (39 keeps teacher onehot only)
    std::map<int, std::vector<int>> pred_prev;        // consumer → previous-step head top-K
    std::map<int, std::vector<int>> exec_prev;        // consumer → previous-step teacher (sticky)
    uint64_t data_base = 0;
    HANDLE fh = INVALID_HANDLE_VALUE, mh = NULL;
    void * view = nullptr;
    bool do_prefetch = false;
    int  trace_every = 32;
    int  stride = 1;          // full head/teacher recompute period (E0_PREF_STRIDE); skipped steps only re-issue prefetch
    bool async = false;       // E0_PREF_ASYNC=0 falls back to synchronous (tax-splitting A/B)
    // Predicted-range VirtualLock: pinned bytes are not trimmed while under budget;
    // lock failure once the cap is hit = natural ceiling. NOTE: VirtualLock normally
    // requires SeLockMemoryPrivilege, absent on consumer machines — treat this as best-effort.
    bool   do_pin = false;
    size_t pin_cap = 3ull << 30, pin_bytes = 0;
    std::map<int, std::vector<std::pair<size_t, size_t>>> pinned;   // consumer → currently pinned (file_off,len)
    // Self-managed L1 over the mmap view (E0_POOL): re-touch at step boundaries, in-table
    // age-based LRU; "evict" = drop from table (stop re-touching; reference-bit aging lets
    // the OS reclaim to standby — a free soft layer, no privileged APIs involved).
    int  pool = 0;                                    // 0=legacy advisory 1=sticky 2=sticky+observation
    struct Pin { size_t off, len; int age; uint8_t reg; };
    std::map<int, std::vector<Pin>> pins;             // consumer → pin table (age = steps since last hit)
    size_t reg_bytes[2] = {0, 0}, reg_cap[2] = {0, 0};   // 0=hot zone (teacher) 1=observation zone (head)
    long pool_hit = 0, pool_tot = 0, lock_fail = 0, evicts = 0, shrinks = 0;
    // Async: m_in row-snapshot mailbox ((NH+1)×H floats, contiguous); worker runs full steps.
    std::vector<float> snap;
    std::atomic<bool> busy{false}, quit{false};
    std::mutex wmtx;
    std::condition_variable cv;
    std::thread worker;
    long hits = 0, sticky = 0, rnd = 0, total = 0, steps = 0;
    long long pool_mb = 0;                        // private-page pool hot-zone MB (>0 enables; E0_POOL_MB)
    uint32_t lcg = 2463534242u;
    uint32_t next() { lcg ^= lcg << 13; lcg ^= lcg >> 17; lcg ^= lcg << 5; return lcg; }
    ~Pref() {
        quit = true; cv.notify_all();
        if (worker.joinable()) worker.join();
        if (!total) return;
        fprintf(stderr, "\n[pref-trace] head   overlap = %ld/%ld = %.1f%%\n", hits, total, 100.0 * hits / total);
        fprintf(stderr, "[pref-trace] sticky overlap = %ld/%ld = %.1f%%\n", sticky, total, 100.0 * sticky / total);
        fprintf(stderr, "[pref-trace] random overlap = %ld/%ld = %.1f%%\n", rnd, total, 100.0 * rnd / total);
        fprintf(stderr, "[pref-trace] head−sticky = %.1f pp | head−random = %.1f pp\n",
                100.0 * (hits - sticky) / total, 100.0 * (hits - rnd) / total);
        e0_pool_trace_final();
        if (pool)
            fprintf(stderr, "[pref-trace] POOL(%s) hit = %ld/%ld = %.1f%% | pins hot=%lluMB obs=%lluMB (cap %llu/%llu) evict=%ld lockfail=%ld shrink=%ld\n",
                    pool == 2 ? "sticky+obs" : "sticky", pool_hit, pool_tot, pool_tot ? 100.0 * pool_hit / pool_tot : 0.0,
                    (unsigned long long) (reg_bytes[0] >> 20), (unsigned long long) (reg_bytes[1] >> 20),
                    (unsigned long long) (reg_cap[0] >> 20), (unsigned long long) (reg_cap[1] >> 20),
                    evicts, lock_fail, shrinks);
        fflush(stderr);
    }
};
Pref g;

float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865475f)); }
float dotf(const float * __restrict a, const float * __restrict b, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];   // /arch:AVX2 auto-FMA reduction; the bottleneck is weight streaming, not ALU
    return s;
}
void topk(const float * v, int n, int k, std::vector<int> & out) {
    out.clear();
    std::vector<char> used(n, 0);
    for (int j = 0; j < k; ++j) {
        int best = -1;
        for (int i = 0; i < n; ++i) if (!used[i] && (best < 0 || v[i] > v[best])) best = i;
        if (best < 0) break;
        used[best] = 1; out.push_back(best);
    }
}
int ovlp(const std::vector<int> & a, const std::vector<int> & b) {
    int n = 0;
    for (int x : a) for (int y : b) if (x == y) { ++n; break; }
    return n;
}

}  // namespace

// ══ Private-page expert pool (E0_POOL_MB): L1 with fill-once slots ══
// Design: decode reads expert bytes **directly from pool pages** via the expert-base
// resolver (patch surface #7), bypassing the view page-fault path entirely — a hit is a
// plain read with no kernel involvement. Slots fill once and never recycle (no eviction
// ⇒ no read-vs-overwrite race, correctness by construction); once the budget is
// exhausted, new experts pass through to the mmap view = graceful degradation to the
// un-pooled behavior. Three acquisition channels:
//   demand = synchronous whole-block ReadFile on the compute thread (first touch)
//   hot    = worker prefill from the teacher set
//   obs    = dedicated quota for head predictions
// Whole-block reads are also fragmentation-proof (1×1MB sequential vs a 256×4K fault
// storm — an order of magnitude apart on consumer SSDs).
static_assert(sizeof(LONG) == 4, "slot state atoms");
namespace {

struct PoolTT {                       // one 3D experts tensor (40 slots measured across all layers)
    uint8_t * arena = nullptr;        // MEM_RESERVE, MEM_COMMIT on demand
    volatile LONG * st = nullptr;     // slot state: 0=empty 1=filling 2=ready 3=not-poolable
    CRITICAL_SECTION * cvlk = nullptr;// per-slot pair: CS lock + CV (filler owns it, readers sleep until woken)
    CONDITION_VARIABLE * cv = nullptr;
    size_t per = 0;                   // bytes per expert
    uint64_t foff = 0;                // file offset of expert 0
};
static volatile LONG g_p_sc_ok = 0, g_p_sc_bad = 0;      // byte-level self-check (fill vs view memcmp sampling)
static bool g_pool_sc = false;                           // E0_POOL_SELFCHECK=1 enables
static void e0_pool_selfcheck(uint8_t * slot, uint64_t foff, size_t per) {
    if (!g_pool_sc || !g.view) return;
    const uint8_t * v = (const uint8_t *) g.view + foff;
    int bad = 0;
    for (size_t i = 0; i < per; i += 4096) if (memcmp(slot + i, v + i, 64) != 0) { bad = 1; break; }
    if (bad) InterlockedIncrement(&g_p_sc_bad); else InterlockedIncrement(&g_p_sc_ok);
}
// ⚠ The tensor *pointer* changes every forward (same tensor registered multiple times,
// measured) ⇒ key by tensor NAME (table built at init, read-only afterwards → lock-free).
std::map<std::string, PoolTT *> g_ptt_by_name;                  // GGUF name → PTT
std::map<int, std::vector<PoolTT *>> g_ptt_by_c;                // consumer → 3 PTTs (for prefill)
std::mutex g_pmtx;                                              // fill/accounting lock (fills are ms-scale, not a spin hotspot)
size_t g_pool_cap[2] = {0, 0};                                  // 0=hot 1=obs (E0_POOL_MB/E0_POOL_OBS_MB)
size_t g_pool_bytes[2] = {0, 0};
volatile LONG g_p_calls = 0;
bool g_pool_on = false;
volatile LONG g_p_hits = 0, g_p_miss_d = 0, g_p_bypass = 0, g_p_waits = 0, g_p_fills = 0;

// Expert-base resolver hook from the patch band (exported by ggml.dll, C decoration).
typedef const char * (*ggml_edge0_mmid_base_fn)(const struct ggml_tensor * src0, int64_t id, size_t nb02);
extern "C" __declspec(dllimport) void ggml_edge0_set_mmid_resolver(ggml_edge0_mmid_base_fn fn);

// Resolver callback (patch surface #7, runs on ggml compute threads): returns the base
// address of expert `id`'s data.
extern "C" static const char * e0_mmid_base_res(const ggml_tensor * s, int64_t id, size_t nb02) {
    LONG c = InterlockedIncrement(&g_p_calls);
    if (c <= 2) { fprintf(stderr, "[pref-trace] POOL2 call#%ld name=%s nb02=%zu\n", c, s->name ? s->name : "?", nb02); fflush(stderr); }
    const char * orig = (const char *) s->data + (size_t) id * nb02;
    auto it = g_ptt_by_name.find(s->name ? s->name : "");
    if (it == g_ptt_by_name.end() || !it->second->st || it->second->per != nb02) {
        LONG b = InterlockedIncrement(&g_p_bypass);
        if (b <= 6)
            fprintf(stderr, "[pref-trace] POOL2 miss#%ld name=%s nb02=%zu in_table=%d table=%zu\n",
                    b, s->name ? s->name : "?", nb02, (int) (it != g_ptt_by_name.end()), g_ptt_by_name.size());
        return orig;
    }
    PoolTT * pt = it->second;
    for (;;) {
        LONG st = pt->st[id];
        if (st == 1) {                                        // another thread is filling: sleep until woken (no spinning)
            InterlockedIncrement(&g_p_waits);
            CRITICAL_SECTION & lk = pt->cvlk[id];
            EnterCriticalSection(&lk);
            if (pt->st[id] == 1) SleepConditionVariableCS(&pt->cv[id], &lk, 2000);
            LeaveCriticalSection(&lk);
            continue;
        }
        if (st == 2) {
            LONG h = InterlockedIncrement(&g_p_hits);
            if (h <= 3 || (h & 0x3FFF) == 0)
                fprintf(stderr, "[pref-trace] POOL2 hit=%ld fillD=%ld bypass=%ld wait=%ld sc=%ld/%ld | hot=%lluMB obs=%lluMB\n",
                        h, g_p_miss_d, g_p_bypass, g_p_waits, (long) g_p_sc_ok, (long) g_p_sc_bad,
                        (unsigned long long) (g_pool_bytes[0] >> 20), (unsigned long long) (g_pool_bytes[1] >> 20));
            fflush(stderr);
            return (const char *) (pt->arena + (size_t) id * pt->per);
        }
        if (st == 3) { InterlockedIncrement(&g_p_bypass); return orig; }
        {                                                                   // claim the fill: CAS + exclusive CV section
            CRITICAL_SECTION & lk = pt->cvlk[id];
            if (InterlockedCompareExchange(&pt->st[id], 1, 0) != 0) continue;
            EnterCriticalSection(&lk);
        }
        LONG fs = InterlockedIncrement(&g_p_fills);
        if (fs <= 3 || (fs & 0x3FF) == 0) {
            fprintf(stderr, "[pref-trace] POOL2 fill#%ld name=%s id=%lld per=%zu hits=%ld\n",
                    fs, s->name ? s->name : "?", (long long) id, pt->per, g_p_hits);
            fflush(stderr);
        }
        // Demand fill on this thread (hot budget).
        bool ok = false;
        {
            std::lock_guard<std::mutex> lk(g_pmtx);
            if (g_pool_bytes[0] + pt->per <= g_pool_cap[0]) {
                uint8_t * dst = pt->arena + (size_t) id * pt->per;
                if (VirtualAlloc(dst, pt->per, MEM_COMMIT, PAGE_READWRITE)) {
                    uint64_t fo = pt->foff + (uint64_t) id * pt->per;
                    LARGE_INTEGER off; off.QuadPart = (LONGLONG) fo;
                    ok = SetFilePointerEx(g.fh, off, nullptr, FILE_BEGIN)
                         && ReadFile(g.fh, dst, (DWORD) pt->per, nullptr, nullptr);
                    if (ok) { g_pool_bytes[0] += pt->per; e0_pool_selfcheck(dst, fo, pt->per); }
                }
            }
            if (ok) InterlockedIncrement(&g_p_miss_d);
        }
        pt->st[id] = ok ? 2 : 3;
        WakeAllConditionVariable(&pt->cv[id]);
        LeaveCriticalSection(&pt->cvlk[id]);
        return ok ? (const char *) (pt->arena + (size_t) id * pt->per) : orig;
    }
}

// Worker prefill (teacher=hot / head=obs): called outside the resolver path; same per-slot CAS pattern.
static void e0_pool_prefill(int c, const std::vector<int> & ids, int reg) {
    auto it = g_ptt_by_c.find(c);
    if (it == g_ptt_by_c.end()) return;
    for (PoolTT * pt : it->second)
        for (int id : ids) {
            if (pt->st[id] != 0) continue;
            CRITICAL_SECTION & lk = pt->cvlk[id];
            if (InterlockedCompareExchange(&pt->st[id], 1, 0) != 0) continue;
            EnterCriticalSection(&lk);
            bool ok = false;
            {
                std::lock_guard<std::mutex> lk(g_pmtx);
                if (g_pool_bytes[reg] + pt->per <= g_pool_cap[reg]) {
                    uint8_t * dst = pt->arena + (size_t) id * pt->per;
                    if (VirtualAlloc(dst, pt->per, MEM_COMMIT, PAGE_READWRITE)) {
                        LARGE_INTEGER off = { (DWORD) (pt->foff + (uint64_t) id * pt->per),
                                              (LONG) ((pt->foff + (uint64_t) id * pt->per) >> 32) };
                        ok = SetFilePointerEx(g.fh, off, nullptr, FILE_BEGIN)
                             && ReadFile(g.fh, dst, (DWORD) pt->per, nullptr, nullptr);
                        if (ok) { g_pool_bytes[reg] += pt->per; e0_pool_selfcheck(dst, pt->foff + (uint64_t) id * pt->per, pt->per); }
                    }
                }
            }
            pt->st[id] = ok ? 2 : 3;
            WakeAllConditionVariable(&pt->cv[id]);
            LeaveCriticalSection(&lk);
        }
}

static void e0_pool_trace_final() {
    if (!g_pool_on) return;
    fprintf(stderr, "[pref-trace] POOL2 hit=%ld fillD=%ld bypass=%ld wait=%ld sc=%ld/%ld | hot=%lluMB obs=%lluMB\n",
            g_p_hits, g_p_miss_d, g_p_bypass, g_p_waits, (long) g_p_sc_ok, (long) g_p_sc_bad,
            (unsigned long long) (g_pool_bytes[0] >> 20), (unsigned long long) (g_pool_bytes[1] >> 20));
}

static void e0_pool_scan(const std::map<std::string, TT> & tinfo) {   // pool all *_exps tensors (independent of consumer shapes)
    for (auto & kv : tinfo) {
        const std::string & nm = kv.first;
        const TT & tt = kv.second;
        auto suff = [&](const char * sfx) -> size_t {           // exact suffix match (returns length on hit)
            size_t len = strlen(sfx), k = nm.rfind(sfx);
            return (k != std::string::npos && k + len == nm.size()) ? len : 0;
        };
        size_t l1 = suff(".ffn_up_exps.weight"), l2 = suff(".ffn_gate_exps.weight"), l3 = suff(".ffn_down_exps.weight");
        size_t sl = l1 ? l1 : (l2 ? l2 : l3);
        if (!sl || tt.dim[2] < 2 || tt.dim[2] > 4096) continue;
        const size_t per = tt_bytes(tt) / (size_t) tt.dim[2];
        if (!per) continue;
        PoolTT * e = new PoolTT();
        e->per = per;
        e->foff = g.data_base + tt.off;
        e->arena = (uint8_t *) VirtualAlloc(nullptr, per * (size_t) tt.dim[2], MEM_RESERVE, PAGE_READWRITE);
        e->st = (volatile LONG *) VirtualAlloc(nullptr, sizeof(LONG) * (size_t) tt.dim[2], MEM_COMMIT, PAGE_READWRITE);
        e->cvlk = new CRITICAL_SECTION[(size_t) tt.dim[2]];
        e->cv = new CONDITION_VARIABLE[(size_t) tt.dim[2]];
        for (size_t z = 0; z < (size_t) tt.dim[2]; ++z) { InitializeCriticalSection(&e->cvlk[z]); InitializeConditionVariable(&e->cv[z]); }
        if (!e->arena || !e->st) { delete[] e->cvlk; delete[] e->cv; e->cvlk = nullptr; e->cv = nullptr; delete e; continue; }
        g_ptt_by_name[nm] = e;
    }
    g_pool_on = !g_ptt_by_name.empty();
    if (g_pool_on) ggml_edge0_set_mmid_resolver(e0_mmid_base_res);   // attach to patch surface #7
    { int k = 0; for (auto & q : g_ptt_by_name) if (k++ < 3) fprintf(stderr, "[pref-trace] POOL2 key%d=%s per=%zu\n", k, q.first.c_str(), q.second->per); }
    fprintf(stderr, "[pref-trace] POOL2 init: tt=%zu caps hot=%lluMB obs=%lluMB resolver=%s\n",
            g_ptt_by_name.size(), (unsigned long long) (g_pool_cap[0] >> 20),
            (unsigned long long) (g_pool_cap[1] >> 20), g_pool_on ? "on" : "FAIL");
    fflush(stderr);
}

}  // namespace

static void enable_capture(llama_context * ctx) {
    for (int L = OWNER0; L <= OWNER0 + NH; ++L) llama_set_embeddings_layer_inp(ctx, (uint32_t) L, true);
}

// ── Pool re-touch ladder: deterministic accounting (zero prediction = sticky; + observation zone = sticky/obs) ──
// Mechanism: VirtualLock needs SeLockMemoryPrivilege, which consumer machines lack ⇒ pinning
// is NOT the product mechanism. What works unprivileged = demand touch + working-set hard cap
// + the OS clock's natural LRU:
//   "pin"   = worker volatile-reads the range in 4K steps (demands pages into the WS; any
//             trimmed page's fault tax is paid inside the worker, off the critical path);
//   "evict" = drop from the table only (stop re-touching → reference-bit ages → OS trims it
//             into standby = free soft layer);
//   "cap"   = at init, SetProcessWorkingSetSizeEx(MAX = baseline + hot + obs + headroom, HARDWS_MAX_ENABLE).
// Hot zone = re-touch of the previous step's teacher ids (same experts in a row = zero-fault
// hit); observation zone = head-guess quota (a bad guess only ages its own pages — the hot
// zone re-touches every step and never ages = contamination ceiling). Low-memory notification
// ⇒ shrink the tables (hot to 75%, observation to empty).
static volatile char g_touch_sink;                  // prevent DCE (every page is really read)
static void e0_touch(const void * view, size_t off, size_t len) {
    const volatile char * p = (const volatile char *) view + off;
    char s = 0;
    for (size_t i = 0; i < len; i += 4096) s += p[i];
    g_touch_sink = s;
}
static void e0_pool_evict(int R) {                  // evict the oldest entry in region R (age==0 entries are exempt)
    int wc = -1, wi = -1, ba = 0;
    for (auto & pc : g.pins)
        for (int k = 0; k < (int) pc.second.size(); ++k) {
            auto & p = pc.second[k];
            if (p.reg == R && p.age > ba) { ba = p.age; wc = pc.first; wi = k; }
        }
    if (wc < 0) return;
    g.reg_bytes[R] -= g.pins[wc][wi].len; ++g.evicts;
    g.pins[wc].erase(g.pins[wc].begin() + wi);      // dropping from the table = stop re-touching; physical reclaim is the OS clock's job
}
static void e0_issue_pool() {
    if (!g.view) return;
    const int nreg = g.pool == 2 ? 2 : 1;
    std::map<int, std::vector<std::pair<size_t, size_t>>> want[2];   // 0=hot(teacher) 1=observation(head)
    auto ranges = [&](const std::map<int, std::vector<int>> & src, int R) {
        for (auto & pc : src) {
            auto & v = want[R][pc.first];
            for (const TT & tt : g.exps[pc.first].t) {
                const size_t per = tt_bytes(tt) / EK;
                for (int e : pc.second)
                    v.push_back({ (size_t) g.data_base + tt.off + (size_t) e * per, per });
            }
        }
    };
    ranges(g.exec_prev, 0);
    if (nreg > 1) {
        std::map<int, std::vector<int>> preds;
        { std::lock_guard<std::mutex> lk(g.wmtx); preds = g.pred_prev; }
        ranges(preds, 1);
    }
    for (auto & kv : g.pins) for (auto & p : kv.second) ++p.age;     // uniform step-rate aging
    std::vector<std::pair<size_t, size_t>> touch;                     // this step's demand set (≤ table cap)
    for (int R = 0; R < nreg; ++R) {
        for (auto & wv : want[R]) {
            auto & cur = g.pins[wv.first];
            std::vector<char> hit(wv.second.size(), 0);
            for (size_t i = 0; i < wv.second.size(); ++i)
                for (auto & p : cur)
                    if (p.reg == R && p.off == wv.second[i].first) { p.age = 0; hit[i] = 1; if (!R) ++g.pool_hit; break; }
            if (!R) g.pool_tot += (long) wv.second.size();           // denominator = all hot-zone demand this step
            for (size_t i = 0; i < wv.second.size(); ++i) {
                if (hit[i]) continue;
                auto & w = wv.second[i];
                while (g.reg_bytes[R] + w.second > g.reg_cap[R]) {
                    size_t b = g.reg_bytes[R];
                    e0_pool_evict(R);
                    if (g.reg_bytes[R] == b) { ++g.lock_fail; break; }  // nothing evictable ⇒ hard ceiling (refused-pin counter)
                }
                if (g.reg_bytes[R] + w.second > g.reg_cap[R]) { ++g.lock_fail; continue; }
                cur.push_back({ w.first, w.second, 0, (uint8_t) R });
                g.reg_bytes[R] += w.second;
            }
        }
    }
    for (auto & pc : g.pins) for (auto & p : pc.second) e0_touch(g.view, p.off, p.len);  // re-touch the whole table
    typedef HANDLE(WINAPI * QMRN)(LPCWSTR, void *);                   // memory-pressure cooperation (table-side fast shrink)
    typedef BOOL(WINAPI * GMRN_T)(HANDLE, PBOOL);
    static QMRN qmrn = (QMRN) GetProcAddress(GetModuleHandleA("kernel32"), "QueryMemoryResourceNotification");
    static GMRN_T gmrn = (GMRN_T) GetProcAddress(GetModuleHandleA("kernel32"), "GetMemoryResourceNotification");
    static HANDLE hn = qmrn ? qmrn(L"MemoryLow", nullptr) : nullptr;
    if (hn && gmrn) {
        BOOL low = FALSE;
        if (gmrn(hn, &low) && low) {
            ++g.shrinks;
            for (int R = 0; R < nreg; ++R)
                while (g.reg_bytes[R] && (R || g.reg_bytes[R] * 4 > g.reg_cap[R] * 3)) {
                    size_t b = g.reg_bytes[R];
                    e0_pool_evict(R);
                    if (g.reg_bytes[R] == b) break;
                }
        }
    }
}

// ── Advisory issue: PVM soft warm-up + VirtualLock hard retention (lock failure once the
// budget is hit = natural ceiling). Callable from the worker thread or the main thread
// (synchronous / stride-skipped); pred_prev is copied under a short wmtx lock.
static void e0_issue() {
    if (g.pool) { e0_issue_pool(); return; }          // pool accounting takes over (legacy pin/prefetch path retired)
    if (!g.view || (!g.do_prefetch && !g.do_pin)) return;
    std::map<int, std::vector<int>> preds;
    { std::lock_guard<std::mutex> lk(g.wmtx); preds = g.pred_prev; }
    static WIN32_MEMORY_RANGE_ENTRY win[4 * 3 * NH];
    int nw = 0;
    for (auto & pc : preds) {
        const int c = pc.first;
        std::vector<std::pair<size_t, size_t>> want;
        for (const TT & tt : g.exps[c].t) {
            const size_t per = tt_bytes(tt) / EK;
            for (int e : pc.second) want.push_back({ (size_t) g.data_base + tt.off + (size_t) e * per, per });
        }
        if (g.do_pin) {
            auto & cur = g.pinned[c];
            for (auto & r : cur) {                                   // release stale pins
                bool keep = false;
                for (auto & w : want) if (w.first == r.first) { keep = true; break; }
                if (!keep) { VirtualUnlock((char *) g.view + r.first, r.second); g.pin_bytes -= r.second; }
            }
            for (auto & w : want) {                                  // pin new predictions (skipped once capped)
                bool have = false;
                for (auto & r : cur) if (r.first == w.first) { have = true; break; }
                if (have || g.pin_bytes + w.second > g.pin_cap) continue;
                if (VirtualLock((char *) g.view + w.first, w.second)) g.pin_bytes += w.second;
            }
            cur = want;   // remember current pin state (entries skipped by the cap unlock harmlessly next round)
        }
        if (g.do_prefetch)
            for (auto & w : want) {
                if (nw == 4 * 3 * NH) break;
                win[nw].VirtualAddress = (PVOID) ((char *) g.view + w.first);
                win[nw].NumberOfBytes  = w.second;
                ++nw;
            }
    }
    if (nw) {
        static decltype(&PrefetchVirtualMemory) pvm =
            (decltype(&PrefetchVirtualMemory)) GetProcAddress(GetModuleHandleA("kernel32"), "PrefetchVirtualMemory");
        if (pvm) pvm(GetCurrentProcess(), nw, win, 0);
    }
}

// ── Full-step computation: (1) teacher (2) three-way accounting (3) head → preds + roll history (4) issue ──
// rows = snapshot buffer, row order = layers OWNER0..OWNER0+NH, HH floats per row.
// This function has no ctx dependency ⇒ it can run on the worker thread (keeps the ~90ms
// head GEMV off the critical path).
static void e0_full_step(const float * rows) {
    std::map<int, std::vector<int>> teacher;
    for (int c : consumers) {
        const float * x = rows + (size_t) (c - OWNER0) * HH;
        std::vector<float> tl(EK);
        const float * gw = g.gate[c].data();
        for (int e = 0; e < EK; ++e) tl[e] = dotf(gw + (size_t) e * HH, x, HH);
        topk(tl.data(), EK, KDIM, teacher[c]);
    }
    // Three-way settlement: pred copy (short lock) vs teacher @ this snapshot.
    std::map<int, std::vector<int>> preds_copy;
    { std::lock_guard<std::mutex> lk(g.wmtx); preds_copy = g.pred_prev; }
    for (int c : consumers) {
        auto pp = preds_copy.find(c);
        if (pp != preds_copy.end()) {
            g.hits += ovlp(pp->second, teacher[c]); g.total += KDIM;
            auto sp = g.exec_prev.find(c);
            if (sp != g.exec_prev.end()) g.sticky += ovlp(sp->second, teacher[c]);
            std::vector<int> r;
            for (int tries = 0; tries < 64 && (int) r.size() < KDIM; ++tries) {
                int v = (int) (g.next() % (uint32_t) EK);
                bool dup = false; for (int y : r) if (y == v) { dup = true; break; }
                if (!dup) r.push_back(v);
            }
            g.rnd += ovlp(r, teacher[c]);
        }
        g.exec_prev[c] = teacher[c];
    }
    if (g_pool_on) for (auto & kv : g.exec_prev) e0_pool_prefill(kv.first, kv.second, 0);   // teacher → hot zone
    // head: owner o features = [m_in|cur_oh|prev_oh] → new pred (consumer = o+1)
    std::map<int, std::vector<int>> new_preds;
    std::vector<float> feat(FE), act(512), lg(EK);
    for (int o = OWNER0; o < OWNER0 + NH; ++o) {
        const int c = o + 1;
        auto & h = g.hist[o];
        memcpy(feat.data(),          rows + (size_t) (o - OWNER0) * HH, HH * 4);
        memcpy(feat.data() + HH,     h.cur.data(),  EK * 4);
        memcpy(feat.data() + HH + EK, h.prev.data(), EK * 4);
        const Head & hd = g.heads[o];
        for (int j = 0; j < 512; ++j) act[j] = gelu_erf(dotf(hd.fc1.data() + (size_t) j * FE, feat.data(), FE));
        for (int e = 0; e < EK; ++e)
            lg[e] = dotf(hd.fc2.data() + (size_t) e * 512, act.data(), 512)
                  + dotf(hd.lin.data() + (size_t) e * FE, feat.data(), FE);
        topk(lg.data(), EK, KDIM, new_preds[c]);
    }
    if (g_pool_on && g_pool_cap[1]) for (auto & kv : new_preds) e0_pool_prefill(kv.first, kv.second, 1);  // head → observation zone
    { std::lock_guard<std::mutex> lk(g.wmtx); g.pred_prev = std::move(new_preds); }
    // roll the onehot history (teacher→cur, cur→prev)
    for (auto & kv : g.hist) {
        auto t = teacher.find(kv.first);
        if (t == teacher.end()) continue;
        kv.second.prev = kv.second.cur;
        kv.second.cur.assign(EK, 0.f);
        for (int e : t->second) kv.second.cur[e] = 1.f;
    }
    e0_issue();
    if (g.trace_every && g.steps % g.trace_every == 0 && g.total) {
        fprintf(stderr, "[pref-trace] step %ld: head %.1f%% sticky %.1f%% rand %.1f%% pinned=%lluMB (n=%ld)\n",
                g.steps, 100.0 * g.hits / g.total, 100.0 * g.sticky / g.total,
                100.0 * g.rnd / g.total, (unsigned long long) (g.pin_bytes >> 20), g.total);
        if (g.pool)
            fprintf(stderr, "[pref-trace]  POOL hit=%.1f%% (n=%ld) pins %llu+%lluMB evict=%ld fail=%ld shrink=%ld\n",
                    100.0 * g.pool_hit / (g.pool_tot ? g.pool_tot : 1), g.pool_tot,
                    (unsigned long long) (g.reg_bytes[0] >> 20), (unsigned long long) (g.reg_bytes[1] >> 20),
                    g.evicts, g.lock_fail, g.shrinks);
        fflush(stderr);
    }
}

static void e0_worker() {
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g.wmtx);
            g.cv.wait(lk, [&] { return g.quit.load() || g.busy.load(); });
            if (g.quit) return;
        }
        if (g.busy.load()) {
            e0_full_step(g.snap.data());
            g.busy = false;
        }
    }
}

bool edge0_pref_init(llama_model * model, llama_context * ctx, int pool_mb, const char * gguf_path) {
    (void) model;
    if (g.active) { if (!g.gate.empty()) enable_capture(ctx); return true; }   // hot reload (only head mode needs capture)
    const char * side = getenv("E0_PREROUTER");
    const char * gg   = (gguf_path && *gguf_path) ? gguf_path : getenv("E0_GGUF");   // primary CLI channel = model path
    if (pool_mb > 0) g.pool_mb = pool_mb;                                        // --pool-mb formal CLI
    else if (const char * pmb = getenv("E0_POOL_MB")) g.pool_mb = atoll(pmb);    // env fallback (harness/debug); runnable without the sidecar
    const bool headson = side && *side;
    if ((!headson && !g.pool_mb) || !gg || !*gg) return false;

    if (consumers.empty()) for (int c = OWNER0 + 1; c <= OWNER0 + NH; ++c) consumers.push_back(c);  // 7..39

    // ── GGUF header: skip-scan KVs (for general.alignment) + build the tensor table ──
    FILE * f = fopen(gg, "rb");
    if (!f) return false;
    struct { uint32_t magic, ver; uint64_t nt, nkv; } h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(&h.magic, "GGUF", 4)) { fclose(f); return false; }
    size_t alignment = 32;
    for (uint64_t i = 0; i < h.nkv; ++i) {
        std::string k = rd_str(f);
        uint32_t t; if (fread(&t, 4, 1, f) != 1) { fclose(f); return false; }
        if (k == "general.alignment" && t == 4) fread(&alignment, 4, 1, f);
        else skip_val(f, t);
    }
    std::map<std::string, TT> tinfo;
    for (uint64_t i = 0; i < h.nt; ++i) {
        TT x; x.name = rd_str(f);
        uint32_t nd; fread(&nd, 4, 1, f);
        for (uint32_t d = 0; d < nd && d < 4; ++d) fread(&x.dim[d], 8, 1, f);
        fread(&x.type, 4, 1, f); fread(&x.off, 8, 1, f);
        tinfo[x.name] = x;
    }
    g.data_base = align_up((uint64_t) ftell(f), alignment);

    // Register exps metadata (needed by both pool and head) + copy the F32 gate weights host-side (head mode only).
    for (int c : consumers) {
        for (const char * suf : {"ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"}) {
            auto et = tinfo.find("blk." + std::to_string(c) + "." + suf + ".weight");
            if (et != tinfo.end() && et->second.dim[2] == (uint64_t) EK) g.exps[c].t.push_back(et->second);
        }
        if (!headson) continue;
        auto it = tinfo.find("blk." + std::to_string(c) + ".ffn_gate_inp.weight");
        if (it == tinfo.end() || it->second.type != 0) {
            fclose(f); fprintf(stderr, "[pref-trace] gate missing/bad type c=%d => dormant\n", c); return false;
        }
        const TT & gt = it->second;
        size_t nb = tt_bytes(gt);
        std::vector<float> buf(nb / 4);
        _fseeki64(f, (long long) (g.data_base + gt.off), SEEK_SET);
        if (fread(buf.data(), 1, nb, f) != nb) { fclose(f); return false; }
        g.gate[c] = std::move(buf);
    }
    fclose(f);

    // sidecar head weights (head mode only)
    if (headson) {
    std::map<std::string, SEnt> st;
    if (!st_load(side, st)) { fprintf(stderr, "[pref-trace] sidecar parse failed => dormant\n"); return false; }
    FILE * sf = fopen(side, "rb");
    if (!sf) return false;
    uint64_t shn = 0; fread(&shn, 8, 1, sf);
    uint64_t sdata = 8 + shn;
    for (int o = OWNER0; o < OWNER0 + NH; ++o) {
        Head hd;
        auto key = [&](const char * w) { return "layers." + std::to_string(o) + "." + w + ".weight"; };
        bool ok = grab_f16(sf, sdata, st[key("fc1")], hd.fc1)
               && grab_f16(sf, sdata, st[key("fc2")], hd.fc2)
               && grab_f16(sf, sdata, st[key("linear_init")], hd.lin);
        if (!ok) { fclose(sf); fprintf(stderr, "[pref-trace] sidecar missing layer o=%d => dormant\n", o); return false; }
        g.heads[o] = std::move(hd);
    }
    fclose(sf);
    }   // headson

    // Dedicated read handle (pool fills via g.fh) + a self-owned read-only mapping (PVM anchor; head mode only).
    if (headson || g.pool_mb) {
        g.fh = CreateFileA(gg, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (g.fh != INVALID_HANDLE_VALUE && headson) {
            g.mh = CreateFileMappingA(g.fh, NULL, PAGE_READONLY, 0, 0, NULL);
            if (g.mh) g.view = MapViewOfFile(g.mh, FILE_MAP_READ, 0, 0, 0);
        }
    }
    g.do_prefetch = getenv("E0_PREFETCH") != nullptr;
    if (const char * tn = getenv("E0_PREF_TRACE_N")) g.trace_every = atoi(tn);
    if (const char * sn = getenv("E0_PREF_STRIDE")) g.stride = atoi(sn) > 0 ? atoi(sn) : 1;
    const char * an = getenv("E0_PREF_ASYNC");
    g.async = !(an && *an == '0');                     // async is the default; =0 falls back to synchronous (tax isolation)
    g.do_pin = getenv("E0_PREF_PIN") != nullptr;
    if (const char * pm = getenv("E0_PIN_MAX_MB")) g.pin_cap = (size_t) atoll(pm) << 20;
    if (const char * pl = getenv("E0_POOL")) {                      // sticky=zone1  obs=zone1+2
        if (!strcmp(pl, "obs")) g.pool = 2;
        else if (*pl && strcmp(pl, "0")) g.pool = 1;
    }
    if (g.pool) {
        g.reg_cap[0] = g.pin_cap;                                   // hot zone = E0_PIN_MAX_MB (default 3G)
        const char * om = getenv("E0_POOL_OBS_MB");
        g.reg_cap[1] = (size_t) (om ? atoll(om) : 1024) << 20;      // observation zone default 1G
        g.do_pin = false;                                           // the advisory pin path does not run in parallel
        // Working-set hard cap = startup baseline WS + pool quotas + headroom (the residency contract without SeLockMemory).
        PROCESS_MEMORY_COUNTERS_EX pmc = {};
        pmc.cb = sizeof(pmc);
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
            const char * hm = getenv("E0_POOL_WSHEAD_MB");
            size_t head = (size_t) (hm ? atoll(hm) : 1024) << 20;
            SIZE_T wmax = (SIZE_T) pmc.WorkingSetSize + g.reg_cap[0] + g.reg_cap[1] + head;
            SIZE_T qmin = 0, qmax = 0; DWORD qf = 0;      // the Ex variant rejects min=-1 (error 87) ⇒ read current quotas and keep them
            GetProcessWorkingSetSizeEx(GetCurrentProcess(), &qmin, &qmax, &qf);
            BOOL r = SetProcessWorkingSetSizeEx(GetCurrentProcess(), qmin, wmax,
                                                QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE);
            fprintf(stderr, "[pref-trace] POOL ws-cap: base=%lluMB max=%lluMB set=%d err=%lu\n",
                    (unsigned long long) (pmc.WorkingSetSize >> 20), (unsigned long long) (wmax >> 20),
                    (int) r, r ? 0UL : GetLastError());
        }
    }
    g.snap.assign((size_t) (NH + 1) * HH, 0.f);

    // Open the layer-input capture channel (same staging API as speculative decode; head mode only).
    if (headson) {
        enable_capture(ctx);
        for (int L = OWNER0; L <= OWNER0 + NH; ++L) g.hist[L] = Hist{ std::vector<float>(EK, 0.f), std::vector<float>(EK, 0.f) };
    }
    if (g.pool_mb) {
        g_pool_cap[0] = (size_t) g.pool_mb << 20;
        if (const char * pom = getenv("E0_POOL_OBS_MB")) g_pool_cap[1] = (size_t) atoll(pom) << 20;
        g_pool_sc = getenv("E0_POOL_SELFCHECK") != nullptr;
        if (g_pool_sc && !g.view) {      // self-check needs the view: map it even in pool-only mode
            g.mh = CreateFileMappingA(g.fh, NULL, PAGE_READONLY, 0, 0, NULL);
            if (g.mh) g.view = MapViewOfFile(g.mh, FILE_MAP_READ, 0, 0, 0);
        }
        e0_pool_scan(tinfo);
        for (int c : consumers)
            for (const TT & tt : g.exps[c].t) {
                auto it = g_ptt_by_name.find(tt.name);
                if (it != g_ptt_by_name.end()) g_ptt_by_c[c].push_back(it->second);
            }
    }
    g.active = headson || g_pool_on;
    if (g.active && !headson) { g.async = false; }        // pool-only mode: no worker (demand is self-sufficient)
    if (g.active && headson && g.async) g.worker = std::thread(e0_worker);
    fprintf(stderr, "[pref-trace] INIT ok: heads=%zu gates=%zu exps=%zu map=%s prefetch=%s pin=%s(async=%d) stride=%d K=%d E=%d pool=%d(hot=%lluMB obs=%lluMB)\n",
            g.heads.size(), g.gate.size(), g.exps.size(), g.view ? "ok" : "FAIL",
            g.do_prefetch ? "on" : "off", g.do_pin ? "on" : "off", (int) g.async, g.stride, KDIM, EK,
            (int) g_pool_on, (unsigned long long) (g_pool_cap[0] >> 20), (unsigned long long) (g_pool_cap[1] >> 20));
    return g.active;
}

void edge0_pref_on_step(llama_context * ctx, int64_t last_row) {
    if (!g.active) return;
    if (g.gate.empty()) return;                 // pool-only mode: no capture channel, nothing to account — demand is autonomous
    llama_synchronize(ctx);
    ++g.steps;

    if (g.stride > 1 && (g.steps % g.stride) != 0) {   // downsampled: reuse existing preds, just re-issue (µs scale)
        if (!g.async && !g.pool) e0_issue();            // async=worker already issued on cadence; pool=skipped steps neither age nor evict
        return;
    }
    if (g.async) {
        if (g.busy.load()) return;                     // worker busy: drop this snapshot (preds already valid, lossless)
        float * dst = g.snap.data();
        for (int L = OWNER0; L <= OWNER0 + NH; ++L, dst += HH) {
            float * buf = llama_get_embeddings_layer_inp(ctx, (uint32_t) L);
            if (!buf) return;                          // channel not ready
            memcpy(dst, buf + last_row * HH, (size_t) HH * 4);
        }
        { std::lock_guard<std::mutex> lk(g.wmtx); g.busy = true; }
        g.cv.notify_one();
        return;
    }
    // Synchronous path: snapshot + full step inline (reference/fallback).
    float * dst = g.snap.data();
    for (int L = OWNER0; L <= OWNER0 + NH; ++L, dst += HH) {
        float * buf = llama_get_embeddings_layer_inp(ctx, (uint32_t) L);
        if (!buf) return;
        memcpy(dst, buf + last_row * HH, (size_t) HH * 4);
    }
    e0_full_step(g.snap.data());
}
