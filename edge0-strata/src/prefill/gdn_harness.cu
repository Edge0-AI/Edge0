// src/prefill/gdn_harness.cu - production-linked parity and benchmark harness for the prompt GDN path (CUDA).
//
// Links the real strata_prefill library and calls the real gdn_gates / gdn_conv / gdn_recurrence: there is not one
// copy of a kernel in this file.  Which recurrence runs is chosen through the GdnProbe (gdn_probe.hpp): a path can be
// forced, the path that actually ran and the reason a wanted path did not run are read back after every call, and a
// forced path that is not eligible launches nothing and is reported INELIGIBLE (it is never timed as a fallback).
// Fixtures and the FP64 oracle are tests/gdn/gdn_fixture.hpp (shared with the other backends' harnesses).
// Usage and output format: Desktop\30thOctober\results\gdn\HARNESS.md and docs/GDN_CONTRACT.md.
#include "gdn_fixture.hpp"
#include "strata/prefill/gdn_probe.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef GDN_HARNESS_SHA
#define GDN_HARNESS_SHA "unknown"
#endif

using namespace gdnfx;
namespace sp = strata::prefill;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> T* dupload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

struct Args {
    int64_t T = 4096;
    std::vector<int64_t> Ts;                  // --T a,b,c : several lengths (one case each)
    int64_t chunk = 2048;                     // tokens per engine call
    std::vector<int64_t> cuts;                // explicit first call sizes (unaligned splits), then `chunk`
    std::vector<int> paths;                   // default: auto + every forced path
    int incumbent = sp::kGdnKeyHead;          // the path others are compared with (falls to pipe where ineligible)
    bool parity = true, bench = false, ref = true;
    std::vector<int> ref_heads = {0, 17, 34, 5, 22, 47};
    int reps = 7, warmup = 2;
    bool any_sms = false;
    int gsb = 0;
    int tiles = 0;
    bool fallback_incumbent = false;          // a forced path takes the incumbent for calls it cannot run (mirrors production)
    int64_t ld16 = 0;
    FixtureSpec fx;
    std::string json;
    std::string label;
    bool stress = false;                      // run the stress fixtures (parity only)
};

std::vector<int64_t> parse_list(const char* s) {
    std::vector<int64_t> v;
    for (const char* p = s; *p;) {
        char* e = nullptr;
        long long x = std::strtoll(p, &e, 10);
        if (e == p) break;
        if (*e == 'k' || *e == 'K') { x *= 1024; ++e; }
        v.push_back(x);
        p = *e == ',' ? e + 1 : e;
    }
    return v;
}
int path_from_name(const std::string& n) {
    for (int p = 0; p < sp::kGdnPathCount; ++p)
        if (n == sp::gdn_path_name(p)) return p;
    return -2;
}

struct Dev {   // device-resident fixture
    int64_t T = 0;
    float *qkv = nullptr, *ab = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv_w = nullptr, *z = nullptr, *gamma = nullptr;
    float* state0 = nullptr;   // [S*HV*S state | C*3 history]
    float *gate = nullptr, *beta = nullptr, *h = nullptr, *y = nullptr, *state = nullptr;
    uint16_t* y16 = nullptr;
    int64_t ld16 = 0, chunk_max = 0;
    char* flush = nullptr;
    size_t flush_bytes = 256u << 20;
};
constexpr size_t kStateFloats = (size_t) S * HV * S;

Dev upload(const Fixture& f, int64_t chunk_max, int64_t ld16) {
    Dev d;
    d.T = f.T;
    d.chunk_max = chunk_max;
    d.ld16 = ld16 > 0 ? ld16 : (int64_t) HV * S;
    d.qkv = dupload(f.qkv);
    d.ab = dupload(f.ab);
    d.dt = dupload(f.dt);
    d.ssm_a = dupload(f.ssm_a);
    d.conv_w = dupload(f.conv_w);
    d.z = dupload(f.z);
    d.gamma = dupload(f.gamma);
    std::vector<float> st(kStateFloats + (size_t) C * 3);
    std::memcpy(st.data(), f.state0.data(), kStateFloats * 4);
    std::memcpy(st.data() + kStateFloats, f.hist.data(), (size_t) C * 3 * 4);
    d.state0 = dupload(st);
    d.state = dalloc<float>(st.size());
    d.gate = dalloc<float>((size_t) chunk_max * HV);
    d.beta = dalloc<float>((size_t) chunk_max * HV);
    d.h = dalloc<float>((size_t) chunk_max * C);
    d.y = dalloc<float>((size_t) chunk_max * HV * S);
    d.y16 = dalloc<uint16_t>((size_t) f.T * d.ld16 + 64);
    ck(cudaMemset(d.y16, 0xAB, ((size_t) f.T * d.ld16 + 64) * 2), "memset y16 sentinel");
    d.flush = dalloc<char>(d.flush_bytes);
    return d;
}
void release(Dev& d) {
    for (void* p : {(void*) d.qkv, (void*) d.ab, (void*) d.dt, (void*) d.ssm_a, (void*) d.conv_w, (void*) d.z, (void*) d.gamma,
                    (void*) d.state0, (void*) d.state, (void*) d.gate, (void*) d.beta, (void*) d.h, (void*) d.y,
                    (void*) d.y16, (void*) d.flush})
        cudaFree(p);
    d = Dev();
}

std::vector<int64_t> plan_calls(int64_t T, const Args& a) {
    std::vector<int64_t> v;
    int64_t left = T;
    for (int64_t c : a.cuts) {
        if (left <= 0) break;
        v.push_back(std::min(c, left));
        left -= v.back();
    }
    while (left > 0) { v.push_back(std::min(a.chunk, left)); left -= v.back(); }
    return v;
}

struct RunOut {
    int path = 0;
    bool eligible = true;
    std::string reason;
    std::set<std::string> logged;             // distinct "wanted->selected [reason]" over the calls
    int fallback_calls = 0;
    double ms_gates = 0, ms_conv = 0, ms_rec = 0;
    std::vector<uint16_t> y16;                // dense [T][HV*S]
    std::vector<float> state;
};

// Runs the layer pipeline for every call of the plan.  timed: events around each stage.
RunOut run_path(Dev& d, const Args& a, const std::vector<int64_t>& calls, int path, bool timed, bool download) {
    RunOut r;
    r.path = path;
    ck(cudaMemcpy(d.state, d.state0, (kStateFloats + (size_t) C * 3) * 4, cudaMemcpyDeviceToDevice), "reset state");
    float* conv_hist = d.state + kStateFloats;
    sp::GdnProbe probe;
    probe.any_sms = a.any_sms;
    probe.gsb = a.gsb;
    probe.tc_tiles = a.tiles;
    sp::gdn_probe_install(&probe);
    std::vector<cudaEvent_t> ev;
    if (timed) {
        ev.resize(calls.size() * 4);
        for (auto& e : ev) ck(cudaEventCreate(&e), "event");
    }
    int64_t t0 = 0;
    size_t ci = 0;
    for (int64_t T : calls) {
        if (timed) ck(cudaEventRecord(ev[ci * 4]), "ev");
        sp::gdn_gates(d.ab + t0 * 2 * HV, d.dt, d.ssm_a, d.gate, d.beta, T, nullptr);
        if (timed) ck(cudaEventRecord(ev[ci * 4 + 1]), "ev");
        sp::gdn_conv(conv_hist, d.qkv + t0 * C, d.conv_w, d.h, T, EPS, nullptr);
        if (timed) ck(cudaEventRecord(ev[ci * 4 + 2]), "ev");
        probe.force = path;
        sp::gdn_recurrence(d.state, d.h, d.gate, d.beta, d.z + t0 * HV * S, d.gamma, EPS, d.y, d.y16 + t0 * d.ld16, T, nullptr, d.ld16);
        if (probe.selected == sp::kGdnNone && a.fallback_incumbent && path != a.incumbent) {
            probe.force = a.incumbent;
            sp::gdn_recurrence(d.state, d.h, d.gate, d.beta, d.z + t0 * HV * S, d.gamma, EPS, d.y, d.y16 + t0 * d.ld16, T, nullptr, d.ld16);
            if (probe.selected == sp::kGdnNone) probe.force = sp::kGdnPipe, sp::gdn_recurrence(d.state, d.h, d.gate, d.beta, d.z + t0 * HV * S, d.gamma, EPS, d.y, d.y16 + t0 * d.ld16, T, nullptr, d.ld16);
            ++r.fallback_calls;
        }
        if (timed) ck(cudaEventRecord(ev[ci * 4 + 3]), "ev");
        if (probe.selected == sp::kGdnNone) {
            r.eligible = false;
            r.reason = std::string("call of ") + std::to_string((long long) T) + " tokens: " + probe.reason;
            break;
        }
        char b[200];
        std::snprintf(b, sizeof b, "%s->%s%s%s", sp::gdn_path_name(probe.wanted), sp::gdn_path_name(probe.selected),
                      probe.reason[0] ? " [" : "", probe.reason[0] ? probe.reason : "");
        std::string s = b;
        if (probe.reason[0]) s += "]";
        r.logged.insert(s);
        t0 += T;
        ++ci;
    }
    ck(cudaDeviceSynchronize(), "run_path sync");
    sp::gdn_probe_install(nullptr);
    if (timed && r.eligible)
        for (size_t i = 0; i < ci; ++i) {
            float x = 0;
            cudaEventElapsedTime(&x, ev[i * 4], ev[i * 4 + 1]); r.ms_gates += x;
            cudaEventElapsedTime(&x, ev[i * 4 + 1], ev[i * 4 + 2]); r.ms_conv += x;
            cudaEventElapsedTime(&x, ev[i * 4 + 2], ev[i * 4 + 3]); r.ms_rec += x;
        }
    for (auto e : ev) cudaEventDestroy(e);
    if (download && r.eligible) {
        r.y16.resize((size_t) d.T * HV * S);
        if (d.ld16 == (int64_t) HV * S) {
            ck(cudaMemcpy(r.y16.data(), d.y16, r.y16.size() * 2, cudaMemcpyDeviceToHost), "y16 down");
        } else {
            ck(cudaMemcpy2D(r.y16.data(), (size_t) HV * S * 2, d.y16, (size_t) d.ld16 * 2, (size_t) HV * S * 2, d.T, cudaMemcpyDeviceToHost), "y16 down 2d");
            // padding columns of a padded stride must be untouched (sentinel 0xABAB)
            std::vector<uint16_t> pad((size_t) d.T * d.ld16);
            ck(cudaMemcpy(pad.data(), d.y16, pad.size() * 2, cudaMemcpyDeviceToHost), "pad down");
            for (int64_t t = 0; t < d.T; ++t)
                for (int64_t c = HV * S; c < d.ld16; ++c)
                    if (pad[(size_t) t * d.ld16 + c] != 0xABAB) { r.eligible = false; r.reason = "PADDING OVERWRITTEN (stride contract broken)"; return r; }
        }
        r.state.resize(kStateFloats + (size_t) C * 3);
        ck(cudaMemcpy(r.state.data(), d.state, r.state.size() * 4, cudaMemcpyDeviceToHost), "state down");
    }
    return r;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

struct Cmp { double y_nmax = 0, y_rl2 = 0, s_nmax = 0, s_rl2 = 0; long nonfinite = 0; };

Cmp vs_ref(const RunOut& r, const RefResult& ref, int64_t T) {
    Cmp c;
    ErrStat ey, es;
    for (size_t n = 0; n < ref.heads.size(); ++n) {
        const int vh = ref.heads[n];
        for (int64_t t = 0; t < T; ++t)
            for (int col = 0; col < S; ++col)
                ey.add(half_to_float(r.y16[(size_t) t * HV * S + (size_t) vh * S + col]), ref.y[n][(size_t) t * S + col]);
        for (int i = 0; i < S; ++i)
            for (int col = 0; col < S; ++col) es.add(r.state[((size_t) i * HV + vh) * S + col], ref.state[n][(size_t) i * S + col]);
    }
    c.y_nmax = ey.normalized_max(); c.y_rl2 = ey.rel_l2(); c.s_nmax = es.normalized_max(); c.s_rl2 = es.rel_l2();
    c.nonfinite = ey.nonfinite + es.nonfinite;
    return c;
}
// candidate against the incumbent over every head, token and state element (not a subset)
Cmp vs_incumbent(const RunOut& r, const RunOut& inc, int64_t T) {
    Cmp c;
    ErrStat ey, es;
    for (size_t i = 0; i < r.y16.size(); ++i) ey.add(half_to_float(r.y16[i]), half_to_float(inc.y16[i]));
    for (size_t i = 0; i < kStateFloats; ++i) es.add(r.state[i], inc.state[i]);
    c.y_nmax = ey.normalized_max(); c.y_rl2 = ey.rel_l2(); c.s_nmax = es.normalized_max(); c.s_rl2 = es.rel_l2();
    c.nonfinite = ey.nonfinite + es.nonfinite;
    (void) T;
    return c;
}

std::string join(const std::set<std::string>& s) {
    std::string o;
    for (auto& x : s) { if (!o.empty()) o += "; "; o += x; }
    return o;
}

void json_line(const Args& a, const char* kind, const char* fixture, int64_t T, const RunOut& r, const Cmp* ref, const Cmp* inc,
               bool exact, double ms_med, double ms_min, const char* verdict, const cudaDeviceProp& pr) {
    if (a.json.empty()) return;
    FILE* f = std::fopen(a.json.c_str(), "a");
    if (!f) return;
    std::fprintf(f, "{\"kind\":\"%s\",\"label\":\"%s\",\"device\":\"%s\",\"sms\":%d,\"cc\":\"%d.%d\",\"sha\":\"%s\",\"fixture\":\"%s\","
                    "\"T\":%lld,\"chunk\":%lld,\"path\":\"%s\",\"eligible\":%s,\"reason\":\"%s\",\"log\":\"%s\",\"fallback_calls\":%d,"
                    "\"ms_gates\":%.4f,\"ms_conv\":%.4f,\"ms_rec\":%.4f,\"ms_total_med\":%.4f,\"ms_total_min\":%.4f,\"exact_vs_incumbent\":%s,\"verdict\":\"%s\"",
                 kind, a.label.c_str(), pr.name, pr.multiProcessorCount, pr.major, pr.minor, GDN_HARNESS_SHA, fixture, (long long) T,
                 (long long) a.chunk, sp::gdn_path_name(r.path), r.eligible ? "true" : "false", r.reason.c_str(), join(r.logged).c_str(),
                 r.fallback_calls, r.ms_gates, r.ms_conv, r.ms_rec, ms_med, ms_min, exact ? "true" : "false", verdict);
    if (ref) std::fprintf(f, ",\"ref_y_nmax\":%.3e,\"ref_y_rl2\":%.3e,\"ref_s_nmax\":%.3e,\"ref_s_rl2\":%.3e,\"nonfinite\":%ld", ref->y_nmax, ref->y_rl2, ref->s_nmax, ref->s_rl2, ref->nonfinite);
    if (inc) std::fprintf(f, ",\"inc_y_nmax\":%.3e,\"inc_y_rl2\":%.3e,\"inc_s_nmax\":%.3e,\"inc_s_rl2\":%.3e", inc->y_nmax, inc->y_rl2, inc->s_nmax, inc->s_rl2);
    std::fprintf(f, "}\n");
    std::fclose(f);
}

// One case: a fixture of T tokens, every wanted path.  Returns the number of failed verdicts.
int run_case(const Args& a, const FixtureSpec& spec, const char* fname, const cudaDeviceProp& pr) {
    const int64_t T = spec.T;
    const std::vector<int64_t> calls = plan_calls(T, a);
    int64_t cmax = 0;
    for (auto c : calls) cmax = std::max(cmax, c);
    Fixture f = make_fixture(spec);
    Dev d = upload(f, cmax, a.ld16);
    std::printf("\n=== case %s T=%lld calls=%zu (first %lld, max %lld) ld16=%lld  device %s (%d SMs, cc %d.%d)\n", fname, (long long) T,
                calls.size(), (long long) calls[0], (long long) cmax, (long long) d.ld16, pr.name, pr.multiProcessorCount, pr.major, pr.minor);
    RefResult ref;
    if (a.parity && a.ref) {
        auto t0 = std::chrono::steady_clock::now();
        ref = ref_run(f, a.ref_heads);
        std::printf("FP64 oracle for %zu value heads: %.1f s\n", a.ref_heads.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    // the incumbent first (it may itself be ineligible: then pipe, which always runs)
    std::vector<int> order = a.paths;
    RunOut inc = run_path(d, a, calls, a.incumbent, false, true);
    int inc_path = a.incumbent;
    if (!inc.eligible) {
        std::printf("incumbent %s INELIGIBLE (%s): using pipe\n", sp::gdn_path_name(a.incumbent), inc.reason.c_str());
        inc = run_path(d, a, calls, sp::kGdnPipe, false, true);
        inc_path = sp::kGdnPipe;
    }
    std::map<int, RunOut> runs;
    for (int p : order) {
        RunOut r = (p == inc_path) ? inc : run_path(d, a, calls, p, false, true);
        runs[p] = std::move(r);
    }
    int fails = 0;
    std::map<int, std::vector<double>> total_ms;
    std::map<int, RunOut> timed_last;
    if (a.bench) {
        for (int w = 0; w < a.warmup + a.reps; ++w)
            for (int p : order) {
                if (!runs[p].eligible) continue;
                ck(cudaMemset(d.flush, w & 255, d.flush_bytes), "flush");
                RunOut t = run_path(d, a, calls, p, true, false);
                if (w >= a.warmup) {
                    total_ms[p].push_back(t.ms_gates + t.ms_conv + t.ms_rec);
                    timed_last[p].ms_gates += t.ms_gates / a.reps;
                    timed_last[p].ms_conv += t.ms_conv / a.reps;
                    timed_last[p].ms_rec += t.ms_rec / a.reps;
                }
            }
    }
    std::printf("%-8s %-6s %-9s %9s %9s %9s %9s  %-34s %s\n", "path", "elig", "bitwise", "ref_y", "ref_state", "inc_y", "inc_state", "ms gates/conv/rec (mean)", "ran");
    for (int p : order) {
        RunOut& r = runs[p];
        const char* name = sp::gdn_path_name(p);
        if (!r.eligible) {
            std::printf("%-8s %-6s %s\n", name, "NO", r.reason.c_str());
            json_line(a, "case", fname, T, r, nullptr, nullptr, false, 0, 0, "INELIGIBLE", pr);
            continue;
        }
        Cmp cr{}, ci{};
        const bool have_ref = a.parity && a.ref;
        if (have_ref) cr = vs_ref(r, ref, T);
        ci = vs_incumbent(r, inc, T);
        const bool exact = r.y16 == inc.y16 && std::memcmp(r.state.data(), inc.state.data(), (kStateFloats + C * 3) * 4) == 0;
        // conv history after the last call is a copy of inputs: it must match the oracle's exactly
        bool hist_ok = true;
        if (have_ref) hist_ok = std::memcmp(r.state.data() + kStateFloats, ref.hist.data(), (size_t) C * 3 * 4) == 0;
        // verdict: finite, history exact, and not worse than max(floor, 3 x the incumbent's own error vs the oracle)
        const char* verdict = "PASS";
        if (have_ref) {
            const Cmp ir = vs_ref(inc, ref, T);
            const double ys = std::max(2e-3, 3 * ir.y_nmax), ss = std::max(1e-4, 3 * ir.s_nmax);
            if (cr.nonfinite || !hist_ok) verdict = "FAIL-nonfinite-or-history";
            else if (cr.y_nmax > ys) verdict = "FAIL-output-vs-oracle";
            else if (cr.s_nmax > ss) verdict = "FAIL-state-vs-oracle";
            else if (exact) verdict = "PASS-bitwise";
        } else if (exact) verdict = "PASS-bitwise";
        else verdict = "NO-ORACLE";
        if (!std::strncmp(verdict, "FAIL", 4)) ++fails;
        double med = 0, mn = 0;
        if (a.bench && !total_ms[p].empty()) {
            med = median(total_ms[p]);
            mn = *std::min_element(total_ms[p].begin(), total_ms[p].end());
            r.ms_gates = timed_last[p].ms_gates; r.ms_conv = timed_last[p].ms_conv; r.ms_rec = timed_last[p].ms_rec;
        }
        char tb[120] = "";
        if (a.bench) std::snprintf(tb, sizeof tb, "%.3f/%.3f/%.3f (tot med %.3f min %.3f)", r.ms_gates, r.ms_conv, r.ms_rec, med, mn);
        std::printf("%-8s %-6s %-9s %9.2e %9.2e %9.2e %9.2e  %-34s %s  %s h=%08llx\n", name, "yes", exact ? "identical" : "differs", cr.y_nmax, cr.s_nmax,
                    ci.y_nmax, ci.s_nmax, tb, join(r.logged).c_str(), verdict,
                    (unsigned long long) (fnv1a(r.y16.data(), r.y16.size() * 2) & 0xffffffffULL));
        if (r.fallback_calls) std::printf("         (%d call(s) ran on the incumbent: the forced path could not take them)\n", r.fallback_calls);
        json_line(a, "case", fname, T, r, have_ref ? &cr : nullptr, &ci, exact, med, mn, verdict, pr);
    }
    release(d);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    bool paths_given = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", k.c_str()); std::exit(2); } return argv[++i]; };
        if (k == "--T") { a.Ts = parse_list(val()); }
        else if (k == "--chunk") a.chunk = parse_list(val())[0];
        else if (k == "--cuts") a.cuts = parse_list(val());
        else if (k == "--paths") {
            paths_given = true;
            std::string s = val();
            for (size_t p = 0; p <= s.size();) {
                size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                const int id = path_from_name(s.substr(p, e - p));
                if (id < 0) { std::fprintf(stderr, "unknown path %s (auto,keyhead,pipe,cols,chunked,tc,tcs)\n", s.substr(p, e - p).c_str()); return 2; }
                a.paths.push_back(id);
                p = e + 1;
            }
        }
        else if (k == "--incumbent") a.incumbent = path_from_name(val());
        else if (k == "--parity") a.parity = true;
        else if (k == "--no-parity") a.parity = false;
        else if (k == "--bench") a.bench = true;
        else if (k == "--no-ref") a.ref = false;
        else if (k == "--ref-heads") { a.ref_heads.clear(); for (auto x : parse_list(val())) a.ref_heads.push_back((int) x); }
        else if (k == "--reps") a.reps = std::atoi(val());
        else if (k == "--warmup") a.warmup = std::atoi(val());
        else if (k == "--any-sms") a.any_sms = true;
        else if (k == "--gsb") a.gsb = std::atoi(val());
        else if (k == "--tiles") a.tiles = std::atoi(val());
        else if (k == "--fallback-incumbent") a.fallback_incumbent = true;
        else if (k == "--ld16") a.ld16 = parse_list(val())[0];
        else if (k == "--state") a.fx.state = argv[++i];
        else if (k == "--v-scale") a.fx.v_scale = (float) std::atof(val());
        else if (k == "--gate-scale") a.fx.gate_scale = (float) std::atof(val());
        else if (k == "--coherent-keys") a.fx.coherent_keys = true;
        else if (k == "--alt-beta") a.fx.alt_beta = true;
        else if (k == "--tiny-k") a.fx.tiny_k = true;
        else if (k == "--stress") a.stress = true;
        else if (k == "--seed") a.fx.seed = (uint64_t) std::atoll(val());
        else if (k == "--json") a.json = val();
        else if (k == "--label") a.label = val();
        else {
            std::fprintf(stderr, "unknown option %s (see Desktop\\30thOctober\\results\\gdn\\HARNESS.md)\n", k.c_str());
            return 2;
        }
    }
    if (a.Ts.empty()) a.Ts = {a.T};
    if (!paths_given) a.paths = {sp::kGdnAuto, sp::kGdnKeyHead, sp::kGdnPipe, sp::kGdnChunked, sp::kGdnTc, sp::kGdnTcSplit};
    if (a.incumbent < 0) { std::fprintf(stderr, "bad --incumbent\n"); return 2; }
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    cudaDeviceProp pr{};
    ck(cudaGetDeviceProperties(&pr, dev), "props");
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    std::printf("gdn_harness sha=%s device=%s sms=%d cc=%d.%d driver=%d runtime=%d smem_optin=%zu\n", GDN_HARNESS_SHA, pr.name,
                pr.multiProcessorCount, pr.major, pr.minor, drv, rt, (size_t) pr.sharedMemPerBlockOptin);
    for (const char* e : {"STRATA_GDN_CHUNKED", "STRATA_GDN_TC", "STRATA_GDN_KEYHEAD", "STRATA_GDN_PIPELINE", "STRATA_GDN_REC_WARP",
                          "STRATA_GDN_CONVL2", "STRATA_GDN_CONV_SERIAL", "STRATA_GDN_REC_HEADS", "STRATA_PF_PAD"})
        if (const char* v = std::getenv(e)) std::printf("env %s=%s\n", e, v);
    int fails = 0;
    for (int64_t T : a.Ts) {
        FixtureSpec s = a.fx;
        s.T = T;
        fails += run_case(a, s, "default", pr);
        if (a.stress) {
            struct { const char* n; const char* st; float v, g; bool coh, alt, tiny; } cases[] = {
                {"state-zero", "zero", 1, 1, false, false, false}, {"state-big", "big", 1, 1, false, false, false},
                {"state-small", "small", 1, 1, false, false, false}, {"v-x50", "random", 50, 1, false, false, false},
                {"slow-gates", "random", 1, 0.005f, false, false, false}, {"fast-gates", "random", 1, 3, false, false, false},
                {"coherent-keys", "random", 1, 1, true, false, false}, {"beta-endpoints", "random", 1, 1, false, true, false},
                {"tiny-keys", "random", 1, 1, false, false, true}};
            for (auto& c : cases) {
                FixtureSpec t = a.fx;
                t.T = T; t.state = c.st; t.v_scale = c.v; t.gate_scale = c.g; t.coherent_keys = c.coh; t.alt_beta = c.alt; t.tiny_k = c.tiny;
                fails += run_case(a, t, c.n, pr);
            }
        }
    }
    std::printf("\n%s: %d failed verdict(s)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
