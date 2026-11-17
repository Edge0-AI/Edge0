// src/core/session.cpp - one token through all 48 layers.  See the header for why the graphs are per-layer.
#include "strata/core/session.hpp"
#include "strata/core/pin_budget.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/core/progress.hpp"

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/kernels/ngram.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The doorbell spin.  x86 pauses, aarch64 yields (include/strata/platform/cpu_relax.hpp).
#include "strata/platform/cpu_relax.hpp"
#define STRATA_SPIN_PAUSE() strata_cpu_pause()

namespace strata::core {
namespace {

constexpr uint64_t SESSION_STATE_ALIGN = 256;

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

/// The floats one GDN layer's recurrent + conv state needs.  `gdn_buffers_bytes` carves them for ONE layer and
/// `GdnBuffers::state`/`conv_state` point INTO that carve, so a session with 36 GDN layers has to give each one
/// its own - they cannot share, because the recurrence is the whole point.
uint64_t gdn_state_floats(const ModelGeometry& g) {
    return (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
           (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
}

}  // namespace

/// `NG_HIST` rows of `hc_dim` floats: the PLE conv's history, which is the ONLY PLE state that lives in the
/// session arena.  The table and the weights are model-level and the caller owns them.
static uint64_t ple_hist_bytes() {
    return (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * sizeof(float);
}

uint64_t session_bytes(const ModelGeometry& g, int64_t max_cells, int64_t k, int64_t layer_lo, int64_t layer_hi) {
    if (layer_hi < 0 || layer_hi > g.n_layers) layer_hi = g.n_layers;
    if (layer_lo < 0) layer_lo = 0;
    // QSA layers are `l % interval == interval-1`, so exactly `bound / interval` of them live below `bound`
    const int64_t I = std::max<int64_t>(g.qsa_interval, 1);
    const int64_t q_lo = layer_lo / I, q_hi = layer_hi / I;
    const int64_t q_n = std::max<int64_t>(q_hi - q_lo, g.n_qsa_layers() > 0 ? 1 : 0);
    const int64_t gdn_n = std::max<int64_t>((layer_hi - layer_lo) - std::max<int64_t>(q_hi - q_lo, 0), 0);
    uint64_t n = 0;
    n += gdn_buffers_bytes(g);
    n += (uint64_t) gdn_n * gdn_state_floats(g) * 4;
    // One QSA state carries the RoPE table; the others borrow it (P7: 64 MiB per layer at 262K).
    if (g.n_qsa_layers() > 0)
        n += qsa_state_bytes(g, max_cells, true) + (uint64_t) (q_n - 1) * qsa_state_bytes(g, max_cells, false);
    n += qsa_buffers_bytes(g, max_cells);
    n += moe_buffers_bytes(g, k);
    n += block_buffers_bytes(g);
    n += ple_hist_bytes();                       // the PLE's NG_HIST normalized history rows
    return align_up(n, SESSION_STATE_ALIGN) + 4096;
}

uint64_t session_init(const ModelGeometry& g, int64_t max_cells, int64_t k, void* base, SessionState& s,
                      int64_t layer_lo, int64_t layer_hi) {
    if (layer_hi < 0 || layer_hi > g.n_layers) layer_hi = g.n_layers;
    if (layer_lo < 0) layer_lo = 0;
    uint8_t* p = (uint8_t*) base;
    uint64_t used = 0;
    auto take = [&](uint64_t bytes) {
        uint8_t* r = p + used;
        used = align_up(used + bytes, SESSION_STATE_ALIGN);
        return r;
    };

    s.max_cells = max_cells;
    s.k = k;
    s.layer_lo = layer_lo;
    s.layer_hi = layer_hi;
    const int64_t I = std::max<int64_t>(g.qsa_interval, 1);
    const int64_t q_n_range = layer_hi / I - layer_lo / I;
    s.qsa_ord0 = layer_lo / I;
    s.qsa_alloc = std::max<int64_t>(q_n_range, g.n_qsa_layers() > 0 ? 1 : 0);
    s.gdn_ord0 = layer_lo - layer_lo / I;
    s.gdn_alloc = std::max<int64_t>((layer_hi - layer_lo) - q_n_range, 0);

    gdn_buffers_init(g, take(gdn_buffers_bytes(g)), s.gdn);
    s.gdn_state = (float*) take((uint64_t) s.gdn_alloc * gdn_state_floats(g) * 4);

    // the QSA states are separate allocations carved from one arena, because `QsaState` is a struct of
    // pointers and `qsa_state_init` writes them - a contiguous array would need the arena to be laid out the
    // same way, which is a coupling with nothing to gain.  Only the range's ordinals are initialized; the
    // rest stay value-initialized nulls.  The FIRST ALLOCATED one (the session's primary, ordinal qsa_ord0)
    // owns the RoPE table that the others - and the prefill staging identity, and the MTP drafter - borrow.
    const uint64_t first = qsa_state_bytes(g, max_cells, true), rest = qsa_state_bytes(g, max_cells, false);
    s.qsa_state_arena = take(g.n_qsa_layers() > 0 ? first + (uint64_t) (s.qsa_alloc - 1) * rest : 0);
    s.qsa_states = new QsaState[(size_t) g.n_qsa_layers()]();
    s.qsa_buf_arena = take(qsa_buffers_bytes(g, max_cells));

    uint8_t* qp = (uint8_t*) s.qsa_state_arena;
    // a layer whose pinned RAM could not be had (KV streaming's host copy) is half-built: going on would have the
    // attention read null host pointers at the first request ("illegal memory access"), so the session fails here
    for (int64_t j = 0; j < s.qsa_alloc; ++j)
        if (qsa_state_init(g, max_cells, qp + (j == 0 ? 0 : first + (uint64_t) (j - 1) * rest),
                           s.qsa_states[s.qsa_ord0 + j], j == 0 ? nullptr : &s.qsa_states[s.qsa_ord0]) == 0)
            return 0;
    qsa_buffers_init(g, max_cells, s.qsa_buf_arena, s.qsa_bufs);

    s.moe_arena = take(moe_buffers_bytes(g, k));
    moe_buffers_init(g, k, s.moe_arena, s.moe);
    s.block_arena = take(block_buffers_bytes(g));
    block_buffers_init(g, s.block_arena, s.block);
    // THE PLE HISTORY: NG_HIST rows of hc_dim floats, row-fastest.  Sequence state, carved here and zeroed by
    // `session_zero`; it survives every token, which is the whole point of a conv history.
    s.ple_hist = (float*) take(ple_hist_bytes());
    s.R = s.block.R;
    return used;
}

void session_release(SessionState& s) {
    for (int64_t j = 0; s.qsa_states != nullptr && j < s.qsa_alloc; ++j)
        if (s.qsa_states[j].owns_rope) {
            strata::kernels::rope_table_release(s.qsa_states[j].cos_tab);
            s.qsa_states[j].owns_rope = false;
        }
}

#if defined(STRATA_GPU_METAL)
}  // namespace strata::core
namespace strata::metal_engine {
void window_reset(const strata::core::SessionState* ss);   // metal/src/engine/window.cpp: that session's running state (the main one or a batch slot's) back to a new sequence's
}  // namespace strata::metal_engine
namespace strata::core {
#endif

void session_zero(SessionState& s, const ModelGeometry& g, const float* R_init, void* stream) {
#if defined(STRATA_GPU_METAL)
    // the running state lives in the Metal decode window; the session's own device buffers are not read there
    (void) g; (void) R_init; (void) stream;
    strata::metal_engine::window_reset(&s);
    s.ple_prev[0] = -1;
    s.ple_prev[1] = -1;
    s.ple_token = -1;
    return;
#endif
    cudaStream_t cs = (cudaStream_t) stream;
    // the residual: `hc` copies of the one vector a caller hands in.  A real sequence's first token is the
    // embedding broadcast to every stream, which is the reference's own initial condition.
    if (R_init != nullptr) {
        for (int64_t c = 0; c < g.hc; ++c)
            cudaMemcpyAsync(s.block.R + (size_t) c * g.n_embd, R_init, (size_t) g.n_embd * 4,
                            cudaMemcpyDeviceToDevice, cs);
    } else {
        cudaMemsetAsync(s.block.R, 0, (size_t) g.hc * g.n_embd * 4, cs);
    }
    // every owned GDN layer's recurrence and conv history
    cudaMemsetAsync(s.gdn_state, 0, (size_t) s.gdn_alloc * gdn_state_floats(g) * 4, cs);
    // and every owned QSA layer's cache and indexer
    for (int64_t j = 0; j < s.qsa_alloc; ++j) qsa_state_zero(s.qsa_states[s.qsa_ord0 + j], g, stream);
    // **AND THE PLE'S CONV HISTORY AND TOKEN WINDOW.**  A sequence that started with a warm history would
    // convolve over rows belonging to a different sequence - the conv reads NG_HIST previous NORMALIZED rows,
    // so a stale one is a real contribution and not a zero.  The token window resets to `NG_HIST`-many nulls
    // for the same reason: `ngram_rows` treats a missing predecessor as the EOS cut, which is what a sequence
    // boundary IS.
    cudaMemsetAsync(s.ple_hist, 0, (size_t) ple_hist_bytes(), cs);
    s.ple_prev[0] = -1;
    s.ple_prev[1] = -1;
    s.ple_token = -1;
}

/// Sets `s.gdn.state`/`conv_state` for `layer`, which is what makes one layer's GDN state its own.  Shared by
/// the direct and captured paths so the two cannot disagree about which slice a layer owns.
void gdn_point_at(const ModelGeometry& g, int64_t layer, SessionState& s) {
    if (is_qsa_layer(g, layer)) return;
    int64_t gdn_index = 0;
    for (int64_t l = 0; l < layer; ++l) if (!is_qsa_layer(g, l)) ++gdn_index;
    s.gdn.state = s.gdn_state + (size_t) (gdn_index - s.gdn_ord0) * gdn_state_floats(g);
    s.gdn.conv_state = s.gdn.state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
}

/// The per-token staging every QSA layer's captured H2D reads FROM.  Must run before each replay: the graphs
/// captured the SOURCE POINTER, not the value, and that is exactly why the buffers are pinned and fixed.
void stage_token(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s) {
    const strata::kernels::QsaShapes sh = qsa_shapes(g);
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        QsaState& q = s.qsa_states[s.qsa_ord0 + j];
        qsa_step_fill(q.host_step, pos, sh);
        for (int64_t h = 0; h < g.n_head; ++h) q.host_pos[h] = (int32_t) (pos_base + pos);
    }
}

bool session_capture(const WeightTable& tables, const ModelGeometry& g, SessionState& s, const float* parts,
                     SessionGraphs& gr, std::string& err, bool split, int64_t layer_lo, int64_t layer_hi) {
    if (gr.captured) return true;
    if (layer_hi < 0 || layer_hi > g.n_layers) layer_hi = g.n_layers;
    if (layer_lo < 0) layer_lo = 0;
    gr.execs = new cudaGraphExec_t[(size_t) g.n_layers]();
    gr.posts = new cudaGraphExec_t[(size_t) g.n_layers]();
    for (int64_t i = 0; i < g.n_layers; ++i) gr.posts[i] = nullptr;
    if (split) {
        gr.preA = new cudaGraphExec_t[(size_t) g.n_layers]();
        gr.preB = new cudaGraphExec_t[(size_t) g.n_layers]();
        for (int64_t i = 0; i < g.n_layers; ++i) { gr.preA[i] = nullptr; gr.preB[i] = nullptr; }
        for (int k = 0; k < 5; ++k) {
            gr.preP[k] = new cudaGraphExec_t[(size_t) g.n_layers]();
            for (int64_t i = 0; i < g.n_layers; ++i) gr.preP[k][i] = nullptr;
        }
    }
    gr.split_captured = split;
    gr.n = 0;

    int64_t qsa_index = s.qsa_ord0;
    for (int64_t l = layer_lo; l < layer_hi; ++l) {
        gdn_point_at(g, l, s);
        const bool qsa = is_qsa_layer(g, l);
        QsaState& qst = qsa ? s.qsa_states[qsa_index] : s.qsa_states[s.qsa_primary()];

        // **TWO GRAPHS PER LAYER, SPLIT AT THE ROUTER.**  `pre` ends with the doorbell rung; the host then runs
        // the CPU pool on what it published; `post` combines those experts with THIS layer's weights.  Capturing
        // them together is what the old single graph could not do, and the reason it could not is that
        // `moe_combine` needs the pool's answer and the pool needs the router's.
        auto capture = [&](bool post, cudaGraphExec_t* out, const char* what, int half = 0,
                           int stage_prefix = 0) -> bool {
            cudaStream_t cs = nullptr;
            if (cudaStreamCreate(&cs) != cudaSuccess) {
                err = std::string("session_capture: stream create failed");
                return false;
            }
            if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
                err = "session_capture: begin failed at layer " + std::to_string(l);
                return false;
            }
            err.clear();
            // R0.9: `half` is 0 for every mode except the split capture, where it selects a prefix of the
            // stages so the mixer and the FFN-front-plus-router can be timed separately.
            const bool ok = post
                                ? block_layer_post(tables, g, l, s.k, s.moe, s.block, parts, (void*) cs, err)
                                : block_layer_pre(tables, g, l, 0, 0, s.gdn, qst, s.qsa_bufs, s.moe, s.k,
                                                  s.block, (void*) cs, err, s.db, s.ple.ready() ? &s.ple : nullptr,
                                                  half, stage_prefix);
            if (!ok) {
                err = "session_capture: " + std::string(what) + " layer " + std::to_string(l) + ": " + err;
                return false;
            }
            cudaGraph_t graph = nullptr;
            const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
            cudaStreamDestroy(cs);
            if (ce != cudaSuccess) {
                err = "session_capture: " + std::string(what) + " layer " + std::to_string(l) + ": " +
                      cudaGetErrorString(ce) + " (a synchronous call in the layer?)";
                return false;
            }
            if (cudaGraphInstantiate(out, graph, 0) != cudaSuccess) {
                err = "session_capture: instantiate failed at layer " + std::to_string(l);
                return false;
            }
            cudaGraphDestroy(graph);
            return true;
        };
        if (!capture(/*post=*/false, &gr.execs[l], "pre")) return false;
        if (!capture(/*post=*/true, &gr.posts[l], "post")) return false;
        if (split) {
            if (!capture(/*post=*/false, &gr.preA[l], "preA", /*half=*/1)) return false;
            if (!capture(/*post=*/false, &gr.preB[l], "preB", /*half=*/2)) return false;
            // Prefixes 1..5, ALL through the new parameter - including the fifth, so that it and the fourth
            // differ only in the one stage between them and not in how many nodes their graphs carry.
            for (int k = 1; k <= 5; ++k)
                if (!capture(/*post=*/false, &gr.preP[k - 1][l], "preP", /*half=*/0, /*stage_prefix=*/k))
                    return false;
        }
        ++gr.n;
        if (qsa) ++qsa_index;
    }
    gr.captured = true;
    gr.parts_dev = const_cast<float*>(parts);   // the address the graphs baked in; the loop copies here
    return true;
}

bool session_replay(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s, SessionGraphs& gr,
                    void* stream, std::string& err) {
    if (!gr.captured || gr.n != g.n_layers) { err = "session_replay: not captured"; return false; }
    cudaStream_t cs = (cudaStream_t) stream;
    stage_token(g, pos, pos_base, s);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const cudaError_t e = cudaGraphLaunch(gr.execs[l], cs);
        if (e != cudaSuccess) {
            err = "session_replay: layer " + std::to_string(l) + ": " + cudaGetErrorString(e);
            return false;
        }
    }
    return true;
}

bool session_replay_full(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                         SessionGraphs& gr, void* stream, std::string& err) {
    if (!gr.captured || gr.n != g.n_layers || gr.posts == nullptr) {
        err = "session_replay_full: not captured with post graphs";
        return false;
    }
    cudaStream_t cs = (cudaStream_t) stream;
    stage_token(g, pos, pos_base, s);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        // `pre[l]` then `post[l]`, in the order `session_loop` uses.  The two are ordered on one stream, and
        // `post[l]` reads what `pre[l]` wrote, so they cannot be reordered or run concurrently.
        cudaError_t e = cudaGraphLaunch(gr.execs[l], cs);
        if (e != cudaSuccess) {
            err = "session_replay_full: pre[" + std::to_string(l) + "]: " + cudaGetErrorString(e);
            return false;
        }
        e = cudaGraphLaunch(gr.posts[l], cs);
        if (e != cudaSuccess) {
            err = "session_replay_full: post[" + std::to_string(l) + "]: " + cudaGetErrorString(e);
            return false;
        }
    }
    return true;
}

bool session_replay_stages_per_layer(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                                     SessionGraphs& gr, void* stream, std::vector<double>& mixer_per_layer,
                                     double& ms_ffn, double& ms_post, std::string& err) {
    if (!gr.split_captured || gr.preA == nullptr || gr.preB == nullptr) {
        err = "session_replay_stages: not captured with the split";
        return false;
    }
    cudaStream_t cs = (cudaStream_t) stream;
    stage_token(g, pos, pos_base, s);

    // Four events PER LAYER rather than four reused ones: reusing them would need a synchronisation after
    // every layer to read them before the next launch overwrote them, and a per-layer sync would report the
    // serialised time rather than the graph's.
    const int64_t n = g.n_layers;
    std::vector<cudaEvent_t> ev((size_t) (n * 4));
    for (auto& e : ev) {
        if (cudaEventCreate(&e) != cudaSuccess) { err = "session_replay_stages: event create"; return false; }
    }
    struct Free {
        std::vector<cudaEvent_t>* v;
        ~Free() { for (auto& e : *v) cudaEventDestroy(e); }
    } frees{&ev};

    for (int64_t l = 0; l < n; ++l) {
        cudaEventRecord(ev[(size_t) (l * 4 + 0)], cs);
        if (cudaGraphLaunch(gr.preA[l], cs) != cudaSuccess) { err = "stages: preA"; return false; }
        cudaEventRecord(ev[(size_t) (l * 4 + 1)], cs);
        if (cudaGraphLaunch(gr.preB[l], cs) != cudaSuccess) { err = "stages: preB"; return false; }
        cudaEventRecord(ev[(size_t) (l * 4 + 2)], cs);
        if (cudaGraphLaunch(gr.posts[l], cs) != cudaSuccess) { err = "stages: post"; return false; }
        cudaEventRecord(ev[(size_t) (l * 4 + 3)], cs);
    }
    if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "stages: final sync"; return false; }

    mixer_per_layer.assign((size_t) n, 0.0);
    ms_ffn = 0; ms_post = 0;
    for (int64_t l = 0; l < n; ++l) {
        float a = 0, b = 0, c = 0;
        cudaEventElapsedTime(&a, ev[(size_t) (l * 4 + 0)], ev[(size_t) (l * 4 + 1)]);
        cudaEventElapsedTime(&b, ev[(size_t) (l * 4 + 1)], ev[(size_t) (l * 4 + 2)]);
        cudaEventElapsedTime(&c, ev[(size_t) (l * 4 + 2)], ev[(size_t) (l * 4 + 3)]);
        mixer_per_layer[(size_t) l] = (double) a;
        ms_ffn += b; ms_post += c;
    }
    return true;
}

bool session_replay_stages(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                           SessionGraphs& gr, void* stream, double& ms_mixer, double& ms_ffn, double& ms_post,
                           std::string& err) {
    std::vector<double> per;
    if (!session_replay_stages_per_layer(g, pos, pos_base, s, gr, stream, per, ms_ffn, ms_post, err)) return false;
    ms_mixer = 0;
    for (double v : per) ms_mixer += v;
    return true;
}

bool session_replay_stage_sweep(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                                SessionGraphs& gr, void* stream, int k, double& ms, std::string& err) {
    if (!gr.split_captured || gr.preP[0] == nullptr) {
        err = "session_replay_stage_sweep: not captured with the split";
        return false;
    }
    if (k < 1 || k > 5) { err = "session_replay_stage_sweep: k must be 1..5"; return false; }
    cudaStream_t cs = (cudaStream_t) stream;
    stage_token(g, pos, pos_base, s);

    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) {
        err = "session_replay_stage_sweep: event create";
        return false;
    }
    cudaEventRecord(a, cs);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const cudaError_t e = cudaGraphLaunch(gr.preP[k - 1][l], cs);
        if (e != cudaSuccess) {
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            err = std::string("session_replay_stage_sweep: launch: ") + cudaGetErrorString(e);
            return false;
        }
    }
    cudaEventRecord(b, cs);
    if (cudaStreamSynchronize(cs) != cudaSuccess) {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        err = "session_replay_stage_sweep: sync";
        return false;
    }
    float v = 0;
    cudaEventElapsedTime(&v, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    ms = (double) v;
    return true;
}

bool session_replay_stage_prefixes(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                                   SessionGraphs& gr, void* stream, std::vector<double>& stage_ms,
                                   std::vector<double>& mixer_per_layer, std::string& err) {
    if (!gr.split_captured || gr.preP[0] == nullptr) {
        err = "session_replay_stage_prefixes: not captured with the split";
        return false;
    }
    cudaStream_t cs = (cudaStream_t) stream;
    stage_token(g, pos, pos_base, s);
    const int64_t n = g.n_layers;
    const size_t r_floats = (size_t) g.hc * (size_t) g.n_embd;

    float* saved = nullptr;
    if (cudaMalloc((void**) &saved, r_floats * sizeof(float)) != cudaSuccess) {
        err = "session_replay_stage_prefixes: could not save the residual";
        return false;
    }
    std::vector<cudaEvent_t> ev((size_t) (n * 6));
    bool ev_ok = true;
    for (auto& e : ev) if (cudaEventCreate(&e) != cudaSuccess) { ev_ok = false; break; }
    if (!ev_ok) {
        for (auto& e : ev) cudaEventDestroy(e);
        cudaFree(saved);
        err = "session_replay_stage_prefixes: event create";
        return false;
    }
    struct Cleanup {
        std::vector<cudaEvent_t>* v;
        float* p;
        ~Cleanup() { for (auto& e : *v) cudaEventDestroy(e); cudaFree(p); }
    } cleanup{&ev, saved};

    double t[5] = {0, 0, 0, 0, 0};
    mixer_per_layer.assign((size_t) n, 0.0);
    for (int64_t l = 0; l < n; ++l) {
        // The residual as this layer receives it, before any prefix has advanced it.
        cudaMemcpyAsync(saved, s.block.R, r_floats * sizeof(float), cudaMemcpyDeviceToDevice, cs);
        for (int k = 1; k <= 5; ++k) {
            if (k > 1) cudaMemcpyAsync(s.block.R, saved, r_floats * sizeof(float), cudaMemcpyDeviceToDevice, cs);
            cudaEventRecord(ev[(size_t) (l * 6 + k - 1)], cs);
            const cudaGraphExec_t ge = (k == 5) ? gr.execs[l] : gr.preP[k - 1][l];
            const cudaError_t e = cudaGraphLaunch(ge, cs);
            if (e != cudaSuccess) {
                err = std::string("session_replay_stage_prefixes: launch prefix ") + std::to_string(k) + ": " +
                      cudaGetErrorString(e);
                return false;
            }
        }
        cudaEventRecord(ev[(size_t) (l * 6 + 5)], cs);
    }
    if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "prefixes: final sync"; return false; }

    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    for (int64_t l = 0; l < n; ++l) {
        float a = 0, b = 0, c = 0, d = 0, e = 0;
        cudaEventElapsedTime(&a, ev[(size_t) (l * 6 + 0)], ev[(size_t) (l * 6 + 1)]);
        cudaEventElapsedTime(&b, ev[(size_t) (l * 6 + 1)], ev[(size_t) (l * 6 + 2)]);
        cudaEventElapsedTime(&c, ev[(size_t) (l * 6 + 2)], ev[(size_t) (l * 6 + 3)]);
        cudaEventElapsedTime(&d, ev[(size_t) (l * 6 + 3)], ev[(size_t) (l * 6 + 4)]);
        cudaEventElapsedTime(&e, ev[(size_t) (l * 6 + 4)], ev[(size_t) (l * 6 + 5)]);
        t1 += a; t2 += b; t3 += c; t4 += d; t5 += e;
        mixer_per_layer[(size_t) l] = (double) c;   // prefix 3 IS the mixer: stages 0..2
    }
    t[0] = t1;
    t[1] = t2 - t1;
    t[2] = t3 - t2;
    t[3] = t4 - t3;
    t[4] = t5 - t4;
    stage_ms.assign(t, t + 5);
    return true;
}

void session_graphs_free(SessionGraphs& gr) {
    if (gr.execs) {
        for (int64_t i = 0; i < gr.n; ++i) cudaGraphExecDestroy(gr.execs[i]);
        delete[] gr.execs;
    }
    if (gr.posts) {
        for (int64_t i = 0; i < gr.n; ++i)
            if (gr.posts[i] != nullptr) cudaGraphExecDestroy(gr.posts[i]);
        delete[] gr.posts;
    }
    if (gr.preA) {
        for (int64_t i = 0; i < gr.n; ++i)
            if (gr.preA[i] != nullptr) cudaGraphExecDestroy(gr.preA[i]);
        delete[] gr.preA;
    }
    if (gr.preB) {
        for (int64_t i = 0; i < gr.n; ++i)
            if (gr.preB[i] != nullptr) cudaGraphExecDestroy(gr.preB[i]);
        delete[] gr.preB;
    }
    for (int k = 0; k < 5; ++k) {
        if (gr.preP[k] == nullptr) continue;
        for (int64_t i = 0; i < gr.n; ++i)
            if (gr.preP[k][i] != nullptr) cudaGraphExecDestroy(gr.preP[k][i]);
        delete[] gr.preP[k];
        gr.preP[k] = nullptr;
    }
    gr.preA = nullptr;
    gr.preB = nullptr;
    gr.split_captured = false;
    gr.posts = nullptr;
    gr.execs = nullptr;
    gr.parts_dev = nullptr;
    gr.n = 0;
    gr.captured = false;
}

bool SessionLoopScratch::init(size_t parts_bytes_in, std::string& err) {
    if (y_miss != nullptr || probe != nullptr) {
        err = "SessionLoopScratch::init: already initialised";
        return false;
    }
    parts_bytes = parts_bytes_in;
    // MAPPED as well as pinned: the token graph's handoff kernel reads it through its device pointer.
    if (strata::core::host_alloc_guarded((void**) &y_miss, parts_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        err = "SessionLoopScratch: cudaHostAlloc for the pool's staging failed";
        return false;
    }
    std::memset(y_miss, 0, parts_bytes);
    if (cudaEventCreate(&probe) != cudaSuccess) {
        err = "SessionLoopScratch: cudaEventCreate failed";
        free();
        return false;
    }
    // **PIN THE HOST ONCE, NOT ONCE PER TOKEN.**  `ExpertPool` builds its workers from `physical_cores(true)`,
    // which drops the first physical core so the host loop can spin without taking a worker's cycles - and
    // nothing in the pool can pin the host, so if this does not happen the spin is free to land on a worker's
    // core or its SMT sibling.  The symptom is not an error: it is a CPU path at 26.9 GB/s where the same pool
    // runs at 36.32.  It was being done and undone on EVERY token, which is a syscall pair on the critical path
    // for a property that wants to hold for the whole session.
    // The host's core is the pool's reserved one: the first physical core, or the last with --host-core last (F12),
    // or the first core's SMT sibling with --host-core sibling.
    const int host_core = strata::kernels::cpu::planned_host_core();
    if (host_core >= 0) {
        pinned_core = strata::kernels::cpu::pin_current_thread(host_core);
        pinned = pinned_core.valid;
    } else {
        const std::vector<int> cores = strata::kernels::cpu::physical_cores(false);
        if (!cores.empty()) {
            pinned_core = strata::kernels::cpu::pin_current_thread(cores[0]);
            pinned = pinned_core.valid;
        }
    }
    strata::platform::name_this_thread("strata-host");
    strata::aux_cpus::note_owned_thread();   // the host: --aux-cpus leaves it where it is
    return true;
}

void SessionLoopScratch::free() {
    // Restores the caller's affinity on every path, including teardown: a library that silently leaves its
    // caller's thread nailed to one core is worse than one that never pinned at all.
    if (pinned) {
        strata::kernels::cpu::restore_thread_affinity(pinned_core);
        pinned = false;
        pinned_core = {};
    }
    if (probe != nullptr) { cudaEventDestroy(probe); probe = nullptr; }
    if (y_miss != nullptr) { cudaFreeHost(y_miss); y_miss = nullptr; }
    parts_bytes = 0;
}

bool session_loop(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s, SessionGraphs& gr,
                  PoolFn pool, HitFn hits, void* user, bool overlap, void* stream, std::string& err,
                  float* dump_layers, SessionLoopScratch* scratch) {
    if (!gr.captured || gr.n != g.n_layers) { err = "session_loop: not captured"; return false; }
    if (gr.parts_dev == nullptr) { err = "session_loop: the graphs were captured without a parts buffer"; return false; }
    if (s.db == nullptr) { err = "session_loop: no doorbell; the loop has nothing to poll"; return false; }

    cudaStream_t cs = (cudaStream_t) stream;
    const int64_t k = s.k;
    const size_t parts_bytes = (size_t) k * g.n_embd * 4;
    ++gr.calls_total;

    // ================================ THE POSITION, WHICH THIS LOOP NEVER STAGED ================================
    //
    // **EVERY TOKEN AFTER THE FIRST USED TO REPLAY POSITION 0.**  The QSA layers read their per-token counts from
    // `st.step` and `st.pos_dev`, which are DEVICE buffers filled by an H2D copy captured INTO each graph.  A
    // graph bakes the SOURCE POINTER, not the value - so the copy re-reads the pinned host buffers on every
    // replay, and those hold whatever was last written there.  `session_capture` left them at position 0.
    //
    // `layer.hpp` documents the requirement this loop was missing: "The per-token staging every QSA layer's
    // captured H2D reads FROM.  Must run before each replay: the graphs captured the SOURCE POINTER, not the
    // value, and that is exactly why the buffers are pinned and fixed."  `session_replay` calls `stage_token`;
    // `session_loop` did not.  **A single-token test cannot see a position that never advances** - which is why
    // the whole suite passed while a 20-token prompt rotated every token at position 0.
    stage_token(g, pos, pos_base, s);

    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // Everything this loop needs on the host - the pinned staging for the pool's answer, the doorbell probe
    // event, and the host pin - lives in a `SessionLoopScratch` owned by the SESSION and passed in.  It used to
    // be created here, once per token, and `cudaFreeHost` at the end of the call IMPLICITLY SYNCHRONISES THE
    // DEVICE, so every token finished with a device-wide sync that nothing had asked for.  A comment here said
    // "allocated once"; once per token is not once.
    //
    // Passing `nullptr` keeps the old behaviour so existing callers and tests are unaffected: the loop builds a
    // local scratch and frees it on every exit path below, including the error returns.
    SessionLoopScratch local;
    if (scratch == nullptr) {
        if (!local.init(parts_bytes, err)) return false;
        scratch = &local;
    }
    struct LocalFree {
        SessionLoopScratch* p;
        ~LocalFree() { if (p != nullptr) p->free(); }
    } local_free{scratch == &local ? &local : nullptr};
    if (scratch->parts_bytes != parts_bytes) {
        err = "session_loop: the scratch was sized for a different k or n_embd";
        return false;
    }

    // HOST-side staging for the pool's answer.  It is copied to `gr.parts_dev` before the next launch, on the
    // same stream, so the ordering is the stream's and no captured node is needed for it.
    //
    // ---- **PINNED, BECAUSE A COPY FROM PAGEABLE MEMORY IS NOT ASYNCHRONOUS AT ALL.**
    //
    // `cudaMemcpyAsync` with a pageable source cannot DMA: the driver first memcpys the bytes into an internal
    // pinned bounce buffer, SYNCHRONOUSLY, on this thread, and only then enqueues the transfer.  So the "async"
    // copy would be a blocking memcpy plus a driver round trip, on the critical path of all 48 layers.  The
    // doorbell's `h_x_f` is `cudaHostAlloc(Mapped)` for exactly this reason and this buffer is the same kind of
    // object.
    float* y_miss = scratch->y_miss;
    cudaEvent_t probe = scratch->probe;

    // the first layer has no previous layer's experts: `y_miss` starts at zero, which is what "hits are empty
    // in this phase" means once every miss has been computed
    cudaMemcpyAsync(gr.parts_dev, y_miss, parts_bytes, cudaMemcpyHostToDevice, cs);
    doorbell_reset(*s.db);
    uint32_t expected = 0;

    // **`pre[0]` IS LAUNCHED BEFORE THE LOOP SO EVERY ITERATION HAS THE SAME SHAPE**: poll the ring that
    // `pre[l]` published, run the pool on it, combine, then start layer `l+1`'s `pre`.
    //
    // SLOT 0 OF THE DUMP IS THE INPUT, not a layer output.  `s.R` holds the embedding broadcast to every stream
    // when this function is entered, which is the reference's `hc_init` node - so dumping it here is what splits
    // "the input to layer 0 is already wrong" from "layer 0 is wrong".
    if (dump_layers != nullptr) {
        const size_t n = (size_t) g.hc * (size_t) g.n_embd;
        const cudaError_t de = cudaMemcpyAsync(dump_layers, s.R, n * sizeof(float), cudaMemcpyDeviceToHost, cs);
        if (de != cudaSuccess) {
            err = "session_loop: dump the input residual: " + std::string(cudaGetErrorString(de));
            return false;
        }
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        const cudaError_t le = cudaGraphLaunch(gr.execs[0], cs);
        if (le != cudaSuccess) {
            err = "session_loop: launch pre[0]: " + std::string(cudaGetErrorString(le));
            return false;
        }
        cudaEventRecord(probe, cs);
        (void) t0;
    }

    for (int64_t l = 0; l < g.n_layers; ++l) {
        const auto t_launch = std::chrono::steady_clock::now();

        // ---- poll for the ring.  **THE DRIVER CALL IS NOW A FALLBACK, NOT THE MECHANISM.**
        const uint32_t want = ++expected;
        bool rang = false;
        bool mid_graph = false;
        // VOLATILE, because the device writes this through mapped pinned memory and a cached host line would
        // never see it.  Read through a volatile pointer so the compiler re-issues the load every iteration -
        // otherwise this whole spin collapses into `while (true) { }`.
        volatile uint32_t* const seq = s.db->h_seq;
        for (;;) {
            // **THE PER-ITERATION DRIVER CALL IS REQUIRED, AND THREE EXPERIMENTS NOW SAY SO.**  Round 287
            // throttled this to one call in 64 and broke tests 51 and 56; adding `__threadfence_system()` to
            // `doorbell_ring_kernel` and a volatile read here, then throttling, broke them again.  So the call
            // is not merely flushing submission (round 195's reading) and it is not a missing fence either: on
            // this driver, the device's write to mapped pinned memory becomes host-visible only when the driver
            // is entered.  The fence and the volatile read are kept because they are correct and cost nothing -
            // without them the ring's ordering against `x_f`/`ids`/`weights` is unstated - but they do not
            // remove the need for the call, and nothing here should be read as claiming they do.
            const cudaError_t q = cudaEventQuery(probe);
            if (!rang && *seq >= want) {
                rang = true;
                mid_graph = (q != cudaSuccess);
            }
            if (q == cudaSuccess) break;                 // the graph has ended
            if (q != cudaErrorNotReady) {
                err = "session_loop: query at layer " + std::to_string(l) + ": " + cudaGetErrorString(q);
                return false;
            }
            if (rang) break;                             // rung, and the graph is still running: the overlap
            STRATA_SPIN_PAUSE();
        }
        strata_load_fence();   // the ring was seen: the GPU's x_f / ids / weights are read after it (a no-op on x86)
        if (mid_graph) ++gr.rings_mid_graph;
        // the window, from just before the launch to the instant the ring was seen
        const auto t_ring = std::chrono::steady_clock::now();
        gr.ms_to_ring += std::chrono::duration<double, std::milli>(t_ring - t_launch).count();
        if (!rang) {
            // The graph finished without the ring being observed.  `h_seq` is monotonic, so this is not a race
            // - it means the ring never happened, which is the failure round 199 found in a captured event.
            err = "session_loop: layer " + std::to_string(l) + " ended without ringing";
            return false;
        }

        // ---- **THE GPU'S HALF GOES FIRST, SO IT RUNS WHILE THE CPU DOES ITS HALF.**  `Launch` only enqueues:
        // the quantize and the grouped expert kernel land on `main_cs` and the GPU starts on them immediately,
        // while the host is still inside `pool` below.  Nothing here waits.
        if (hits != nullptr) hits(user, cs, HitPhase::Launch, s.db->h_ids, k);

        // ---- the pool, on the bytes the doorbell published.  **THESE ARE LAYER `l`'s EXPERTS.**
        if (pool != nullptr) pool(user, s.db->h_x_f, s.db->h_ids, s.db->h_weights, g.n_embd, k, y_miss);

        // ---- **AND THEY ARE COMBINED BY LAYER `l`, NOT BY LAYER `l+1`.**
        //
        // This copy and the launch below are the whole of the L100 fix.  `moe_combine` multiplies `parts` by
        // THIS layer's `b.weights`, so `parts` must be THIS layer's expert outputs; the previous form copied
        // the pool's answer into `parts_dev` for the NEXT layer, which computed
        // `sum_j w_{l+1}[j] * expert_{ids_l,j}(x_l)` - both the selection and the input one layer stale while
        // the weights were current.  It produced finite, fluent, deterministic tokens that were not the
        // model's, and no timing test could see it.
        cudaMemcpyAsync(gr.parts_dev, y_miss, parts_bytes, cudaMemcpyHostToDevice, cs);
        // ---- **AND THEN THE COMBINE.**  Stream-ordered after the copy above, so `hit_out` is added to misses
        // that are already in `parts`, and before `post[l]`, whose `moe_combine` reads the sum.  A no-op when
        // there is no VRAM tier or nothing is resident, which is every layer until the cache warms.
        if (hits != nullptr) hits(user, cs, HitPhase::Combine, nullptr, 0);
        {
            const cudaError_t pe = cudaGraphLaunch(gr.posts[l], cs);
            if (pe != cudaSuccess) {
                err = "session_loop: launch post[" + std::to_string(l) + "]: " + cudaGetErrorString(pe);
                return false;
            }
        }
        // ---- the C1 oracle.  `s.R` now holds layer `l`'s residual and keeps it until `post[l+1]` writes it, so
        // a copy enqueued HERE - after `post[l]`, before `pre[l+1]` - is ordered correctly by the stream alone.
        // Nothing is synchronised: the loop's existing final sync is what completes these.  See the note on
        // `dump_layers` in session.hpp for why this is an enqueue and not a read.
        if (dump_layers != nullptr) {
            const size_t n = (size_t) g.hc * (size_t) g.n_embd;
            const cudaError_t de = cudaMemcpyAsync(dump_layers + (size_t) (l + 1) * n, s.R, n * sizeof(float),
                                                   cudaMemcpyDeviceToHost, cs);
            if (de != cudaSuccess) {
                err = "session_loop: dump layer " + std::to_string(l) + ": " + cudaGetErrorString(de);
                return false;
            }
        }
        if (!overlap) {
            // THE COMPARISON ARM.  The same sequence with the pipeline removed: the CPU does not start until the
            // layer is over, so nothing overlaps and the difference is attributable to the doorbell alone.
            if (cudaStreamSynchronize(cs) != cudaSuccess) {
                err = "session_loop: sync at layer " + std::to_string(l);
                return false;
            }
        }
        // ---- and layer l+1's routing starts.  `pre[l+1]` touches none of the buffers `post[l]` reads, because
        // the two are ordered on one stream and `pre[l+1]`'s first use of `bb.mixed`/`bb.inject` is its own
        // `gr_read`, which comes after `post[l]` has consumed them.
        if (l + 1 < g.n_layers) {
            const cudaError_t ne = cudaGraphLaunch(gr.execs[l + 1], cs);
            if (ne != cudaSuccess) {
                err = "session_loop: launch pre[" + std::to_string(l + 1) + "]: " + cudaGetErrorString(ne);
                return false;
            }
            cudaEventRecord(probe, cs);
        }
        // ---- **THE SECOND HALF OF THE ROUND TRIP, AND THE HALF NOTHING WAS MEASURING.**
        //
        // From seeing the ring to having queued the next `pre`.  Everything in here is a driver call, and on
        // the token path the GPU has nothing left to run for most of it - `pre[l]` has already rung, and
        // `post[l]` cannot start until this code launches it.  So this interval is GPU-idle time, and it is
        // the number that decides whether the fix is R2.4 (fewer launches per layer) or a faster kernel.
        //
        // It INCLUDES the `!overlap` synchronisation when that arm is selected, which is deliberate: that arm
        // exists to show what the pipeline is worth, and hiding its cost here would defeat the comparison.
        gr.ms_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ring).count();
    }
    if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "session_loop: final sync"; return false; }
    return true;
}

bool session_token(const WeightTable& tables, const ModelGeometry& g, int64_t pos, int32_t pos_base,
                   SessionState& s, const float* parts, void* stream, bool sync_every_layer,
                   std::string& err) {
    cudaStream_t cs = (cudaStream_t) stream;
    int64_t qsa_index = 0;
    int64_t gdn_index = 0;

    for (int64_t l = 0; l < g.n_layers; ++l) {
        // THE GDN LAYERS EACH GET THEIR OWN STATE, and `GdnBuffers` carries it - so the session points the
        // shared scratch at the right slice before each call.  Sharing one state across 36 layers would make
        // every layer start from the previous layer's recurrence, which produces a perfectly finite answer.
        if (!is_qsa_layer(g, l)) {
            s.gdn.state = s.gdn_state + (size_t) gdn_index * gdn_state_floats(g);
            s.gdn.conv_state = s.gdn.state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
            ++gdn_index;
        }

        err.clear();
        const bool ok = is_qsa_layer(g, l)
            ? block_layer(tables, g, l, pos, pos_base, s.gdn, s.qsa_states[qsa_index], s.qsa_bufs, s.moe, s.k,
                          s.block, parts, stream, err, nullptr, s.ple.ready() ? &s.ple : nullptr)
            : block_layer(tables, g, l, pos, pos_base, s.gdn, s.qsa_states[0], s.qsa_bufs, s.moe, s.k,
                          s.block, parts, stream, err, nullptr, s.ple.ready() ? &s.ple : nullptr);
        if (!ok) {
            err = "layer " + std::to_string(l) + ": " + err;
            return false;
        }
        if (is_qsa_layer(g, l)) ++qsa_index;

        if (sync_every_layer) {
            // THE DEBUG MODE, and its cost is the point: it is a real synchronisation after every layer, which
            // P2.X3 forbids in the fast path.  It exists to turn an ASYNCHRONOUS error - a kernel reading a
            // buffer a later layer wrote, or an out-of-range launch - into a failure AT THE LAYER THAT CAUSED
            // IT, instead of a wrong number 40 layers later or at the end of the token.
            const cudaError_t e = cudaStreamSynchronize(cs);
            if (e != cudaSuccess) {
                err = "layer " + std::to_string(l) + ": " + cudaGetErrorString(e);
                return false;
            }
        }
    }
    return true;
}

}  // namespace strata::core

namespace strata::core {

bool session_capture_token(const WeightTable& tables, const ModelGeometry& g, SessionState& s, float* parts_dev,
                           const float* y_miss_host, size_t parts_bytes, TokenGraph& tg, std::string& err,
                           const TokenHits* hits) {
    if (hits != nullptr && !hits->on()) { err = "session_capture_token: incomplete hit configuration"; return false; }
    if (tg.captured) return true;
    if (s.db == nullptr || s.db->d_flag == nullptr || s.db->d_seq == nullptr) {
        err = "session_capture_token: the doorbell has no flag";
        return false;
    }
    if (parts_dev == nullptr || y_miss_host == nullptr || parts_bytes == 0) {
        err = "session_capture_token: parts buffers are required";
        return false;
    }
    float* y_dev = nullptr;
    if (cudaHostGetDevicePointer((void**) &y_dev, const_cast<float*>(y_miss_host), 0) != cudaSuccess || !y_dev) {
        err = "session_capture_token: the parts staging is not mapped pinned memory";
        return false;
    }
    cudaStream_t cs = nullptr;
    if (cudaStreamCreate(&cs) != cudaSuccess) { err = "session_capture_token: stream create failed"; return false; }
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        cudaStreamDestroy(cs);
        err = "session_capture_token: begin capture failed";
        return false;
    }
    int64_t qsa_index = 0;
    bool ok = true;
    for (int64_t l = 0; l < g.n_layers && ok; ++l) {
        gdn_point_at(g, l, s);
        const bool qsa = is_qsa_layer(g, l);
        QsaState& qst = qsa ? s.qsa_states[qsa_index] : s.qsa_states[0];
        err.clear();
        ok = block_layer_pre(tables, g, l, 0, 0, s.gdn, qst, s.qsa_bufs, s.moe, s.k, s.block, (void*) cs, err, s.db,
                             s.ple.ready() ? &s.ple : nullptr);
        if (!ok) { err = "session_capture_token: pre layer " + std::to_string(l) + ": " + err; break; }
        if (hits != nullptr) {
            // After the ring (and the shared expert): the GPU's experts run while the CPU computes the misses.
            strata::kernels::moe_hit_select(s.moe.ids, hits->d_res + l * hits->n_expert, (int) s.k,
                                            (int) hits->n_expert, hits->d_slot, hits->d_dst, hits->d_count, (void*) cs);
            strata::kernels::quantize_q8_0_scaled(s.block.mixed, hits->x_q8, hits->x_scale, g.n_embd, (void*) cs);
            strata::kernels::moe_hit_grouped_s2_dev(hits->cache_base, hits->d_slot, hits->d_dst, hits->d_count, s.k,
                                                    hits->blob, hits->x_q8, hits->scratch, hits->hit_out, (void*) cs,
                                                    hits->x_scale);
        }
        strata::kernels::doorbell_wait(s.db->d_flag, s.db->d_seq, (void*) cs);
        // A kernel, not a memcpy node: a copy-engine node splits the WDDM submission (measured 67 flushes/token).
        strata::kernels::copy_from_mapped(parts_dev, y_dev, (int64_t) (parts_bytes / sizeof(float)), (void*) cs);
        if (hits != nullptr)
            strata::kernels::moe_hit_add(parts_dev, hits->hit_out, hits->d_dst, hits->d_count, s.k, g.n_embd, (void*) cs);
        ok = block_layer_post(tables, g, l, s.k, s.moe, s.block, parts_dev, (void*) cs, err);
        if (!ok) { err = "session_capture_token: post layer " + std::to_string(l) + ": " + err; break; }
        if (qsa) ++qsa_index;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    cudaStreamDestroy(cs);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess) {
        err = std::string("session_capture_token: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const cudaError_t ie = cudaGraphInstantiate(&tg.exec, graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        err = std::string("session_capture_token: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    tg.captured = true;
    tg.n_layers = g.n_layers;
    tg.y_src = y_miss_host;
    tg.parts_bytes = parts_bytes;
    return true;
}

bool session_run_token(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s, TokenGraph& tg,
                       PoolFn pool, void* user, float* y_miss_host, void* stream, std::string& err) {
    if (!tg.captured) { err = "session_run_token: not captured"; return false; }
    if (y_miss_host != tg.y_src) { err = "session_run_token: the staging buffer is not the captured one"; return false; }
    cudaStream_t cs = (cudaStream_t) stream;
    ++tg.calls;
    stage_token(g, pos, pos_base, s);
    doorbell_reset(*s.db);
    const cudaError_t le = cudaGraphLaunch(tg.exec, cs);
    if (le != cudaSuccess) { err = std::string("session_run_token: launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs);                     // one flush, so WDDM submits the graph now
    static const int flush_us = [] {
        const char* e = std::getenv("STRATA_TG_FLUSH_US");
        return e ? std::atoi(e) : 2000;   // 24 Sep: 0 flushes run as fast as 5 us ones; this only notices faults
    }();
    volatile uint32_t* const seq = s.db->h_seq;
    volatile uint32_t* const flag = s.db->h_flag;
    using Clock = std::chrono::steady_clock;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const uint32_t want = (uint32_t) (l + 1);
        const auto t0 = Clock::now();
        auto last_flush = t0;
        uint32_t spins = 0;
        progress_at("token: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            STRATA_SPIN_PAUSE();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(flush_us)) {
                // A slow ring: flush the submission queue once more, and notice a fault or a finished graph.
                last_flush = now;
                ++tg.flushes;
                const cudaError_t q = cudaStreamQuery(cs);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "session_run_token: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - t0 > std::chrono::seconds(20)) {
                err = "session_run_token: timed out waiting for layer " + std::to_string(l);
                return false;
            }
        }
        strata_load_fence();   // as above, for the token loop's wait
        const auto t1 = Clock::now();
        progress_at("token: the CPU experts of layer", l);
        if (pool != nullptr) pool(user, s.db->h_x_f, s.db->h_ids, s.db->h_weights, g.n_embd, s.k, y_miss_host);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        strata_store_fence();
        *flag = want;
        const auto t2 = Clock::now();
        tg.ms_wait += std::chrono::duration<double, std::milli>(t1 - t0).count();
        tg.ms_pool += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    progress_at("token: waiting for the GPU to finish the token");
    const cudaError_t se = cudaStreamSynchronize(cs);
    if (se != cudaSuccess) { err = std::string("session_run_token: ") + cudaGetErrorString(se); return false; }
    progress_at("decode");
    progress_beat();
    return true;
}

void token_graph_free(TokenGraph& tg) {
    if (tg.exec) cudaGraphExecDestroy(tg.exec);
    tg = TokenGraph{};
}

}  // namespace strata::core
