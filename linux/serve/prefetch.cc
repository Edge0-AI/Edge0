#include "prefetch.h"
#include "prefetch_internal.h"
#include "expert_pool.h"

#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "ggml-cpu/ggml-cpu-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
struct PrefetchState {
    std::unique_ptr<edge0::PrefetchWorker> worker;
    std::unique_ptr<edge0::ExpertPool> pool;
    llama_context * context = nullptr;
};
// Never destroyed: edge0_pref_shutdown() releases it; a forced exit must not run destructors that take locks.
PrefetchState & prefetch_state() { static auto & state = *new PrefetchState; return state; }
const char * pool_resolve(const ggml_tensor * tensor, int64_t expert, size_t stride) {
    auto & state = prefetch_state();
    if (state.pool) {
        if (const char * data = state.pool->resolve(tensor->data, expert, stride)) return data;
    }
    return static_cast<const char *>(tensor->data) + expert * stride;
}
}

bool edge0_pref_init(llama_model * model, llama_context * ctx, int pool_mib, const char * gguf_path) {
    edge0_pref_shutdown();
    const char * sidecar = std::getenv("E0_PREROUTER");
    size_t pool_bytes = 0;
    if (!edge0::resolve_mib(pool_mib, std::getenv("E0_POOL_MB"), pool_bytes)) {
        std::fprintf(stderr, "[pref-trace] invalid pool MiB value; advisory disabled\n");
        return false;
    }
    if ((!sidecar || !*sidecar) && !pool_bytes) return false;
    const char * mode = std::getenv("E0_PREFETCH");
    if (mode && std::strcmp(mode, "0") && std::strcmp(mode, "1")) {
        std::fprintf(stderr, "[pref-trace] E0_PREFETCH must be 0 or 1; advisory disabled\n");
        return false;
    }
    if (!model || !ctx) return false;
    const char * path = gguf_path && *gguf_path ? gguf_path : std::getenv("E0_GGUF");
    if (!path || !*path) { std::fprintf(stderr, "[pref-trace] GGUF path required\n"); return false; }
    try {
        auto & state = prefetch_state();
        auto engine = std::make_unique<edge0::PrefetchEngine>([&state](const edge0::ReadOnlyFile & file, edge0::ByteRange range) {
            if (state.pool) state.pool->prefetch(range);
            return file.advise(range);
        });
        if (!engine->load(path, sidecar, mode && !std::strcmp(mode, "1"))) return false;
        const auto & shape = engine->shape();
        if (shape.hidden != llama_model_n_embd(model) || shape.layers != llama_model_n_layer(model) ||
            shape.experts != llama_model_n_expert(model)) {
            std::fprintf(stderr, "[pref-trace] loaded model/sidecar mismatch; advisory disabled\n");
            return false;
        }
        if (pool_bytes) {
            state.pool = std::make_unique<edge0::ExpertPool>(engine->file());
            if (!state.pool->start(pool_bytes)) { state.pool.reset(); return false; }
            size_t enrolled = 0;
            for (const auto & tensor : engine->tensors()) {
                const ggml_tensor * loaded = model->get_tensor(tensor.name.c_str());
                if (loaded && state.pool->add(loaded->data, tensor)) ++enrolled;
            }
            std::fprintf(stderr, "[pref-trace] pool enabled: %zu MiB tensors=%zu (immutable slots, mmap fallback)\n",
                         pool_bytes >> 20, enrolled);
        }
        std::fprintf(stderr, "[pref-trace] INIT ok: H=%d E=%d K=%d layers=%zu prefetch=%s\n",
                     shape.hidden, shape.experts, shape.topk, engine->capture_layers().size(),
                     mode && !std::strcmp(mode, "1") ? "on" : "off");
        const char * async = std::getenv("E0_PREF_ASYNC");
        const bool heads = !engine->capture_layers().empty();
        state.worker = std::make_unique<edge0::PrefetchWorker>(std::move(engine), heads && (!async || std::strcmp(async, "0")));
        for (int layer : state.worker->engine().capture_layers()) llama_set_embeddings_layer_inp(ctx, layer, true);
        state.context = ctx;
        if (state.pool) ggml_edge0_set_mmid_resolver(pool_resolve);
        return true;
    } catch (const std::exception & error) {
        edge0_pref_shutdown();
        std::fprintf(stderr, "[pref-trace] init failed: %s; advisory disabled\n", error.what());
        return false;
    }
}

void edge0_pref_on_step(llama_context * ctx, int64_t last_row) {
    auto & state = prefetch_state();
    if (!state.worker || ctx != state.context) return;
    const auto & engine = state.worker->engine();
    if (engine.capture_layers().empty()) return;
    if (last_row < 0 || static_cast<uint64_t>(last_row) >= llama_n_batch(ctx)) {
        std::fprintf(stderr, "[pref-trace] invalid capture row; snapshot skipped\n");
        return;
    }
    llama_synchronize(ctx);
    const size_t hidden = engine.shape().hidden;
    try {
        std::vector<float> rows(engine.capture_layers().size() * hidden);
        for (size_t i = 0; i < engine.capture_layers().size(); ++i) {
            const float * data = llama_get_embeddings_layer_inp(ctx, engine.capture_layers()[i]);
            if (!data) return;
            std::copy_n(data + static_cast<size_t>(last_row) * hidden, hidden, rows.data() + i * hidden);
        }
        state.worker->submit(rows);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[pref-trace] advisory snapshot failed: %s; inference continues\n", error.what());
    }
}

void edge0_pref_shutdown() {
    auto & state = prefetch_state();
    if (state.pool) ggml_edge0_set_mmid_resolver(nullptr);
    if (state.worker && state.context) {
        for (int layer : state.worker->engine().capture_layers()) llama_set_embeddings_layer_inp(state.context, layer, false);
    }
    state.worker.reset();
    state.pool.reset();
    state.context = nullptr;
}
