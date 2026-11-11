// qwen_image_2_1_native.hpp - Strata's native Qwen-Image-2.1 DiT runner for stable-diffusion.cpp (CUDA / HIP builds).
//
// Copied into src/model/diffusion/ by tools/image/sdcpp_native/apply.py and chosen by model_builders.cpp when the
// model args contain qwen_image_2_1_native=1.  sd.cpp keeps the text encoder, the VAE, the sampler and the RNG; this
// runner replaces only QwenImage21Runner::compute (latent + timestep + text condition in, velocity out, on the host).
//
// Per step: the 32 blocks run on the image tokens only.  The text prefix (causal within the text, timestep-free:
// its modulation row is the t = 0 one) goes through all blocks once per prompt and leaves each layer's K/V (fp16)
// behind - the same prefix cache sd.cpp keeps.  GEMMs are ggml graphs on the runner's backend (the MMQ / BLAS
// kernels sd.cpp uses, same numerics); LayerNorm+modulate, gate+residual, q/k RMSNorm+RoPE, SwiGLU and (on gfx11)
// the image attention are Strata kernels (ggml/src/ggml-cuda/qi21-native.cu) on the backend's stream.  Anything this
// runner does not cover (reference images, an unfused MLP, a CPU / Vulkan backend, too little VRAM) goes to the
// stock runner.
#ifndef __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_NATIVE_H__
#define __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_NATIVE_H__

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "model/diffusion/qwen_image_2_1.hpp"

extern "C" {
void* qi21_stream(ggml_backend_t backend);
int qi21_has_wmma_attn(ggml_backend_t backend);
int qi21_res_ln_mod(void* stream, float* x, const float* h, const float* g, float* out, const float* m, int T, int C,
                    float eps);
int qi21_qkv_post(void* stream, const float* q, const float* k, const float* v, int ld, const float* wq, const float* wk,
                  const float* pe, void* qo, void* qlo, int q_f16, int q_tstride, void* ko, void* vo, int kv_tstride,
                  int kv_off, int T, int H, float eps);
int qi21_swiglu(void* stream, const float* gu, float* out, int T, int F);
int qi21_tanh(void* stream, const float* in, float* out, int n);
int qi21_res_ln_mod2(void* stream, float* x, const float* h, const float* tg, float* out, const float* m, int T, int C,
                     float eps);
int qi21_attn_causal(void* stream, const float* q, int q_tstride, const void* k, const void* v, int kv_tstride,
                     float* out, int H, int Lq, int Lk, int causal);
int qi21_attn_wmma(void* stream, const void* q, const void* qlo, int q_tstride, const void* k, const void* v,
                   int kv_tstride, float* out, int H, int nq, int nkv);
int qi21_memcpy_d2d(void* stream, void* dst, const void* src, size_t n);
}

namespace Qwen {

    struct QwenImage21NativeRunner : public QwenImage21Runner {
        static constexpr int64_t C = 4096, D = 128, H = 32, F = 12288;
        std::string dit_path;
        bool disabled = false;           // a failure: the stock runner from here on
        bool loaded   = false;
        bool wmma_attn = false;
        bool timing    = false;
        bool said_why  = false;
        bool qsplit    = false;
        bool exact_attn = false;
        bool fast_ln    = true;
        bool fuse_qkv   = true;

        // weights (our own copy: the stock runner's params are never loaded, they load lazily on first use)
        ggml_context* wctx               = nullptr;
        ggml_backend_buffer_t wbuf       = nullptr;
        std::map<std::string, ggml_tensor*> W;

        // activations, sized for t_cap image tokens and l_cap prefix tokens
        ggml_context* actx         = nullptr;
        ggml_backend_buffer_t abuf = nullptr;
        int64_t t_cap = 0, l_cap = 0;
        ggml_tensor *x, *h, *q, *k, *v, *att, *gu, *mlp, *q16, *kw, *vw, *pe_img, *pe_txt, *latent, *outp, *timein,
            *mod, *gates, *scale_out, *ctx_in, *pk, *pv;

        // the prefix cache: per layer K/V of the text tokens (fp16, head-major), for one condition
        uint64_t prefix_key = 0;
        int64_t prefix_len  = -1;

        std::vector<char> gbuf;          // ggml graph metadata arena

        QwenImage21NativeRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix,
                                std::shared_ptr<RunnerWeightManager> weight_manager, const char* model_args,
                                std::string dit_path)
            : QwenImage21Runner(backend, weights, prefix, weight_manager, model_args), dit_path(std::move(dit_path)) {
            timing = getenv("STRATA_QI_TIMING") != nullptr;
            qsplit     = getenv("STRATA_QI_QSPLIT") != nullptr;      // measured: no accuracy gain, +7% step time
            exact_attn = getenv("STRATA_QI_EXACT_ATTN") != nullptr;
            fuse_qkv   = getenv("STRATA_QI_NO_QKV_FUSE") == nullptr;
            fast_ln    = getenv("STRATA_QI_FAST_LN") != nullptr;    // measured: no gain on the 3060 (2.10 vs 2.10 s/step)
            gbuf.resize(64 << 20);
        }

        ~QwenImage21NativeRunner() override { release(); }

        std::string get_desc() override { return "qwen_image_2_1 (strata native)"; }

        void release() {
            if (abuf) ggml_backend_buffer_free(abuf);
            if (actx) ggml_free(actx);
            if (wbuf) ggml_backend_buffer_free(wbuf);
            if (wctx) ggml_free(wctx);
            abuf = nullptr; actx = nullptr; wbuf = nullptr; wctx = nullptr;
            W.clear();
            loaded = false;
            t_cap = l_cap = 0;
            prefix_len = -1;
        }

        void fail(const char* why) {
            LOG_WARN("Qwen Image 2.1 native: %s; using the stock runner", why);
            release();
            disabled = true;
        }

        ggml_tensor* w(const std::string& name) {
            auto it = W.find("model.diffusion_model." + name);
            return it == W.end() ? nullptr : it->second;
        }

        // ---------------------------------------------------------------- weights
        bool load_weights() {
            ggml_context* meta = nullptr;
            gguf_init_params gp{true, &meta};
            gguf_context* g = gguf_init_from_file(dit_path.c_str(), gp);
            if (g == nullptr) return false;
            const int64_t n = gguf_get_n_tensors(g);
            wctx = ggml_init({(size_t) (n + 8) * ggml_tensor_overhead(), nullptr, true});
            struct Piece { ggml_tensor* t; size_t file_off, dst_off; };
            std::vector<Piece> plan;
            std::map<ggml_tensor*, ggml_type> widened;
            const size_t base = gguf_get_data_offset(g);
            // q, k and v of a block in one tensor when they have the same type (one GEMM, one activation quantize):
            // rows stay rows, so the three weights are simply laid one after the other
            auto qkv_name = [](int b, const char* w) {
                return "model.diffusion_model.transformer_blocks." + std::to_string(b) + ".attn.to_" + w + ".weight";
            };
            std::map<std::string, std::pair<ggml_tensor*, size_t>> fused_part;   // q/k/v name -> (fused, offset)
            if (fuse_qkv) {
                for (int b = 0; b < config.num_layers; ++b) {
                    ggml_tensor* tq = ggml_get_tensor(meta, qkv_name(b, "q").c_str());
                    ggml_tensor* tk = ggml_get_tensor(meta, qkv_name(b, "k").c_str());
                    ggml_tensor* tv = ggml_get_tensor(meta, qkv_name(b, "v").c_str());
                    if (!tq || !tk || !tv || tq->type != tk->type || tq->type != tv->type || tq->ne[0] != tk->ne[0] ||
                        tq->ne[1] != tk->ne[1] || tq->ne[1] != tv->ne[1])
                        continue;
                    ggml_tensor* f = ggml_new_tensor_2d(wctx, tq->type, tq->ne[0], 3 * tq->ne[1]);
                    const std::string fname = "model.diffusion_model.transformer_blocks." + std::to_string(b) +
                                              ".attn.qkv.weight";
                    ggml_set_name(f, fname.c_str());
                    W[fname] = f;
                    const size_t nb = ggml_nbytes(tq);
                    fused_part[qkv_name(b, "q")] = {f, 0};
                    fused_part[qkv_name(b, "k")] = {f, nb};
                    fused_part[qkv_name(b, "v")] = {f, 2 * nb};
                }
            }
            for (int64_t i = 0; i < n; ++i) {
                const char* name = gguf_get_tensor_name(g, i);
                if (strncmp(name, "model.diffusion_model.", 22) != 0) continue;
                auto fp = fused_part.find(name);
                if (fp != fused_part.end()) {
                    plan.push_back({fp->second.first, base + gguf_get_tensor_offset(g, i), fp->second.second});
                    continue;
                }
                ggml_tensor* src = ggml_get_tensor(meta, name);
                // a 1-D norm weight in BF16 / F16 is kept as F32 (elementwise ops on it then work on every backend)
                const bool widen = ggml_n_dims(src) == 1 && src->type != GGML_TYPE_F32;
                ggml_tensor* t   = widen ? ggml_new_tensor_1d(wctx, GGML_TYPE_F32, src->ne[0]) : ggml_dup_tensor(wctx, src);
                if (widen) widened[t] = src->type;
                ggml_set_name(t, name);
                W[name] = t;
                plan.push_back({t, base + gguf_get_tensor_offset(g, i), 0});
            }
            wbuf = ggml_backend_alloc_ctx_tensors(wctx, runtime_backend);
            if (wbuf == nullptr) {
                gguf_free(g);
                ggml_free(meta);
                return false;
            }
            FILE* f = fopen(dit_path.c_str(), "rb");
            std::vector<char> stage(64 << 20);
            bool ok = f != nullptr;
            for (auto& [t, off, dst_off] : plan) {
                if (!ok) break;
                auto wi = widened.find(t);
                const bool part = std::string(ggml_get_name(t)).find(".attn.qkv.") != std::string::npos;
                const size_t nb = part ? ggml_nbytes(t) / 3
                                       : wi == widened.end() ? ggml_nbytes(t) : ggml_row_size(wi->second, t->ne[0]);
#ifdef _WIN32
                ok = _fseeki64(f, (long long) off, SEEK_SET) == 0;
#else
                ok = fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
                if (wi != widened.end()) {
                    std::vector<char> raw(nb);
                    std::vector<float> f32(t->ne[0]);
                    ok = fread(raw.data(), 1, nb, f) == nb;
                    if (ok) {
                        ggml_get_type_traits(wi->second)->to_float(raw.data(), f32.data(), t->ne[0]);
                        ggml_backend_tensor_set(t, f32.data(), 0, f32.size() * sizeof(float));
                    }
                    continue;
                }
                for (size_t done = 0; ok && done < nb;) {
                    const size_t c = std::min(stage.size(), nb - done);
                    ok = fread(stage.data(), 1, c, f) == c;
                    if (ok) ggml_backend_tensor_set(t, stage.data(), dst_off + done, c);
                    done += c;
                }
            }
            if (f) fclose(f);
            gguf_free(g);
            ggml_free(meta);
            if (!ok) return false;
            for (int i = 0; i < config.num_layers; ++i) {
                const std::string b = "transformer_blocks." + std::to_string(i) + ".";
                if (w(b + "attn.qkv.weight") == nullptr &&
                    (!w(b + "attn.to_q.weight") || !w(b + "attn.to_k.weight") || !w(b + "attn.to_v.weight")))
                    return false;
                for (const char* s : {"attn.to_out.0.weight",
                                      "attn.norm_q.weight", "attn.norm_k.weight", "img_mlp.gate_up.weight",
                                      "img_mlp.out.weight"}) {
                    if (w(b + s) == nullptr) return false;
                }
                if (w(b + "attn.norm_q.weight")->type != GGML_TYPE_F32 || w(b + "attn.norm_k.weight")->type != GGML_TYPE_F32)
                    return false;
            }
            for (const char* s : {"img_in.weight", "txt_in.text_norm.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight",
                                  "time_text_embed.timestep_embedder.linear_1.weight",
                                  "time_text_embed.timestep_embedder.linear_2.weight", "modulation.1.weight",
                                  "norm_out.linear.weight", "proj_out.weight"}) {
                if (w(s) == nullptr) return false;
            }
            LOG_INFO("Qwen Image 2.1 native: %zu tensors, %.2f GB on %s%s", W.size(),
                     ggml_backend_buffer_get_size(wbuf) / 1e9, ggml_backend_name(runtime_backend),
                     wmma_attn ? " (Strata attention)" : "");
            return true;
        }

        // ---------------------------------------------------------------- activations
        bool alloc_acts(int64_t T, int64_t L) {
            if (T <= t_cap && L <= l_cap) return true;
            if (abuf) ggml_backend_buffer_free(abuf);
            if (actx) ggml_free(actx);
            t_cap = std::max(T, t_cap);
            l_cap = std::max(L, l_cap);
            const int64_t R = std::max(t_cap, l_cap);     // rows of the shared work buffers
            actx  = ggml_init({64 * ggml_tensor_overhead(), nullptr, true});
            x     = ggml_new_tensor_2d(actx, GGML_TYPE_F32, C, R);
            h     = ggml_new_tensor_2d(actx, GGML_TYPE_F32, C, R);
            q     = ggml_new_tensor_2d(actx, GGML_TYPE_F32, C, R);
            k     = ggml_new_tensor_2d(actx, GGML_TYPE_F32, C, R);
            v     = ggml_new_tensor_2d(actx, GGML_TYPE_F32, C, R);
            att   = ggml_new_tensor_2d(actx, GGML_TYPE_F32, C, R);
            gu    = ggml_new_tensor_2d(actx, GGML_TYPE_F32, 2 * F, R);
            mlp   = ggml_new_tensor_2d(actx, GGML_TYPE_F32, F, R);
            q16   = ggml_new_tensor_3d(actx, GGML_TYPE_F32, D, R, H);     // fp16 or fp32 queries, head-major
            kw    = ggml_new_tensor_3d(actx, GGML_TYPE_F16, D, l_cap + t_cap, H);
            vw    = ggml_new_tensor_3d(actx, GGML_TYPE_F16, D, l_cap + t_cap, H);
            pe_img = ggml_new_tensor_2d(actx, GGML_TYPE_F32, 4 * D / 2, t_cap);
            pe_txt = ggml_new_tensor_2d(actx, GGML_TYPE_F32, 4 * D / 2, std::max<int64_t>(l_cap, 1));
            latent = ggml_new_tensor_2d(actx, GGML_TYPE_F32, config.in_channels, t_cap);
            outp   = ggml_new_tensor_2d(actx, GGML_TYPE_F32, config.out_channels, t_cap);
            timein = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 2);
            mod    = ggml_new_tensor_2d(actx, GGML_TYPE_F32, 4 * C, 2);
            gates  = ggml_new_tensor_2d(actx, GGML_TYPE_F32, 4 * C, 2);     // tanh(mod), the gate rows used
            scale_out = ggml_new_tensor_1d(actx, GGML_TYPE_F32, C);
            ctx_in = ggml_new_tensor_2d(actx, GGML_TYPE_F32, config.context_dim, std::max<int64_t>(l_cap, 1));
            pk     = ggml_new_tensor_3d(actx, GGML_TYPE_F16, D * H, std::max<int64_t>(l_cap, 1), config.num_layers);
            pv     = ggml_new_tensor_3d(actx, GGML_TYPE_F16, D * H, std::max<int64_t>(l_cap, 1), config.num_layers);
            abuf   = ggml_backend_alloc_ctx_tensors(actx, runtime_backend);
            prefix_len = -1;                               // the cache lived in the old buffer
            return abuf != nullptr;
        }

        // ---------------------------------------------------------------- ggml graph helpers
        struct G {
            ggml_context* ctx;
            ggml_cgraph* gf;
        };
        G begin() {
            ggml_context* c = ggml_init({gbuf.size(), gbuf.data(), true});
            return {c, ggml_new_graph_custom(c, 256, false)};
        }
        static ggml_tensor* rows(ggml_context* c, ggml_tensor* t, int64_t n, int64_t first = 0) {
            return ggml_view_2d(c, t, t->ne[0], n, t->nb[1], first * t->nb[1]);
        }
        static void bind(ggml_tensor* node, ggml_tensor* storage, size_t offset = 0) {
            node->data   = (char*) storage->data + offset;
            node->buffer = storage->buffer;
        }
        // compute a graph on the backend's stream (no wait).  Nodes the caller did not bind get scratch space in
        // `gu` (only the small prelude / head / txt_in graphs have such nodes, and gu is free then); views get
        // their source's memory.
        void run(G& g, std::initializer_list<ggml_tensor*> outs) {
            for (auto* o : outs) ggml_build_forward_expand(g.gf, o);
            size_t off = 0;
            for (ggml_tensor* t = ggml_get_first_tensor(g.ctx); t; t = ggml_get_next_tensor(g.ctx, t)) {
                if (t->view_src != nullptr) {
                    t->data   = (char*) t->view_src->data + t->view_offs;
                    t->buffer = t->view_src->buffer;
                } else if (t->data == nullptr) {
                    bind(t, gu, off);
                    off += GGML_PAD(ggml_nbytes(t), 256);
                    GGML_ASSERT(off <= ggml_nbytes(gu));
                }
            }
            ggml_backend_graph_compute_async(runtime_backend, g.gf);
            ggml_free(g.ctx);
        }
        // dst[N x n] = W . src[K x n] (rows `first..`)
        void gemm(G& g, ggml_tensor* weight, ggml_tensor* src, ggml_tensor* dst, int64_t n) {
            ggml_tensor* y = ggml_mul_mat(g.ctx, weight, rows(g.ctx, src, n));
            bind(y, dst);
            ggml_build_forward_expand(g.gf, y);
        }

        // ---------------------------------------------------------------- the transformer blocks
        // rows [0, n) of x are the tokens; mrow = 0 (image) or 1 (text prefix); pe: per-token rope table;
        // attention: image -> [cached prefix K/V of the layer, these tokens' K/V]; prefix -> causal over itself, and
        // each layer's K/V goes into the prefix cache.
        bool blocks_pass(int64_t n, int mrow, ggml_tensor* pe, int64_t L) {
            void* s       = qi21_stream(runtime_backend);
            const float* m = (const float*) mod->data + mrow * 4 * C;
            const float *m0 = m, *m1 = m + C, *m2 = m + 2 * C, *m3 = m + 3 * C;
            const float* tgr = (const float*) gates->data + mrow * 4 * C;   // tanh of the same row
            const float *t1 = tgr + C, *t3 = tgr + 3 * C;
            auto res_ln = [&](float* xx, const float* hh, const float* graw, const float* tgate, float* out,
                              const float* mm) {
                return fast_ln ? qi21_res_ln_mod2(s, xx, hh, tgate, out, mm, (int) n, (int) C, 1e-6f)
                               : qi21_res_ln_mod(s, xx, hh, graw, out, mm, (int) n, (int) C, 1e-6f);
            };
            const bool prefix = mrow == 1;
            const int64_t kvt = prefix ? n : L + n;       // keys per head in the work K/V
            const bool use_wmma = !prefix && wmma_attn && !exact_attn;
            // fp16 queries carry ~5e-4 relative error into logits that reach the tens here: the low half of each
            // query (q - fp16(q)) goes through a second WMMA (STRATA_QI_QSPLIT=0 drops it).  It sits in the fp32-sized
            // q16 buffer right after the fp16 high half.
            const size_t qlo_off = (size_t) H * n * D * sizeof(ggml_fp16_t);
            float* X = (float*) x->data;
            float* Hh = (float*) h->data;
            int rc = res_ln(X, nullptr, nullptr, nullptr, Hh, m0);
            for (int i = 0; i < config.num_layers && rc == 0; ++i) {
                const std::string b = "transformer_blocks." + std::to_string(i) + ".";
                ggml_tensor* wqkv = w(b + "attn.qkv.weight");
                const float *qs = (const float*) q->data, *ks = (const float*) k->data, *vs = (const float*) v->data;
                int ldq = (int) C;
                {
                    G g = begin();
                    if (wqkv != nullptr) {                 // [3C x n] into gu (free here), q | k | v per token
                        gemm(g, wqkv, h, gu, n);
                        qs = (const float*) gu->data;
                        ks = qs + C;
                        vs = qs + 2 * C;
                        ldq = (int) (3 * C);
                    } else {
                        gemm(g, w(b + "attn.to_q.weight"), h, q, n);
                        gemm(g, w(b + "attn.to_k.weight"), h, k, n);
                        gemm(g, w(b + "attn.to_v.weight"), h, v, n);
                    }
                    run(g, {});
                }
                const int64_t off = prefix ? 0 : L;
                rc = qi21_qkv_post(s, qs, ks, vs, ldq,
                                   (const float*) w(b + "attn.norm_q.weight")->data,
                                   (const float*) w(b + "attn.norm_k.weight")->data, (const float*) pe->data, q16->data,
                                   use_wmma && qsplit ? (char*) q16->data + qlo_off : nullptr, use_wmma ? 1 : 0, (int) n, kw->data, vw->data, (int) kvt, (int) off, (int) n, (int) H,
                                   1e-6f);
                if (rc) break;
                const size_t pbytes = (size_t) L * D * sizeof(ggml_fp16_t);   // one head's prefix rows
                if (prefix) {
                    // keep this layer's text K/V (head-major rows 0..n) in the prefix cache, as [H][n][D]
                    for (int64_t hh = 0; hh < H && rc == 0; ++hh) {
                        char* pkd = (char*) pk->data + i * pk->nb[2] + hh * n * D * sizeof(ggml_fp16_t);
                        char* pvd = (char*) pv->data + i * pv->nb[2] + hh * n * D * sizeof(ggml_fp16_t);
                        rc = qi21_memcpy_d2d(s, pkd, (char*) kw->data + hh * kvt * D * sizeof(ggml_fp16_t),
                                             n * D * sizeof(ggml_fp16_t));
                        rc |= qi21_memcpy_d2d(s, pvd, (char*) vw->data + hh * kvt * D * sizeof(ggml_fp16_t),
                                              n * D * sizeof(ggml_fp16_t));
                    }
                    if (rc) break;
                    rc = qi21_attn_causal(s, (const float*) q16->data, (int) n, kw->data, vw->data, (int) kvt,
                                          (float*) att->data, (int) H, (int) n, (int) n, 1);
                } else {
                    for (int64_t hh = 0; hh < H && L > 0 && rc == 0; ++hh) {
                        const char* pks = (const char*) pk->data + i * pk->nb[2] + hh * pbytes;
                        const char* pvs = (const char*) pv->data + i * pv->nb[2] + hh * pbytes;
                        rc = qi21_memcpy_d2d(s, (char*) kw->data + hh * kvt * D * sizeof(ggml_fp16_t), pks, pbytes);
                        rc |= qi21_memcpy_d2d(s, (char*) vw->data + hh * kvt * D * sizeof(ggml_fp16_t), pvs, pbytes);
                    }
                    if (rc) break;
                    if (exact_attn) {
                        // reference: fp32 attention (fp16 K/V as every path has them), slow; STRATA_QI_EXACT_ATTN=1
                        rc = qi21_attn_causal(s, (const float*) q16->data, (int) n, kw->data, vw->data, (int) kvt,
                                              (float*) att->data, (int) H, (int) n, (int) kvt, 0);
                    } else if (use_wmma) {
                        rc = qi21_attn_wmma(s, q16->data, qsplit ? (char*) q16->data + qlo_off : nullptr, (int) n,
                                            kw->data, vw->data, (int) kvt, (float*) att->data, (int) H, (int) n,
                                            (int) kvt);
                    } else {
                        // ggml's flash attention (what sd.cpp runs on this backend)
                        G g = begin();
                        ggml_tensor* qt = ggml_view_3d(g.ctx, q16, D, n, H, D * sizeof(float), n * D * sizeof(float), 0);
                        ggml_tensor* kt = ggml_view_3d(g.ctx, kw, D, kvt, H, D * sizeof(ggml_fp16_t),
                                                       kvt * D * sizeof(ggml_fp16_t), 0);
                        ggml_tensor* vt = ggml_view_3d(g.ctx, vw, D, kvt, H, D * sizeof(ggml_fp16_t),
                                                       kvt * D * sizeof(ggml_fp16_t), 0);
                        ggml_tensor* o  = ggml_flash_attn_ext(g.ctx, qt, kt, vt, nullptr, 1.f / sqrtf((float) D), 0, 0);
                        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
                        if (!ggml_backend_supports_op(runtime_backend, o)) {
                            ggml_free(g.ctx);
                            rc = 3;
                            break;
                        }
                        bind(o, att);
                        run(g, {o});
                    }
                }
                if (rc) break;
                {
                    G g = begin();
                    gemm(g, w(b + "attn.to_out.0.weight"), att, h, n);
                    run(g, {});
                }
                rc = res_ln(X, Hh, m1, t1, Hh, m2);
                if (rc) break;
                {
                    G g = begin();
                    gemm(g, w(b + "img_mlp.gate_up.weight"), h, gu, n);
                    run(g, {});
                }
                rc = qi21_swiglu(s, (const float*) gu->data, (float*) mlp->data, (int) n, (int) F);
                if (rc) break;
                {
                    G g = begin();
                    gemm(g, w(b + "img_mlp.out.weight"), mlp, h, n);
                    run(g, {});
                }
                const bool last = i + 1 == config.num_layers;
                rc = res_ln(X, Hh, m3, t3, last ? nullptr : Hh, m0);
            }
            if (rc) LOG_WARN("Qwen Image 2.1 native: a kernel failed (%d)", rc);
            return rc == 0;
        }

        // ---------------------------------------------------------------- compute
        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            if (disabled) return QwenImage21Runner::compute(n_threads, inputs);
            const auto& xin     = tensor_or_empty(inputs.x);
            const auto& context = tensor_or_empty(inputs.context);
            const auto* extra   = std::get_if<QwenImage21DiffusionExtra>(&inputs.extra);
            const bool refs     = inputs.ref_latents && inputs.ref_image_params.pass_to_dit && !inputs.ref_latents->empty();
            const char* why = nullptr;
            if (xin.empty() || context.empty() || xin.dim() != 4 || xin.shape()[3] != 1 ||
                xin.shape()[2] != config.in_channels || context.shape()[0] != config.context_dim ||
                tensor_or_empty(inputs.timesteps).numel() != 1)
                why = "unexpected input shapes";
            else if (refs)
                why = "reference images";
            else if (!config.fused_mlp)
                why = "an unfused MLP";
            else if (!ggml_backend_is_cuda(runtime_backend))
                why = "not a CUDA / HIP backend";
            if (why != nullptr) {
                if (!said_why) LOG_INFO("Qwen Image 2.1 native: %s; this step runs on the stock runner", why);
                said_why = true;
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (!loaded) {
                wmma_attn = qi21_has_wmma_attn(runtime_backend) && getenv("STRATA_QI_GGML_ATTN") == nullptr;
                if (!load_weights()) {
                    fail("could not load the DiT weights onto the GPU");
                    return QwenImage21Runner::compute(n_threads, inputs);
                }
                loaded = true;
            }
            const int64_t Wd = xin.shape()[0], Hd = xin.shape()[1], T = Wd * Hd, Lt = context.shape()[1];
            if (!alloc_acts(T, Lt)) {
                fail("not enough GPU memory for the activations");
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            // rope tables: sd.cpp's own layout and embedding for [text, image]
            std::vector<std::pair<int64_t, int64_t>> shapes{{Hd, Wd}};
            QwenImage21Layout layout;
            try {
                layout = QwenImage21Layout::build(Lt, tensor_or_empty(extra ? extra->image_slots : nullptr), shapes);
            } catch (const std::exception&) {
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            if (layout.segments.size() != 2 || layout.segments[0].image_index >= 0 || layout.prefix_length != Lt) {
                if (!said_why) LOG_INFO("Qwen Image 2.1 native: images in the prompt; this step runs on the stock runner");
                said_why = true;
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            Rope::Embedding emb;
            emb.ids.assign(layout.positions.begin(), layout.positions.end());
            emb.positions.token_count = emb.ids.size();
            for (auto region : layout.rope_layout.images) emb.positions.images.push_back(region);
            emb.values = Rope::embed_nd(emb.ids, 1, 10000.f, config.axes_dim, emb.layout, &emb.frequencies);
            std::vector<float> pe_all = finish_rope_pe(std::move(emb));
            const size_t per_tok = 4 * D / 2;
            if (pe_all.size() != (size_t) (Lt + T) * per_tok) {
                fail("unexpected rope table size");
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            ggml_backend_tensor_set(pe_img, pe_all.data() + Lt * per_tok, 0, T * per_tok * sizeof(float));
            // timestep embedding -> modulation rows (image: t, prefix: 0) and the final scale
            const float tv[2] = {inputs.timesteps->data()[0], 0.f};
            ggml_backend_tensor_set(timein, tv, 0, sizeof(tv));
            {
                G g = begin();
                ggml_tensor* te = ggml_timestep_embedding(g.ctx, timein, 256, 10000);
                te = ggml_mul_mat(g.ctx, w("time_text_embed.timestep_embedder.linear_1.weight"), te);
                te = ggml_silu(g.ctx, te);
                te = ggml_mul_mat(g.ctx, w("time_text_embed.timestep_embedder.linear_2.weight"), te);
                te = ggml_silu(g.ctx, te);
                ggml_tensor* md = ggml_mul_mat(g.ctx, w("modulation.1.weight"), te);
                ggml_tensor* sc = ggml_mul_mat(g.ctx, w("norm_out.linear.weight"),
                                               ggml_view_1d(g.ctx, te, C, 0));
                bind(md, mod);
                bind(sc, scale_out);
                run(g, {md, sc});
            }
            if (qi21_tanh(qi21_stream(runtime_backend), (const float*) mod->data, (float*) gates->data,
                          (int) (8 * C)) != 0) {
                fail("the gate kernel failed");
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            // the text prefix: once per condition
            uint64_t key = extra != nullptr ? extra->prefix_id : 0;
            {
                uint64_t hsh = 1469598103934665603ull;
                const unsigned char* p = (const unsigned char*) context.data();
                const size_t nb = context.numel() * sizeof(float);
                for (size_t i = 0; i < nb; i += 61) hsh = (hsh ^ p[i]) * 1099511628211ull;
                key = key * 31 + hsh;
            }
            if (key != prefix_key || prefix_len != Lt) {
                ggml_backend_tensor_set(pe_txt, pe_all.data(), 0, Lt * per_tok * sizeof(float));
                ggml_backend_tensor_set(ctx_in, context.data(), 0, Lt * config.context_dim * sizeof(float));
                {
                    // txt_in: zero-centered RMSNorm (weight + 1) -> in_layer -> GELU -> out_layer, into x
                    G g = begin();
                    ggml_tensor* t  = ggml_rms_norm(g.ctx, rows(g.ctx, ctx_in, Lt), 1e-6f);
                    ggml_tensor* nw = ggml_scale_bias(g.ctx, w("txt_in.text_norm.weight"), 1.f, 1.f);
                    t               = ggml_mul(g.ctx, t, nw);
                    t               = ggml_mul_mat(g.ctx, w("txt_in.in_layer.weight"), t);
                    t               = ggml_gelu(g.ctx, t);
                    t               = ggml_mul_mat(g.ctx, w("txt_in.out_layer.weight"), t);
                    bind(t, x);
                    run(g, {t});
                }
                if (!blocks_pass(Lt, 1, pe_txt, 0)) {
                    fail("the text prefix pass failed");
                    return QwenImage21Runner::compute(n_threads, inputs);
                }
                prefix_key = key;
                prefix_len = Lt;
            }
            // the image tokens: patchify (patch 1: token h*W + w, channels), img_in, the blocks, the output head
            {
                std::vector<float> tok((size_t) T * config.in_channels);
                const float* xs = xin.data();
                for (int64_t c = 0; c < config.in_channels; ++c)
                    for (int64_t t = 0; t < T; ++t) tok[(size_t) t * config.in_channels + c] = xs[(size_t) c * T + t];
                ggml_backend_tensor_set(latent, tok.data(), 0, tok.size() * sizeof(float));
                G g = begin();
                gemm(g, w("img_in.weight"), latent, x, T);
                run(g, {});
            }
            if (!blocks_pass(T, 0, pe_img, Lt)) {
                fail("the image pass failed");
                return QwenImage21Runner::compute(n_threads, inputs);
            }
            {
                G g = begin();
                ggml_tensor* t = ggml_norm(g.ctx, rows(g.ctx, x, T), 1e-6f);
                t              = ggml_mul(g.ctx, t, ggml_scale_bias(g.ctx, scale_out, 1.f, 1.f));
                t              = ggml_mul_mat(g.ctx, w("proj_out.weight"), t);
                bind(t, outp);
                run(g, {t});
            }
            ggml_backend_synchronize(runtime_backend);
            std::vector<float> o((size_t) T * config.out_channels);
            ggml_backend_tensor_get(outp, o.data(), 0, o.size() * sizeof(float));
            sd::Tensor<float> result({Wd, Hd, config.out_channels, 1});
            float* r = result.data();
            for (int64_t c = 0; c < config.out_channels; ++c)
                for (int64_t t = 0; t < T; ++t) r[(size_t) c * T + t] = o[(size_t) t * config.out_channels + c];
            if (timing) {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                LOG_INFO("Qwen Image 2.1 native: step %.1f ms (%lld image tokens, prefix %lld)", ms, (long long) T,
                         (long long) Lt);
            }
            return restore_trailing_singleton_dims(std::move(result), xin.dim());
        }
    };

}  // namespace Qwen

#endif  // __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_NATIVE_H__
