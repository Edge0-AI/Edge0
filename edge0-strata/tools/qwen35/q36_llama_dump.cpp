// tools/qwen35/q36_llama_dump.cpp - llama.cpp's side of the Qwen3.6 parity check (NEW-MODELS-PLAN.md T1): the same GGUF,
// the same token ids, llama.cpp's own graph.  Prints the prompt's ids and the greedy ids, and dumps every processed
// position's layer outputs ("l_out-N", through the scheduler's eval callback) in strata-q36ref's format
// ([pos][layer][n_embd] f32) plus each greedy step's top-2 (id, logit).
//
// Built against a llama.cpp checkout (not part of Strata's build):
//   g++ -O2 -std=c++17 q36_llama_dump.cpp -I<llama>/include -I<llama>/ggml/include -L<llama>/build/bin -lllama -lggml
//       -lggml-base -Wl,-rpath,<llama>/build/bin -o q36_llama_dump
// Usage: q36_llama_dump --model GGUF (--prompt TEXT | --tokens ID,ID,...) [--n 64] [--ngl 99] [--dump-layers F]
//                       [--dump-logits F] [--force ID,ID,...] [--dump-full-logits F]
//   --force feeds that continuation instead of the greedy ids (teacher forcing); --dump-full-logits writes every
//   step's logits (f32, [step][vocab]).
//   --prompt is tokenized with special tokens parsed and no BOS added (Qwen adds none).
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {
struct Capture {
    int n_embd = 0, n_layer = 0, n_tok = 0;
    std::map<int, std::vector<float>> layers;   // layer -> [n_tok][n_embd] of the current batch
};

bool on_tensor(struct ggml_tensor* t, bool ask, void* user) {
    auto* c = (Capture*) user;
    const char* name = ggml_get_name(t);
    if (std::strncmp(name, "l_out-", 6) != 0) return ask ? false : true;
    if (ask) return true;
    const int il = std::atoi(name + 6);
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd) return true;
    std::vector<float>& v = c->layers[il];
    v.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return true;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') ++p;
        if (!*p) break;
        v.push_back(std::atoi(p));
        while (*p && *p != ',' && *p != ' ') ++p;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::string model, prompt, tokens, dump_layers, dump_logits, force, dump_full;
    int n_gen = 64, ngl = 99;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--prompt") prompt = v;
        else if (a == "--tokens") tokens = v;
        else if (a == "--n") n_gen = std::atoi(v.c_str());
        else if (a == "--ngl") ngl = std::atoi(v.c_str());
        else if (a == "--dump-layers") dump_layers = v;
        else if (a == "--dump-logits") dump_logits = v;
        else if (a == "--force") force = v;
        else if (a == "--dump-full-logits") dump_full = v;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    // STRATA_DUMP_NGL: the GPU layers when the caller asks for the GPU (--ngl > 0) but the file does not fit (UD-Q4_K_XL
    // on a 12 GB card); the other layers run on llama.cpp's CPU backend
    if (const char* g = std::getenv("STRATA_DUMP_NGL"); g != nullptr && ngl > 0) ngl = std::atoi(g);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) { std::fprintf(stderr, "cannot load %s\n", model.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(mdl);
    std::vector<llama_token> ids;
    if (!tokens.empty()) {
        for (int t : parse_ids(tokens)) ids.push_back(t);
    } else {
        ids.resize(prompt.size() + 16);
        const int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ids.data(), (int) ids.size(), false, true);
        if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
        ids.resize(n);
    }
    Capture cap;
    cap.n_embd = llama_model_n_embd(mdl);
    cap.n_layer = llama_model_n_layer(mdl);
    llama_context_params cp = llama_context_default_params();
    // STRATA_DUMP_CTX: a smaller context / batch (the parity sequences are a few hundred tokens) for a file whose
    // weights leave no VRAM for the 4096-token compute buffers (UD-Q6_K_XL on a 32 GB card)
    const char* cx = std::getenv("STRATA_DUMP_CTX");
    const int nctx = cx ? std::atoi(cx) : 4096;
    cp.n_ctx = nctx;
    cp.n_batch = cp.n_ubatch = nctx;
    cp.op_offload = ngl > 0;   // --ngl 0 is the CPU backend: no prompt matmuls offloaded to a GPU either
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &cap;
    llama_context* ctx = llama_init_from_model(mdl, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create the context\n"); return 1; }
    FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    const std::vector<int> forced = parse_ids(force);
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    const int V = llama_vocab_n_tokens(vocab);
    auto write_layers = [&](int n_tok) {
        if (!fl) return;
        // a layer may keep fewer rows than the batch (the last layer keeps only the output rows): those are the
        // batch's LAST positions; the positions it did not compute are written as zeros
        std::vector<float> zero(cap.n_embd, 0.f);
        for (int t = 0; t < n_tok; ++t)
            for (int l = 0; l < cap.n_layer; ++l) {
                auto it = cap.layers.find(l);
                const int rows = it == cap.layers.end() ? 0 : (int) (it->second.size() / cap.n_embd);
                const int r = t - (n_tok - rows);
                const float* src = r >= 0 ? it->second.data() + (size_t) r * cap.n_embd : zero.data();
                std::fwrite(src, 4, cap.n_embd, fl);
            }
        cap.layers.clear();
    };
    std::printf("prompt:");
    for (int t : ids) std::printf(" %d", t);
    std::printf("\n");
    std::vector<llama_token> gen;
    llama_batch b = llama_batch_get_one(ids.data(), (int) ids.size());
    if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    write_layers((int) ids.size());
    for (int i = 0; i < n_gen; ++i) {
        const float* lg = llama_get_logits_ith(ctx, -1);
        int a = 0, c = -1;
        for (int j = 1; j < V; ++j) if (lg[j] > lg[a]) a = j;
        for (int j = 0; j < V; ++j) if (j != a && (c < 0 || lg[j] > lg[c])) c = j;
        if (fg) { const float rec[4] = {(float) a, lg[a], (float) c, lg[c]}; std::fwrite(rec, 4, 4, fg); }
        gen.push_back(a);
        if (ff) std::fwrite(lg, 4, V, ff);
        if (i + 1 == n_gen) break;
        llama_token t = i < (int) forced.size() ? forced[i] : a;
        llama_batch one = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, one) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
        write_layers(1);
    }
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    llama_free(ctx);
    llama_model_free(mdl);
    llama_backend_free();
    return 0;
}
