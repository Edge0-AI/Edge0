// tools/qwen35/q36_mtmd_dump.cpp - llama.cpp's side of the Qwen3.6 PICTURE check: the same GGUF and mmproj, llama.cpp's own
// mtmd library and graph (the M-RoPE positions of mtmd_helper_eval_chunk_single, qwen35moe's IMROPE).  Prints the prompt's
// chunks (the text ids, the picture's token count) and the greedy ids, and dumps every step's full logits (f32,
// [step][vocab]) so the first token's logits can be compared with strata-q36's (--dump-full-logits there).
//
// Built against a llama.cpp checkout with its mtmd library (not part of Strata's build):
//   g++ -O2 -std=c++17 q36_mtmd_dump.cpp -I<llama>/include -I<llama>/ggml/include -I<llama>/tools/mtmd -L<llama>/build/bin
//       -lllama -lmtmd -lggml -lggml-base -Wl,-rpath,<llama>/build/bin -o q36_mtmd_dump
// Usage: q36_mtmd_dump --model GGUF --mmproj F --image F [--image F2 ...] --prompt TEXT [--n 32] [--ngl 99] [--max-tokens N] [--mm-cpu 1]
//                      [--dump-full-logits F] [--force ID,ID,...]
//   TEXT holds one <__media__> marker where the picture goes (the chat template's user turn); it is tokenized with
//   special tokens parsed and no BOS.  --force feeds that continuation instead of the greedy ids (teacher forcing).
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int> parse_ids(const std::string& s) {
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

int main(int argc, char** argv) {
    std::string model, mmproj, prompt, dump_full, force;
    std::vector<std::string> images;
    int n_gen = 32, ngl = 99, max_tokens = 0, mm_cpu = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--mmproj") mmproj = v;
        else if (a == "--image") images.push_back(v);   // repeat for several pictures (one <__media__> each)
        else if (a == "--prompt") prompt = v;
        else if (a == "--n") n_gen = std::atoi(v.c_str());
        else if (a == "--ngl") ngl = std::atoi(v.c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(v.c_str());
        else if (a == "--mm-cpu") mm_cpu = std::atoi(v.c_str());   // the encoder on the CPU (as strata-vision's default)
        else if (a == "--dump-full-logits") dump_full = v;
        else if (a == "--force") force = v;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) { std::fprintf(stderr, "cannot load %s\n", model.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(mdl);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = cp.n_ubatch = 2048;
    cp.op_offload = ngl > 0;
    llama_context* ctx = llama_init_from_model(mdl, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create the context\n"); return 1; }
    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = ngl > 0 && !mm_cpu;
    mcp.print_timings = false;
    mcp.warmup = false;
    if (max_tokens > 0) mcp.image_max_tokens = max_tokens;
    mtmd_context* mctx = mtmd_init_from_file(mmproj.c_str(), mdl, mcp);
    if (!mctx) { std::fprintf(stderr, "cannot load %s\n", mmproj.c_str()); return 1; }
    std::vector<mtmd_helper_bitmap_wrapper> bws;
    std::vector<const mtmd_bitmap*> bms;
    for (const std::string& image : images) {
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mctx, image.c_str(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::fprintf(stderr, "cannot read %s\n", image.c_str()); return 1; }
        bws.push_back(bw);
        bms.push_back(bw.bitmap);
    }
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    if (mtmd_tokenize(mctx, chunks, &txt, bms.data(), bms.size()) != 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    const size_t nc = mtmd_input_chunks_size(chunks);
    llama_pos n_past = 0;
    for (size_t c = 0; c < nc; ++c) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(ch, &n);
            std::printf("text:");
            for (size_t i = 0; i < n; ++i) std::printf(" %d", (int) t[i]);
            std::printf("\n");
        } else {
            std::printf("image: %d tokens, %d positions\n", (int) mtmd_input_chunk_get_n_tokens(ch),
                        (int) mtmd_input_chunk_get_n_pos(ch));
        }
        llama_pos np = n_past;
        if (mtmd_helper_eval_chunk_single(mctx, ctx, ch, n_past, 0, (int32_t) cp.n_batch, c + 1 == nc, &np) != 0) {
            std::fprintf(stderr, "eval failed at chunk %zu\n", c);
            return 1;
        }
        n_past = np;
    }
    const std::vector<int> forced = parse_ids(force);
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    const int V = llama_vocab_n_tokens(vocab);
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<llama_token> gen;
    for (int i = 0; i < n_gen; ++i) {
        const float* lg = llama_get_logits_ith(ctx, -1);
        int a = 0;
        for (int j = 1; j < V; ++j) if (lg[j] > lg[a]) a = j;
        if (i == 0) {
            int b = -1;
            for (int j = 0; j < V; ++j) if (j != a && (b < 0 || lg[j] > lg[b])) b = j;
            std::printf("first: top1 %d (%.4f) top2 %d (%.4f)\n", a, lg[a], b, lg[b]);
        }
        gen.push_back(a);
        if (ff) std::fwrite(lg, 4, V, ff);
        if (i + 1 == n_gen) break;
        llama_token t = i < (int) forced.size() ? forced[i] : a;
        llama_pos pp = n_past++;   // the text after a picture continues at n_past (max(nx, ny) past its first position)
        llama_batch one = llama_batch_get_one(&t, 1);
        one.pos = &pp;
        if (llama_decode(ctx, one) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    }
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    if (ff) std::fclose(ff);
    mtmd_input_chunks_free(chunks);
    for (auto& bw : bws) mtmd_bitmap_free(bw.bitmap);
    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(mdl);
    llama_backend_free();
    return 0;
}
