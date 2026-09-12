// Compile against llama.cpp 6c84c7d5d8833c6e0df69628f75a0f599797934e.
// Usage: reference MODEL OUTPUT.f32 [comma-separated token IDs] [chunk size]
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

static bool trace(ggml_tensor * tensor, bool ask, void * data) {
    const std::string name = ggml_get_name(tensor);
    bool selected = name.find("l_last") == 0 || name == "model.input_embed" ||
        (name.size() >= 2 && name.substr(name.size() - 2) == "-0");
    if (ask) return selected;
    if (selected && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor)) {
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        std::ofstream output(std::string(static_cast<const char *>(data)) + "/" + name + ".f32", std::ios::binary);
        output.write(bytes.data(), bytes.size());
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    const bool tokenize_only = argc > 4 && std::string(argv[3]) == "--tokenize";
    mp.vocab_only = tokenize_only;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    if (tokenize_only) {
        const auto * vocab = llama_model_get_vocab(model);
        std::string text = argv[4];
        int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
        std::vector<llama_token> tokens(n);
        n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, false, true);
        std::ofstream output(argv[2]);
        output << "[";
        for (int i = 0; i < n; ++i) output << (i ? "," : "") << tokens[i];
        output << "]\n";
        llama_model_free(model);
        llama_backend_free();
        return n < 0 || !output.good() ? 6 : 0;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (const char * dir = std::getenv("EDGE0_REFERENCE_TRACE_DIR")) {
        cp.cb_eval = trace;
        cp.cb_eval_user_data = const_cast<char *>(dir);
    }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 4; }
    std::vector<llama_token> tokens;
    std::stringstream input(argc > 3 ? argv[3] : "1,3,7,4,2,256,12,10,16");
    std::string item;
    while (std::getline(input, item, ',')) tokens.push_back(std::stoi(item));
    int chunk = argc > 4 ? std::stoi(argv[4]) : 512;
    if (chunk < 1 || tokens.empty()) return 2;
    for (size_t start = 0; start < tokens.size(); start += chunk) {
        auto batch = llama_batch_get_one(tokens.data() + start,
            std::min<size_t>(chunk, tokens.size() - start));
        if (llama_decode(ctx, batch)) { llama_free(ctx); llama_model_free(model); return 5; }
    }
    int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx, -1)), vocab * sizeof(float));
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return output.good() ? 0 : 6;
}
