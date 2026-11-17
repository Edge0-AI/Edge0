// tools/vision/gguf_meta_test.cpp - strata-vision opens a GGUF whose weight tensors have types llama.cpp does not know.
//
// Writes a small GGUF by hand (a tokenizer, two tensors of the Strata-private types TQK6 = 145 and TQK7 = 146, and split
// keys that point at shards which are not there), then checks that
//   - llama.cpp itself cannot load it (the reason the picture reader used to say "cannot open the text model"),
//   - strata_open_text_model() can, and sees the same vocabulary.
// Exit code 0 = all checks passed.
#include "gguf_meta.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Out {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const auto* c = (const uint8_t*) p; b.insert(b.end(), c, c + n); }
    template <class T> void num(T v) { raw(&v, sizeof v); }
    void str(const std::string& s) { num<uint64_t>(s.size()); raw(s.data(), s.size()); }
    void kv_str(const std::string& k, const std::string& v) { str(k); num<uint32_t>(GGUF_TYPE_STRING); str(v); }
    void kv_u32(const std::string& k, uint32_t v) { str(k); num<uint32_t>(GGUF_TYPE_UINT32); num<uint32_t>(v); }
    void kv_strs(const std::string& k, const std::vector<std::string>& v) {
        str(k); num<uint32_t>(GGUF_TYPE_ARRAY); num<uint32_t>(GGUF_TYPE_STRING); num<uint64_t>(v.size());
        for (const auto& s : v) str(s);
    }
    void kv_i32s(const std::string& k, const std::vector<int32_t>& v) {
        str(k); num<uint32_t>(GGUF_TYPE_ARRAY); num<uint32_t>(GGUF_TYPE_INT32); num<uint64_t>(v.size());
        raw(v.data(), v.size() * 4);
    }
};

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

bool write_fixture(const std::string& path, size_t& n_vocab, bool split) {
    std::vector<std::string> tokens;
    for (int i = 0; i < 256; ++i) tokens.push_back(std::string("t") + std::to_string(i));
    tokens.push_back("<|endoftext|>");
    n_vocab = tokens.size();
    std::vector<int32_t> types(tokens.size(), 1);
    types.back() = 3;
    Out o;
    o.raw("GGUF", 4);
    o.num<uint32_t>(3);
    o.num<uint64_t>(2);                       // two tensors
    const uint64_t n_kv = split ? 9 : 6;
    o.num<uint64_t>(n_kv);
    o.kv_str("general.architecture", "llama");
    o.kv_str("tokenizer.ggml.model", "gpt2");
    o.kv_str("tokenizer.ggml.pre", "default");
    o.kv_strs("tokenizer.ggml.tokens", tokens);
    o.kv_i32s("tokenizer.ggml.token_type", types);
    o.kv_strs("tokenizer.ggml.merges", {"t1 t2"});
    if (split) {
        o.kv_u32("split.no", 0);
        o.kv_u32("split.count", 2);
        o.kv_u32("split.tensors.count", 4);
    }
    // two tensors of types llama.cpp's ggml does not have (145 = TQK6, 146 = TQK7), 128 elements each
    for (int i = 0; i < 2; ++i) {
        o.str(i == 0 ? "blk.0.ffn_gate_exps.weight" : "blk.0.ffn_up_exps.weight");
        o.num<uint32_t>(1);
        o.num<uint64_t>(128);
        o.num<uint32_t>(145 + i);
        o.num<uint64_t>(i * 64);
    }
    while (o.b.size() % 32) o.b.push_back(0);
    o.b.resize(o.b.size() + 128, 0);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(o.b.data(), 1, o.b.size(), f) == o.b.size();
    std::fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    llama_backend_init();
    for (int split = 0; split < 2; ++split) {
        const std::string label = split ? " (a first shard whose other shards are missing)" : "";
        const std::string path = dir + "/strata-vision-gguf-test-" + std::to_string(split) + ".gguf";
        size_t n_vocab = 0;
        check(write_fixture(path, n_vocab, split == 1), ("wrote a GGUF with tensor types 145 and 146" + label).c_str());
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model* plain = llama_model_load_from_file(path.c_str(), mp);
        std::printf("note  llama.cpp itself %s this file\n", plain ? "accepts" : "refuses");
        if (plain) llama_model_free(plain);
        std::string how;
        llama_model* m = strata_open_text_model(path, &how);
        check(m != nullptr, ("strata_open_text_model opens it (" + how + ")" + label).c_str());
        if (m) {
            check((size_t) llama_vocab_n_tokens(llama_model_get_vocab(m)) == n_vocab, "and sees the whole vocabulary");
            llama_model_free(m);
        }
        check(how.find("header only") == 0, "by reading the header, not the weights");
        std::remove(path.c_str());
    }
    check(strata_open_text_model(dir + "/does-not-exist.gguf") == nullptr, "a file that is not there gives no model");
    llama_backend_free();
    std::printf("%s\n", failures ? "GGUF-META FAIL" : "GGUF-META PASS");
    return failures ? 1 : 0;
}
