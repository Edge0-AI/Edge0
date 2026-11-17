// tests/core/gguf_geometry_test.cpp - the model's geometry from its GGUF metadata (strata::read_geometry) and the
// check against the shape the kernels are compiled for (strata::geometry_unsupported, check_architecture), on
// synthetic GGUFs: no model, no GPU.
//
// The full key set is the one Unsloth's Qwen3.8-Flash-Next UD-IQ4_XS carries (read from its first shard).  Then
// the cases: a file with only the six keys the guard has always required, a pruned expert count, keys that
// describe one thing twice and disagree, and a shape the kernels are not compiled for.
//
// `gguf_geometry_test --real SHARD1`: the geometry a real model's metadata shard declares, and the verdict.
#include "gguf_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::ModelGeometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-gguf-geometry-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::vector<uint64_t> ratios(int64_t layers, int64_t interval, uint64_t block) {
    std::vector<uint64_t> r((size_t) layers, 0);
    for (int64_t l = 0; l < layers; ++l)
        if (l % interval == interval - 1) r[(size_t) l] = block;
    return r;
}

/// Qwen3.8-Flash-Next's metadata as its GGUF carries it (the PLE and rope-section keys the reader does not use
/// are left out), with a vocabulary of `vocab` tokens.
std::vector<fixture::Kv> full_keys(size_t vocab = 1000) {
    using namespace fixture;
    const std::string p = "qwen4exp.";
    std::vector<std::string> tokens(vocab);
    for (size_t i = 0; i < vocab; ++i) tokens[i] = "t" + std::to_string(i);
    return {str("general.architecture", "qwen4exp"),
            u32(p + "block_count", 48),
            u32(p + "context_length", 262144),
            u32(p + "embedding_length", 2560),
            u32(p + "attention.head_count", 24),
            u32(p + "attention.head_count_kv", 2),
            f32(p + "rope.freq_base", 1e7f),
            f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
            u32(p + "expert_count", 512),
            u32(p + "expert_used_count", 10),
            u32(p + "attention.key_length", 256),
            u32(p + "attention.value_length", 256),
            u32(p + "expert_feed_forward_length", 640),
            u32(p + "expert_shared_feed_forward_length", 640),
            u32(p + "ssm.conv_kernel", 4),
            u32(p + "ssm.state_size", 128),
            u32(p + "ssm.group_count", 16),
            u32(p + "ssm.time_step_rank", 48),
            u32(p + "ssm.inner_size", 6144),
            u32(p + "full_attention_interval", 4),
            u32(p + "rope.dimension_count", 64),
            u32(p + "hyper_connection.count", 4),
            u32(p + "hyper_connection.low_rank", 320),
            u32(p + "attention.indexer.head_count", 4),
            u32(p + "attention.indexer.key_length", 128),
            u32(p + "attention.indexer.top_k", 2048),
            u32s(p + "attention.compress_ratios", ratios(48, 4, 4)),
            str("tokenizer.ggml.model", "gpt2"),
            strs("tokenizer.ggml.tokens", tokens)};
}

/// `kv` with `key` replaced by `v` (or removed, when `v` is null).
std::vector<fixture::Kv> with(std::vector<fixture::Kv> kv, const std::string& key, const fixture::Kv* v) {
    for (auto it = kv.begin(); it != kv.end(); ++it)
        if (it->key == key) {
            if (v) *it = *v;
            else kv.erase(it);
            return kv;
        }
    if (v) kv.push_back(*v);
    return kv;
}

struct Result {
    std::string read_err, unsupported, guard;
    ModelGeometry g;
};
Result run(const fs::path& dir, const std::vector<fixture::Kv>& kv) {
    static int n = 0;
    const fs::path p = dir / ("m" + std::to_string(n++) + ".gguf");
    fixture::write(p, kv, {});
    Result r;
    const strata::GgufFile f(p.string());
    r.read_err = strata::read_geometry(f, r.g);
    if (r.read_err.empty()) r.unsupported = strata::geometry_unsupported(r.g);
    r.guard = strata::check_architecture(f);
    return r;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

bool same_shape(const ModelGeometry& a, const ModelGeometry& b) {
    return a.n_embd == b.n_embd && a.n_layers == b.n_layers && a.qsa_interval == b.qsa_interval &&
           a.n_vocab == b.n_vocab && a.ssm_state_size == b.ssm_state_size && a.ssm_k_heads == b.ssm_k_heads &&
           a.ssm_v_heads == b.ssm_v_heads && a.ssm_d_conv == b.ssm_d_conv &&
           a.ssm_conv_channels == b.ssm_conv_channels && a.ssm_value_dim == b.ssm_value_dim && a.n_head == b.n_head &&
           a.n_head_kv == b.n_head_kv && a.head_dim == b.head_dim && a.n_rot == b.n_rot &&
           a.idx_q_heads == b.idx_q_heads && a.idx_key_dim == b.idx_key_dim && a.idx_block == b.idx_block &&
           a.idx_top_k == b.idx_top_k && a.hc == b.hc && a.hc_lr == b.hc_lr && a.n_expert == b.n_expert &&
           a.n_expert_used == b.n_expert_used && a.n_ff == b.n_ff;
}

int real_model(const std::string& shard) {
    try {
        const strata::GgufFile f(shard);
        ModelGeometry g;
        const std::string err = strata::read_geometry(f, g);
        if (!err.empty()) {
            std::printf("  refused: %s\n", err.c_str());
            return 1;
        }
        std::printf("  n_embd %lld, layers %lld (attention every %lld), vocab %lld, rms_eps %g\n", (long long) g.n_embd,
                    (long long) g.n_layers, (long long) g.qsa_interval, (long long) g.n_vocab, g.rms_eps);
        std::printf("  GDN: state %lld, k heads %lld, v heads %lld, conv %lld (%lld channels, values %lld)\n",
                    (long long) g.ssm_state_size, (long long) g.ssm_k_heads, (long long) g.ssm_v_heads,
                    (long long) g.ssm_d_conv, (long long) g.ssm_conv_channels, (long long) g.ssm_value_dim);
        std::printf("  QSA: heads %lld/%lld x %lld, rope %lld, indexer %lld x %lld, block %lld, top-k %lld\n",
                    (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim, (long long) g.n_rot,
                    (long long) g.idx_q_heads, (long long) g.idx_key_dim, (long long) g.idx_block,
                    (long long) g.idx_top_k);
        std::printf("  HC %lld (rank %lld); MoE %lld experts, %lld per token, ff %lld\n", (long long) g.hc,
                    (long long) g.hc_lr, (long long) g.n_expert, (long long) g.n_expert_used, (long long) g.n_ff);
        const std::string why = strata::geometry_unsupported(g);
        std::printf("  the compiled kernels: %s\n", why.empty() ? "can run it" : why.c_str());
        return why.empty() ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("gguf_geometry_test\n");
    TempDir d;
    const std::string p = "qwen4exp.";
    {
        const Result r = run(d.path, full_keys());
        ModelGeometry want = strata::core::kCompiledGeometry;
        want.n_vocab = 1000;
        check(r.read_err.empty(), "the full key set reads" + (r.read_err.empty() ? "" : ": " + r.read_err));
        check(same_shape(r.g, want), "  and gives the compiled shape (vocabulary from tokenizer.ggml.tokens)");
        check(r.g.ssm_conv_channels == 10240, "  conv channels derived: 2 x 128 x 16 + 6144 = 10240");
        check(r.unsupported.empty() && r.guard.empty(), "  which the kernels run, and the guard passes");
    }
    {
        // the six keys tests/core/native_dense_ple_key_test.cpp writes: everything else is the compiled value
        const std::vector<fixture::Kv> six = {
            fixture::str("general.architecture", "qwen4exp"), fixture::u32(p + "block_count", 48),
            fixture::u32(p + "embedding_length", 2560),       fixture::u32(p + "expert_count", 512),
            fixture::u32(p + "expert_used_count", 10),        fixture::u32(p + "attention.head_count", 24),
            fixture::u32(p + "attention.head_count_kv", 2)};
        const Result r = run(d.path, six);
        check(r.read_err.empty() && same_shape(r.g, strata::core::kCompiledGeometry) && r.guard.empty(),
              "only the six required keys: the rest keeps the compiled values");
        for (const char* key : {"block_count", "embedding_length", "expert_count", "expert_used_count",
                                "attention.head_count", "attention.head_count_kv"}) {
            const Result m = run(d.path, with(six, p + key, nullptr));
            check(has(m.read_err, "missing") && has(m.read_err, key) && m.guard == m.read_err,
                  std::string("a file without ") + key + " is refused, naming it");
        }
    }
    {
        // GSQ-RCO Coder: fewer experts, read from the file and run by the same kernels
        const fixture::Kv e = fixture::u32(p + "expert_count", 384), k = fixture::u32(p + "expert_used_count", 8);
        const Result r = run(d.path, with(with(full_keys(), e.key, &e), k.key, &k));
        check(r.read_err.empty() && r.g.n_expert == 384 && r.g.n_expert_used == 8 && r.guard.empty(),
              "a pruned variant: 384 experts, 8 per token, read and accepted");
        const fixture::Kv k65 = fixture::u32(p + "expert_used_count", 65);
        check(has(run(d.path, with(full_keys(), k65.key, &k65)).read_err, "expert_used_count = 65"),
              "65 experts per token is refused (the router takes 1..64)");
        const fixture::Kv k0 = fixture::u32(p + "expert_used_count", 0);
        check(has(run(d.path, with(full_keys(), k0.key, &k0)).read_err, "must be positive"),
              "0 experts per token is refused");
    }
    {
        // keys that describe one thing twice must agree
        const fixture::Kv vl = fixture::u32(p + "attention.value_length", 128);
        check(has(run(d.path, with(full_keys(), vl.key, &vl)).read_err, "attention.value_length = 128"),
              "a value head width other than the key head width is refused");
        const fixture::Kv sh = fixture::u32(p + "expert_shared_feed_forward_length", 1280);
        check(has(run(d.path, with(full_keys(), sh.key, &sh)).read_err, "expert_shared_feed_forward_length = 1280"),
              "a shared expert wider than the routed ones is refused");
        const fixture::Kv in = fixture::u32(p + "ssm.inner_size", 4096);
        check(has(run(d.path, with(full_keys(), in.key, &in)).read_err, "square heads"),
              "ssm.inner_size other than state_size x time_step_rank is refused");
        std::vector<uint64_t> shifted = ratios(48, 4, 4);
        shifted[3] = 0;
        shifted[2] = 4;
        const fixture::Kv cr = fixture::u32s(p + "attention.compress_ratios", shifted);
        check(has(run(d.path, with(full_keys(), cr.key, &cr)).read_err, "compress_ratios[2] = 4"),
              "compress_ratios that disagree with full_attention_interval are refused at the first layer");
        const fixture::Kv short_cr = fixture::u32s(p + "attention.compress_ratios", ratios(40, 4, 4));
        check(has(run(d.path, with(full_keys(), short_cr.key, &short_cr)).read_err, "one entry for each of the 48"),
              "compress_ratios of another length are refused");
        std::vector<uint64_t> mixed = ratios(48, 4, 4);
        mixed[7] = 8;
        const fixture::Kv mx = fixture::u32s(p + "attention.compress_ratios", mixed);
        check(has(run(d.path, with(full_keys(), mx.key, &mx)).read_err, "compress_ratios[7] = 8"),
              "two pooling blocks in one model are refused");
        // #2396: 48 zeros are a converter's placeholder (orcarouter's Uncensored GGUFs): read as absent, the compiled block stays
        const fixture::Kv zeros = fixture::u32s(p + "attention.compress_ratios", std::vector<uint64_t>(48, 0));
        const Result z = run(d.path, with(full_keys(), zeros.key, &zeros));
        check(z.read_err.empty() && z.guard.empty() && z.g.idx_block == 4, "an all-zero compress_ratios list is treated as absent");
        const fixture::Kv zshort = fixture::u32s(p + "attention.compress_ratios", std::vector<uint64_t>(40, 0));
        check(has(run(d.path, with(full_keys(), zshort.key, &zshort)).read_err, "one entry for each of the 48"),
              "an all-zero list of another length is still refused");
        std::vector<uint64_t> one_set(48, 0);
        one_set[3] = 4;                      // a real (but partial) list: layer 7 would be full attention with a 0
        const fixture::Kv partial = fixture::u32s(p + "attention.compress_ratios", one_set);
        check(has(run(d.path, with(full_keys(), partial.key, &partial)).read_err, "compress_ratios[7] = 0"),
              "a list with one non-zero entry is checked in full");
        const fixture::Kv fl = fixture::f32(p + "block_count", 48.0f);
        check(has(run(d.path, with(full_keys(), fl.key, &fl)).read_err, "is not an integer"),
              "a float where a count belongs is refused");
    }
    {
        // a consistent file of another shape reads, and the kernels refuse it naming the key
        const fixture::Kv bc = fixture::u32(p + "block_count", 40);
        const fixture::Kv cr = fixture::u32s(p + "attention.compress_ratios", ratios(40, 4, 4));
        const Result r = run(d.path, with(with(full_keys(), bc.key, &bc), cr.key, &cr));
        check(r.read_err.empty() && r.g.n_layers == 40 && r.g.n_qsa_layers() == 10, "40 layers read: 10 QSA, 30 GDN");
        check(r.unsupported == "qwen4exp.block_count = 40, the kernels are compiled for 48" && r.guard == r.unsupported,
              "  and refused: " + r.unsupported);
        const fixture::Kv hd = fixture::u32(p + "attention.key_length", 128), vd = fixture::u32(p + "attention.value_length", 128);
        const Result h = run(d.path, with(with(full_keys(), hd.key, &hd), vd.key, &vd));
        check(h.read_err.empty() && h.g.head_dim == 128 && has(h.guard, "attention.key_length = 128"),
              "head width 128 reads and is refused: " + h.guard);
        const fixture::Kv tk = fixture::u32(p + "attention.indexer.top_k", 1024);
        check(has(run(d.path, with(full_keys(), tk.key, &tk)).guard, "attention.indexer.top_k = 1024"),
              "an indexer budget of 1024 is refused");
        const fixture::Kv eps = fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-5f);
        check(has(run(d.path, with(full_keys(), eps.key, &eps)).guard, "layer_norm_rms_epsilon = 1e-05"),
              "an rms epsilon of 1e-5 is refused");
        const fixture::Kv vocab = fixture::u32(p + "embedding_length", 2560);
        check(run(d.path, with(full_keys(151936), vocab.key, &vocab)).guard.empty(),
              "another vocabulary size is accepted (the head and embedding take their tensors' shapes)");
        const fixture::Kv arch = fixture::str("general.architecture", "glm5next");
        const Result a = run(d.path, with(full_keys(), arch.key, &arch));
        check(a.read_err == "architecture is 'glm5next', this engine requires 'qwen4exp'" && a.guard == a.read_err,
              "another architecture is refused by name, by the reader and the guard");
    }
    {
        // Qwen3.8-27B's own keys (qwen35, dense): the reader names the architecture, not the first qwen4exp key it
        // misses (which is what generate printed before the reader checked the name)
        const std::vector<fixture::Kv> dense = {
            fixture::str("general.architecture", "qwen35"), fixture::u32("qwen35.block_count", 65),
            fixture::u32("qwen35.embedding_length", 5120), fixture::u32("qwen35.feed_forward_length", 17408),
            fixture::u32("qwen35.attention.head_count", 24), fixture::u32("qwen35.attention.head_count_kv", 4)};
        const Result r = run(d.path, dense);
        check(r.read_err == "architecture is 'qwen35', this engine requires 'qwen4exp'",
              "a dense qwen35 file is refused by its architecture: " + r.read_err);
        const Result none = run(d.path, with(dense, "general.architecture", nullptr));
        check(none.read_err == "missing general.architecture", "a file without general.architecture says so");
    }
    std::printf(g_fail ? "gguf_geometry_test: %d FAILED\n" : "gguf_geometry_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
