// tests/core/gguf_split_test.cpp - a split model's shards (strata::gguf_split_paths) and their joint view
// (strata::GgufModel), on synthetic GGUFs: no model, no GPU.
//
// The layout is Unsloth's UD-Q4_K_XL in miniature: shard 1 holds the metadata and no tensor, shards 2-4 the
// tensors, and layer 11's down sits in shard 2 while its gate and up are in shard 3.  Then the ways a real
// download goes wrong: a shard missing, a shard of another model (split keys that disagree), a tensor in two
// shards, a directory that does not add up to split.tensors.count, a truncated shard, and shard 1 opened alone.
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_split.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-gguf-split-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::string shard_name(int i, int n) {
    char b[64];
    std::snprintf(b, sizeof b, "Model-UD-Q4_K_XL-%05d-of-%05d.gguf", i, n);
    return b;
}

using fixture::Tensor;
const uint32_t F32 = 0, Q8_0 = 8;

/// Writes the 4-shard model; `mutate(i, kv, tensors)` may change shard i (0-based) before it is written.
template <class F> std::vector<std::string> write_model(const fs::path& dir, F mutate, uint64_t total = 7) {
    std::vector<std::string> paths;
    for (int i = 0; i < 4; ++i) {
        std::vector<fixture::Kv> kv;
        if (i == 0) kv.push_back(fixture::str("general.architecture", "qwen4exp"));
        for (const auto& k : fixture::split_keys((uint64_t) i, 4, total)) kv.push_back(k);
        std::vector<Tensor> ts;
        if (i == 1) ts = {{"output.weight", {64, 4}, Q8_0, 1}, {"token_embd.weight", {64, 4}, Q8_0, 2},
                          {"blk.11.ffn_down_exps.weight", {32, 32, 2}, F32, 3}};
        if (i == 2) ts = {{"blk.11.ffn_gate_exps.weight", {32, 32, 2}, F32, 4},
                          {"blk.11.ffn_up_exps.weight", {32, 32, 2}, F32, 5}, {"blk.12.attn_q.weight", {32, 8}, F32, 6}};
        if (i == 3) ts = {{"output_norm.weight", {32}, F32, 7}};
        uint64_t cut = 0;
        mutate(i, kv, ts, cut);
        const fs::path p = dir / shard_name(i + 1, 4);
        fixture::write(p, kv, ts, cut);
        paths.push_back(p.string());
    }
    return paths;
}
auto same = [](int, std::vector<fixture::Kv>&, std::vector<Tensor>&, uint64_t&) {};

std::string error_of(const std::vector<std::string>& paths) {
    try {
        strata::GgufModel m(paths);
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}
}  // namespace

// `gguf_split_test --real SHARD`: the same view of a real model (headers only): its shards, the architecture
// check on the metadata shard, and where the tensors the engine reads by name live.
int real_model(const std::string& any) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(any);
        size_t tensors = 0;
        for (size_t i = 0; i < m.size(); ++i) {
            std::printf("  shard %zu: %s, %zu tensors, data at %llu\n", i + 1, m.shard(i).path().c_str(),
                        m.shard(i).tensors().size(), (unsigned long long) m.shard(i).data_start());
            tensors += m.shard(i).tensors().size();
        }
        const std::string arch = strata::check_architecture(m.meta());
        std::printf("  %zu tensors; architecture check on the metadata shard: %s\n", tensors,
                    arch.empty() ? "ok" : arch.c_str());
        int bad = arch.empty() ? 0 : 1;
        for (const char* name : {"output.weight", "token_embd.weight", "per_layer_token_embd.weight",
                                 "blk.1.ple_key.weight", "blk.11.ffn_gate_exps.weight", "blk.11.ffn_up_exps.weight",
                                 "blk.11.ffn_down_exps.weight"}) {
            size_t s = 0;
            const strata::TensorInfo* t = m.find(name, &s);
            if (!t) { std::printf("  %-30s absent\n", name); continue; }
            const bool ok = m.in_bounds(*t, s);
            bad += !ok;
            std::printf("  %-30s shard %zu, %s, %llu B, %s\n", name, s + 1, t->type_name(),
                        (unsigned long long) strata::tensor_payload_bytes(*t), ok ? "in bounds" : "OUT OF BOUNDS");
        }
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("gguf_split_test\n");
    {
        TempDir d;
        const auto paths = write_model(d.path, same);
        // any shard names the family; the list is in split order, metadata shard first
        std::vector<std::string> got;
        try { got = strata::gguf_split_paths(paths[2]); } catch (...) {}
        check(got == paths, "gguf_split_paths from shard 3: the 4 shards in order");
        try { got = strata::gguf_split_paths(paths[0]); } catch (...) { got.clear(); }
        check(got == paths, "gguf_split_paths from shard 1: the same 4 shards");
        check(strata::gguf_split_paths((d.path / "plain.gguf").string()).size() == 1,
              "a name without -NNNNN-of-MMMMM is a model of one file");
        std::string err;
        try {
            const strata::GgufModel m(paths);
            size_t s_out = 9, s_gate = 9, s_down = 9, s_up = 9;
            const strata::TensorInfo* out = m.find("output.weight", &s_out);
            const strata::TensorInfo* gate = m.find("blk.11.ffn_gate_exps.weight", &s_gate);
            const strata::TensorInfo* up = m.find("blk.11.ffn_up_exps.weight", &s_up);
            const strata::TensorInfo* down = m.find("blk.11.ffn_down_exps.weight", &s_down);
            check(m.size() == 4 && m.meta().get("general.architecture") && m.meta().tensors().empty(),
                  "shard 1: the metadata, no tensor");
            check(out && s_out == 1 && m.in_bounds(*out, s_out), "output.weight found in shard 2, in bounds");
            check(gate && up && down && s_gate == 2 && s_up == 2 && s_down == 1,
                  "layer 11 per role: gate/up in shard 3, down in shard 2");
            check(m.find("blk.99.ffn_gate_exps.weight") == nullptr, "an absent tensor is nullptr");
            check(strata::check_architecture(m.meta()).find("block_count") != std::string::npos,
                  "the architecture guard reads the metadata shard (and wants its keys)");
        } catch (const std::exception& e) {
            err = e.what();
        }
        check(err.empty(), "the 4-shard model opens" + (err.empty() ? std::string() : ": " + err));
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, same);
        fs::remove(paths[3]);
        std::string err;
        try { strata::gguf_split_paths(paths[0]); } catch (const std::exception& e) { err = e.what(); }
        check(err.find("missing model shard") != std::string::npos && err.find(shard_name(4, 4)) != std::string::npos,
              "a missing shard is an error that names it");
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, [](int i, auto&, auto& ts, uint64_t&) {
            if (i == 3) ts.push_back({"blk.11.ffn_up_exps.weight", {32, 32, 2}, F32, 9});
        }, 8);
        const std::string err = error_of(paths);
        check(err.find("in two shards") != std::string::npos && err.find(shard_name(3, 4)) != std::string::npos &&
                  err.find(shard_name(4, 4)) != std::string::npos,
              "a tensor in two shards is refused, naming both");
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, [](int i, auto& kv, auto&, uint64_t&) {
            if (i == 2) kv = fixture::split_keys(1, 4, 7);    // says it is shard 2
        });
        check(error_of(paths).find("does not declare itself shard 3 of 4") != std::string::npos,
              "a shard whose split.no disagrees with its place is refused");
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, [](int i, auto& kv, auto&, uint64_t&) {
            if (i == 3) kv = fixture::split_keys(3, 4, 9);    // another model's split.tensors.count
        });
        check(!error_of(paths).empty(), "a shard of another model (split.tensors.count differs) is refused");
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, [](int i, auto&, auto& ts, uint64_t&) {
            if (i == 3) ts.clear();                         // a directory short of split.tensors.count
        });
        check(error_of(paths).find("split.tensors.count is 7") != std::string::npos,
              "shards that do not add up to split.tensors.count are refused");
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, [](int i, auto& kv, auto&, uint64_t&) {
            if (i == 0) kv.erase(kv.begin());               // no general.architecture in shard 1
        });
        check(error_of(paths).find("general.architecture") != std::string::npos,
              "a split model whose first shard has no metadata is refused");
        check(error_of({paths[0]}).find("opened as a whole model") != std::string::npos,
              "shard 1 of 4 opened as a whole model is refused");
    }
    {
        TempDir d;
        const auto paths = write_model(d.path, [](int i, auto&, auto&, uint64_t& cut) {
            if (i == 2) cut = 100;                          // the last tensor of shard 3 is cut short
        });
        bool truncated = false, first_ok = false;
        try {
            const strata::GgufModel m(paths);
            size_t s = 0;
            const strata::TensorInfo* q = m.find("blk.12.attn_q.weight", &s);
            const strata::TensorInfo* g = m.find("blk.11.ffn_gate_exps.weight");
            truncated = q && !m.in_bounds(*q, s);
            first_ok = g && m.in_bounds(*g, 2);
        } catch (...) {}
        check(truncated && first_ok, "a truncated shard: its last tensor is out of bounds, the others are not");
    }
    std::printf(g_fail ? "gguf_split_test: %d FAILED\n" : "gguf_split_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
