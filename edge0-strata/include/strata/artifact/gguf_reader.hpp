// include/strata/artifact/gguf_reader.hpp - generated from src/artifact/gguf_reader.cpp by
// scripts/split_artifact.py.  Header-only on purpose: the reader is one translation unit's worth of
// code with no state to hide, and a header-only split cannot introduce a duplicate-symbol or
// missing-declaration bug in code that is already validated.
#pragma once
// src/artifact/gguf_reader.cpp - P1.S2: GGUF v3 reader, mmap, no ggml dependency.
//
// The C++ counterpart of tools/gguf_reader.py, which has been the reference since P0.S6 (written
// because gguf-py cannot represent type 42 / Q2_0). Per P1.S2 it must:
//   * mmap the file and parse header, metadata KV (ALL value types incl. arrays), tensor directory
//   * be multi-shard aware (split.* keys)
//   * carry an architecture guard: general.architecture == "qwen4exp" and the compiled-in constants
//     must match, refusing with a precise error otherwise
//   * have no ggml dependency
//
// This file is the reader plus a `--check` mode that validates it the way the Python one is
// validated: parse the tiny model AND both real shards, and report counts that other harnesses have
// already established independently (1,223 / 1 tensors; 202 Q2_0 in shard 1). Agreement with those
// numbers is the test.
//
// Build: scripts/build_artifact.bat        Run: gguf_reader.exe <file.gguf> [--check]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>

#include "strata/artifact/gguf_split.hpp"
#include "strata/core/geometry.hpp"
#include "strata/artifact/model_family.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // windows.h's min/max macros would break std::min/std::max in every file that includes this one
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {

// ---- ggml type ids we care about. 42 = Q2_0, the PrismML ternary 2-bit encoding this engine targets.
inline const char* ggml_type_name(uint32_t t) {
    switch (t) {
    case 0:
        return "F32";
    case 1:
        return "F16";
    case 2:
        return "Q4_0";
    case 3:
        return "Q4_1";
    case 6:
        return "Q5_0";
    case 7:
        return "Q5_1";
    case 8:
        return "Q8_0";
    case 9:
        return "Q8_1";
    case 10:
        return "Q2_K";
    case 11:
        return "Q3_K";
    case 12:
        return "Q4_K";
    case 13:
        return "Q5_K";
    case 14:
        return "Q6_K";
    case 15:
        return "Q8_K";
    case 16:
        return "IQ2_XXS";
    case 17:
        return "IQ2_XS";
    case 18:
        return "IQ3_XXS";
    case 19:
        return "IQ1_S";
    case 20:
        return "IQ4_NL";
    case 21:
        return "IQ3_S";
    case 22:
        return "IQ2_S";
    case 23:
        return "IQ4_XS";
    case 24:
        return "I8";
    case 30:
        return "BF16";
    case 34:
        return "TQ1_0";
    case 35:
        return "TQ2_0";
    case 39:
        return "MXFP4";
    case 40:
        return "NVFP4";
    case 41:
        return "Q1_0";
    case 42:
        return "Q2_0";
    // Agention's llama.cpp fork: the APR trellis expert types (docs/GYRO.md)
    case 143:
        return "PTQ1_0";
    case 144:
        return "TQ2_T";
    case 145:
        return "TQK6";
    case 146:
        return "TQK7";
    case 147:
        return "TQP5";
    case 148:
        return "TQP6";
    case 149:
        return "TQP7";
    default:
        return "?";
    }
}

// Block geometry: (elements per block, bytes per block). Q2_0 is 64/18 - proven from this artifact's
// own offset brackets in P0.S6 and recorded in docs/q2_0-contract.md.
inline bool block_geometry(uint32_t t, int& elems, int& bytes) {
    switch (t) {
    case 0:
        elems = 1;
        bytes = 4;
        return true;
    case 1:
    case 30:
        elems = 1;
        bytes = 2;
        return true;
    case 2:
        elems = 32;
        bytes = 18;
        return true;
    case 3:
        elems = 32;
        bytes = 20;
        return true;
    case 6:
        elems = 32;
        bytes = 22;
        return true;
    case 7:
        elems = 32;
        bytes = 24;
        return true;
    case 8:
        elems = 32;
        bytes = 34;
        return true;
    case 9:
        elems = 32;
        bytes = 36;
        return true;
    case 10:
        elems = 256;
        bytes = 84;
        return true;
    case 11:
        elems = 256;
        bytes = 110;
        return true;
    case 12:
        elems = 256;
        bytes = 144;
        return true;
    case 13:
        elems = 256;
        bytes = 176;
        return true;
    case 14:
        elems = 256;
        bytes = 210;
        return true;
    case 16:
        elems = 256;
        bytes = 66;
        return true;
    case 17:
        elems = 256;
        bytes = 74;
        return true;
    case 18:
        elems = 256;
        bytes = 98;
        return true;
    case 20:
        elems = 32;
        bytes = 18;
        return true;
    case 21:   // IQ3_S
        elems = 256;
        bytes = 110;
        return true;
    case 22:   // IQ2_S
        elems = 256;
        bytes = 82;
        return true;
    case 23:
        elems = 256;
        bytes = 136;
        return true;
    case 29:   // IQ1_M
        elems = 256;
        bytes = 56;
        return true;
    case 24:   // I8: raw bytes (the FP8 PLE table of tools/ple_fp8_pack.py)
        elems = 1;
        bytes = 1;
        return true;
    case 40:   // NVFP4: 64 values = 4 UE4M3 sub-block scales + 32 bytes of E2M1 codes
        elems = 64;
        bytes = 36;
        return true;
    case 39:   // MXFP4: one E8M0 scale + 16 bytes of 4-bit values (Qwen3.6's MXFP4_MOE experts)
        elems = 32;
        bytes = 17;
        return true;
    case 42:
        elems = 64;
        bytes = 18;
        return true;
    // APR trellis types: 128 values = fp16 scale + 32 steps of K bits (K = 8 TQ2_T, 6 TQK6, 7 TQK7);
    // PTQ1_0 = 26 B of base-3 digits + fp16 scale
    case 143:
        elems = 128;
        bytes = 28;
        return true;
    case 144:
        elems = 128;
        bytes = 34;
        return true;
    case 145:
        elems = 128;
        bytes = 26;
        return true;
    case 146:
        elems = 128;
        bytes = 30;
        return true;
    // Gyro-L pair trellis: 128 values = fp16 scale + 64 steps of K bits, 2 weights per step (8*K bytes)
    case 147:
        elems = 128;
        bytes = 42;
        return true;
    case 148:
        elems = 128;
        bytes = 50;
        return true;
    case 149:
        elems = 128;
        bytes = 58;
        return true;
    default:
        return false;
    }
}

// The Edge0 container uses HF tensor names; the engine's native path uses the
// llama.cpp qwen4exp names.  Map the names the loader asks for onto the names
// the file carries, without touching the tensor bytes.
inline std::string edge0_tensor_alias(const std::string& n) {
    if (n == "token_embd.weight") return "model.language_model.embed_tokens.weight";
    if (n == "output.weight") return "lm_head.weight";
    if (n == "output_hc_norm.weight") return "model.language_model.hyper_connection_mixer.hc_norm.weight";
    if (n == "output_hc_down.weight") return "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight";
    if (n == "output_hc_up.weight") return "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight";
    if (n.rfind("blk.", 0) != 0) return n;
    const size_t dot = n.find('.', 4);
    if (dot == std::string::npos) return n;
    const std::string layer = n.substr(4, dot - 4);
    const std::string s = n.substr(dot + 1);
    const std::string pre = "model.language_model.layers." + layer + ".";
    if (s == "attn_q.weight") return pre + "self_attn.q_proj.weight";
    if (s == "attn_q_norm.weight") return pre + "self_attn.q_norm.weight";
    if (s == "attn_k.weight") return pre + "self_attn.k_proj.weight";
    if (s == "attn_k_norm.weight") return pre + "self_attn.k_norm.weight";
    if (s == "attn_v.weight") return pre + "self_attn.v_proj.weight";
    if (s == "attn_output.weight") return pre + "self_attn.o_proj.weight";
    if (s == "hc_attn_norm.weight") return pre + "attn_hyper_connection.hc_norm.weight";
    if (s == "hc_attn_down.weight") return pre + "attn_hyper_connection.input_mix_weight_down.weight";
    if (s == "hc_attn_up.weight") return pre + "attn_hyper_connection.input_mix_weight_up.weight";
    if (s == "hc_attn_inject.weight") return pre + "attn_hyper_connection.block_inject_weight.weight";
    if (s == "hc_ffn_norm.weight") return pre + "mlp_hyper_connection.hc_norm.weight";
    if (s == "hc_ffn_down.weight") return pre + "mlp_hyper_connection.input_mix_weight_down.weight";
    if (s == "hc_ffn_up.weight") return pre + "mlp_hyper_connection.input_mix_weight_up.weight";
    if (s == "hc_ffn_inject.weight") return pre + "mlp_hyper_connection.block_inject_weight.weight";
    if (s == "ffn_gate_inp.weight") return pre + "mlp.gate.weight";
    if (s == "ffn_gate_inp_shexp.weight") return pre + "mlp.shared_expert_gate.weight";
    if (s == "ffn_gate_shexp.weight") return pre + "mlp.shared_expert.gate_proj.weight";
    if (s == "ffn_up_shexp.weight") return pre + "mlp.shared_expert.up_proj.weight";
    if (s == "ffn_down_shexp.weight") return pre + "mlp.shared_expert.down_proj.weight";
    if (s == "ssm_a") return pre + "linear_attn.A_log";
    if (s == "ssm_dt.bias") return pre + "linear_attn.dt_bias";
    if (s == "ssm_conv1d.weight") return pre + "linear_attn.conv1d.weight";
    if (s == "ssm_norm.weight") return pre + "linear_attn.norm.weight";
    if (s == "ssm_alpha.weight") return pre + "linear_attn.in_proj_a.weight";
    if (s == "ssm_beta.weight") return pre + "linear_attn.in_proj_b.weight";
    if (s == "ssm_out.weight") return pre + "linear_attn.out_proj.weight";
    if (s == "attn_qkv.weight") return pre + "linear_attn.in_proj_qkv.weight";
    if (s == "indexer.q_proj.weight") return pre + "self_attn.indexer.index_qk_proj.weight";
    if (s == "indexer.k_proj.weight") return pre + "self_attn.indexer.index_qk_proj.weight";
    if (s == "indexer.q_norm.weight") return pre + "self_attn.indexer.q_layernorm.weight";
    if (s == "indexer.k_norm.weight") return pre + "self_attn.indexer.k_layernorm.weight";
    if (s == "ple_key.weight") return pre + "ple.key_proj.weight";
    if (s == "ple_value.weight") return pre + "ple.value_proj.weight";
    if (s == "ple_norm_key.weight") return pre + "ple.norm_key.weight";
    if (s == "ple_norm_query.weight") return pre + "ple.norm_query.weight";
    if (s == "ple_norm_conv.weight") return pre + "ple.norm_conv.weight";
    if (s == "ple_conv1d.weight") return pre + "ple.conv1d.weight";
    return n;
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape; // GGUF order: dim 0 varies fastest
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t elements() const {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    const char* type_name() const { return ggml_type_name(type); }
};

// A bounds-checked cursor over the mmapped header region. Every read is checked, so a truncated or
// corrupt file produces a precise error rather than a segfault.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t size) : base_(base), size_(size) {}
    void need(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("GGUF: unexpected end of file in header");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = read<uint64_t>();
        need((size_t)n);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), (size_t)n);
        pos_ += (size_t)n;
        return s;
    }
    size_t pos() const { return pos_; }

private:
    const uint8_t* base_;
    size_t size_;
    size_t pos_ = 0;
};

enum class MetaType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 };

struct MetaValue {
    MetaType type = MetaType::U32;
    uint64_t u = 0;                // integer payloads
    double f = 0;                  // float payloads
    std::string s;                 // string payloads
    MetaType elem = MetaType::U32; // arrays
    uint64_t count = 0;
    std::vector<MetaValue> items;
    bool is_num() const { return type != MetaType::STRING && type != MetaType::ARRAY; }
    double num() const { return (type == MetaType::F32 || type == MetaType::F64) ? f : (double)u; }
};

inline MetaValue read_value(Cursor& c, MetaType t, int depth = 0) {
    if (depth > 2) throw std::runtime_error("GGUF: array nesting too deep");
    MetaValue v;
    v.type = t;
    switch (t) {
    case MetaType::U8:
        v.u = c.read<uint8_t>();
        break;
    case MetaType::I8:
        v.u = (uint64_t)(int64_t)c.read<int8_t>();
        break;
    case MetaType::U16:
        v.u = c.read<uint16_t>();
        break;
    case MetaType::I16:
        v.u = (uint64_t)(int64_t)c.read<int16_t>();
        break;
    case MetaType::U32:
        v.u = c.read<uint32_t>();
        break;
    case MetaType::I32:
        v.u = (uint64_t)(int64_t)c.read<int32_t>();
        break;
    case MetaType::F32: {
        float x = c.read<float>();
        v.f = x;
        v.u = 0;
        break;
    }
    case MetaType::BOOL:
        v.u = c.read<uint8_t>() ? 1 : 0;
        break;
    case MetaType::STRING:
        v.s = c.str();
        break;
    case MetaType::U64:
        v.u = c.read<uint64_t>();
        break;
    case MetaType::I64:
        v.u = (uint64_t)c.read<int64_t>();
        break;
    case MetaType::F64:
        v.f = c.read<double>();
        break;
    case MetaType::ARRAY: {
        v.elem = (MetaType)c.read<uint32_t>();
        v.count = c.read<uint64_t>();
        if (v.count > (1u << 24)) throw std::runtime_error("GGUF: implausible array length");
        v.items.reserve((size_t)std::min<uint64_t>(v.count, 64));
        for (uint64_t i = 0; i < v.count; ++i) {
            MetaValue e = read_value(c, v.elem, depth + 1);
            if (i < 64) v.items.push_back(std::move(e)); // keep a sample; the count is what matters
        }
        break;
    }
    default:
        throw std::runtime_error("GGUF: unknown metadata value type");
    }
    return v;
}

class GgufFile {
public:
    explicit GgufFile(const std::string& path) : path_(path) {
        try { open(); }
        catch (...) { close(); throw; }
    }
    ~GgufFile() { close(); }
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, MetaValue>& metadata() const { return meta_; }
    uint32_t version() const { return version_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t file_size() const { return size_; }
    uint64_t alignment() const { return alignment_; }
    const std::string& path() const { return path_; }

    const TensorInfo* find(const std::string& name) const {
        for (const auto& t : tensors_)
            if (t.name == name) return &t;
        const std::string alias = edge0_tensor_alias(name);
        if (alias != name)
            for (const auto& t : tensors_)
                if (t.name == alias) return &t;
        return nullptr;
    }
    uint64_t count_type(const char* tn) const {
        uint64_t n = 0;
        for (const auto& t : tensors_)
            if (std::strcmp(t.type_name(), tn) == 0) ++n;
        return n;
    }
    const MetaValue* get(const std::string& key) const {
        auto it = meta_.find(key);
        // The Edge0 student keeps the Flash-Next shell but names its architecture "edge0"; the
        // shell metadata is the same keys under the edge0 prefix.  Alias them at the read.
        if (it == meta_.end() && key.rfind("qwen4exp.", 0) == 0)
            it = meta_.find("edge0." + key.substr(9));
        return it == meta_.end() ? nullptr : &it->second;
    }
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + data_start_ + t.offset; }

private:
    void open() {
#ifdef _WIN32
        HANDLE h = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path_);
        LARGE_INTEGER li{};
        GetFileSizeEx(h, &li);
        size_ = (uint64_t)li.QuadPart;
        HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m) {
            CloseHandle(h);
            throw std::runtime_error("CreateFileMapping failed");
        }
        base_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base_) {
            CloseHandle(m);
            CloseHandle(h);
            throw std::runtime_error("MapViewOfFile failed");
        }
        map_ = m;
        file_ = h;
#else
        int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path_);
        fd_ = fd;
        struct stat st{};
        if (fstat(fd, &st) != 0) throw std::runtime_error("fstat failed");
        size_ = (uint64_t)st.st_size;
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base_ = (const uint8_t*)p;
#endif
        parse();
    }
    void close() {
#ifdef _WIN32
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle((HANDLE)map_);
        if (file_) CloseHandle((HANDLE)file_);
#else
        if (base_) munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    void parse() {
        Cursor c(base_, size_);
        const uint32_t magic = c.read<uint32_t>();
        if (magic != 0x46554747u) throw std::runtime_error("not a GGUF file (bad magic)");
        version_ = c.read<uint32_t>();
        if (version_ != 3)
            throw std::runtime_error("GGUF v" + std::to_string(version_) + ", this reader handles v3");
        const uint64_t n_tensors = c.read<uint64_t>();
        const uint64_t n_kv = c.read<uint64_t>();

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.str();
            MetaType t = (MetaType)c.read<uint32_t>();
            meta_.emplace(std::move(key), read_value(c, t));
        }
        tensors_.reserve((size_t)n_tensors);
        // GGUF has no index to arbitrate between two tensors of one name: find() is first-match, so a
        // duplicate would silently win by position.  Refuse the file at open instead, naming both.
        std::set<std::string> names;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = c.str();
            if (!names.insert(t.name).second)
                throw std::runtime_error("GGUF: duplicate tensor name '" + t.name + "' in " + path_);
            const uint32_t nd = c.read<uint32_t>();
            if (nd == 0 || nd > 4) throw std::runtime_error("GGUF: bad n_dims for " + t.name);
            t.shape.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.shape[d] = c.read<uint64_t>();
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        if (data_start_ > size_) {
            // A file with tensors needs its data section.  One without (a split model's metadata-only first
            // shard) may end at its header, before the aligned start - unsloth's UD-Q5_K_XL shard 1 ends 6 bytes
            // short, and llama.cpp's gguf-py reads it.  Nothing is read there; the clamp keeps
            // `file_size() - data_start()` (the payload checks downstream) at 0 instead of wrapping.
            if (n_tensors != 0) throw std::runtime_error("GGUF: data section starts past EOF");
            data_start_ = size_;
        }
    }

    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0, data_start_ = 0, alignment_ = 32;
    uint32_t version_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* map_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::vector<TensorInfo> tensors_;
    std::map<std::string, MetaValue> meta_;
};

// Bytes of a tensor's payload from its shape and block geometry; 0 when the type is unknown, a row is not whole
// blocks, or the count overflows.
inline uint64_t tensor_payload_bytes(const TensorInfo& t) {
    int be = 0, bb = 0;
    if (t.shape.empty() || !block_geometry(t.type, be, bb) || t.shape[0] % (uint64_t) be) return 0;
    uint64_t elements = 1;
    for (uint64_t d : t.shape) {
        if (d == 0 || elements > (std::numeric_limits<uint64_t>::max)() / d) return 0;
        elements *= d;
    }
    const uint64_t blocks = elements / (uint64_t) be;
    if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) bb) return 0;
    return blocks * (uint64_t) bb;
}

// The shards of one model (strata::gguf_split_paths), opened together; tensors are looked up across all of them.
// From eddoursul/Strata 8029fa9, with the split-key validation of #255 (gopinath87607) made a property of the
// model rather than of one loader: a split GGUF carries the model's metadata (general.architecture and the
// qwen4exp keys) in its FIRST shard only, and the later shards just declare split.count / split.no /
// split.tensors.count.  Unsloth's UD-Q4_K_XL is the extreme case: shard 1 holds the metadata and no tensor at
// all, and output.weight, token_embd.weight and the PLE table live in shard 2.  So the architecture is checked
// on `meta()` and a tensor is read from the shard that holds it.
//
// Refused at construction: a later shard whose split keys disagree with shard 1's (another model's shard, or a
// shard renamed into the family), a split model whose tensor directories do not add up to split.tensors.count,
// and a tensor name present in two shards (GGUF has no index to say which one is meant).
class GgufModel {
public:
    explicit GgufModel(const std::vector<std::string>& paths) {
        if (paths.empty()) throw std::runtime_error("GGUF: a model needs at least one shard");
        for (const auto& p : paths) shards_.push_back(std::make_unique<GgufFile>(p));
        validate_split();
        for (size_t i = 0; i < shards_.size(); ++i)
            for (const auto& t : shards_[i]->tensors()) {
                const auto ins = index_.emplace(t.name, std::make_pair(i, &t));
                if (!ins.second)
                    throw std::runtime_error("GGUF: tensor " + t.name + " is in two shards (" +
                                             shards_[ins.first->second.first]->path() + " and " +
                                             shards_[i]->path() + ")");
            }
    }
    /// Opens every shard of the model that `any_shard` belongs to (throws when one is missing).
    static GgufModel open(const std::string& any_shard) { return GgufModel(gguf_split_paths(any_shard)); }

    size_t size() const { return shards_.size(); }
    const GgufFile& shard(size_t i) const { return *shards_[i]; }
    /// The metadata shard: general.architecture and the model's keys.
    const GgufFile& meta() const { return *shards_[0]; }
    /// The tensor named `name` (its shard index in `*shard`), or nullptr.
    const TensorInfo* find(const std::string& name, size_t* shard = nullptr) const {
        auto it = index_.find(name);
        if (it == index_.end()) {
            const std::string alias = edge0_tensor_alias(name);
            if (alias != name) it = index_.find(alias);
        }
        if (it == index_.end()) return nullptr;
        if (shard) *shard = it->second.first;
        return it->second.second;
    }
    /// Whether `t` (a tensor of shard `s`) has a known byte count that lies inside its file.
    bool in_bounds(const TensorInfo& t, size_t s) const {
        const GgufFile& g = *shards_[s];
        const uint64_t bytes = tensor_payload_bytes(t);
        const uint64_t payload = g.file_size() - g.data_start();
        return bytes != 0 && t.offset <= payload && bytes <= payload - t.offset;
    }

private:
    void validate_split() const {
        const size_t n = shards_.size();
        const MetaValue* count0 = shards_[0]->get("split.count");
        if (n == 1) {
            if (count0 && count0->u > 1)
                throw std::runtime_error("GGUF: " + shards_[0]->path() + " is shard 1 of " + std::to_string(count0->u) +
                                         ", but it was opened as a whole model");
            return;
        }
        if (!shards_[0]->get("general.architecture"))
            throw std::runtime_error("GGUF: " + shards_[0]->path() + " has no general.architecture; the first shard "
                                     "of a split model carries the metadata");
        const MetaValue* total = shards_[0]->get("split.tensors.count");
        uint64_t tensors = 0;
        for (size_t i = 0; i < n; ++i) {
            const GgufFile& g = *shards_[i];
            const MetaValue* count = g.get("split.count");
            const MetaValue* no = g.get("split.no");
            const MetaValue* tc = g.get("split.tensors.count");
            if (!count || !no || count->u != n || no->u != i || (total && (!tc || tc->u != total->u)))
                throw std::runtime_error("GGUF: " + g.path() + " does not declare itself shard " + std::to_string(i + 1) +
                                         " of " + std::to_string(n) + " of this model (split.count / split.no / "
                                         "split.tensors.count)");
            tensors += g.tensors().size();
        }
        if (total && tensors != total->u)
            throw std::runtime_error("GGUF: the " + std::to_string(n) + " shards hold " + std::to_string(tensors) +
                                     " tensors, but split.tensors.count is " + std::to_string(total->u));
    }

    std::vector<std::unique_ptr<GgufFile>> shards_;
    std::map<std::string, std::pair<size_t, const TensorInfo*>> index_;
};

/// The family of the model whose metadata shard is `g` (Unknown when it has no or another architecture).
inline ModelFamily model_family(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    return arch ? model_family_from_arch(arch->s) : ModelFamily::Unknown;
}

/// Qwen3.6-35B-A3B's guard: every key the geometry is read from must be present (core::geometry_from_gguf),
/// and the few that the layer pattern depends on must have the values the model has.
inline std::string check_architecture_qwen35moe(const GgufFile& g) {
    static const char* const keys[] = {
        "qwen35moe.block_count", "qwen35moe.embedding_length", "qwen35moe.attention.head_count",
        "qwen35moe.attention.head_count_kv", "qwen35moe.attention.key_length", "qwen35moe.attention.value_length",
        "qwen35moe.expert_count", "qwen35moe.expert_used_count", "qwen35moe.expert_feed_forward_length",
        "qwen35moe.expert_shared_feed_forward_length", "qwen35moe.ssm.conv_kernel", "qwen35moe.ssm.state_size",
        "qwen35moe.ssm.group_count", "qwen35moe.ssm.time_step_rank", "qwen35moe.ssm.inner_size",
        "qwen35moe.full_attention_interval", "qwen35moe.nextn_predict_layers",
    };
    for (const char* k : keys) {
        const MetaValue* v = g.get(k);
        if (!v) return std::string("missing ") + k;
        if (!v->is_num()) return std::string(k) + " is not a number";
    }
    const MetaValue* nl = g.get("qwen35moe.block_count");
    const MetaValue* mtp = g.get("qwen35moe.nextn_predict_layers");
    if (mtp->u >= nl->u) return "qwen35moe.nextn_predict_layers is not smaller than qwen35moe.block_count";
    if (g.get("qwen35moe.full_attention_interval")->u < 2)
        return "qwen35moe.full_attention_interval must be at least 2 (the GDN / attention pattern)";
    return {};
}

// ---- the model's geometry from its metadata, and the architecture guard (P1.S2).  The engine runs ONE
// architecture, qwen4exp.  Its dimensions are read from the file (llama.cpp's key names under the arch prefix)
// and then checked against the shape the kernels are compiled for (core/geometry.hpp), so a file of another
// shape is refused with the key that differs rather than silently mis-run.

/// Fills `g` from the metadata shard `f`.  Returns "" or the first problem.  Another architecture is refused by
/// name first: its keys carry another prefix, and "missing qwen4exp.block_count" would hide what is wrong.  The six
/// keys every qwen4exp file has carried since the first guard are required; any other key the file does not carry
/// keeps the compiled value.  The keys that describe one thing twice (the interval and `compress_ratios`, the key
/// and value head widths, the routed and shared expert widths) must agree.
inline std::string read_geometry(const GgufFile& f, core::ModelGeometry& g) {
    g = core::kCompiledGeometry;
    const MetaValue* arch = f.get("general.architecture");
    if (!arch) return "missing general.architecture";
    const bool edge0_arch = arch->s == "edge0";
    if (arch->s != "qwen4exp" && !edge0_arch)
        return "architecture is '" + arch->s + "', this engine requires 'qwen4exp' or 'edge0'";
    const std::string p = edge0_arch ? "edge0." : "qwen4exp.";
    std::string err;
    // an integer key into `out`: present and positive, or absent and not `required`
    auto read_int = [&](const char* key, int64_t& out, bool required) {
        const MetaValue* v = f.get(p + key);
        if (!v) {
            if (required) err = "missing " + p + key;
            return !required;
        }
        if (!v->is_num() || v->type == MetaType::F32 || v->type == MetaType::F64 || v->type == MetaType::BOOL) {
            err = p + key + " is not an integer";
            return false;
        }
        if ((int64_t) v->u <= 0) {
            err = p + key + " = " + std::to_string((int64_t) v->u) + ", it must be positive";
            return false;
        }
        out = (int64_t) v->u;
        return true;
    };
    struct Key {
        const char* key;
        int64_t core::ModelGeometry::*field;
        bool required;
    };
    const Key keys[] = {
        {"block_count", &core::ModelGeometry::n_layers, true},
        {"embedding_length", &core::ModelGeometry::n_embd, true},
        {"expert_count", &core::ModelGeometry::n_expert, true},
        {"expert_used_count", &core::ModelGeometry::n_expert_used, true},
        {"attention.head_count", &core::ModelGeometry::n_head, true},
        {"attention.head_count_kv", &core::ModelGeometry::n_head_kv, true},
        {"full_attention_interval", &core::ModelGeometry::qsa_interval, false},
        {"ssm.state_size", &core::ModelGeometry::ssm_state_size, false},
        {"ssm.group_count", &core::ModelGeometry::ssm_k_heads, false},
        {"ssm.time_step_rank", &core::ModelGeometry::ssm_v_heads, false},
        {"ssm.conv_kernel", &core::ModelGeometry::ssm_d_conv, false},
        {"ssm.inner_size", &core::ModelGeometry::ssm_value_dim, false},
        {"attention.key_length", &core::ModelGeometry::head_dim, false},
        {"rope.dimension_count", &core::ModelGeometry::n_rot, false},
        {"attention.indexer.head_count", &core::ModelGeometry::idx_q_heads, false},
        {"attention.indexer.key_length", &core::ModelGeometry::idx_key_dim, false},
        {"attention.indexer.top_k", &core::ModelGeometry::idx_top_k, false},
        {"hyper_connection.count", &core::ModelGeometry::hc, false},
        {"hyper_connection.low_rank", &core::ModelGeometry::hc_lr, false},
        {"expert_feed_forward_length", &core::ModelGeometry::n_ff, false},
    };
    for (const Key& k : keys)
        if (!read_int(k.key, g.*k.field, k.required && !edge0_arch)) return err;
    if (edge0_arch) {
        // Edge0 carries its latent metadata under edge0.*; the shell geometry is Flash-Next's
        // compiled defaults, overlaid with the few dimensions it does record.
        auto edge_num = [&](const char* key) -> int64_t {
            const MetaValue* v = f.get(std::string("edge0.") + key);
            return v && v->is_num() ? (int64_t) v->u : 0;
        };
        if (int64_t v = edge_num("num_layers")) g.n_layers = v;
        if (int64_t v = edge_num("num_experts")) g.n_expert = v;
        if (int64_t v = edge_num("top_k")) g.n_expert_used = v;
    }

    int64_t value_length = g.head_dim, ff_shexp = g.n_ff;
    if (!read_int("attention.value_length", value_length, false) ||
        !read_int("expert_shared_feed_forward_length", ff_shexp, false))
        return err;
    if (value_length != g.head_dim)
        return p + "attention.value_length = " + std::to_string(value_length) + " and attention.key_length = " +
               std::to_string(g.head_dim) + ": the attention kernels use one head width for both";
    if (ff_shexp != g.n_ff)
        return p + "expert_shared_feed_forward_length = " + std::to_string(ff_shexp) +
               " and expert_feed_forward_length = " + std::to_string(g.n_ff) +
               ": the shared expert is read with the routed experts' width";
    if (const MetaValue* v = f.get(p + "attention.layer_norm_rms_epsilon")) {
        if (!v->is_num() || !(v->num() > 0)) return p + "attention.layer_norm_rms_epsilon must be a positive number";
        g.rms_eps = v->num();
    }
    if (const MetaValue* v = f.get("tokenizer.ggml.tokens")) {
        if (v->type != MetaType::ARRAY || v->count == 0) return "tokenizer.ggml.tokens is not a list of tokens";
        g.n_vocab = (int64_t) v->count;
    }

    if (g.n_head % g.n_head_kv != 0)
        return p + "attention.head_count = " + std::to_string(g.n_head) + " is not a multiple of head_count_kv = " +
               std::to_string(g.n_head_kv);
    // GDN's value heads are as wide as its key heads (the state is square)
    if (g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads)
        return p + "ssm.inner_size = " + std::to_string(g.ssm_value_dim) + ", but ssm.state_size x ssm.time_step_rank = " +
               std::to_string(g.ssm_state_size * g.ssm_v_heads) + ": the GDN kernels take square heads";
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    if (g.n_expert_used > g.n_expert || g.n_expert_used > 64)
        return p + "expert_used_count = " + std::to_string(g.n_expert_used) + ", the router takes 1.." +
               std::to_string(std::min<int64_t>(g.n_expert, 64)) + " of " + std::to_string(g.n_expert) + " experts";
    if (g.n_rot > g.head_dim || g.n_rot > g.idx_key_dim || g.n_rot % 2)
        return p + "rope.dimension_count = " + std::to_string(g.n_rot) + " is not an even slice of the head widths";

    // The layer split, twice over: the interval, and a ratio per layer that is non-zero (the indexer's pooling
    // block) exactly on the full-attention layers.
    if (const MetaValue* v = f.get(p + "attention.compress_ratios")) {
        if (v->type != MetaType::ARRAY || v->count != (uint64_t) g.n_layers || v->items.size() != v->count)
            return p + "attention.compress_ratios must list one entry for each of the " + std::to_string(g.n_layers) +
                   " layers";
        // A list of zeros only is a converter's placeholder, not a layer split (#2396: orcarouter's
        // Qwen3.8-Flash-Next-Uncensored GGUFs carry 48 zeros; 0.1.43 never read the key): it is treated as absent
        // and the compiled pooling block is used.  A list with any non-zero entry is checked in full below.
        bool given = false;
        for (int64_t l = 0; l < g.n_layers && !given; ++l) given = v->items[(size_t) l].u != 0;
        if (!given) {
            static bool said = false;   // the guard runs several times in one process: one line
            if (!said) {
                said = true;
                std::fprintf(stderr, "strata: %sattention.compress_ratios is all zeros (a converter's placeholder): "
                                     "using the compiled pooling block of %lld cells (#2396)\n",
                             p.c_str(), (long long) g.idx_block);
            }
            return {};
        }
        int64_t block = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const int64_t r = (int64_t) v->items[(size_t) l].u;
            if ((r != 0) != core::is_qsa_layer(g, l) || r < 0 || (block && r && r != block))
                return p + "attention.compress_ratios[" + std::to_string(l) + "] = " + std::to_string(r) +
                       ", but full_attention_interval = " + std::to_string(g.qsa_interval) + " makes layer " +
                       std::to_string(l) + (core::is_qsa_layer(g, l) ? " full attention" : " linear attention") +
                       (block && r ? " (and the other full-attention layers pool " + std::to_string(block) + ")" : "");
            if (r) block = r;
        }
        if (block) g.idx_block = block;
    }
    return {};
}

/// Returns "" when the compiled kernels can run `g`, else the first metadata key whose value they are not
/// compiled for.  Only the expert count and the experts per token are free (the router and the expert paths
/// take them at run time); the vocabulary is free because the head and the embedding take their tensors' shapes.
inline std::string geometry_unsupported(const core::ModelGeometry& g) {
    const core::ModelGeometry& c = core::kCompiledGeometry;
    struct Pin {
        const char* key;
        int64_t core::ModelGeometry::*field;
    };
    const Pin pins[] = {
        {"block_count", &core::ModelGeometry::n_layers},
        {"embedding_length", &core::ModelGeometry::n_embd},
        {"full_attention_interval", &core::ModelGeometry::qsa_interval},
        {"ssm.state_size", &core::ModelGeometry::ssm_state_size},
        {"ssm.group_count", &core::ModelGeometry::ssm_k_heads},
        {"ssm.time_step_rank", &core::ModelGeometry::ssm_v_heads},
        {"ssm.conv_kernel", &core::ModelGeometry::ssm_d_conv},
        {"ssm.inner_size", &core::ModelGeometry::ssm_value_dim},
        {"attention.head_count", &core::ModelGeometry::n_head},
        {"attention.head_count_kv", &core::ModelGeometry::n_head_kv},
        {"attention.key_length", &core::ModelGeometry::head_dim},
        {"rope.dimension_count", &core::ModelGeometry::n_rot},
        {"attention.indexer.head_count", &core::ModelGeometry::idx_q_heads},
        {"attention.indexer.key_length", &core::ModelGeometry::idx_key_dim},
        {"attention.compress_ratios", &core::ModelGeometry::idx_block},
        {"attention.indexer.top_k", &core::ModelGeometry::idx_top_k},
        {"hyper_connection.count", &core::ModelGeometry::hc},
        {"hyper_connection.low_rank", &core::ModelGeometry::hc_lr},
        {"expert_feed_forward_length", &core::ModelGeometry::n_ff},
    };
    for (const Pin& pin : pins)
        if (g.*pin.field != c.*pin.field)
            return std::string("qwen4exp.") + pin.key + " = " + std::to_string(g.*pin.field) +
                   ", the kernels are compiled for " + std::to_string(c.*pin.field);
    // one constant in the norm kernels; the files carry it as an f32, so compare at f32 precision
    if (std::abs(g.rms_eps - c.rms_eps) > 1e-6 * c.rms_eps) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "qwen4exp.attention.layer_norm_rms_epsilon = %g, the kernels are compiled for %g",
                      g.rms_eps, c.rms_eps);
        return buf;
    }
    return {};
}

/// The architecture guard: "" when the engine can read the model whose metadata shard is `f`.  Qwen3.8-Flash-Next
/// (qwen4exp) is read and checked against the shape the kernels are compiled for; Qwen3.6-35B-A3B (qwen35moe) must
/// carry every key its geometry is read from (core::geometry_from_gguf).
inline std::string check_architecture(const GgufFile& f) {
    const MetaValue* arch = f.get("general.architecture");
    if (!arch) return "missing general.architecture";
    const ModelFamily family = model_family_from_arch(arch->s);
    if (family == ModelFamily::Qwen35Moe) return check_architecture_qwen35moe(f);
    core::ModelGeometry g;   // anything else is refused by name by read_geometry
    std::string err = read_geometry(f, g);
    if (err.empty()) err = geometry_unsupported(g);
    return err;
}

/// The engine's guard: `check_architecture`, then a family this engine has a path for.  A recognised family
/// it does not run yet (Qwen3.6 on SYCL) is refused with a message that says so.
inline std::string check_engine_architecture(const GgufFile& g) {
    std::string err = check_architecture(g);
    if (!err.empty()) return err;
    return model_family_not_supported(model_family(g));
}

} // namespace strata
