// tests/core/gguf_fixture.hpp - synthetic GGUF v3 files for the loader tests (no gguf-py, no model, no GPU).
//
// A file is a header, metadata (strings, integers, f32, and arrays of u32 or strings), a tensor directory whose offsets are 32-byte aligned
// in the data section, and the tensors' payloads, each filled with a pattern the test can recognise.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fixture {

struct Kv {
    std::string key;
    uint32_t type = 8;         ///< GGUF value type: 8 string, 2 u16, 4 u32, 5 i32, 6 f32, 9 array
    std::string s;
    uint64_t u = 0;
    float f = 0;
    uint32_t elem = 4;                 ///< arrays: 4 u32 (`arr`) or 8 string (`sarr`)
    std::vector<uint64_t> arr;
    std::vector<std::string> sarr;
};
// fields set by name: a braced Kv{...} that stops short warns under -Wmissing-field-initializers (clang -Wextra)
inline Kv make_kv(const std::string& k, uint32_t type) {
    Kv r;
    r.key = k;
    r.type = type;
    return r;
}
inline Kv str(const std::string& k, const std::string& v) { Kv r = make_kv(k, 8); r.s = v; return r; }
inline Kv u16(const std::string& k, uint64_t v) { Kv r = make_kv(k, 2); r.u = v; return r; }
inline Kv i32(const std::string& k, uint64_t v) { Kv r = make_kv(k, 5); r.u = v; return r; }
inline Kv u32(const std::string& k, uint64_t v) { Kv r = make_kv(k, 4); r.u = v; return r; }
inline Kv f32(const std::string& k, float v) { Kv r = make_kv(k, 6); r.f = v; return r; }
inline Kv u32s(const std::string& k, std::vector<uint64_t> v) { Kv r = make_kv(k, 9); r.arr = std::move(v); return r; }
inline Kv strs(const std::string& k, std::vector<std::string> v) {
    Kv r = make_kv(k, 9);
    r.elem = 8;
    r.sarr = std::move(v);
    return r;
}
inline Kv str_array(const std::string& k, std::vector<std::string> v) { return strs(k, std::move(v)); }

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;   ///< GGUF order: dim 0 varies fastest
    uint32_t type = 0;
    uint8_t seed = 1;              ///< payload byte i is (seed * 131 + i * 7 + i / 4093) & 0xff
};

inline uint8_t pattern(uint8_t seed, uint64_t i) { return (uint8_t) ((seed * 131u + i * 7u + i / 4093u) & 0xffu); }

/// The split keys llama.cpp's gguf-split writes (u16 split.no / split.count, i32 split.tensors.count).
inline std::vector<Kv> split_keys(uint64_t no, uint64_t count, uint64_t tensors) {
    return {u16("split.no", no), u16("split.count", count), i32("split.tensors.count", tensors)};
}

struct Written {
    uint64_t data_start = 0;
    std::vector<uint64_t> offsets;   ///< per tensor, from data_start
    uint64_t size = 0;
};

/// Writes `path`; `cut` bytes are dropped from the end (a truncated download).
inline Written write(const std::filesystem::path& path, const std::vector<Kv>& kv, const std::vector<Tensor>& ts,
                     uint64_t cut = 0) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto put32 = [&](uint32_t v) { put(&v, 4); };
    auto put64 = [&](uint64_t v) { put(&v, 8); };
    auto puts = [&](const std::string& s) { put64(s.size()); put(s.data(), s.size()); };
    put32(0x46554747u);
    put32(3);
    put64(ts.size());
    put64(kv.size());
    for (const auto& k : kv) {
        puts(k.key);
        put32(k.type);
        if (k.type == 8) puts(k.s);
        else if (k.type == 2) { const uint16_t v = (uint16_t) k.u; put(&v, 2); }
        else if (k.type == 4 || k.type == 5) { const uint32_t v = (uint32_t) k.u; put(&v, 4); }
        else if (k.type == 6) put(&k.f, 4);
        else if (k.type == 9 && k.elem == 4) {
            put32(4);
            put64(k.arr.size());
            for (uint64_t x : k.arr) put32((uint32_t) x);
        } else if (k.type == 9 && k.elem == 8) {
            put32(8);
            put64(k.sarr.size());
            for (const auto& x : k.sarr) puts(x);
        } else throw std::runtime_error("fixture: unsupported metadata type");
    }
    Written w;
    std::vector<uint64_t> bytes;
    uint64_t at = 0;
    for (const auto& t : ts) {
        strata::TensorInfo info;
        info.shape = t.shape;
        info.type = t.type;
        const uint64_t n = strata::tensor_payload_bytes(info);
        if (n == 0) throw std::runtime_error("fixture: " + t.name + " has no whole-block byte count");
        puts(t.name);
        put32((uint32_t) t.shape.size());
        for (uint64_t d : t.shape) put64(d);
        put32(t.type);
        put64(at);
        w.offsets.push_back(at);
        bytes.push_back(n);
        at = (at + n + 31) / 32 * 32;
    }
    w.data_start = (b.size() + 31) / 32 * 32;
    b.resize((size_t) w.data_start, 0);
    for (size_t i = 0; i < ts.size(); ++i) {
        b.resize((size_t) (w.data_start + w.offsets[i]), 0);
        for (uint64_t j = 0; j < bytes[i]; ++j) b.push_back(pattern(ts[i].seed, j));
    }
    if (cut > b.size()) cut = b.size();
    b.resize(b.size() - (size_t) cut);
    w.size = b.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char*) b.data(), (std::streamsize) b.size());
    if (!f) throw std::runtime_error("fixture: cannot write " + path.string());
    return w;
}

}  // namespace fixture
