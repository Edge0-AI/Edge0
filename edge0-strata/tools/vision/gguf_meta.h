// tools/vision/gguf_meta.h - open a model GGUF for its vocabulary only, whatever its weight tensors are.
//
// The picture reader needs the text model for one thing: its tokenizer (and the header values that go with it).  llama.cpp's
// loader reads the whole tensor directory first and refuses a file with a tensor type it does not know - and a Gyro
// quantization (TQK6 / TQK7, ids 145 and 146) and other Strata-private types are exactly that - so even a vocab-only
// load failed with "cannot open the text model".  strata_open_text_model() therefore copies the key/value header of the
// GGUF (everything before the tensor directory) into a small temporary GGUF without tensors, and loads THAT vocab-only.
// The weights are never read, mapped or type-checked, and the split keys are dropped so the loader does not go looking
// for the other shards.  If the header cannot be read (a type this reader does not know), it falls back to loading the
// file itself, as before.
#pragma once

#include "gguf.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define STRATA_GETPID _getpid
#else
#include <unistd.h>
#define STRATA_GETPID getpid
#endif

namespace strata_gguf {

struct Reader {
    FILE* f = nullptr;
    std::string err;
    bool take(void* dst, size_t n) {
        if (std::fread(dst, 1, n, f) != n) { err = "the GGUF header is cut short"; return false; }
        return true;
    }
    template <class T> bool num(T& v) { return take(&v, sizeof v); }
    bool str(std::string& s, uint64_t cap = (1ull << 26)) {
        uint64_t n = 0;
        if (!num(n)) return false;
        if (n > cap) { err = "a string in the GGUF header is too long"; return false; }
        s.resize((size_t) n);
        return n == 0 || take(&s[0], (size_t) n);
    }
};

inline size_t scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

inline bool dropped_key(const std::string& k) {
    return k == "split.no" || k == "split.count" || k == "split.tensors.count";
}

// Copy the key/value header of `src` into `dst` as a GGUF with no tensors.  False (and `err`) when the header holds
// something this reader does not handle.
inline bool copy_meta(const std::string& src, const std::string& dst, std::string& err) {
    Reader r;
    r.f = std::fopen(src.c_str(), "rb");
    if (!r.f) { err = "cannot open " + src; return false; }
    struct Closer { FILE* f; ~Closer() { if (f) std::fclose(f); } } closer{r.f};
    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!r.take(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) { err = r.err.empty() ? "not a GGUF file" : r.err; return false; }
    if (!r.num(version) || !r.num(n_tensors) || !r.num(n_kv)) { err = r.err; return false; }
    if (version < 2 || version > 3) { err = "GGUF version " + std::to_string(version) + " is not handled"; return false; }
    if (n_kv > 100000) { err = "too many keys in the GGUF header"; return false; }
    gguf_context* out = gguf_init_empty();
    if (!out) { err = "out of memory"; return false; }
    struct Freer { gguf_context* g; ~Freer() { gguf_free(g); } } freer{out};
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!r.str(key, 1 << 16) || !r.num(type)) { err = r.err; return false; }
        const bool keep = !dropped_key(key);
        if (type == GGUF_TYPE_STRING) {
            std::string v;
            if (!r.str(v)) { err = r.err; return false; }
            if (keep) gguf_set_val_str(out, key.c_str(), v.c_str());
        } else if (type == GGUF_TYPE_ARRAY) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!r.num(et) || !r.num(n)) { err = r.err; return false; }
            if (n > 50'000'000ull) { err = "an array in the GGUF header is too long"; return false; }
            if (et == GGUF_TYPE_STRING) {
                std::vector<std::string> items((size_t) n);
                for (auto& s : items) if (!r.str(s)) { err = r.err; return false; }
                if (keep) {
                    std::vector<const char*> ptrs;
                    ptrs.reserve(items.size());
                    for (const auto& s : items) ptrs.push_back(s.c_str());
                    gguf_set_arr_str(out, key.c_str(), ptrs.data(), ptrs.size());
                }
            } else if (scalar_size(et)) {
                std::vector<uint8_t> raw((size_t) (n * scalar_size(et)));
                if (!raw.empty() && !r.take(raw.data(), raw.size())) { err = r.err; return false; }
                if (keep) gguf_set_arr_data(out, key.c_str(), (gguf_type) et, raw.data(), (size_t) n);
            } else { err = "an array of arrays in the GGUF header is not handled"; return false; }
        } else if (scalar_size(type)) {
            uint64_t raw = 0;
            if (!r.take(&raw, scalar_size(type))) { err = r.err; return false; }   // little endian, as the file
            if (!keep) continue;
            switch (type) {
                case GGUF_TYPE_UINT8: gguf_set_val_u8(out, key.c_str(), (uint8_t) raw); break;
                case GGUF_TYPE_INT8: gguf_set_val_i8(out, key.c_str(), (int8_t) raw); break;
                case GGUF_TYPE_UINT16: gguf_set_val_u16(out, key.c_str(), (uint16_t) raw); break;
                case GGUF_TYPE_INT16: gguf_set_val_i16(out, key.c_str(), (int16_t) raw); break;
                case GGUF_TYPE_UINT32: gguf_set_val_u32(out, key.c_str(), (uint32_t) raw); break;
                case GGUF_TYPE_INT32: gguf_set_val_i32(out, key.c_str(), (int32_t) raw); break;
                case GGUF_TYPE_FLOAT32: { float v; std::memcpy(&v, &raw, 4); gguf_set_val_f32(out, key.c_str(), v); break; }
                case GGUF_TYPE_BOOL: gguf_set_val_bool(out, key.c_str(), raw != 0); break;
                case GGUF_TYPE_UINT64: gguf_set_val_u64(out, key.c_str(), raw); break;
                case GGUF_TYPE_INT64: gguf_set_val_i64(out, key.c_str(), (int64_t) raw); break;
                case GGUF_TYPE_FLOAT64: { double v; std::memcpy(&v, &raw, 8); gguf_set_val_f64(out, key.c_str(), v); break; }
                default: break;
            }
        } else { err = "a value type in the GGUF header is not handled"; return false; }
    }
    if (!gguf_write_to_file(out, dst.c_str(), true)) { err = "cannot write " + dst; return false; }
    return true;
}

}  // namespace strata_gguf

// The text model for the picture reader: vocabulary and header only.  `how` says which way it was opened.
inline llama_model* strata_open_text_model(const std::string& model, std::string* how = nullptr) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    const char* tmpdir = std::getenv("TMPDIR");
    if (!tmpdir || !*tmpdir) tmpdir = std::getenv("TEMP");
    if (!tmpdir || !*tmpdir) tmpdir = std::getenv("TMP");
    if (!tmpdir || !*tmpdir) tmpdir = "/tmp";
    const std::string meta = std::string(tmpdir) + "/strata-vision-meta-" + std::to_string((long) STRATA_GETPID()) + ".gguf";
    std::string err;
    llama_model* m = nullptr;
    if (strata_gguf::copy_meta(model, meta, err)) {
        m = llama_model_load_from_file(meta.c_str(), mp);
        if (how) *how = m ? "header only" : "header copy was not accepted";
    } else if (how) *how = "header copy failed: " + err;
    std::remove(meta.c_str());
    if (!m) {   // the file as it is (a type of header this reader does not handle, or an llama.cpp that wants more)
        m = llama_model_load_from_file(model.c_str(), mp);
        if (how) *how += m ? ", loaded the file itself" : ", the file itself did not load either";
    }
    return m;
}
