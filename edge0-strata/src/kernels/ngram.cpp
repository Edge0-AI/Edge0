// src/kernels/ngram.cpp - P2.S4: the PLE n-gram hash and the IQ4_NL table read.
//
// See include/strata/kernels/ngram.hpp for the semantics, the rival readings, and the note on MADV_RANDOM.
#include "strata/kernels/ngram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/ngram/ple_reader.hpp"
#include "strata/platform/memory.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <stdexcept>

#if defined(_WIN32)
// `PrefetchVirtualMemory` (memoryapi.h, Windows 8+) is the whole point of the change in `gather` below.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace strata::kernels {

namespace {
/// The A/B arm.  Host-token-path only, so it needs no atomics; see the note on `ple_prefetch_enable`.
bool g_ple_prefetch = true;
}  // namespace

void ple_prefetch_enable(bool on) { g_ple_prefetch = on; }
bool ple_prefetch_enabled() { return g_ple_prefetch; }

PleConsts ple_artifact_consts() {
    // docs/gguf-dump-shard1.txt, verbatim.  Written once here rather than derived, because `head_offsets` is
    // ALSO in the metadata as its own array - and deriving one from the other would hide a mismatch between
    // them instead of surfacing it.  The parity test checks the two agree.
    PleConsts c{};
    c.mult[0] = 23703573157769ull;
    c.mult[1] = 20109073645365ull;
    c.mult[2] = 8052911324071ull;
    const uint64_t vocab[PLE_N_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    const uint64_t offset[PLE_N_HEADS] = {
        0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    for (int i = 0; i < PLE_N_HEADS; ++i) {
        c.vocab[i] = vocab[i];
        c.offset[i] = offset[i];
    }
    return c;
}

uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    // The first term is an ASSIGNMENT and the rest are XORed into it, which is how the source writes it
    // (`uint64_t mixed = ctx[0]*m[0]; for j=1.. mixed ^= ctx[j]*m[j];`).  Every product wraps mod 2^64,
    // which is what the `(uint64_t)` casts in the source make explicit.
    uint64_t mixed = (uint64_t) ctx[0] * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];
    return mixed;
}

void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out) {
    const int n_prev = NGRAM_SIZE - 1;
    for (int i = 0; i < n_tokens; ++i) {
        int64_t ctx[NGRAM_SIZE];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < NGRAM_SIZE; ++s) {
            // `prev` is OLDEST FIRST, so predecessor `s` positions back is entry (n_prev - s): s=1 reads the
            // NEWEST.  Reading index (s-1) instead walks the window backwards, which still produces indices
            // in range and so cannot be caught by a range check - only by an oracle.
            const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];
            // The cut is evaluated BEFORE the value is stored, so the position whose predecessor was EOS is
            // itself EOS.  Storing first and then cutting would leave position s holding the real token while
            // position s+1 became EOS - one token of history too much.
            cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
            ctx[s] = cut ? PLE_EOS_TOKEN_ID : t;
        }
        for (int n = 2; n <= NGRAM_SIZE; ++n) {
            const uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * HEADS_PER_NGRAM;
            for (int g = 0; g < HEADS_PER_NGRAM; ++g) {
                const int h = base + g;
                out[i * PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

namespace {
const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
}

int iq4nl_code(int code) { return kIq4Nl[code & 15]; }

void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d = f32_from_f16(dbits);
        const uint8_t* qs = blk + 2;
        // SPLIT HALVES: qs[j] holds elements j and j+16, not 2j and 2j+1.
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + j] = d * (float) kIq4Nl[qs[j] & 0x0F];
            out160[b * 32 + j + 16] = d * (float) kIq4Nl[qs[j] >> 4];
        }
    }
}

namespace {
struct Fp8Table {
    float v[256];
    Fp8Table() {
        for (int x = 0; x < 256; ++x) {
            const int s = x >> 7, e = (x >> 3) & 15, m = x & 7;
            float f;
            if (e == 15 && m == 7) f = 0.0f;                     // NaN in e4m3fn; never in a real table, read as 0
            else if (e == 0) f = std::ldexp((float) m / 8.0f, -6);
            else f = std::ldexp(1.0f + (float) m / 8.0f, e - 7);
            v[x] = s ? -f : f;
        }
    }
};
const Fp8Table kFp8;
}  // namespace

void fp8_e4m3_dequant_row(const uint8_t* row, float scale, float* out160) {
    for (int j = 0; j < PLE_HEAD_DIM; ++j) out160[j] = kFp8.v[row[j]] * scale;
}

void bf16_dequant_row(const uint8_t* row, float* out160) {
    // bfloat16 IS the top 16 bits of a float32, so widening is exact for every value, normal or not - no rounding,
    // no clamp, no scale. Bytes are assembled little-endian explicitly rather than reinterpreted, so this is the
    // same on a big-endian host.
    for (int j = 0; j < PLE_HEAD_DIM; ++j) {
        const uint32_t bits = (uint32_t) row[2 * j] | ((uint32_t) row[2 * j + 1] << 8);
        const uint32_t wide = bits << 16;
        float v;
        std::memcpy(&v, &wide, sizeof v);
        out160[j] = v;
    }
}

// NVIDIA's FP4-Local table: a row of 160 values is 80 bytes of E2M1 codes (element 2i in the low nibble, 2i+1 in the
// high one) followed by 10 E4M3 block scales (one per 16 values, sign bit always 0); one F32 scales the whole table.
// value = e2m1(code) * e4m3(block scale) * table scale, exactly what vLLM's nvfp4 PLE lookup computes.
void nvfp4_row_dequant(const uint8_t* row, float scale, float* out160) {
    static const float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int j = 0; j < PLE_HEAD_DIM; ++j) {
        const uint8_t byte = row[j >> 1];
        const uint8_t code = (j & 1) ? (uint8_t) (byte >> 4) : (uint8_t) (byte & 15);
        const float v = kE2M1[code & 7];
        const float s = kFp8.v[row[PLE_HEAD_DIM / 2 + (j >> 4)]];
        out160[j] = ((code & 8) ? -v : v) * s * scale;
    }
}

// ---------------------------------------------------------------------------------------------------
// THE FORMAT TABLE (see PleFormatInfo). One entry per table type; the block formats decode 5 blocks of 32.
namespace {
template <void (*Block)(const uint8_t*, float*), int BlockBytes>
void dequant_blocks(const uint8_t* row, float /*scale*/, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) Block(row + (size_t) b * BlockBytes, out160 + b * 32);
}
void dequant_iq4_nl(const uint8_t* row, float, float* out160) { iq4nl_dequant_row(row, out160); }
void dequant_fp8(const uint8_t* row, float scale, float* out160) { fp8_e4m3_dequant_row(row, scale, out160); }
void dequant_bf16(const uint8_t* row, float, float* out160) { bf16_dequant_row(row, out160); }
void dequant_nvfp4(const uint8_t* row, float scale, float* out160) { nvfp4_row_dequant(row, scale, out160); }

const PleFormatInfo kPleFormats[] = {
    {PleFormat::IQ4_NL, "IQ4_NL", "IQ4_NL", PLE_ROW_BYTES, false, dequant_iq4_nl},
    {PleFormat::Q5_0, "Q5_0", "Q5_0", (PLE_HEAD_DIM / 32) * 22, false, dequant_blocks<strata::dequantize_q5_0, 22>},
    {PleFormat::F8_E4M3, "F8_E4M3", "I8", PLE_ROW_BYTES_FP8, true, dequant_fp8},
    // Ordinary GGUFs ship the table in other types: Unsloth's UD-Q6_K_XL and Swift-1.5 Q4_K_L as Q8_0, a Q5_K_M finetune as Q5_1
    {PleFormat::Q5_1, "Q5_1", "Q5_1", PLE_ROW_BYTES_Q5_1, false, dequant_blocks<strata::dequantize_q5_1, 24>},
    {PleFormat::Q8_0, "Q8_0", "Q8_0", PLE_ROW_BYTES_Q8_0, false, dequant_blocks<strata::dequantize_q8_0, 34>},
    // Q4_0 (plain llama-quantize Q4_0 files, #599): 90-byte rows like IQ4_NL (18-byte blocks), a linear 4-bit grid
    {PleFormat::Q4_0, "Q4_0", "Q4_0", PLE_ROW_BYTES, false, dequant_blocks<strata::dequantize_q4_0, 18>},
    // BF16 (#586): the checkpoint's own table at full precision, 320-byte rows; self-describing, so no scale and no
    // metadata to trust (tools/ple_fp8_pack.py writes the FP8 form of the same table)
    {PleFormat::BF16, "BF16", "BF16", PLE_ROW_BYTES_BF16, false, dequant_bf16},
    // NVFP4 x16 (NVIDIA's Qwen3.8-Flash-Next-FP4-Local, tools/nvfp4_local_pack.py ple): 90-byte rows, I8 bytes marked
    // strata.ple.format = nvfp4x16 with strata.ple.scale. The tensor is [90, rows]: ne0 is the row's byte count.
    {PleFormat::NVFP4, "NVFP4", "I8", PLE_ROW_BYTES_NVFP4, true, dequant_nvfp4},
};
constexpr int kPleFormatCount = (int) (sizeof kPleFormats / sizeof kPleFormats[0]);
}  // namespace

const PleFormatInfo* ple_formats() { return kPleFormats; }
int ple_format_count() { return kPleFormatCount; }
const PleFormatInfo& ple_format_info(PleFormat f) { return kPleFormats[(int) f]; }
const PleFormatInfo* ple_format_for_type(const char* gguf_type_name) {
    for (const PleFormatInfo& f : kPleFormats)
        if (std::strcmp(f.gguf_type, gguf_type_name) == 0) return &f;
    return nullptr;
}
std::string ple_format_list() {
    std::string out;
    for (int i = 0; i < kPleFormatCount; ++i) {
        const PleFormatInfo& f = kPleFormats[i];
        if (i > 0) out += i + 1 == kPleFormatCount ? " or " : ", ";
        out += std::strcmp(f.gguf_type, "I8") == 0 ? (f.id == PleFormat::NVFP4 ? "NVFP4 (I8)" : "FP8 (I8)") : f.name;
    }
    return out;
}

static_assert(PLE_ROW_BYTES_MAX >= PLE_ROW_BYTES && PLE_ROW_BYTES_MAX >= PLE_ROW_BYTES_FP8, "row buffers too small");

struct PleTable::Impl {
    GgufFile* file = nullptr;
    const uint8_t* data = nullptr;
    uint64_t n_rows = 0;
    const PleFormatInfo* fmt = &ple_format_info(PleFormat::IQ4_NL);   // the table's format (#296: Q5_0 is one of them)
    mutable uint64_t bytes_read = 0;
    // Direct mode (plan v0.3 P2): the mapping above is released after the header parse and every row comes
    // from an unbuffered SSD read into `raw`.
    PleIo mode = PleIo::Mmap;
    strata::ngram::PleReader reader;
    strata::ngram::PleReader::Ticket ticket;
    bool pending = false;
    bool locked = false;
    // the table is shared by the prompt path and the decode windows; they can use it at once (one on a worker thread)
    std::recursive_mutex mu;
    uint32_t rows[PLE_N_HEADS] = {};
    uint8_t raw[PLE_N_HEADS * PLE_ROW_BYTES_MAX] = {};
    static constexpr size_t kMaxPrefetch = 16;
    size_t n_prefetch = 0;
    strata::ngram::PleReader::Ticket prefetch_tickets[kMaxPrefetch] = {};
    uint32_t prefetch_keys[kMaxPrefetch][PLE_N_HEADS] = {};
    uint8_t prefetch_raw[kMaxPrefetch][PLE_N_HEADS * PLE_ROW_BYTES_MAX] = {};
    float scale = 1.0f;               // the FP8 table's one scale
    uint32_t rb = PLE_ROW_BYTES;      // bytes per row
    void decode(const uint8_t* row, float* out160) const { fmt->dequant(row, scale, out160); }
    // STRATA_PLE_CHECK=1 (a diagnostic for #3602, off by default): the file and the table's offset, to re-read a row whose decoded
    // values are not plausible and compare the bytes the direct reader delivered with the file's own
    std::string path;
    uint64_t table_offset = 0;
    mutable int check_reports = 0;
    static bool check_on() {
        static const bool on = [] { const char* v = std::getenv("STRATA_PLE_CHECK"); return v != nullptr && std::atoi(v) != 0; }();
        return on;
    }
    /// One decoded row (`out160`, from `raw`): when a value is not finite or beyond 1e5 (a real PLE value is orders smaller; the
    /// largest an IQ4_NL row can decode to is 8.3e6), say which row, where, and whether the delivered bytes are the file's.
    void check_row(const char* where, uint32_t row, const uint8_t* raw, const float* out160) const {
        if (check_reports >= 8 || row >= n_rows) return;
        int bad = -1;
        for (int j = 0; j < PLE_HEAD_DIM; ++j)
            if (!std::isfinite(out160[j]) || std::fabs(out160[j]) > 1e5f) { bad = j; break; }
        if (bad < 0) return;
        ++check_reports;
        const uint64_t at = table_offset + (uint64_t) row * rb;
        std::vector<uint8_t> file(rb, 0);
        bool read = false;
        {
            std::ifstream f(path, std::ios::binary);
            if (f) {
                f.seekg((std::streamoff) at);
                f.read((char*) file.data(), (std::streamsize) rb);
                read = (uint32_t) f.gcount() == rb;
            }
        }
        std::fprintf(stderr, "strata ple check: %s: row %u (file offset %llu, page offset %llu): value %d of 160 decodes to %g", where,
                     row, (unsigned long long) at, (unsigned long long) (at % 4096), bad, (double) out160[bad]);
        if (!read) {
            std::fprintf(stderr, "; the file could not be re-read\n");
            return;
        }
        uint32_t first = rb;
        for (uint32_t b = 0; b < rb; ++b)
            if (raw[b] != file[b]) { first = b; break; }
        if (first == rb) {
            std::fprintf(stderr, "; the delivered bytes ARE the file's bytes (the table itself holds this value)\n");
        } else {
            std::fprintf(stderr, "; THE DELIVERED BYTES DIFFER FROM THE FILE from byte %u (delivered 0x%02x, file 0x%02x); first 8 delivered:", first,
                         raw[first], file[first]);
            for (int b = 0; b < 8 && (uint32_t) b < rb; ++b) std::fprintf(stderr, " %02x", raw[b]);
            std::fprintf(stderr, "; first 8 of the file:");
            for (int b = 0; b < 8 && (uint32_t) b < rb; ++b) std::fprintf(stderr, " %02x", file[b]);
            std::fprintf(stderr, "\n");
        }
    }
};

PleTable::PleTable() : impl_(new Impl) {}
PleTable::~PleTable() { close(); delete impl_; }

bool PleTable::open(const std::string& gguf_path, std::string& err) {
    return open(gguf_path, err, PleIoOptions{});
}

bool PleTable::open(const std::string& gguf_path, std::string& err, const PleIoOptions& io) {
    close();
    try {
        impl_->file = new GgufFile(gguf_path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    const TensorInfo* t = impl_->file->find("per_layer_token_embd.weight");
    if (t == nullptr) {
        err = "per_layer_token_embd.weight is not in " + gguf_path;
        close();
        return false;
    }
    // [160, 320001536]: ne0 = 160 is the FAST axis, so the ROW index is shape[1] and a row is contiguous. The NVFP4 table
    // is I8 bytes with ne0 = 90 (the row's byte count), marked strata.ple.format = nvfp4x16.
    const MetaValue* fmt_key = impl_->file->get("strata.ple.format");
    const bool nvfp4x16 = std::strcmp(t->type_name(), "I8") == 0 && fmt_key != nullptr && fmt_key->s == "nvfp4x16";
    const uint64_t want_ne0 = nvfp4x16 ? (uint64_t) PLE_ROW_BYTES_NVFP4 : (uint64_t) PLE_HEAD_DIM;
    if (t->shape.size() != 2 || t->shape[0] != want_ne0) {
        err = "per_layer_token_embd.weight has an unexpected shape";
        close();
        return false;
    }
    // The table's type is looked up in the format list (ple_formats()): IQ4_NL (ISTA-DASLab's shard 2, the original's
    // own GGUF), Q5_0 (#296: OrcaRouter's GGUF), or the FP8 table as shipped: I8 bytes marked strata.ple.format =
    // f8_e4m3 with strata.ple.scale (tools/ple_fp8_pack.py)
    const PleFormatInfo* fmt = nvfp4x16 ? &ple_format_info(PleFormat::NVFP4) : ple_format_for_type(t->type_name());
    if (fmt == nullptr) {
        err = std::string("per_layer_token_embd.weight is ") + t->type_name() + ", not " + ple_format_list();
        close();
        return false;
    }
    if (fmt->needs_scale) {
        const MetaValue* f = impl_->file->get("strata.ple.format");
        const MetaValue* s = impl_->file->get("strata.ple.scale");
        const char* want = nvfp4x16 ? "nvfp4x16" : "f8_e4m3";
        if (f == nullptr || f->s != want || s == nullptr || !(s->num() > 0.0)) {
            err = std::string("per_layer_token_embd.weight is I8 without strata.ple.format = ") + want + " and a positive strata.ple.scale";
            close();
            return false;
        }
        impl_->scale = (float) s->num();
    }
    impl_->fmt = fmt;
    impl_->rb = fmt->row_bytes;
    // PleReader's row_bytes has been a runtime parameter since the FP8 table (160 B rows) needed it; Q5_0's
    // 110 B rows go through the exact same generic path (ple_reader_test --selftest covers both 90 and 110 B
    // rows: straddling, caching, in-flight tickets, keep-alive). This refusal was stale.
    impl_->n_rows = t->shape[1];

    // THE CHECK THAT MAKES THE OFFSET FALSIFIABLE.  The manifest's `shard2_tensor.offset` is 0, but that is
    // the offset within the GGUF's DATA SECTION: the file's first 192 bytes are a header, and reading at 0
    // would decode the header plus 192 bytes of shifted rows - still plausible IQ4_NL, and wrong for every
    // row.  `GgufFile` parses the header, so `tensor_data` is already correct; this asserts the tensor
    // exactly fills the file from there, which is what makes the whole arrangement checkable rather than
    // assumed.  A wrong data offset would leave a different remainder.
    // A shard may hold other tensors too (Swift 1.5's shard 1 holds layers 0-12 and the table): the table must
    // then fit inside the file at its own offset; alone in its shard (the original's shard 2) it fills it exactly.
    // The size arithmetic is checked before it is used (#865): a header that claims enough rows to wrap the 64-bit
    // product, or a data section that starts past the end of the file, must be refused, not compared after wrapping.
    if (impl_->n_rows > std::numeric_limits<uint64_t>::max() / impl_->rb ||
        impl_->file->data_start() > impl_->file->file_size()) {
        err = "PLE table size overflow or invalid data offset in " + gguf_path;
        close();
        return false;
    }
    const uint64_t need = impl_->n_rows * (uint64_t) impl_->rb;
    const uint64_t have = impl_->file->file_size() - impl_->file->data_start();
    const bool alone = impl_->file->tensors().size() == 1;
    if (t->offset > have || need > have - t->offset || (alone && (t->offset != 0 || need != have))) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "PLE table size mismatch: %llu rows x %d B = %llu at offset %llu, but the file holds %llu from "
                      "data_start %llu",
                      (unsigned long long) impl_->n_rows, (int) impl_->rb, (unsigned long long) need,
                      (unsigned long long) t->offset, (unsigned long long) have,
                      (unsigned long long) impl_->file->data_start());
        err = buf;
        if (alone && have > need)   // #657: extra bytes after the table: a damaged or wrong file, not a format question
            err += "; the file is longer than its table - delete it and its .done mark and run setup again";
        close();
        return false;
    }
    impl_->data = impl_->file->tensor_data(*t);        // only a table that fits the file is mapped
    if (io.mode == PleIo::Direct) {
        // The parse above is the validated source of the offset; the mapping itself is not kept, so no page of
        // the table can enter this process's working set or the file cache through it.
        const uint64_t table_offset = impl_->file->data_start() + t->offset;
        const uint64_t n_rows = impl_->n_rows;
        impl_->path = gguf_path;
        impl_->table_offset = table_offset;
        delete impl_->file;
        impl_->file = nullptr;
        impl_->data = nullptr;
        if (!impl_->reader.open(gguf_path, table_offset, n_rows, io.max_inflight, io.cache_rows, err, io.io_thread,
                                impl_->rb)) {
            close();
            return false;
        }
        impl_->reader.set_keepalive(io.keepalive_ms, io.keepalive_window_s);
        impl_->n_rows = n_rows;
    }
    if (io.mode == PleIo::Mmap && io.lock && impl_->data != nullptr) {
#if defined(_WIN32)
        // Windows has no mlock: the platform helper raises the process's minimum working set and
        // VirtualLocks the mapped table instead (ordinary accounts hold the privilege). A failure
        // falls back to touching the pages, i.e. plain mmap behaviour, with a warning - same as POSIX.
        // A table that is more than half of the RAM leaves the rest of the process little (the locked pages cannot be
        // paged out): say so, and go on (recommend, never force).
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms) && need > ms.ullTotalPhys / 2)
            std::fprintf(stderr, "strata: warning: --ple-io ram locks %.1f GiB, more than half of this PC's %.1f GiB of RAM\n",
                         (double) need / (1ull << 30), (double) ms.ullTotalPhys / (1ull << 30));
        const strata::platform::LockResult lr =
            strata::platform::lock_resident((void*) impl_->data, need);
        if (lr.ok && lr.locked_bytes >= need) {
            impl_->locked = true;
        } else {
            // a partial lock is not a lock: the unlocked rest would fault on the token path, so give it back
            if (lr.locked_bytes > 0) strata::platform::unlock_resident((void*) impl_->data, lr.locked_bytes);
            std::fprintf(stderr, "strata: PLE table lock failed (%s): touching its pages instead\n",
                         lr.note.c_str());
            volatile uint8_t sink = 0;
            for (uint64_t off = 0; off < need; off += 4096) sink = sink + impl_->data[off];
            (void) sink;
        }
#else
        const uint64_t page = 4096;
        const uintptr_t a0 = (uintptr_t) impl_->data & ~(uintptr_t) (page - 1);
        const uintptr_t a1 = (uintptr_t) impl_->data + (uintptr_t) need;
        madvise((void*) a0, a1 - a0, MADV_WILLNEED);
        // Fault the table in with several threads first: mlock (and a single toucher) brings it in one page at a time
        // from one thread, ~0.5 GB/s from a cold file - 106 s for a 54 GB Q8_0 table.  The pages are then resident
        // and mlock only pins them.
        {
            constexpr uintptr_t kPiece = 64ull << 20;
            const uintptr_t pieces = (a1 - a0 + kPiece - 1) / kPiece;
            const unsigned threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
            std::atomic<uintptr_t> next{0};
            auto touch = [&] {
                volatile uint8_t sink = 0;
                for (uintptr_t i; (i = next.fetch_add(1)) < pieces;)
                    for (uintptr_t p = a0 + i * kPiece, e = std::min(a1, p + kPiece); p < e; p += page)
                        sink = sink + *(const volatile uint8_t*) p;
                (void) sink;
            };
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < threads; ++t) pool.emplace_back(touch);
            for (auto& t : pool) t.join();
        }
        if (mlock((const void*) a0, a1 - a0) == 0) {
            impl_->locked = true;
        } else {
            std::fprintf(stderr, "strata: PLE table mlock failed (%s; raise `ulimit -l`): its pages stay faulted in "
                                 "but may be reclaimed\n", std::strerror(errno));
        }
#endif
    }
    impl_->mode = io.mode;
    return true;
}

void PleTable::wait_prefetches() {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (impl_->n_prefetch == 0) return;
    std::string dummy;
    for (size_t i = 0; i < impl_->n_prefetch; ++i) (void) impl_->reader.collect(impl_->prefetch_tickets[i], dummy);
    impl_->n_prefetch = 0;
}

void PleTable::prefetch_rows(const uint32_t* rows16) {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (impl_->mode == PleIo::Direct && impl_->reader.is_open()) {
        for (size_t i = 0; i < impl_->n_prefetch; ++i)
            if (std::memcmp(impl_->prefetch_keys[i], rows16, sizeof(uint32_t) * PLE_N_HEADS) == 0) return;
        if (impl_->n_prefetch >= Impl::kMaxPrefetch) wait_prefetches();
        const size_t slot = impl_->n_prefetch++;
        std::memcpy(impl_->prefetch_keys[slot], rows16, sizeof(uint32_t) * PLE_N_HEADS);
        impl_->prefetch_tickets[slot] = impl_->reader.issue(impl_->prefetch_keys[slot], PLE_N_HEADS, impl_->prefetch_raw[slot]);
        return;
    }
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * impl_->rb);
            ranges[n].NumberOfBytes = impl_->rb;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
}

void PleTable::close() {
    wait_prefetches();
    impl_->reader.close();
    impl_->pending = false;
    impl_->locked = false;   // the unmap below releases the lock
    impl_->mode = PleIo::Mmap;
    delete impl_->file;
    impl_->file = nullptr;
    impl_->data = nullptr;
    impl_->n_rows = 0;
    impl_->rb = PLE_ROW_BYTES;
    impl_->fmt = &ple_format_info(PleFormat::IQ4_NL);
    impl_->scale = 1.0f;
    impl_->bytes_read = 0;
}

bool PleTable::is_open() const { return impl_->data != nullptr || impl_->reader.is_open(); }
bool PleTable::locked() const { return impl_->locked; }
bool PleTable::unlock() {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (!impl_->locked || impl_->data == nullptr) return false;
    strata::platform::unlock_resident((void*) impl_->data, impl_->n_rows * (uint64_t) impl_->rb);
    impl_->locked = false;
    return true;
}
const char* PleTable::format() const { return impl_->fmt->name; }
PleIo PleTable::mode() const { return impl_->mode; }
uint64_t PleTable::rows() const { return impl_->n_rows; }
uint64_t PleTable::table_bytes() const { return impl_->n_rows * (uint64_t) impl_->rb; }
uint64_t PleTable::bytes_read() const { return impl_->bytes_read; }

void PleTable::read_row(uint32_t row, float* out160) const {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (impl_->mode == PleIo::Direct && impl_->reader.is_open()) {
        uint8_t raw[PLE_ROW_BYTES_MAX];
        std::string err;
        const auto t = impl_->reader.issue(&row, 1, raw);
        if (!impl_->reader.collect(t, err)) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
            return;
        }
        impl_->decode(raw, out160);
        if (Impl::check_on()) impl_->check_row("read_row", row, raw, out160);
        impl_->bytes_read += impl_->rb;
        return;
    }
    if (impl_->data == nullptr || row >= impl_->n_rows) {
        std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
        return;
    }
    impl_->decode(impl_->data + (size_t) row * impl_->rb, out160);
    impl_->bytes_read += impl_->rb;
}

bool PleTable::issue(const uint32_t* rows16) {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    wait_prefetches();
    std::memcpy(impl_->rows, rows16, sizeof impl_->rows);
    if (impl_->mode == PleIo::Direct) {
        if (impl_->pending) return false;              // one token in flight per table
        impl_->ticket = impl_->reader.issue(impl_->rows, PLE_N_HEADS, impl_->raw);
        impl_->pending = true;
        return true;
    }
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * impl_->rb);
            ranges[n].NumberOfBytes = impl_->rb;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    impl_->pending = true;
    return true;
}

bool PleTable::collect(float* out2560, std::string& err) {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (!impl_->pending) { err = "PleTable::collect without issue"; return false; }
    impl_->pending = false;
    if (impl_->mode == PleIo::Direct) {
        if (!impl_->reader.collect(impl_->ticket, err)) return false;
        for (int h = 0; h < PLE_N_HEADS; ++h)
            impl_->decode(impl_->raw + (size_t) h * impl_->rb, out2560 + (size_t) h * PLE_HEAD_DIM);
        if (Impl::check_on())
            for (int h = 0; h < PLE_N_HEADS; ++h)
                impl_->check_row("collect", impl_->rows[h], impl_->raw + (size_t) h * impl_->rb, out2560 + (size_t) h * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) PLE_N_HEADS * impl_->rb;
        return true;
    }
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(impl_->rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
    return true;
}

bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err) {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (impl_->pending) { err = "PleTable::gather_batch while a token is in flight"; return false; }
    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
    if (impl_->mode == PleIo::Direct) {
        // Fast path: match each requested token to any in-flight/completed prefetch slot, then collect directly
        // without issuing a second PleReader ticket or copying through RowCache.
        if (n_tokens <= Impl::kMaxPrefetch) {
            size_t matched[Impl::kMaxPrefetch];
            bool exact_match = true;
            for (size_t t = 0; t < n_tokens; ++t) {
                size_t i = 0;
                while (i < impl_->n_prefetch &&
                       std::memcmp(impl_->prefetch_keys[i], rows + t * PLE_N_HEADS,
                                   sizeof(uint32_t) * PLE_N_HEADS) != 0) ++i;
                if (i == impl_->n_prefetch) { exact_match = false; break; }
                matched[t] = i;   // duplicate requests may share one immutable prefetch slot
            }
            if (exact_match) {
                bool collect_ok = true;
                for (size_t i = 0; i < impl_->n_prefetch; ++i) {
                    std::string ignored;
                    if (!impl_->reader.collect(impl_->prefetch_tickets[i], collect_ok ? err : ignored))
                        collect_ok = false;   // retain the first error, but drain every output buffer
                }
                impl_->n_prefetch = 0;
                if (!collect_ok) return false;
                for (size_t t = 0; t < n_tokens; ++t) {
                    for (int h = 0; h < PLE_N_HEADS; ++h) {
                        impl_->decode(impl_->prefetch_raw[matched[t]] + (size_t) h * impl_->rb,
                                      out + (t * PLE_N_HEADS + (size_t) h) * PLE_HEAD_DIM);
                        if (Impl::check_on())
                            impl_->check_row("gather_batch (prefetched)", rows[t * PLE_N_HEADS + (size_t) h],
                                             impl_->prefetch_raw[matched[t]] + (size_t) h * impl_->rb,
                                             out + (t * PLE_N_HEADS + (size_t) h) * PLE_HEAD_DIM);
                    }
                }
                impl_->bytes_read += (uint64_t) n * impl_->rb;
                return true;
            }
        }
        wait_prefetches();
        std::vector<uint8_t> raw(n * impl_->rb);
        const auto ticket = impl_->reader.issue(rows, n, raw.data());
        if (!impl_->reader.collect(ticket, err)) return false;
        for (size_t i = 0; i < n; ++i) {
            impl_->decode(raw.data() + i * impl_->rb, out + i * PLE_HEAD_DIM);
            if (Impl::check_on()) impl_->check_row("gather_batch", rows[i], raw.data() + i * impl_->rb, out + i * PLE_HEAD_DIM);
        }
        impl_->bytes_read += (uint64_t) n * impl_->rb;
        return true;
    }
    wait_prefetches();
    // #1984 #1971: a mapped table that is not in the file cache served this loop as serial demand faults, one random
    // 4 KiB page at a time (16 rows a token, 16 x the chunk's tokens for a prompt), and the first chunk of a prompt has
    // only layer 0 to hide it behind: a cold 27 GB table at ~0.2-1 ms a fault is minutes with no progress mark, which
    // is what the stall watchdog stops.  Ask the OS for the next window's pages in one go (PrefetchVirtualMemory /
    // MADV_WILLNEED, a hint that cannot change a value: an unfetched page is faulted by the read) while this window is read.
    constexpr size_t kWin = 4096;   // rows a window
    auto hint = [&](size_t a, size_t b) {
        if (impl_->data == nullptr || !g_ple_prefetch) return;
#if defined(_WIN32)
        WIN32_MEMORY_RANGE_ENTRY ranges[kWin];
        ULONG_PTR k = 0;
        for (size_t i = a; i < b; ++i) {
            if (rows[i] >= impl_->n_rows) continue;
            ranges[k].VirtualAddress = (PVOID) (impl_->data + (size_t) rows[i] * impl_->rb);
            ranges[k].NumberOfBytes = impl_->rb;
            ++k;
        }
        if (k > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), k, ranges, 0);
#else
        for (size_t i = a; i < b; ++i) {
            if (rows[i] >= impl_->n_rows) continue;
            const uintptr_t r0 = (uintptr_t) (impl_->data + (size_t) rows[i] * impl_->rb);
            const uintptr_t p0 = r0 & ~(uintptr_t) 4095;
            (void) madvise((void*) p0, (size_t) (r0 + impl_->rb - p0), MADV_WILLNEED);
        }
#endif
    };
    hint(0, std::min(n, kWin));
    for (size_t w = 0; w < n; w += kWin) {
        if (w + kWin < n) hint(w + kWin, std::min(n, w + 2 * kWin));
        const size_t e = std::min(n, w + kWin);
        for (size_t i = w; i < e; ++i) read_row(rows[i], out + i * PLE_HEAD_DIM);
    }
    return true;
}

bool PleTable::gather_raw_iq4nl(const uint32_t* rows, size_t n_tokens, uint8_t* out) const {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);
    if (impl_->data == nullptr || impl_->fmt->id != PleFormat::IQ4_NL || impl_->rb != PLE_ROW_BYTES) return false;
    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* o = out + i * PLE_ROW_BYTES;
        if (rows[i] >= impl_->n_rows) {
            // a row past the table reads as +0.0f (read_row): scale 0 and code 8 (value +1) in every nibble decode to +0.0f
            // on the device, where all-zero bytes would decode to -0.0f (scale 0 times code 0's value -127)
            for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
                uint8_t* blk = o + (size_t) b * 18;
                std::memset(blk, 0, 2);
                std::memset(blk + 2, 0x88, 16);
            }
        } else std::memcpy(o, impl_->data + (size_t) rows[i] * PLE_ROW_BYTES, PLE_ROW_BYTES);
    }
    impl_->bytes_read += (uint64_t) n * PLE_ROW_BYTES;
    return true;
}

void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_delay_us(us); }

std::string PleTable::io_report() const {
    if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
    const strata::ngram::ReaderStats s = impl_->reader.snapshot();
    char buf[400];
    int n = std::snprintf(buf, sizeof buf,
                  "ple io: %llu rows, %.1f%% row-cache hits, %llu SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
                  "blocked %.3f ms total (submit %.3f ms), cache %llu/%llu rows",
                  (unsigned long long) s.requests, s.requests ? 100.0 * (double) s.cache_hits / (double) s.requests : 0.0,
                  (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                  s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                  (unsigned long long) impl_->reader.cache_capacity());
    if (s.keepalive_reads > 0 && n > 0 && n < (int) sizeof buf)
        std::snprintf(buf + n, sizeof buf - (size_t) n, ", SSD kept awake by %llu reads (slowest %.1f ms)",
                      (unsigned long long) s.keepalive_reads, s.keepalive_us_max / 1000.0);
    return buf;
}

void PleTable::gather(const uint32_t* rows16, float* out2560) const {
    std::lock_guard<std::recursive_mutex> lk(impl_->mu);   // issue + collect as one
    if (impl_->mode == PleIo::Direct) {
        // `gather` stays const for its existing callers; the reader's state is the table's I/O state.
        PleTable* self = const_cast<PleTable*>(this);
        std::string err;
        if (!self->issue(rows16) || !self->collect(out2560, err)) {
            std::fprintf(stderr, "PleTable::gather: %s\n", err.empty() ? "a token is already in flight" : err.c_str());
            std::memset(out2560, 0, (size_t) NG_N_EMBD * sizeof(float));
        }
        return;
    }
    // ================================ SIXTEEN SERIAL PAGE FAULTS, MEASURED ================================
    //
    // **THIS COST 2.10-2.61 ms PER TOKEN AND HAD NEVER BEEN IN THE PLAN'S BUDGET AT ALL.**  The round-309
    // `token host phases` line put it second behind the layer loop among avoidable terms, and the arithmetic
    // says why: the table is 320,001,536 rows of `PLE_ROW_BYTES` = 90 B in a 26.8 GB mapping, so the sixteen
    // rows a token needs are 1,440 B - **0.5 MB/s**.  That is not bandwidth, it is latency: sixteen reads into
    // sixteen different 4 KB pages scattered across 26.8 GB, taken ONE AT A TIME, and on this machine the PLE
    // shard is 26.8 GB against 63 GB of RAM that the 31.6 GB expert arena is also competing for, so they are
    // not in the OS cache.  Sixteen serial NVMe reads at ~150 us is 2.4 ms, which is the measurement.
    //
    // `PrefetchVirtualMemory` issues all sixteen in ONE call and lets them complete in parallel.  It is a hint
    // and cannot change the answer - a range it does not fetch is simply faulted in by the read that follows -
    // so the only risk is that it does nothing.
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            // Out-of-range rows are handled by `read_row` as zeros and have no address to prefetch.
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * impl_->rb);
            ranges[n].NumberOfBytes = impl_->rb;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    // HEAD-SLOWEST, which is what `ggml_get_rows` does and what the source's own comment says: head h's 160
    // values occupy [h*160, (h+1)*160).  A head-fastest layout would put element (d, h) at d*16 + h and needs
    // a real transpose - a reshape of the same flat buffer compares equal and would make the check vacuous.
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(rows16[h], out2560 + (size_t) h * PLE_HEAD_DIM);
}

}  // namespace strata::kernels
