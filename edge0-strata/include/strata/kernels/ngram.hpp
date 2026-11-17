// include/strata/kernels/ngram.hpp - P2.S4: the PLE n-gram hash and the per-layer token table.
//
// Two halves that share nothing but the row indices they exchange:
//
//   * `ngram_rows` - the HOST hash.  Sixteen row indices per token from the last three token ids, computed
//     with 64-bit multiply/xor and no tensor op at all.  `qwen4exp.cpp` L1092-1124, where the source says
//     outright that it is host-side because "ggml has no int64 and no xor".
//   * `PleTable`  - the row gather.  `per_layer_token_embd.weight` is IQ4_NL and is NOT IN THE PACK: it is
//     51.2e9 elements (28.8 GB) in the ORIGINAL second GGUF shard, and it is the only tensor this engine
//     reads from the GGUF rather than from the canonical pack.
//
// THE HASH IS THE ONE PLACE IN THIS MODEL WHERE A WRONG BIT IS SILENTLY PLAUSIBLE.  Every one of its
// properties has a rival reading that produces a valid index in range:
//
//   * XOR, not sum.  Two equal tokens with equal multipliers cancel to zero under XOR and double under sum.
//   * `%` vocab_size, not `& (vocab_size-1)`.  The vocab sizes are 20000003, 20000023, 20000033, ... - not
//     powers of two, so a mask would be a different function.
//   * the EOS cut propagates FORWARD: an EOS at any position replaces that position AND every older one.
//   * a missing predecessor is `LLAMA_TOKEN_NULL = -1`, tested as `t < 0`.  Token id 0 is a REAL token.
//   * the token's OWN eos does not cut its own context - only the predecessors are examined.
//
// All of those are asserted observable in `ple_parity.cpp` rather than trusted.
#pragma once

#include <cstdint>
#include <string>

namespace strata::kernels {

// ---- geometry, every value confirmed against the artifact (docs/gguf-metadata.txt) ----
inline constexpr int NG_N_EMBD = 2560;          // n_embd
inline constexpr int PLE_HEAD_DIM = 160;        // qwen4exp.embedding_length_per_layer_input
inline constexpr int NGRAM_SIZE = 3;            // qwen4exp.ple.ngram_size
inline constexpr int HEADS_PER_NGRAM = 8;       // qwen4exp.ple.heads_per_ngram
inline constexpr int PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM;   // 16
inline constexpr int PLE_CONV_KERNEL = 4;       // qwen4exp.ple.conv_kernel
inline constexpr int32_t PLE_EOS_TOKEN_ID = 248044;
inline constexpr int NG_HC = 4;                 // hyper_connection.count
inline constexpr int NG_HC_DIM = NG_HC * NG_N_EMBD;                      // 10240
inline constexpr int NG_HIST = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE;       // 9
inline constexpr int32_t TOKEN_NULL = -1;       // LLAMA_TOKEN_NULL
inline constexpr float NG_RMS_EPS = 1e-6f;

// The table: [160, 320001536] IQ4_NL.  ne0 = 160 is the FAST axis, so one row is 160 contiguous elements =
// 5 blocks of 32 at 18 bytes = 90 bytes.  The head-slowest flatten then makes 16 rows exactly n_embd = 2560.
inline constexpr uint64_t PLE_TABLE_ROWS = 320001536ull;
inline constexpr int PLE_ROW_BYTES = (PLE_HEAD_DIM / 32) * 18;           // 90: an IQ4_NL row
inline constexpr int PLE_ROW_BYTES_FP8 = PLE_HEAD_DIM;                   // 160: an F8_E4M3 row, one byte a value
inline constexpr int PLE_ROW_BYTES_Q5_1 = (PLE_HEAD_DIM / 32) * 24;     // 120: a Q5_1 row (a Q5_K_M finetune)
inline constexpr int PLE_ROW_BYTES_Q8_0 = (PLE_HEAD_DIM / 32) * 34;     // 170: a Q8_0 row (UD-Q6_K_XL, Swift-1.5 Q4_K_L)
inline constexpr int PLE_ROW_BYTES_BF16 = PLE_HEAD_DIM * 2;             // 320: a BF16 row, two bytes a value
inline constexpr int PLE_ROW_BYTES_NVFP4 = PLE_HEAD_DIM / 2 + PLE_HEAD_DIM / 16;   // 90: 80 B of E2M1 codes + 10 E4M3 block scales
inline constexpr int PLE_ROW_BYTES_MAX = PLE_ROW_BYTES_BF16;             // the widest row any format in `ple_formats()` has

/// THE FORMATS OF THE TABLE, ONE ROW EACH. A table type is one entry here (its GGUF type name, the bytes of one
/// 160-value row, how to turn a row into floats) and nothing else: `PleTable::open`, `format()`, the reader's row size,
/// the error string and the buffers' size all read this list, so a new format cannot be in one switch and missing from
/// another. A row is always `PLE_HEAD_DIM` = 160 values = 5 blocks of 32 for the block formats.
enum class PleFormat : uint8_t { IQ4_NL, Q5_0, F8_E4M3, Q5_1, Q8_0, Q4_0, BF16, NVFP4 };

struct PleFormatInfo {
    PleFormat id;
    const char* name;          ///< what `PleTable::format()` returns
    const char* gguf_type;     ///< `TensorInfo::type_name()` of the table tensor ("I8" for F8_E4M3: GGUF has no FP8 type)
    uint32_t row_bytes;        ///< bytes of one 160-value row in the file
    bool needs_scale;          ///< the table has one scale in the GGUF metadata (`strata.ple.scale`): F8_E4M3 and NVFP4
    void (*dequant)(const uint8_t* row, float scale, float* out160);
};

/// Every format, in the order of `PleFormat`; `ple_format_count()` entries.
const PleFormatInfo* ple_formats();
int ple_format_count();
const PleFormatInfo& ple_format_info(PleFormat f);
/// The format whose table tensor has this GGUF type name, or nullptr. F8_E4M3's name is "I8"; the caller checks the
/// `strata.ple.format` marker and the scale.
const PleFormatInfo* ple_format_for_type(const char* gguf_type_name);
/// "IQ4_NL, Q5_0 or FP8 (I8)": the formats for an error message, built from the list.
std::string ple_format_list();

/// The artifact's own hash constants, transcribed from `docs/gguf-dump-shard1.txt`:
///
///     layer_multipliers = [23703573157769, 20109073645365, 8052911324071]
///     head_vocab_sizes  = [20000003, 20000023, ...]          16 entries, NOT an arithmetic sequence
///     head_offsets      = [0, 20000003, 40000026, ...]       the running sum of head_vocab_sizes
///
/// `ref/ngram.py::selftest` calls its own copies "synthetic but STRUCTURALLY REAL"; they are in fact
/// `20000003 + 10*i` for the vocab sizes, which the artifact's are not (20000003, 20000023, 20000033 - the
/// differences are 20, 10, ...).  A selftest is free to use whatever constants it likes, but a PARITY test
/// against the artifact must use these, and the difference is recorded because it is exactly the kind of
/// thing that gets copied from the reference into an engine.
struct PleConsts {
    uint64_t mult[NGRAM_SIZE];
    uint64_t vocab[PLE_N_HEADS];
    uint64_t offset[PLE_N_HEADS];
};
PleConsts ple_artifact_consts();

/// `mixed_n = (ctx[0]*m[0]) ^ (ctx[1]*m[1]) ^ ... ^ (ctx[n-1]*m[n-1])` in uint64.
///
/// Exposed separately so a test can assert the XOR STRUCTURE directly rather than only its consequences.
uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n);

/// The sixteen row indices per token.  `qwen4exp.cpp` L1092-1124.
///
///   tokens  (n_tokens,)      the ubatch's own tokens
///   prev    (n_tokens, 2)    predecessors OLDEST FIRST, `TOKEN_NULL` where there is none
///   out     (n_tokens, 16)   row indices into the PLE table
///
/// The window is built once per token and then only its PREFIXES are used, which is what makes the bigram
/// heads (0..7) and the trigram heads (8..15) agree about the recent history.
void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out);

/// The IQ4_NL codebook: `{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}`,
/// which is `kvalues_iq4nl` from `ggml-common.h` L1121 and also the manifest's `codebooks.IQ4NL`.
///
/// There is NO `-8` in this version of ggml.  Some releases of `dequantize_row_iq4_nl` subtract 8 from the
/// codebook value; at 3cf03257 they do not, and the difference would shift every one of the 51.2e9 elements.
int iq4nl_code(int code);

/// One 90-byte row -> 160 floats.  The canonical rule `value = cb[code] * scale + offset` with offset 0,
/// because IQ4_NL has no min term (`has_offset` is false in the manifest).
///
/// THE NIBBLE ORDER IS SPLIT-HALF, NOT INTERLEAVED, and that is the trap here.  `dequantize_row_iq4_nl` is
///     y[j]           = d * kvalues[qs[j] & 0xf];
///     y[j + QK4_NL/2] = d * kvalues[qs[j] >> 4];
/// so byte j carries elements j and j+16 - NOT 2j and 2j+1, which is the natural reading and which produces
/// a perfectly plausible embedding of the wrong 160 values.
void iq4nl_dequant_row(const uint8_t* row, float* out160);

/// One FP8 row -> 160 floats: each byte an E4M3 value (the "fn" variant: no infinities, 0x7F/0xFF are NaN), times the
/// table's one scale. This is the table as Qwen3.8-Flash-Next ships it (`...ngram_embedding.shard_k`, F8_E4M3, and
/// `weight_scale`), kept byte for byte by tools/ple_fp8_pack.py; IQ4_NL is 8% off it per row.
void fp8_e4m3_dequant_row(const uint8_t* row, float scale, float* out160);

/// One BF16 row -> 160 floats, EXACTLY: bfloat16 is the top half of a float32, so this is a shift and a copy with no
/// rounding anywhere. It is the other published form of this same table - the full checkpoint stores
/// `...ngram_embedding.shard_k` as BF16 (2 B a value, 102.4 GB), the source of record rather than a re-quantization.
/// There is no scale, because BF16 values are the values.
void bf16_dequant_row(const uint8_t* row, float* out160);

/// How the table's rows are read (plan v0.3 P2). `Direct` is the default: unbuffered 4 KiB reads from the SSD,
/// so the table never occupies RAM or the OS file cache. `Mmap` is the earlier memory-mapped path, kept as the
/// A/B arm; it returns the same bytes.
///
/// NEVER KEEP THE SHARD MAPPED WHILE READING IT DIRECT: a live section on the same file serializes the unbuffered
/// reads (311 -> 1,575 us per token, bench/results/2026-09-23-p2-ssd-direct). Direct mode drops its own mapping
/// after the header parse; nothing else in the process may hold one.
enum class PleIo { Direct, Mmap };

struct PleIoOptions {
    PleIo mode = PleIo::Direct;
    uint32_t max_inflight = 256;     ///< outstanding SSD reads (decode needs 16; a prompt chunk saturates the SSD at 256)
    uint64_t cache_rows = 1u << 20;  ///< bounded row cache, in ROWS: 1,048,576 x 90 B ~ 95 MB (IQ4_NL), ~160 MB (FP8),
                                     ///< ~335 MB (BF16); 0 disables
    bool io_thread = true;           ///< reads submitted by a worker thread, not the caller
    /// Mmap mode only (`--ple-io ram`): lock the whole mapped table in RAM at open, so no SSD read ever sits on
    /// the prompt or token path. Needs RAM for the full table (28.8 GB IQ4_NL, 51.2 GB FP8). mlock on POSIX,
    /// the process working-set minimum plus VirtualLock on Windows; `locked()` reports the outcome.
    bool lock = false;
    /// Direct mode with the I/O worker only: keep the SSD awake while rows are asked for - one page of the table
    /// after this long without a read (0 = off), until `keepalive_window_s` after the last request for rows
    /// (see PleReader::set_keepalive).
    double keepalive_ms = 0;
    double keepalive_window_s = 60;
};

/// The PLE table.  Held by pointer-to-impl so this header does not drag `<windows.h>` into every
/// translation unit that wants the hash.
class PleTable {
public:
    PleTable();
    ~PleTable();
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;

    /// Open with an explicit I/O mode. The two-argument `open` below is the default (Direct).
    bool open(const std::string& gguf_path, std::string& err, const PleIoOptions& io);

    /// The split the plan asks for: `issue` as soon as the token id is known, `collect` just before layer 1
    /// needs the rows. `gather` is `issue` followed by `collect`. In Mmap mode `issue` only prefetches.
    bool issue(const uint32_t* rows16);
    bool collect(float* out2560, std::string& err);
    /// Non-blocking prefetch of 16 rows for one token into the reader's row cache while the GPU is busy.
    void prefetch_rows(const uint32_t* rows16);
    void wait_prefetches();
    /// Plan v0.3 P5: the rows of `n_tokens` tokens (16 each, `rows` token-major) into `out` (2560 floats per token),
    /// as ONE reader request - page dedupe and sort across the whole batch, the reader's full queue depth.  Not
    /// while a single-token `issue` is pending.  The mapped mode gathers row by row.
    bool gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err);
    /// STRATA_PLE_COMPACT: the same rows as their raw IQ4_NL bytes (n_tokens x 16 rows of PLE_ROW_BYTES; a row past the
    /// table: scale 0 and code 8 everywhere, which decode to the +0.0f gather_batch writes) for ple_iq4nl_dequant on the device.  False (nothing
    /// written) unless the table is in memory (--ple-io ram / mmap) and IQ4_NL.
    bool gather_raw_iq4nl(const uint32_t* rows, size_t n_tokens, uint8_t* out) const;

    /// Fault injection (Direct mode): every row read completes no earlier than `us` after issue.
    void set_injected_delay_us(double us);

    /// Maps the ORIGINAL second GGUF shard read-only.  The tensor's data does NOT start at file offset 0:
    /// the manifest's `shard2_tensor.offset` is relative to the GGUF's DATA SECTION, and the header before it
    /// is 192 bytes.  Reading at 0 would silently decode the header plus 192 bytes of shifted rows - still
    /// plausible IQ4_NL, which is why the reader takes the offset from a real GGUF parse (`GgufFile`, the
    /// project's validated reader) and checks that the tensor exactly fills the rest of the file.
    bool open(const std::string& gguf_path, std::string& err);
    void close();
    bool is_open() const;
    /// True when `PleIoOptions::lock` was asked for and the OS lock succeeded (false: pages only pre-touched).
    bool locked() const;
    /// Gives a resident lock back (Mmap mode): the pages stay mapped and cached, but the OS may reclaim them again, so the RAM counts as
    /// available.  False when the table was not locked.  Used by the residency policy when the table's lock would leave the
    /// conversation cache no room (#4518).
    bool unlock();
    uint64_t rows() const;
    /// Bytes of the whole table (rows x row bytes) while open; 0 when closed.  The residency policy in
    /// generate.cpp sizes the table with this before it decides to keep it in RAM.
    uint64_t table_bytes() const;
    /// "IQ4_NL" or "F8_E4M3" (a GGUF from tools/ple_fp8_pack.py: type I8, strata.ple.format = f8_e4m3).
    const char* format() const;

    /// 16 row indices -> 2560 floats.  The gathered rows are flattened HEAD-SLOWEST: row h's 160 values
    /// occupy `out[h*160, (h+1)*160)`, which is what `ggml_get_rows` does and what makes the result a plain
    /// n_embd-wide vector for the block that consumes it.
    void gather(const uint32_t* rows16, float* out2560) const;

    /// One row -> 160 floats, for diagnostics and for the parity test's oracle comparison.
    void read_row(uint32_t row, float* out160) const;

    /// Bytes actually touched since `open`, for the "how often is it not ready" measurement P2.S4 asks for.
    uint64_t bytes_read() const;

    /// One line of reader statistics (Direct mode), e.g. for --stats. Empty in Mmap mode.
    std::string io_report() const;
    PleIo mode() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// ================================ THE PLE GATHER'S PREFETCH, AND ITS A/B ARM ================================
//
// `gather` issues its sixteen row reads as sixteen SEPARATE page faults into a 26.8 GB mapping, which measured
// **2.10-2.61 ms per token** and is second only to the layer loop among the token's avoidable terms. The table
// is 90 B per row, so the sixteen rows are 1,440 B - 0.5 MB/s, which is latency and not bandwidth, and it is
// because the sixteen pages are taken one at a time. `PrefetchVirtualMemory` issues them in one call.
//
// **THIS SWITCH EXISTS SO THE CLAIM CAN BE MEASURED RATHER THAN ASSERTED.** Cross-build comparisons in this
// project have repeatedly turned out to be machine drift - `LAYERS` alone moves 48.9 -> 52.3 between runs of
// one binary - so the two arms are alternated inside one session. Off is `--no-ple-prefetch`.
void ple_prefetch_enable(bool on);
bool ple_prefetch_enabled();

// ---- NOTE ON `madvise(MADV_RANDOM)` ------------------------------------------------------------------
// P2.S4 asks for `madvise(MADV_RANDOM)` on the mapping, and on Linux that is exactly right: a token gathers
// 16 rows scattered over 28.8 GB, so read-ahead is pure waste and would evict useful pages.
//
// THERE IS NO LINUX, SO THERE IS NO MADV_RANDOM HERE.  Windows has no equivalent of the advice - the closest
// is `FILE_FLAG_RANDOM_ACCESS` at CreateFile time, which suppresses the cache-manager's read-ahead for the
// whole handle, and `PrefetchVirtualMemory` for the positive case.  `GgufFile` opens with neither, so this
// build gets the DEFAULT sequential read-ahead on a random access pattern.  Saying so is the point: a
// comment claiming MADV_RANDOM here would be a comment about code that does not run.  On Linux the advice
// belongs in `GgufFile::open`, which this module does not own.
// ------------------------------------------------------------------------------------------------------

}  // namespace strata::kernels
