// include/strata/core/weights.hpp - the dense weights, loaded into VRAM in ENGINE form.
//
// The pack is the ARTIFACT's business; this is the ENGINE's.  The differences are deliberate, and they run in
// BOTH directions:
//
//   * SHRINKING.  A tensor whose source type is BF16 is stored in the pack promoted to 32 bits, because that is
//     lossless and it keeps the canonical decoder simple, but a running engine re-rounds it to 16 - which costs
//     nothing at all, because the value CAME from 16 bits.  Per `tools/pack_budget.py` that saving is the
//     difference between the dense weights fitting in VRAM beside the expert cache and not fitting at all.
//
//   * GROWING.  90 of the 303 quantized tensors carry fp16 SCALES in the pack (all 58 Q2_0 expert tensors, plus
//     Q4_0, Q5_0, Q8_0 and IQ4_NL) because that is what the source block holds.  Every kernel here takes
//     `const float* scales`, and fp16 is a subset of f32, so this loader WIDENS them - exactly, changing no
//     value - for 11,407,360 B.  See `docs/pack-format.md` Â§2.4; assuming 4 bytes per scale instead of reading
//     `scales_fp16` is what made the plane arithmetic disagree with the pack for a whole round.
//
//     pack      5.305 GiB dense + 0.444 GiB embd
//     engine    4.067 GiB dense + 0.444 GiB embd + 10.88 MiB widened scales = 4.522 GiB
//
// WHAT THE LOADER DOES NOT DO: it does not know what any tensor MEANS.  It reads a flat index
// (`<pack>/index.txt`, written by `tools/pack_index.py`), copies each tensor into one arena at a recorded
// offset - as a list of PLANE SEGMENTS, because a scale plane can change width and the rest cannot - and hands
// back a name -> WeightRef map.  Geometry, forms and semantics belong to the caller.
//
// WHY AN INDEX FILE AND NOT A JSON PARSER: a JSON parser in the engine would be a new, unaudited component
// whose failure mode is a wrong byte offset - which decodes to a plausible weight and produces plausible
// logits.  Python wrote the manifest and already parses it correctly.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// What the loader had to DO for a tensor, decided by `tools/pack_index.py` and written into the index
/// rather than inferred here.  The four cases are distinguishable in the manifest only by combining
/// `source_type` with `values_fp16`, and getting that combination wrong yields a plausible tensor.
enum class WeightKind : int {
    Verbatim = 0,   ///< quantized planes, or F16 already stored as 2 bytes: copy straight through
    Bf16InF32 = 1,  ///< the pack holds the bf16 value promoted to f32: take the HIGH 16 BITS (exact)
    F32 = 2,        ///< copy 4 bytes per element
    F16InF32 = 3,   ///< an f16 value promoted to f32: a real f32->f16 conversion, not a truncation
};

struct WeightRef {
    const void* data = nullptr;   ///< device pointer inside the arena
    uint64_t bytes = 0;
    int64_t ne0 = 0;              ///< the manifest's own shape, ne0 the CONTIGUOUS axis
    int64_t ne1 = 0;
    int64_t elements = 0;
    WeightKind kind = WeightKind::Verbatim;
    /// The canonical-form attributes, valid only for a quantized VERBATIM tensor (`code_bits != 0`).
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;

    /// THE THREE PLANE SIZES, IN ENGINE FORM, in the order they sit inside `data`.  Recorded by the loader
    /// from what the index TOLD it rather than re-derived by each caller: the one time this layout was
    /// re-derived independently it assumed 4-byte scales for tensors whose pack scales are fp16, and the
    /// only reason that was not a silent wrong answer is that the re-derivation checked itself and refused.
    /// Two places computing one layout is two places to disagree.
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;    ///< always 4 B per scale in the arena, whatever the pack held
    uint64_t offset_bytes = 0;    ///< 0 when the form has no offset plane

    /// WHERE THIS TENSOR CAME FROM, so a caller can check the loader's work against the pack's own bytes
    /// instead of against the loader's opinion of them.  Without `src_off` a test can only compare two
    /// derived numbers; with it, the widening of an fp16 scale plane can be verified value by value against
    /// `<file>.bin` - and the pack files are named, so `file_id` says which one to open.
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    int file_id = 0;
    bool scales_fp16 = false;     ///< the PACK's width, before the loader widened it

    /// WHICH ACTIVATION THIS WEIGHT WANTS - 0 for Q8_0, 1 for Q8_K.  Decided by 	ools/pack_index.py from
    /// source_type and carried here, because **the S-form CANNOT express it**: Q5_0 and Q5_K are both 8-bit
    /// with bias -16, and IQ4_NL and IQ4_XS differ in nothing the S-form holds.  docs/activation-contract.md.
    int act_kind = 0;
    bool wants_q8k() const { return act_kind == 1; }

    bool quantized() const { return code_bits != 0; }

    // Optional native GGUF projection, owned by NativeDense. All references in
    // one table share its scratch and must execute on one ordered session stream.
    const void* native_data = nullptr;
    void* native_q8_1 = nullptr;
    int native_type = -1;
    /// S23 experiment (STRATA_HC_Q8=1): a hyper-connection projection's Q8_0 bytes as the GGUF stores them, owned by
    /// NativeDense, for the verify window's read (every other path keeps the pack's BF16 `data`)
    const void* hc_q8 = nullptr;
    /// Plan v0.3 P1: false when the loader SKIPPED this tensor's canonical bytes because another form serves it
    /// (native GGUF projections, the native head).  The metadata above stays valid; `data` is null.
    bool resident = true;
    /// STRATA_HC_Q8 (experiment): non-null when `data` holds int8 codes [ne1][ne0] of a BF16 hyper-connection
    /// projection quantized Q8_0-style (one fp32 scale per 32 contiguous values, `q8_scale[ne1][ne0 / 32]`).
    const float* q8_scale = nullptr;
};

struct LoadReport {
    size_t tensors = 0;
    uint64_t arena_bytes = 0;
    size_t re_rounded = 0;        ///< tensors written as 16 bits out of a 32-bit container
    uint64_t bytes_saved = 0;     ///< what those conversions saved against the pack
    double read_ms = 0.0;
    double upload_ms = 0.0;
};

class WeightTable {
public:
    /// The arena size the index asks for, readable WITHOUT loading anything - so a caller can size its
    /// `DeviceArena` before committing to the load, and a plan that cannot fit is refused at startup rather
    /// than halfway through a 5 GB upload.
    /// With `skip`, the size of the compacted arena that holds every tensor EXCEPT the named ones.
    static bool pool_bytes(const std::string& pack_dir, uint64_t& out, std::string& err,
                           const std::set<std::string>* skip = nullptr);

    /// The `code_bits` field of one row of `<pack_dir>/index.txt`, readable WITHOUT loading anything (0 = the
    /// pack stores the tensor unquantized, e.g. a --compat-bf16 key; -1 = no such row).  #326: the loader has
    /// to know this before it builds the skip set.
    static bool index_code_bits(const std::string& pack_dir, const std::string& name, int& code_bits,
                                std::string& err);

    /// Load every tensor in `<pack_dir>/index.txt` into `arena_base`.
    ///
    /// `arena_bytes` MUST be at least `pool_bytes`; the loader checks rather than trusting the caller,
    /// because the failure mode otherwise is a device write past the end of the arena.
    /// With `skip`, the named tensors are not read: their rows keep their metadata with `data == nullptr` and
    /// `resident == false`, and the other tensors are packed into the compacted arena `pool_bytes` sized.
    bool load(const std::string& pack_dir, void* arena_base, uint64_t arena_bytes, std::string& err,
              const std::set<std::string>* skip = nullptr);

    /// A SKIPPED bf16 tensor's values read back from the pack file (the promoted-f32 and the raw 16-bit forms), for
    /// the host-side Q8_0 quantizer of STRATA_HC_Q8 (the tensor then has no copy in the arena).
    bool read_bf16_host(const std::string& name, std::vector<uint16_t>& out, std::string& err) const;

    const WeightRef* find(const std::string& name) const;
    /// STRATA_HC_Q8: the hyper-connection projections (`blk.*.hc_{attn,ffn}_{down,up}.weight`) the loader SKIPPED
    /// are read from the pack, quantized to int8 + one fp32 scale per 32 values, and uploaded to their own device
    /// allocation (`q8_scale` set).  Returns the bytes uploaded in `bytes`.
    bool load_hc_q8(const std::string& pack_dir, uint64_t& bytes, std::string& err);
    /// The tensor names `load_hc_q8` converts, for the loader's skip set.
    static void hc_q8_names(int64_t n_layers, std::set<std::string>& out);
    const std::map<std::string, WeightRef>& all() const { return table_; }
    const LoadReport& report() const { return report_; }

private:
    friend class NativeDense;
    std::map<std::string, WeightRef> table_;
    LoadReport report_;
    std::string pack_dir_;
};

}  // namespace strata::core
