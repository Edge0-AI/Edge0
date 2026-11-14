// metal/src/engine/stubs_144.cpp - the 0.1.44 PC entry points the Metal engine does not implement (merge of the Mac branch
// with the PC release): new overloads / new functions that stubs.cpp (generated) does not know.  Each either stops the engine
// with its name (like stubs.cpp) or, where the caller has a documented fallback, answers "not available".  None is reached
// with the Metal defaults: STRATA_COMMIT_BATCH is off on Metal, the NVFP4 CPU rows need AVX2 / AVX-512, and the K-quant /
// NVFP4 prompt paths are CUDA / HIP only.

#include <cstddef>
#include <cstdint>

#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/kernels.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {
void iq_dequant_f16(int, const void*, int64_t, uint16_t*, void*, const float*) { strata::metal_engine::missing("strata::kernels::iq_dequant_f16(scale)"); }
void iq_dequant_gu_f16(int, const void*, const void*, int64_t, int64_t, uint16_t*, void*, const float*) { strata::metal_engine::missing("strata::kernels::iq_dequant_gu_f16(tail)"); }
bool gdn_commit_layers(float*, int64_t, int64_t, const float*, int64_t, const float*, int64_t, const float*, const float*, int64_t,
                       int, int, int, int, int, const int32_t*, void*) { return false; }   // "nothing launched: keep the per-layer loop"
void native_qsa_indexer_append_steps_layers(const float*, int64_t, const int32_t*, int, int, int32_t, float, const QsaIndexerLayers&, int,
                                            const QsaShapes&, int64_t, const RopeScaling&, void*) {
    strata::metal_engine::missing("strata::kernels::native_qsa_indexer_append_steps_layers");
}
}  // namespace strata::kernels

namespace strata::prefill {
void moe_combine(const float*, const int32_t*, const float*, const float*, const float*, float*, int64_t, void*, const float*) {
    strata::metal_engine::missing("strata::prefill::moe_combine(row_sd)");
}
}  // namespace strata::prefill

namespace strata::kernels::cpu {
bool nvfp4_256_fits(int) { return false; }
bool nvfp4_512_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_256_gu_rows");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_256_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_512_rows");
}
}  // namespace strata::kernels::cpu

// STRATA_QSA_SLOTS (PR #2364): the one-launch-over-the-rows QSA launchers have no Metal kernel; false = "run the per-row calls",
// which is what Verifier::record_window does on a decline.
namespace strata::kernels {
bool qsa_block_scores_slots(const float* const*, const float* const*, const float*, const int32_t*, int64_t, int64_t,
                            const QsaShapes&, float*, void*) { return false; }
bool qsa_decode_attn_slots(const float*, const QsaAttnSlotPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                           float*, float*, int64_t, void*) { return false; }
}  // namespace strata::kernels
