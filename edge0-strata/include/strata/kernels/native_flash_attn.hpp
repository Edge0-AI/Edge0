#pragma once
#include "strata/kernels/qsa.hpp"

namespace strata::kernels {

enum NativeFlashAttnStatus : int32_t {
    kNativeFlashAttnSuccess = 0,
    kNativeFlashAttnUnsupportedStep = 1,
};

// Explicitly selected short-context adapter for pinned CUDA F16/F16 vector
// FlashAttention. Geometry is exactly Q24x256, KV2x256, one query/sequence,
// scale=1/16, no ALiBi/softcap/sink. No allocation, synchronization, or KV rewrite.
// q/output: contiguous [24,256] F32. k/v: existing gathered [capacity,2,256]
// F16 buffers, with only step[kStepWidth] initialized rows required. Optional
// mask: 256 F16 additive logits, broadcast over heads; at least one valid key
// must be unmasked. Finite live q/k/v and finite or -inf live mask entries are
// preconditions; unused k/v/mask padding may contain arbitrary bytes.
// Padding is synthesized as zero K/V and -inf mask without reading it.
//
// capacity >=256; max_context in [1,256] is the HOST session limit. The device
// step must satisfy pos+1==n_kv==width in [1,max_context] and n_bid==n_kv/4.
// An invalid step writes status=UnsupportedStep and NaN output, without reading
// q/k/v/mask. status is one caller-owned device int32, overwritten on every call;
// integration MUST inspect it at its normal completion boundary. Host context
// limits must prevent unsupported steps before decoding. This is not a 20K path.
//
// All buffers (including optional mask and status) must have disjoint spans,
// with four-byte alignment for F32/step/status and two-byte alignment for F16.
// A nonnull explicit stream is mandatory. All pointers remain live for replay.
void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                                  const int32_t* step, int64_t capacity, int max_context,
                                  const QsaShapes& shapes, float* output, int32_t* status,
                                  const uint16_t* mask, void* stream);

} // namespace strata::kernels
