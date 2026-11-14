// Shared host code links these entry points. The Metal window owns latent execution.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_q3k.hpp"
#include "strata/prefill/lat_q3k_fused.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {

void copy_bytes_from_mapped(void*, const void*, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::copy_bytes_from_mapped");
}

void doorbell_publish_value(const float*, const int32_t*, const float*, int64_t, int64_t, float*, int32_t*,
    float*, uint32_t*, uint32_t, void*) {
    strata::metal_engine::missing("strata::kernels::doorbell_publish_value");
}

void iq_mmvq(int, const void*, const void*, float*, int, int, int, void*) {
    strata::metal_engine::missing("strata::kernels::iq_mmvq");
}

void lat_colscale_rows(float*, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("strata::kernels::lat_colscale_rows");
}

const float* lat_gu_colscale_dev(int, int) {
    strata::metal_engine::missing("strata::kernels::lat_gu_colscale_dev");
    return {};
}

void latent_plan_rows(int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t, int,
    int, int, void*) {
    strata::metal_engine::missing("strata::kernels::latent_plan_rows");
}

void latent_u_grouped(const uint8_t*, const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, int,
    int, float*, void*, int) {
    strata::metal_engine::missing("strata::kernels::latent_u_grouped");
}

void latent_u_matvec(const uint8_t*, const uint8_t*, const int32_t*, const int32_t*, const int32_t*,
    const int32_t*, int64_t, int64_t, int, int, int, float*, void*, int) {
    strata::metal_engine::missing("strata::kernels::latent_u_matvec");
}

void moe_hit_add_mapped(float*, const float*, const volatile float*, const int32_t*, const int32_t*, int64_t,
    int64_t, const volatile uint32_t*, uint32_t, void*) {
    strata::metal_engine::missing("strata::kernels::moe_hit_add_mapped");
}

void moe_hit_set(float*, const float*, const int32_t*, const int32_t*, int64_t, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::moe_hit_set");
}

void quantize_q8_0_rows(const float*, long long, long long, void*, void*) {
    strata::metal_engine::missing("strata::kernels::quantize_q8_0_rows");
}

}  // namespace strata::kernels

namespace strata::kernels::cpu {

void lat_q4_0_q8_1_rows(const uint8_t*, size_t, int, const void*const*, int, float*const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::lat_q4_0_q8_1_rows");
}

void lat_q8_0_q8_1_rows(const uint8_t*, size_t, int, const void*const*, int, float*const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::lat_q8_0_q8_1_rows");
}

void lat_quant_q8_1_qs_avx2(const float*, const float*, int8_t*) {
    strata::metal_engine::missing("strata::kernels::cpu::lat_quant_q8_1_qs_avx2");
}

}  // namespace strata::kernels::cpu

namespace strata::prefill::fused {

void experts_latent(const WideBatch&, const LatentGeom&, int, int64_t, const void*, const void*,
    const int32_t*, void*, float*, void*) {
    strata::metal_engine::missing("strata::prefill::fused::experts_latent");
}

void group_latent(const int32_t*, const int32_t*, int, void*, size_t) {
    strata::metal_engine::missing("strata::prefill::fused::group_latent");
}

void latent_dense_d(const void*, const void*, int64_t, int, int, int64_t, void*, void*, int) {
    strata::metal_engine::missing("strata::prefill::fused::latent_dense_d");
}

void latent_dense_u(const void*, const void*, const int32_t*, const int32_t*, int, int64_t, int64_t, int,
    int, float*, void*, int) {
    strata::metal_engine::missing("strata::prefill::fused::latent_dense_u");
}

void latent_dense_u_tables(const int32_t*, const int32_t*, int64_t, int, int32_t*, int32_t*, int32_t*,
    void*) {
    strata::metal_engine::missing("strata::prefill::fused::latent_dense_u_tables");
}

bool latent_fused_available() {
    strata::metal_engine::missing("strata::prefill::fused::latent_fused_available");
    return {};
}

size_t latent_group_bytes(int64_t, int) {
    strata::metal_engine::missing("strata::prefill::fused::latent_group_bytes");
    return {};
}

size_t latent_h_off(int64_t, int) {
    strata::metal_engine::missing("strata::prefill::fused::latent_h_off");
    return {};
}

size_t latent_scratch_bytes(int64_t, int, int64_t) {
    strata::metal_engine::missing("strata::prefill::fused::latent_scratch_bytes");
    return {};
}

}  // namespace strata::prefill::fused

namespace strata::prefill::fused_q3k {

void experts_latent_q3k(const fused::WideBatch&, const fused::LatentGeom&, int, int64_t, const void*,
    const void*, const int32_t*, void*, float*, void*) {
    strata::metal_engine::missing("strata::prefill::fused_q3k::experts_latent_q3k");
}

void latent_dense_d(const void*, const void*, int64_t, int, int, int64_t, void*, void*, int, const float*) {
    strata::metal_engine::missing("strata::prefill::fused_q3k::latent_dense_d");
}

}  // namespace strata::prefill::fused_q3k

namespace strata::prefill {

void lat_q3k_fused(const uint8_t*const*, const int32_t*, int, const void*, void*, float*, int64_t, void*,
    const int32_t*, const int32_t*, const int32_t*) {
    strata::metal_engine::missing("strata::prefill::lat_q3k_fused");
}

}  // namespace strata::prefill
