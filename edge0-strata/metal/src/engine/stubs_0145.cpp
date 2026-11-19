// metal/src/engine/stubs_0145.cpp - the kernel-library entry points the 0.1.45 sources added and Metal does not implement yet.
// Same job and shape as stubs.cpp (each stops the engine with its name, metal/src/engine/missing.cpp). Written from the linker's
// undefined-symbol report because tools/gen_stubs.py reads an older clang AST dump format than macOS 27's clang prints.
// A symbol that gets a Metal implementation is deleted from here. fused_gr_persist_init / fused_gr_read_persist / f16_range_read /
// gemm_f16_range_read are opt-in features of the HIP and Turing builds: off on Metal, so they answer "nothing to do" instead of stopping the engine.

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/dequant_s2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/route_prior.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/s2_gemv.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/share_rules.hpp"
#include "strata/prefill/xmx_moe.hpp"
#include "strata/core/arch_defaults.hpp"
#include "strata/core/batch_rows.hpp"
#include "strata/core/conversation_buffer.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/device.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/emulate.hpp"
#include "strata/core/exchange_storage.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/foresight_swap.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/hit_hook.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/logit_bias.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/session.hpp"
#include "strata/core/session_save_reclaim.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/vmm.hpp"
#include "strata/core/weights.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/direct_file.hpp"
#include "strata/platform/host_limits.hpp"
#include "strata/platform/memory.hpp"
#include "strata/ngram/ple_reader.hpp"
#include "strata/spec/controller.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include <type_traits>

#include <type_traits>

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels { void copy_rows_from_mapped_after(unsigned int const* p0, unsigned int p1, float* p2, float const* p3, long long p4, long long p5, int const* p6, int const* p7, void* p8) { strata::metal_engine::missing("strata::kernels::copy_rows_from_mapped_after"); } }
namespace strata::kernels::cpu { void iq256_gu_rows_exact_one(int p0, unsigned char const* p1, unsigned long p2, unsigned long p3, int p4, void const* p5, float* p6, int p7, int p8) { strata::metal_engine::missing("strata::kernels::cpu::iq256_gu_rows_exact_one"); } }
namespace strata::kernels { void fused_gr_persist_init() {  } }
namespace strata::kernels { bool fused_gr_read_persist(strata::kernels::FusedGrArgs const* p0, int p1, void* p2) {  return false; } }
namespace strata::kernels { void gdn_step_norm_groups(strata::kernels::GdnGroups const& p0, float const* p1, int p2, float const* p3, float const* p4, float const* p5, float const* p6, float p7, float* p8, int p9, int p10, bool p11, void* p12) { strata::metal_engine::missing("strata::kernels::gdn_step_norm_groups"); } }
namespace strata::kernels { void handoff_publish(float* p0, float const* p1, long long p2, float const* p3, long long p4, float const* p5, long long p6, unsigned int* p7, unsigned int* p8, void* p9, unsigned int const* p10) { strata::metal_engine::missing("strata::kernels::handoff_publish"); } }
namespace strata::kernels { void hc_q8_to_bf16(void const* p0, float const* p1, unsigned short* p2, long long p3, void* p4) { strata::metal_engine::missing("strata::kernels::hc_q8_to_bf16"); } }
namespace strata::kernels { void native_quantize_q8_1_il(float const* p0, void* p1, void* p2, int p3, int p4, void* p5) { strata::metal_engine::missing("strata::kernels::native_quantize_q8_1_il"); } }
namespace strata::kernels { void qsa_block_scores_rows(strata::kernels::QsaScoreRows const& p0, float const* p1, int const* p2, long long p3, long long p4, strata::kernels::QsaShapes const& p5, float* p6, void* p7) { strata::metal_engine::missing("strata::kernels::qsa_block_scores_rows"); } }
namespace strata::kernels { void qsa_decode_attn_batch_rows(float const* p0, strata::kernels::QsaAttnPoolsRows const& p1, int const* p2, int const* p3, long long p4, strata::kernels::QsaShapes const& p5, float* p6, float* p7, long long p8, void* p9) { strata::metal_engine::missing("strata::kernels::qsa_decode_attn_batch_rows"); } }
namespace strata::kernels { bool qsa_decode_attn_split(float const* p0, strata::kernels::QsaAttnPools const& p1, int const* p2, int const* p3, long long p4, strata::kernels::QsaShapes const& p5, float* p6, unsigned long long p7, float* p8, long long p9, void* p10) { strata::metal_engine::missing("strata::kernels::qsa_decode_attn_split"); return {}; } }
namespace strata::kernels { void res_patch(int* p0, int const* p1, int p2, void* p3) { strata::metal_engine::missing("strata::kernels::res_patch"); } }
namespace strata::kernels { void wait_copy_i32_from_mapped(unsigned int const* p0, unsigned int p1, int* p2, int const* p3, long long p4, void* p5) { strata::metal_engine::missing("strata::kernels::wait_copy_i32_from_mapped"); } }
namespace strata::kernels { void wait_flag_ge_stamped(unsigned int const* p0, unsigned int p1, unsigned long long* p2, unsigned long long* p3, void* p4) { strata::metal_engine::missing("strata::kernels::wait_flag_ge_stamped"); } }
namespace strata::kernels { void wait_handoff(unsigned int const* p0, unsigned int p1, unsigned int* p2, unsigned long long p3, void* p4) { strata::metal_engine::missing("strata::kernels::wait_handoff"); } }
namespace strata::prefill { void moe_scatter_add(float const* p0, int const* p1, int const* p2, int const* p3, float const* p4, long long p5, float* p6, void* p7) { strata::metal_engine::missing("strata::prefill::moe_scatter_add"); } }
namespace strata::prefill { void moe_shared_init(float const* p0, float const* p1, float* p2, long long p3, void* p4) { strata::metal_engine::missing("strata::prefill::moe_shared_init"); } }
namespace strata::prefill { bool f16_range_read(float& p0, unsigned long long& p1, bool p2) {  return false; } }
namespace strata::prefill { bool gemm_f16_range_read(float* p0, unsigned long long* p1, bool p2) {  return false; } }
namespace strata::prefill { void gr_norm(float const* p0, float const* p1, float p2, float* p3, unsigned short* p4, long long p5, void* p6, unsigned short* p7, bool p8) { strata::metal_engine::missing("strata::prefill::gr_norm"); } }
namespace strata::prefill { void gr_norm_rs(float const* p0, float const* p1, float p2, float* p3, unsigned short* p4, long long p5, void* p6, unsigned short* p7, long long p8, bool p9) { strata::metal_engine::missing("strata::prefill::gr_norm_rs"); } }
namespace strata::prefill { void gr_write_cvec_norm_rs(float* p0, float const* p1, float const* p2, long long p3, float const* p4, float const* p5, int const* p6, int p7, float const* p8, float p9, float* p10, unsigned short* p11, long long p12, void* p13, unsigned short* p14, long long p15, bool p16) { strata::metal_engine::missing("strata::prefill::gr_write_cvec_norm_rs"); } }
namespace strata::prefill { void gr_write_norm_rs(float* p0, float const* p1, float const* p2, long long p3, float const* p4, float p5, float* p6, unsigned short* p7, long long p8, void* p9, unsigned short* p10, long long p11, bool p12) { strata::metal_engine::missing("strata::prefill::gr_write_norm_rs"); } }
// pf_wmma (STRATA_PF_WMMA=1, Volta prompt experts) is CUDA only; the prefill code of 148/149 references it, so the Metal link needs the entry point. Off on Metal: "not launched".
namespace strata::kernels { bool gemm_iq_f16_grouped(int p0, void const* p1, unsigned long p2, int p3, int p4, void const* p5, int p6, int const* p7, int p8, int p9, float* p10, int p11, void* p12) { return false; } }
// STRATA_Q8_SM60 (Pascal GP100, opt-in): a no-op on Metal; NativeHead::load()/unload() call these for a Q8_0 head, so they must exist and answer "nothing packed".
namespace strata::kernels { bool native_q8_0_sm60_pack(void const* p0, int p1, int p2, char const* p3) { return false; } }
namespace strata::kernels { void native_q8_0_sm60_release(void const* p0) { } }
// the MTP drafter's STRATA_MTP_Q8_SM60 packing (mtp.cpp): nothing packed on Metal
namespace strata::kernels { bool native_q8_0_sm60_pack_quiet(void const* p0, int p1, int p2, char const* p3) { return false; } }
namespace strata::kernels { bool native_q8_0_sm60_pack_in_place(void* p0, int p1, int p2, char const* p3) { return false; } }
// STRATA_SELECT_LONG (verify.cpp's long-context decode select): CUDA only; never reached on Metal (no scratch asked for)
namespace strata::kernels { long long qsa_topk_multi_scratch_ints(long long p0) { return 0; } }
namespace strata::kernels { void qsa_block_topk_long(float const* p0, int const* p1, long long p2, long long p3, long long p4, strata::kernels::QsaShapes const& p5, int* p6, void* p7, long long p8, int* p9) { strata::metal_engine::missing("strata::kernels::qsa_block_topk_long"); } }
namespace strata::kernels { void qsa_block_scores_long(float const* p0, float const* p1, float const* p2, int const* p3, long long p4, long long p5, strata::kernels::QsaShapes const& p6, float* p7, void* p8, long long p9) { strata::metal_engine::missing("strata::kernels::qsa_block_scores_long"); } }
// only reached with pf_wmma (off on Metal)
namespace strata::prefill { void swiglu_split_f16(float const* p0, unsigned short* p1, long long p2, int p3, bool p4, void* p5) { } }
// STRATA_PLE_COMPACT (opt-in, CUDA / HIP prompt path): prefill.cpp references it; the Metal prompt runs through its window and never reaches it.
namespace strata::kernels { void ple_iq4nl_dequant(const uint8_t* p0, int64_t p1, float* p2, void* p3) { strata::metal_engine::missing("strata::kernels::ple_iq4nl_dequant"); } }
// STRATA_DEQUANT_SIDE (opt-in, CUDA only): off on Metal; the prompt path takes no second scratch and set_dequant_side(nullptr) keeps the serial form.
namespace strata::prefill { bool gemm_dequant_side() { return false; } }
auto strata::prefill::Gemm::set_dequant_side(uint16_t* p0, std::string& p1) -> bool { if (p0 == nullptr) return true; strata::metal_engine::missing("strata::prefill::Gemm::set_dequant_side"); return false; }
