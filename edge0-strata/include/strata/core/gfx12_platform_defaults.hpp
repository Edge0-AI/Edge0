// include/strata/core/gfx12_platform_defaults.hpp - which platforms take the two approved gfx12 prompt defaults.
//
// 148-amd measured and the owner approved STRATA_HIP_WMMA (matrix-core prompt attention) and the native Q2_0 pack's fused
// int8 prompt experts (STRATA_PF_FUSED) as defaults on gfx12.  Both were measured on Linux only: on a Windows HIP build
// they stay off by default until they are tested there, and STRATA_HIP_WMMA=1 / STRATA_PF_FUSED=1 still switch them on.
#pragma once

namespace strata::core {

constexpr bool gfx12_prompt_defaults_on(bool windows) noexcept { return !windows; }

#if defined(_WIN32)
inline constexpr bool kGfx12PromptDefaultsOn = gfx12_prompt_defaults_on(true);
#else
inline constexpr bool kGfx12PromptDefaultsOn = gfx12_prompt_defaults_on(false);
#endif

}  // namespace strata::core
