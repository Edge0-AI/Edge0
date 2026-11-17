// metal/src/engine/exl3_legacy_stub.cpp - v1.1.8's EXL3 decode kernels, absent on Metal.
//
// exl3_legacy.cu exists for NVIDIA GPUs before sm_80, which pick it over the newer tiles; nothing on a Mac can select it
// (use_legacy() is false without CUDA), and the 2.25bpw text model does not use EXL3 at all.  Leaving the file out of the
// target would break the link for these seven entry points, so they are defined here and stop the run naming themselves -
// the same contract as the stub generator, and a loud failure rather than a silent wrong decode.
#include <cstdio>
#include <cstdlib>

#include <strata/dsv41/exl3.hpp>   // MvJob, MoeArgs, WinArgs

namespace strata::dsv41k::exl3::legacy {

[[noreturn]] static void missing(const char* what) {
    std::fprintf(stderr,
                 "strata metal: exl3::legacy::%s has no Metal implementation (v1.1.8's EXL3 kernels; the 2.25bpw text model "
                 "does not use EXL3)\n", what);
    std::abort();
}

int launch_errors() { return 0; }
bool prepare(cudaStream_t) { missing("prepare"); }
bool several_ok() { return false; }
bool mv(const MvJob*, int, cudaStream_t) { missing("mv"); }
void moe_gate_up(const MoeArgs&, const float*, float*, cudaStream_t) { missing("moe_gate_up"); }
void moe_down(const MoeArgs&, const float*, const float*, float*, cudaStream_t) { missing("moe_down"); }
bool win_experts(const WinArgs&, cudaStream_t) { missing("win_experts"); }

}   // namespace strata::dsv41k::exl3::legacy
