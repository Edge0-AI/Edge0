#pragma once
#include <cstdint>

namespace strata::core::proposal {
// Per-call proposal row contract; no process-wide mode.
// Never a vocabulary ID or the existing no-draft marker (-1).
inline constexpr int32_t kProposalSamplerErrorToken = -2147483647;
enum class RowKind { MuOffer, PointOffer, NullTail, PlainBonus };
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_PROPOSAL_HD __host__ __device__
#else
#define STRATA_PROPOSAL_HD
#endif
// Preconditions: 0 <= t < T, 0 <= nq <= T. The proposal API permits nq == T
// for a NULL tail, which has no candidate token even when T == 1.
STRATA_PROPOSAL_HD constexpr RowKind rowkind(int t, int T, int nq) {
    return t == T - 1 ? (nq == T ? RowKind::NullTail : RowKind::PlainBonus)
                      : (t < nq ? RowKind::MuOffer : RowKind::PointOffer);
}
#undef STRATA_PROPOSAL_HD
// qbuf owns immutable masked mu, not a renormalized conditional proposal.
// MuOffer: require mu(D)>0; accept min(1,p(D)/mu(D)), else residual (p-mu)+.
// NullTail: no dtok read and no accept draw; sample residual (p-mu)+ directly.
// PointOffer and PlainBonus retain existing semantics. Target ex is untouched.
// Missing/nonpositive/nonfinite mu(D), or zero NULL residual, emits the unique
// error token above. The verifier rejects it after synchronization.
// This is a packet/interface check, not a replacement for caller law validation.
}  // namespace strata::core::proposal
