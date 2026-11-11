# GDN prompt-path contract (Flash-Next, CUDA reference)

Audited on main `870aedd9e` (v0.2.8), `src/prefill/kernels.cu` and `src/prefill/prefill.cpp`. Everything below is read from
that source; the harness (`gdn_harness`, `tests/gdn/gdn_fixture.hpp`) checks it against an FP64 oracle that starts from the
raw inputs, so a wrong reading shows up as a failed parity case.

## Shapes

`S = 128` (head size), `HK = 16` key heads, `HV = 48` value heads (3 per key head, `VPK = 3`), `C = 2*HK*S + HV*S = 10240`
channels per token. Value head `vh` uses key head `qh = vh % HK` (NOT `vh / 3`; FLA groups the other way, an adapter is needed
for any comparison with FLA). One layer's prompt call has `T` tokens (the engine's prefill chunk, not the whole prompt); the
state and the convolution history carry from call to call.

## One layer, in the order the engine runs it (`prefill.cpp`, `pt.mark` stages)

| Stage (STRATA_PREFILL_TIMING name) | Work | In the harness |
|---|---|---|
| `gdn` (kPfGdn) | attn_qkv, attn_gate (z), ssm_alpha, ssm_beta projections (quantized GEMMs) | no (not a GDN kernel) |
| `gdn conv+gates` (kPfGdnConv) | `gdn_gates`, `gdn_conv` | yes, timed separately |
| `gdn recurrence` (kPfGdnRec) | `gdn_recurrence`: scan + output norm | yes |
| `gdn out proj` (kPfGdnOut) | ssm_out projection from `y16` (quantized GEMM) | no |

Note the stage names: the profile line `gdn` is the INPUT projections, `gdn out proj` is the output projection. The task's
"GDN stage = kPfGdn + Conv + Rec + Out" therefore contains two dense GEMM stages the recurrence work cannot speed up.

## Tensors (all FP32 unless noted, row-major, `t` = token in the call)

| Tensor | Layout | Meaning |
|---|---|---|
| `qkv` | `[T][C]` | projection output, channels `q (HK*S) | k (HK*S) | v (HV*S)` |
| `ab` | `[T][2*HV]` | alpha at `[t][h]`, beta input at `[t][HV+h]` |
| `dt`, `ssm_a` | `[HV]` | per-head bias and (negative) decay base |
| `conv_w` | `[C][4]` | depthwise causal 4-tap filter, tap 3 is the current token |
| `history` | `[C][3]` | the three previous raw `qkv` inputs of every channel (a state object of its own) |
| `state` | `[S][HV][S]` (row i, head, column) | recurrent state, in place; `history` follows it in the engine's layer state |
| `gate`, `beta` | `[T][HV]` | outputs of `gdn_gates` |
| `h` | `[T][C]` | outputs of `gdn_conv` (SiLU, then q and k L2-normalized per head) |
| `z` | `[T][HV*S]` | output gate input |
| `y` | `[T][HV*S]` FP32 | scratch the recurrence writes the raw output to |
| `y16` | FP16, row stride `ld16` | normalized output the out projection reads |

`ld16 = HV*S = 6144`, or `6144 + 64` when STRATA_PF_PAD is on and `T >= max(pf_switch_min_t, 64)`. A kernel must write exactly
the 6144 columns and leave the pad columns untouched (the harness fills them with a sentinel and fails if they change).

## Math (per value head, `g_t = exp(gate_t)`)

- `gate_t = softplus(alpha_t + dt) * ssm_a` (softplus is the identity above 20); `beta_t = sigmoid(beta_input_t)`
- `h_c(t) = silu(sum_k w[c][k] * x_c(t-3+k))` with `x_c(t<0)` from `history`; q and k heads are then scaled by `1/sqrt(sum^2 + 1e-6)`
- `kv = S^T k`, `delta = (v - g*kv) * beta`, `S <- g*S + k delta^T`, `o = S^T q / sqrt(128)` (uses the UPDATED state)
- `y16 = fp16( o * rsqrt(mean(o^2) + 1e-6) * gamma[col] * sigmoid(z) )`
- after the call: `history <- last three raw inputs` (older history where `T < 3`)

## State mutation rules

- A call with `T <= 0` does nothing.
- The call advances `state` and `history` in place. A path that cannot run must decline BEFORE touching the state (the helper
  returns an error and "nothing was launched"); once a kernel has started, there is no replay. A forced path that is not
  eligible launches nothing and reports kGdnNone with the reason (`gdn_probe.hpp`); the harness never times it.
- Calls are sequential per layer state. The chunked path's scratch is one static buffer per device (single prompt stream per
  device); concurrent prompt streams on one device are not supported by it.

## Dispatch on CUDA (production, no probe)

1. `STRATA_GDN_TC=1` and `T >= 128`: `gdn_rec_tc` (tensor-core scan; sm_80+, >= 48 SMs, ~122 KB smem per block). Not mentioned in the
   2026-10-26 research pack: it landed with PR #3779. FP16 operands, FP32 accumulation: mixed-precision tier.
2. `STRATA_GDN_CHUNKED=1` (`=2` skips the SM-count test) and `T >= 128`: `gdn_rec_chunked` (FP32, C = 32, scan grid `HK x S/16 = 128`
   blocks of 384 threads, 93.7 KB smem; sm_80+).
3. else `gdn_rec_kh_kernel` when `gdn_keyhead_ok()` (sm_80+, all 64 blocks resident), otherwise `gdn_rec_cols_pipe_kernel`
   (`STRATA_GDN_PIPELINE=0`: `gdn_rec_cols_kernel`).
4. `gdn_out_norm_kernel` after any of them (it is the part that writes `y16`).
`STRATA_GDN_REC_WARP` and `STRATA_GDN_CHUNK` exist only in `STRATA_EXPERIMENTAL_SM60` builds and act only on a cc 7.0 device at run
time (`gdn_chunk_available()`): they do nothing on sm_60, sm_86 or sm_120.

## Numerical tiers (gdn-research 06)

1. exact: output and state bitwise equal to the incumbent (`kh` and `pipe` are the same bits by construction).
2. reordered FP32 (chunked): same recurrence, other summation order. Verdict in the harness: FP64-oracle error not above
   `max(floor, 3 x the incumbent's own error)` with floors 2e-3 (normalized max of the FP16 output) and 1e-4 (state); finite; the
   conv history exact.
3. mixed precision (tc): same checks, plus the stress fixtures (`--stress`).
The harness prints `normalized_max = max|d| / max|ref|` and relative L2 against the oracle and against the incumbent over ALL
heads, tokens and state elements.

## Test hook

`include/strata/prefill/gdn_probe.hpp`: a thread-local `GdnProbe` installed by a test selects/forces a path, a smaller chunked scratch
segment (`gsb`), or the SM-count bypass, and reads back `selected`, `wanted` and the fallback `reason`. With no probe installed the
dispatch is unchanged. The engine never installs one.
