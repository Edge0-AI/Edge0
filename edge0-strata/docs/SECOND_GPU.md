# CUDA1-3 expert caches and the peer tier (experimental)

This page is for people who tune the engine by hand and want a second, third or fourth card to hold extra experts. These are
experimental engine settings. For ordinary multi-GPU use in the app or `strata launch`, see [MULTI_GPU.md](MULTI_GPU.md). The
settings go into the run config (`strata-*.json`) as `"args"` entries; they are engine flags, not `strata launch` flags.

> **On this page:** [How it works](#how-it-works) · [Helper caches](#helper-caches) · [Faster helper decode](#faster-helper-decode---remote-expert-opt) ·
> [Peer tier](#peer-tier---peer-device) · [Resident RAM mode](#the-resident-ram-mode-with-the-helper-caches) ·
> [Choosing the primary device](#choosing-the-primary-device-on-windows-strata_primary_device) · [History](#history--older-measurements)

## How it works

CUDA0 keeps the dense layers, the KV and state, the draft layer (MTP) and its own expert cache. Up to three independent expert
caches fill CUDA1, CUDA2 and CUDA3 in that order. Each expert has one owner: CUDA0, one helper card, or the CPU pool. Each
helper computes its rows while the CPU pool works, and the results return through private pinned host buffers before CUDA0
continues the layer. Each layer waits for all helpers.

- No peer-to-peer access is needed. Prompt prefill still uses CUDA0; helpers serve decode, including draft verification.
- Each helper card must keep at least 512 MiB free. Enable the tiers in order (CUDA1 before CUDA2).
- The startup log reports slots and GiB per card, and each request reports how many expert entries each helper computed.
  Compare identical requests at one, two and four GPUs. VRAM use alone does not show useful offload.
- Longer contexts reserve more KV on CUDA0 and shrink its automatic expert cache.
- Under WDDM (Windows, WSL2) only 8 GiB of the host expert arena is registered with CUDA when helpers are on
  (`STRATA_ARENA_PIN_GIB` overrides it). On Linux the whole arena is registered.
- The CUDA0 expert-cache GPU hit path gives outputs that diverge from cache-off runs. Treat performance as experimental until the
  generated tokens are validated.

## Helper caches

| Engine flag | Meaning |
| --- | --- |
| `--expert-cache-device1 N\|auto` | Fill CUDA1 with N experts, or as many as fit. |
| `--expert-cache-device2 N\|auto` | Fill CUDA2 with more experts. |
| `--expert-cache-device3 N\|auto` | Fill CUDA3 with more experts. |
| `--expert-cache-remote-placement stripe\|layer` | `stripe` (default) spreads the ranked expert list over the helpers. `layer` gives each layer to one helper (layer number modulo the number of cards), which can reduce device switches on USB4 but may be slower. |

The shipped profile ranks 8,000 `(layer, expert)` pairs. CUDA0 takes its usual `--expert-cache auto` share. With only CUDA1 it
takes the first remaining pairs. With CUDA2 or CUDA3 the remaining ranked pairs go round robin to the helpers, then all other
pairs in expert-then-layer order, so the extra VRAM fills even though unranked experts may rarely be routed.

`auto` fills each helper from its assigned ranking using actual aligned expert bytes and free VRAM, keeping the 512 MiB allowance.
It sizes capacity at startup, not throughput, and a card's truncated tail is not moved to another card. Explicit numbers still work.

Example engine arguments beside the model and profile arguments:

```text
--expert-cache auto --expert-cache-device1 auto --remote-expert-opt
```

To change slot counts without recompiling, edit the values in the stored config and restart.

**Size a helper against the card's real free VRAM.** On Windows `auto` reads the process's WDDM budget. It cannot see the
desktop, the vision encoder (a separate process), a streaming host or other apps. A helper auto-sized on a card that also drives
the display over-commits, and WDDM backs the excess with system RAM. Use explicit counts on a card that shares its VRAM.

## Faster helper decode (`--remote-expert-opt`)

`--remote-expert-opt` (serve mode only) optimizes the helper caches. The primary cache avoids admitting experts a helper already
holds, helpers replace cold experts with frequently routed CPU misses, each helper reduces its expert outputs to a weighted
partial sum on its GPU before returning one vector per token, and tokens with no CPU expert work skip CPU activation
quantization. It does nothing without a helper cache, and a plain layer split runs as before.

- Setup adds it to a config with two or more GPUs and a helper cache. To leave it out, use setup's `--no-remote-expert-opt` or
  `"remote_expert_opt": false` in the config (kept when setup runs again).
- It changes the floating-point summation order, so output is not claimed to be bitwise identical.
- It switches `--pipeline-windows` off ([MULTI_GPU.md](MULTI_GPU.md#pipeline-windows)). Keep one of the two.
- Only two-card CUDA operation has been measured by its author. Three and four cards are not validated.
- It does not apply to the `--peer-device` path below.

## Peer tier (`--peer-device`)

`--peer-device N` puts a second adaptive expert cache on CUDA device N. It takes the ranked pairs CUDA0's cache does not hold, as
many as fit. The peer computes the rows of its own experts for decode windows and for prompt chunks. Activations and results cross
NVLink or another peer-to-peer path. It is an alternative to the helper caches above, not a third tier beside them.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--peer-device N` | off | Enable the tier on CUDA device N (N >= 1). |
| `--peer-reserve-mib M` | 600 | VRAM left free on the peer card. Keep a few hundred MiB free on both cards beyond this, because the prompt path loads its kernels and cuBLAS on first use. |
| `--peer-slots N` | 0 | Cap the tier at N experts; 0 means as many as fit. |
| `--peer-adapt-swaps N` | -1 | Swaps per adaptive round on the peer; -1 uses the primary's `--adapt-swaps`. |
| `--peer-prefill-rows N` | -1 | Share of each prompt chunk's rows the peer computes; -1 is half of chunk x top-k, 0 keeps prompt rows on the primary. |

Whether the two cards can really exchange data is tested once at start with a copy each way. If the copy arrives wrong, the log
says so and the prompt path uses the host route (activations and rows through mapped host memory). `STRATA_PEER_NO_P2P=1` asks
for that route by hand.

Through the server, list both cards in the config's `"gpu"` (for example `[0, 1]`, numbered as `nvidia-smi` numbers them) and add
`--peer-device 1` to `"args"`. The number is the card's position in that list (with `"gpu": [2, 0]`, `--peer-device 1` is
`nvidia-smi`'s card 0). Several GPUs in `"gpu"` are otherwise a layer split; the server adds `--layer-split` only when
`--peer-device` is not among the args.

`--peer-device` requires `--expert-profile` and an enabled expert cache, and the device must be visible. It refuses
`--layer-split` and `--expert-cache-device1..3`:

```text
strata generate: --peer-device cannot be combined with --layer-split (use one or the other)
strata generate: --peer-device cannot be combined with --expert-cache-device1..3 (the peer tier already caches experts there)
```

With a peer, the prompt path keeps the MMQ path (the fused int8 prompt path is not combined with the peer's rows). The tier size
is manual (`--peer-reserve-mib`, `--peer-slots`). With the same expert set split across the two cards, generated tokens are
byte-identical to the single-GPU run under the exactness gate.

## The resident RAM mode with the helper caches

`--resident-experts` (the low-RAM mode) works with the helper caches. The experts a helper holds are left out of the RAM copy,
and the helpers' refills read through the source: the RAM copy when it holds the bytes, the model files otherwise.

## Choosing the primary device on Windows (`STRATA_PRIMARY_DEVICE`)

The engine runs the model on the device the runtime numbers 0. Windows' HIP runtime enumerates cards in a fixed order that
`HIP_VISIBLE_DEVICES` only filters, so when the faster card comes second, the model can only land on the slower one.
`STRATA_PRIMARY_DEVICE=N` picks another visible ordinal as primary. Unset or 0 keeps the default.

- The helper caches (`--expert-cache-device1..3`) take the visible devices other than the primary, in order.
- The peer tier uses the same ordinal.
- A layer split is refused when this is set.
- NVIDIA cards do not need it: `CUDA_VISIBLE_DEVICES` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` already reorders.

## History / older measurements

Each row is one machine, reported by the author of the change or a community member.

| Setting | Measured on | Result |
| --- | --- | --- |
| `--remote-expert-opt` | dual RTX 4090 (PR author) | +63% mixed, +132% code decode over the plain helper path |
| `--remote-expert-opt` | RTX 5090 + RTX 4090 (PR author) | +28% mixed, +63% code |
| Helper PCIe share on HIP | 2x RX 6900 XT, PCIe 4.0 x8 each, IQ3_S (community) | before the fix, decode 40 tok/s with the primary waiting 43 ms per verify window; after, 66 tok/s and 19 ms. `--pcie-frac 0` avoided the cost ([report](../bench/results/2026-10-04-rdna2-helper-pcie-share/README.md)) |
| Resident RAM mode with a helper | RX 6800 (16 GB) + RX 6600 (8 GB), Windows 11, i5-11400F, 32 GB RAM | RX 6800 alone ~37.5 tok/s; with a 3,800-slot helper ~40.6 / 44.2 tok/s on 8K-token requests; CPU pool share of experts down from ~2.9 to ~0.4 entries per layer-window |
