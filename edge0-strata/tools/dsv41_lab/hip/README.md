# HIP port of the trellis GEMV (gfx1201, R9700) - recipe

State: `wmma_emu.hpp` (host, lane by lane) reproduces the double reference to 3e-8 for K4..K7, widths 1/3/5, table codebook and the
computed code `3single` (`wmma_emu_test`, no GPU needed). `tqk_gemv_wmma.hip.cpp` is its line-by-line HIP form. It was **not
compiled or run** anywhere. Do the steps in this order; each one is a hard stop if it fails.

1. **Layout probe** (30 s). `hipcc --offload-arch=gfx1201 -O2 hip/wmma_probe.hip.cpp -o wmma_probe && HIP_VISIBLE_DEVICES=<n> ./wmma_probe`.
   It must print `layout as assumed`. The assumption (wave32, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, 8 halves per lane for A and B,
   8 floats for D, no replication between the two lane halves):
   `A[m][k]` in lane `l`, element `j`: `m = l % 16, k = 8 * (l / 16) + j`; `B[k][n]`: `k = 8 * (l / 16) + j, n = l % 16`; `D[m][n]`: `m = 8 * (l / 16) + i, n = l % 16`.
   If it does not match, change `wmma_emu.hpp` (the specification) and `wmma_emu_test` until it passes with the layout the probe found, then
   mirror the change in the kernel. (gfx11 replicated A/B between the halves and striped D by `2 i + l / 16`; gfx12 does not, as far as the
   LLVM documentation says, but nobody here has measured it.)
2. **Emulator test**: `g++ -O2 -std=c++17 -I third_party/ggml tools/dsv41_lab/wmma_emu_test.cpp tools/trellis/tqenc.cpp -o wmma_emu_test` (and with `TQE_CODE=3single`).
3. **Kernel**: the data flow, all of it in `wmma_emu.hpp::gemv_tqw`:
   * *Why a new layout.* mma.m16n8k16 gives a lane 2 rows of A and 4 floats of D; WMMA gives a lane 1 row of A (8 consecutive k) and 8 rows of D.
     So lane `l = r + 16 kg` owns row `r` of the tile and **16 consecutive steps** (`u = 0..15`, steps `t = 16 kg + u`) of every 128-weight block:
     stream bits `[(16 - 16 kg) K, (32 - 16 kg) K)`. The two blocks of a pair give 2 chunks of `16 K` bits = `K` words per lane
     (`words[((tile * NP + P) * K + j) * 32 + lane]`, coalesced). Scales `sc[((tile*NP+P)*16 + row)*2 + bk]`: a lane reads rows `8 kg..8 kg+7`.
     The bytes are the same as the file layout; the permutation is done at load (a bit shuffle per lane, `repack_tqw`).
   * *k order.* WMMA k-slot `(kg, j)` of mma `m` = weight `4 (16 kg + 2 m) + j` of the block (steps `u = 2m, 2m+1` of the lane). A and B must use the same
     map; B lane `(n, kg)` loads `x[n][blk*128 + 4(16 kg + 2m) + j]`, `j = 0..7` (2 float4), split `x = hi + lo` in fp16 (two WMMAs, error 1e-7).
     `prep_b_tqw` does this once per call into 32 contiguous bytes per (block, m, kg, column).
   * *States.* local step `u` is at chunk-relative bit `(15 - u) K`. Window 0 (u 0..7) = chunk bits `[8K, 16K)` + the low `15 - K` bits of the **partner lane's
     chunk** (`lane ^ 16`, same row and block; circular wrap: for kg = 0 the partner is kg = 1) - one `__shfl_xor(x, 16)`; window 1 (u 8..15) = chunk bits `[0, 7K + 15)`.
     `state(u) = (window >> offset) & 0x7fff`. Both windows fit 64 bits for K <= 7.
   * *A fragment.* computed code `3single`: `w0 = ((s*89226354 + 64248484) & 0x8FFF8FFF) ^ 0x3B603B60`, `w1 = ((s*0x9E3779B1 + 0x7F4A7C15) & ...) ^ ...`;
     the four half values of a step are `lo(w0), hi(w0), lo(w1), hi(w1)` = 2 words of the 8-half A vector. Table codebook: `x = s*0x9e3779b1`,
     words `lut[x >> 21]`, `lut[(x >> 10) & 2047]` from an 8 KB LDS table of half2 (the table is exact in fp16: 0 inexact entries).
   * *Accumulate.* D (8 floats) after the 8 (16 with lo) WMMAs of a block times the row scale (`fp16 d`, 8 rows per lane) is added to the running `tot[8]`; at the end lane
     `(n = l % 16 < NB, kg)` stores rows `8 kg .. 8 kg + 7` of column `n`.
   * Wave32 (the gfx12 default); 128-thread blocks (4 waves), `T = 2` tiles per wave first; VGPR budget: B fragments of one block half (8 m x 8 regs = 64), records `T K`,
     windows, A (4), D (8) - fits well under 256.
4. **Compare with the emulator** on the same random streams; the CUDA harness (`tqk_lab.cu`) has the benchmark shape (12 gate/up + 6 down matrices) to copy.
   Speed targets on gfx1201: K5/3single at 640 GB/s is a 3.7 Tw/s roof; WMMA hi+lo costs 2 x 8 WMMAs per 2048 weights per wave, the ALU side (2 IMAD + 2 LOP3 a step) is
   light, so expect the memory system to be the limit - the opposite of the A4000, where the tensor pipe is at 75 percent.
5. Not decided here: whether the R9700 build should keep the table codebook (K6/K7 are byte-compatible with agentionai and must) or the computed code (K4/K5 new types) -
   see notes-dsv41-cudalab.md.
