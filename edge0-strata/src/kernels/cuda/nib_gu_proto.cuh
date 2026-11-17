// src/kernels/cuda/nib_gu_proto.cuh - STRATA_NIB_GU_PROTO: a staging variant of nib_gu_kernel (STRATA_EXPERT_NIBBLE's
// 4-bit gate/up), included by iq_kernels.cu inside its anonymous namespace after nib_gu_kernel.
//
// The kernel keeps the S skeleton's per-row arithmetic by calling nib_row_dot_s itself (16 lanes a row, lane t's items
// k = t, t+32, t+64 into s and t+16, t+48 into s16, NibLoad<TG>::load then Split<TG>::apply on the staged block_q8_1,
// __fadd_rn(s, s16), xor 8, 4, 2, 1).  Only the staging and the block -> (group, rows) mapping change:
//   - a block takes RB row blocks (16 rows each) of one group: grid (80 / RB, gy).  The group's entries are staged
//     once and the RB row blocks run one after another on them;
//   - the staging reads the entries' q8_1 rows (2,880 B = 180 x 16 B each, 16-aligned when xq is) as 16-byte loads,
//     every load issued before the first store, so only the stores sit between the two barriers.  The S skeleton's
//     word loop (nib_pass_s) has a serial load -> store chain there that stalls all eight warps of the block.
// The outputs are written without nib_pass_s's dynamically indexed s[t] (a select on t), the same values.
// Id 31 = RB 2, the one STRATA_NIB_GU_PROTO=1 selects: one GP100 die (PH402), T=1, gate/up alone, IQ2_S / IQ3_XXS /
// IQ3_S x1.25-1.27 against the S / P skeleton, every gate and up row bitwise equal (iq_nibble_parity (7),
// native_expert_bench --nibble --vs 31).
#pragma once

// The n (1..GRP_NC) entries from e into s_x (block_q8_1 rows of 80).  Block-uniform (e, n): every thread reaches both
// barriers.  16-byte loads, all in flight before the first store.
__device__ __forceinline__ void nibx_stage(block_q8_1* s_x, const int32_t* __restrict__ ent_tok,
                                           const block_q8_1* __restrict__ xq, int e, int n) {
    constexpr int kV = (GRP_NC * 180 + 255) / 256;   // 3
    const int nv = n * 180;
    uint4 v[kV];
#pragma unroll
    for (int j = 0; j < kV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < nv) {
            const int c = i >= 540 ? 3 : i >= 360 ? 2 : i >= 180 ? 1 : 0;
            v[j] = ((const uint4*) (xq + (size_t) ent_tok[e + c] * 80))[i - 180 * c];
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < nv) ((uint4*) s_x)[i] = v[j];
    }
    __syncthreads();
}

// nib_pass_s's compute on the staged entries (nib_row_dot_s, the production function), stored without the dynamically
// indexed s[t]
template<int TG>
__device__ __forceinline__ void nibx_run(const uint8_t* slot, int row, const block_q8_1* s_x, int e, int n, int t,
                                         float* dst, int r) {
    const uint4* crow = (const uint4*) (slot + nibk::kCodesOff) + (size_t) row * nibk::kItems;
    const uint8_t* srow = slot + nibk::kScOff + (size_t) row * nibk::kItems;
    const __half* drow = (const __half*) (slot + nibk::kDOff) + (size_t) row * nibk::kSuper;
    if (n == 1) {
        const int o[1] = { 0 }; float s[1];
        nib_row_dot_s<TG, 1>(crow, srow, drow, s_x, o, t, s);
        if (t == 0) dst[(size_t) e * nibk::kNFf + r] = s[0];
    } else if (n == 2) {
        const int o[2] = { 0, 80 }; float s[2];
        nib_row_dot_s<TG, 2>(crow, srow, drow, s_x, o, t, s);
        if (t < 2) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : s[1];
    } else if (n == 3) {
        const int o[3] = { 0, 80, 160 }; float s[3];
        nib_row_dot_s<TG, 3>(crow, srow, drow, s_x, o, t, s);
        if (t < 3) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : t == 1 ? s[1] : s[2];
    } else {
        const int o[4] = { 0, 80, 160, 240 }; float s[4];
        nib_row_dot_s<TG, 4>(crow, srow, drow, s_x, o, t, s);
        if (t < 4) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : t == 1 ? s[1] : t == 2 ? s[2] : s[3];
    }
}

template<int TG, int RB>
__global__ void __launch_bounds__(256) nibx_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                      const int32_t* __restrict__ grp_start,
                                                      const int32_t* __restrict__ n_groups,
                                                      const int32_t* __restrict__ ent_tok,
                                                      const block_q8_1* __restrict__ xq, float* __restrict__ gate,
                                                      float* __restrict__ up) {
    static_assert(80 % RB == 0, "row blocks per block divide 80");
    const int ng = *n_groups;
    if (blockIdx.y >= (unsigned) ng) return;
    STRATA_SHARED_ALIGN16_U32(s_buf, kNibSWords);
    block_q8_1* s_x = (block_q8_1*) s_buf;
    const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
    const int rb0 = blockIdx.x * RB;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* slot = (const uint8_t*) grp_ptr[g];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        const int nt = e1 - e0;
        if (nt <= 0) continue;   // an empty group: nothing to write (block-uniform, no barrier skipped unevenly)
        if (nt <= GRP_NC) {
            nibx_stage(s_x, ent_tok, xq, e0, nt);
#pragma unroll 1
            for (int j = 0; j < RB; ++j) {
                const int row = (rb0 + j) * 16 + subwarp;
                const bool is_up = row >= nibk::kNFf;
                nibx_run<TG>(slot, row, s_x, e0, nt, t, is_up ? up : gate, is_up ? row - (int) nibk::kNFf : row);
            }
        } else {   // more entries than one pass holds: per row block, the passes as nib_pass_s runs them
#pragma unroll 1
            for (int j = 0; j < RB; ++j) {
                const int row = (rb0 + j) * 16 + subwarp;
                const bool is_up = row >= nibk::kNFf;
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    nibx_stage(s_x, ent_tok, xq, e, n);
                    nibx_run<TG>(slot, row, s_x, e, n, t, is_up ? up : gate, is_up ? row - (int) nibk::kNFf : row);
                }
            }
        }
    }
}

int g_nibx_occ = 0;   // the last attribute query's resident blocks per SM

template<int TG, int RB>
void nibx_launch_one(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                     const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up,
                     cudaFuncAttributes* attr) {
    if (attr) {
        cudaFuncGetAttributes(attr, nibx_gu_kernel<TG, RB>);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&g_nibx_occ, nibx_gu_kernel<TG, RB>, 256, 0);
        return;
    }
    nibx_gu_kernel<TG, RB><<<dim3(80 / RB, grid.y), 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, gate, up);
}

// false: not the gu-proto id (31) or not a converted format.  attr != nullptr: only the kernel's attributes, nothing
// launched.
bool nibx_launch(int gu_type, int id, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr,
                 const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X,
                 float* gate, float* up, cudaFuncAttributes* attr = nullptr) {
    if (id != 31) return false;
    switch (gu_type) {
        case 22: nibx_launch_one<22, 2>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr); return true;
        case 18: nibx_launch_one<18, 2>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr); return true;
        case 21: nibx_launch_one<21, 2>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr); return true;
        default: return false;
    }
}
