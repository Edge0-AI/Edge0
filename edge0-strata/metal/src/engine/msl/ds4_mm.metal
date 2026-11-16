// metal/src/engine/msl/ds4_mm.metal - the DeepSeek prompt chunk's expert GEMMs: moe_mm.metal's mm_gu3 / mm_down3 (matmul2d on dequantized
// weight tiles) over the engine's own expert slots and its UNPADDED sort order.  Compiled after quant/units/mma/moe_mm.metal.
//   tiles[i] = (slot, first row, valid rows): a tile is TM rows of one expert; rows past `valid` (the next expert's) are computed and
//   dropped (the operand rows exist: the scratch carries TM rows of slack), the swiglu is DeepSeek's clamped one.

struct DsMmArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong stride;    // bytes between slots
    uint gu_row;     // bytes per gate/up row (n_embd values)
    uint d_row;      // bytes per down row (n_ff values)
    uint n_embd;
    uint n_ff;
    float limit;
    uint pad0;
};

// xs[r][c] = half(x[idx[r]][c]) for r < rows
kernel void ds4_mm_gather(device const float* x [[buffer(0)]], constant ulong& ldx [[buffer(1)]], device const int* idx [[buffer(2)]],
                          constant uint& n_embd [[buffer(3)]], device half* xs [[buffer(4)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * 256u + tid;
    if (c >= n_embd) return;
    xs[(ulong) tgp.y * n_embd + c] = half(x[(ulong) idx[tgp.y] * ldx + c]);
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_gu(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]], device half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = W + ulong(e) * a.stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    auto accu = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        const uint16_t cap = rg.get_capacity(), half_cap = cap / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rg.get_multidimensional_index(i);
            const int k = k0 + ix[0], n = ix[1];
            device const uchar* rgw = g0 + ulong(n) * a.gu_row;
            device const uchar* ruw = u0 + ulong(n) * a.gu_row;
            half4 wg, wu, wg2, wu2;
            W8<F>::run(rgw, k, tab, wg, wg2);
            W8<F>::run(ruw, k, tab, wu, wu2);
            rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
            ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
            const uint16_t j = i + half_cap;
            rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
            ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
        }
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, rg, accg);
        op.run(tA, ru, accu);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = fmin(accg[i], a.limit), u = fmin(fmax(accu[i], -a.limit), a.limit);
            accg[i] = g / (1.0f + exp(-g)) * u;
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_dn(device const uchar* W [[buffer(0)]], device half* YEH [[buffer(1)]], device const half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles || a.pad0 == 3u) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = W + ulong(e) * a.stride + a.down_off + ulong(n0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    auto rw = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rw)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        const uint16_t half_cap = rw.get_capacity() / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rw.get_multidimensional_index(i);
            half4 w, w2;
            if (a.pad0 == 1u) { w = half4(0); w2 = half4(0); } else if (a.pad0 == 2u) { w = half4(1); w2 = half4(1); W8<F>::run(w0, 0, tab, w, w2); w = half4(0); } else
            W8<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab, w, w2);
            rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
            const uint16_t j = i + half_cap;
            rw[j] = w2.x; rw[j + 1] = w2.y; rw[j + 2] = w2.z; rw[j + 3] = w2.w;
        }
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, rw, acc);
    }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    acc.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) ((device float*) YEH)[ulong(row0 + m) * a.n_embd + n0 + f] = ot[sg][it];
    }
}

#define DS4_MM_INST(F, T)                                                                                                          template [[host_name("ds4_mm_gu_" #T)]] kernel decltype(ds4_mm_gu<F, 32, 32, 32, 4>) ds4_mm_gu<F, 32, 32, 32, 4>;               template [[host_name("ds4_mm_dn_" #T)]] kernel decltype(ds4_mm_dn<F, 32, 32, 32, 4>) ds4_mm_dn<F, 32, 32, 32, 4>;
DS4_MM_INST(T_IQ1_M, 29)
DS4_MM_INST(T_IQ1_S, 19)
DS4_MM_INST(T_IQ2_XXS, 16)
DS4_MM_INST(T_IQ2_XS, 17)
DS4_MM_INST(T_IQ2_S, 22)
DS4_MM_INST(T_IQ3_XXS, 18)
