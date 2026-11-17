// Cache the prefill GEMM's existing FP16 weight operand once per layer.
// Q3_K bytes, scales, dequantization and the K tile of the GEMM stay the same.
struct LatentPromptQ3Args { uint width, ff, row_bytes, expert_bytes; };

kernel void latent_prompt_q3_stage(device const uchar* experts [[buffer(0)]],
                                   device half* weights [[buffer(1)]],
                                   constant LatentPromptQ3Args& a [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    const uint units = a.width / 32;
    if (gid.x >= 2 * a.ff * units) return;
    const uint row = gid.x / units, unit = gid.x % units;
    const uint expert = gid.y;
    const device uchar* source = experts + ulong(expert) * a.expert_bytes + row * a.row_bytes;
    device half* output = weights + ulong(expert * 2 * a.ff + row) * a.width;
    threadgroup uint table[1];
    float values[32];
    int lo, hi;
    DW<T_Q3_K>::run(source, unit, 0, values, lo, hi, table);
    for (uint j = 0; j < 16; ++j) {
        output[lo + j] = half(values[j]);
        output[hi + j] = half(values[16 + j]);
    }
}

kernel void latent_prompt_gu_f16(device const half* weights [[buffer(0)]],
                                  device const half* input [[buffer(1)]],
                                  device half* middle [[buffer(2)]],
                                  device const int2* tiles [[buffer(3)]],
                                  constant MmArgs& a [[buffer(4)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int rows = 32, cols = 64, tile_k = 32;
    threadgroup float output[rows * cols];
    const int2 tile = tiles[tg.y];
    const int row0 = tile.y, f0 = int(tg.x) * cols;
    device half* gate = (device half*)weights + ulong(tile.x * 2 * a.n_ff) * a.n_embd;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> x(
        (device half*)input, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> g(gate, dextents<int32_t, 2>(a.n_embd, a.n_ff));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> u(
        gate + ulong(a.n_ff) * a.n_embd, dextents<int32_t, 2>(a.n_embd, a.n_ff));
    constexpr auto desc = matmul2d_descriptor(rows, cols, tile_k, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<2>> op;
    auto xa = x.template slice<tile_k, rows>(0, row0);
    auto wg = g.template slice<tile_k, cols>(0, f0);
    auto sum_g = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wg), float>();
    auto sum_u = op.template get_destination_cooperative_tensor<decltype(xa), decltype(wg), float>();
    for (ushort i = 0; i < sum_g.get_capacity(); ++i) {
        if (!sum_g.is_valid_element(i)) continue;
        sum_g[i] = 0.0f;
        sum_u[i] = 0.0f;
    }
    for (int k = 0; k < int(a.n_embd); k += tile_k) {
        xa = x.template slice<tile_k, rows>(k, row0);
        wg = g.template slice<tile_k, cols>(k, f0);
        auto wu = u.template slice<tile_k, cols>(k, f0);
        op.run(xa, wg, sum_g);
        op.run(xa, wu, sum_u);
    }
    for (ushort i = 0; i < sum_g.get_capacity(); ++i) {
        if (!sum_g.is_valid_element(i)) continue;
        const float value = sum_g[i];
        sum_g[i] = (value / (1.0f + exp(-value))) * sum_u[i];
    }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> y(output, dextents<int32_t, 2>(cols, rows));
    sum_g.store(y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < rows * cols; i += 64)
        middle[ulong(row0 + i / cols) * a.n_ff + f0 + i % cols] = half(output[i]);
}
