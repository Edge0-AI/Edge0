// Cache the prefill GEMM's existing FP16 weight operand once per layer.
// The expert bytes, scales, dequantization and the K tile of the GEMM stay the same.
struct LatentPromptQ3Args { uint width, ff, row_bytes, expert_bytes; };

// The SSD-streaming arena (as in moe_mm.metal): with function constant 20, expert e's slot is smap[e] in the chunks
// experts, W1..W3; an expert of the layer that is not in the arena (smap ~0: no token of the chunk routes to it) is
// not staged.
#ifndef STRATA_ARENA_METAL
#define STRATA_ARENA_METAL
constant bool kArenaFc [[function_constant(20)]];
constant bool kArena = is_function_constant_defined(kArenaFc) && kArenaFc;
static inline device const uchar* arena_slot(uint v, device const uchar* W, device const uchar* W1, device const uchar* W2,
                                             device const uchar* W3) {
    const uint c = v >> 28;
    device const uchar* b = c == 0 ? W : (c == 1 ? W1 : (c == 2 ? W2 : W3));
    return b + (ulong)(v & 0x0FFFFFFFu) * 16384ul;
}
#endif

template <int F>
kernel void latent_prompt_stage(device const uchar* experts [[buffer(0)]],
                                device half* weights [[buffer(1)]],
                                constant LatentPromptQ3Args& a [[buffer(2)]],
                                device const uint* smap [[buffer(3)]], device const uchar* W1 [[buffer(4)]],
                                device const uchar* W2 [[buffer(5)]], device const uchar* W3 [[buffer(6)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint units = a.width / 32;
    if (gid.x >= 2 * a.ff * units) return;
    const uint row = gid.x / units, unit = gid.x % units;
    const uint expert = gid.y;
    if (kArena && smap[expert] == 0xFFFFFFFFu) return;   // (the whole threadgroup: one expert per grid row)
    const device uchar* base = kArena ? arena_slot(smap[expert], experts, W1, W2, W3) : experts + ulong(expert) * a.expert_bytes;
    const device uchar* source = base + row * a.row_bytes;
    device half* output = weights + ulong(expert * 2 * a.ff + row) * a.width;
    threadgroup uint table[1];
    float values[32];
    int lo, hi;
    DW<F>::run(source, unit, 0, values, lo, hi, table);
    for (uint j = 0; j < 16; ++j) {
        output[lo + j] = half(values[j]);
        output[hi + j] = half(values[16 + j]);
    }
}

// F is the experts' format: NVFP4 scales the two sums by the expert's {s_gate, s_up} (buffer 5, the expert file; with the
// arena, the expert's slot: buffers 6..9 as in latent_prompt_stage).
template <int F>
kernel void latent_prompt_gu_f16(device const half* weights [[buffer(0)]],
                                 device const half* input [[buffer(1)]],
                                 device half* middle [[buffer(2)]],
                                 device const int2* tiles [[buffer(3)]],
                                 constant MmArgs& a [[buffer(4)]],
                                 device const uchar* experts [[buffer(5)]],
                                 device const uint* smap [[buffer(6)]], device const uchar* W1 [[buffer(7)]],
                                 device const uchar* W2 [[buffer(8)]], device const uchar* W3 [[buffer(9)]],
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
    float2 scale = float2(1.0f);
    if (F == T_NVFP4) {
        device const uchar* E = kArena ? arena_slot(smap[tile.x], experts, W1, W2, W3) : experts + ulong(tile.x) * a.gu_stride;
        device const float* tail = expert_tail(E, a);
        scale = float2(tail[0], tail[1]);
    }
    for (ushort i = 0; i < sum_g.get_capacity(); ++i) {
        if (!sum_g.is_valid_element(i)) continue;
        float value = sum_g[i], up = sum_u[i];
        if (F == T_NVFP4) { value *= scale.x; up *= scale.y; }
        sum_g[i] = (value / (1.0f + exp(-value))) * up;
    }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> y(output, dextents<int32_t, 2>(cols, rows));
    sum_g.store(y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < rows * cols; i += 64)
        middle[ulong(row0 + i / cols) * a.n_ff + f0 + i % cols] = half(output[i]);
}

template [[host_name("latent_prompt_q3_stage")]] kernel decltype(latent_prompt_stage<T_Q3_K>) latent_prompt_stage<T_Q3_K>;
template [[host_name("latent_prompt_nv_stage")]] kernel decltype(latent_prompt_stage<T_NVFP4>) latent_prompt_stage<T_NVFP4>;
template [[host_name("latent_prompt_gu_f16")]] kernel decltype(latent_prompt_gu_f16<T_Q3_K>) latent_prompt_gu_f16<T_Q3_K>;
template [[host_name("latent_prompt_gu_f16_nv")]] kernel decltype(latent_prompt_gu_f16<T_NVFP4>) latent_prompt_gu_f16<T_NVFP4>;
