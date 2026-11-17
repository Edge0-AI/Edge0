// metal/kernels/gdn_tprobe.metal - go / no-go probe for a tensor-op (MPP matmul2d) chunk-state kernel: the five products of one GDN
// chunk step (W S, Q S, M delta, K^T delta' accumulated into S) on fp16 operands, S kept in fp32 threadgroup memory, 64 chunk steps
// a threadgroup, grid h_v * 4 (the shape of gdn_cstate2).  No numerics: timing only.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant int PROBE [[function_constant(0)]];
constant int probe = is_function_constant_defined(PROBE) ? PROBE : 0;   // 1 = skip the products (SIMT parts only)

kernel void gdn_tprobe(device float* out [[buffer(0)]], device half* Wh [[buffer(1)]], device half* Qh [[buffer(2)]],
                       device half* Kh [[buffer(3)]], device half* Mh [[buffer(4)]], constant int& NC [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half Sh[128 * 32];
    threadgroup half Dh[32 * 32];
    threadgroup float X[32 * 32], Y[32 * 32], Z[32 * 32];
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tSh(Sh, dextents<int32_t, 2>(32, 128));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tDh(Dh, dextents<int32_t, 2>(32, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX(X, dextents<int32_t, 2>(32, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(32, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tZ(Z, dextents<int32_t, 2>(32, 32));
    constexpr auto d_ws = matmul2d_descriptor(32, 32, 128, false, false, false);   // [32 x 128] . [128 x 32]
    constexpr auto d_md = matmul2d_descriptor(32, 32, 32, false, false, false);    // [32 x 32] . [32 x 32]
    constexpr auto d_up = matmul2d_descriptor(128, 32, 32, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);   // K^T delta'
    matmul2d<d_ws, execution_simdgroups<4>> op_ws;
    matmul2d<d_md, execution_simdgroups<4>> op_md;
    matmul2d<d_up, execution_simdgroups<4>> op_up;
    const ulong hb = ulong(tg.x) / 4;
    using KT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using DT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    auto cS = op_up.get_destination_cooperative_tensor<KT, DT, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cS.get_capacity(); ++i)
        if (cS.is_valid_element(i)) cS[i] = 0.001f * float(i & 15);
    for (int c = 0; c < NC; ++c) {
        const ulong cb = (hb * NC + c);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cS.get_capacity(); ++i)
            if (cS.is_valid_element(i)) {
                auto ids = cS.get_multidimensional_index(i);
                Sh[ids[1] * 32 + ids[0]] = half(cS[i]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (probe != 1) {
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tW(Wh + cb * 32 * 128, dextents<int32_t, 2>(128, 32));
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Qh + cb * 32 * 128, dextents<int32_t, 2>(128, 32));
            op_ws.run(tW, tSh, tX);
            op_ws.run(tQ, tSh, tY);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 32 * 32; i += 128) Dh[i] = half(0.01f * X[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (probe != 1) {
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tM(Mh + cb * 32 * 32, dextents<int32_t, 2>(32, 32));
            op_md.run(tM, tDh, tZ);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 32 * 32; i += 128) Dh[i] = half(Z[i] + 0.01f * Y[i]);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cS.get_capacity(); ++i)
            if (cS.is_valid_element(i)) cS[i] *= 0.99f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (probe != 1) {
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kh + cb * 32 * 128, dextents<int32_t, 2>(128, 32));
            op_up.run(tK, tDh, cS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cS.get_capacity(); ++i)
        if (cS.is_valid_element(i)) {
            auto ids = cS.get_multidimensional_index(i);
            out[ulong(tg.x) * 4096 + ids[1] * 32 + ids[0]] = cS[i];
        }
}
