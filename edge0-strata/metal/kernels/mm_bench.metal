// metal/kernels/mm_bench.metal - matrix throughput probes (mm_bench.cpp): C[M][N] = A[M][K] . B[N][K]^T, fp32 out.
//   mpp_mm<T, TM, TN, SG>   Metal 4 MetalPerformancePrimitives matmul2d (the M5's neural accelerators), a TM x TN
//                           tile of C per threadgroup, K dynamic, A and B straight from device memory
//   sg_mm<T>                simdgroup_matrix 8x8, a 32 x 32 tile per simdgroup (4 x 4 accumulators), 4 simdgroups
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmArgs { int M, N, K; };

template <typename T, int TM, int TN, int SG>
kernel void mpp_mm(device T* A [[buffer(0)]], device T* B [[buffer(1)]], device float* C [[buffer(2)]],
                   constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    // extents are (inner, outer): A is M rows of K, B is N rows of K, C is M rows of N
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mC = tC.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mC);
}

#define MPP(T, M, N, S) template [[host_name("mpp_mm_" #T "_" #M "_" #N "_" #S)]] kernel decltype(mpp_mm<T, M, N, S>) mpp_mm<T, M, N, S>;
MPP(half, 64, 32, 4) MPP(half, 64, 64, 4) MPP(half, 128, 64, 4) MPP(half, 32, 64, 4) MPP(half, 64, 128, 4)
MPP(half, 16, 64, 2) MPP(half, 128, 128, 4)
MPP(half, 64, 16, 2) MPP(half, 64, 16, 4) MPP(half, 128, 16, 4) MPP(half, 128, 32, 4) MPP(half, 64, 32, 2) MPP(half, 32, 16, 2)
MPP(half, 32, 32, 2) MPP(half, 16, 32, 1) MPP(half, 16, 16, 1) MPP(half, 128, 48, 4) MPP(half, 64, 48, 4)
MPP(float, 64, 32, 4) MPP(float, 64, 64, 4) MPP(float, 128, 64, 4)
MPP(bfloat, 64, 64, 4) MPP(bfloat, 128, 64, 4)

// simdgroup_matrix: threadgroup tile 64 x 64 (4 simdgroups x 32 x 32), K in steps of 8, operands from device memory
template <typename T>
kernel void sg_mm(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]], device float* C [[buffer(2)]],
                  constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int m0 = int(tg.y) * 64 + (sg >> 1) * 32, n0 = int(tg.x) * 64 + (sg & 1) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < a.K; k += 8) {
        simdgroup_matrix<T, 8, 8> ma[4], mb[4];
        for (int i = 0; i < 4; ++i) simdgroup_load(ma[i], A + (m0 + 8 * i) * a.K + k, a.K);
        for (int j = 0; j < 4; ++j) simdgroup_load(mb[j], B + (n0 + 8 * j) * a.K + k, a.K, ulong2(0, 0), true);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], C + (m0 + 8 * i) * a.N + n0 + 8 * j, a.N);
}
template [[host_name("sg_mm_half")]] kernel decltype(sg_mm<half>) sg_mm<half>;
template [[host_name("sg_mm_float")]] kernel decltype(sg_mm<float>) sg_mm<float>;


// the element layout of single-simdgroup matmul2d cooperative tensors (right input, left input, destination) for a
// TM x TN x TK descriptor: out[((which * 32 + lane) * 128 + i) * 4 + (0, 1)] = the two indices, [2] = capacity
template <int TM, int TN, int TK>
kernel void coop_layout(device int* out [[buffer(0)]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto r = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto l = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto dd = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(l)>,
                                                             __tensor_ops_detail::__remove_addrspace_t<decltype(r)>, float>();
    for (uint16_t i = 0; i < dd.get_capacity() && i < 128; ++i) {
        auto ix = dd.get_multidimensional_index(i);
        device int* o = out + ((2 * 32 + lane) * 128 + i) * 4;
        o[0] = ix[0]; o[1] = ix[1]; o[2] = dd.get_capacity();
    }
    for (uint16_t i = 0; i < r.get_capacity() && i < 128; ++i) {
        auto ix = r.get_multidimensional_index(i);
        device int* o = out + ((0 * 32 + lane) * 128 + i) * 4;
        o[0] = ix[0]; o[1] = ix[1]; o[2] = r.get_capacity();
    }
    for (uint16_t i = 0; i < l.get_capacity() && i < 128; ++i) {
        auto ix = l.get_multidimensional_index(i);
        device int* o = out + ((1 * 32 + lane) * 128 + i) * 4;
        o[0] = ix[0]; o[1] = ix[1]; o[2] = l.get_capacity();
    }
}
template [[host_name("coop_layout_16_32_16")]] kernel decltype(coop_layout<16, 32, 16>) coop_layout<16, 32, 16>;
template [[host_name("coop_layout_32_32_16")]] kernel decltype(coop_layout<32, 32, 16>) coop_layout<32, 32, 16>;
template [[host_name("coop_layout_16_64_32")]] kernel decltype(coop_layout<16, 64, 32>) coop_layout<16, 64, 32>;
template [[host_name("coop_layout_16_32_32")]] kernel decltype(coop_layout<16, 32, 32>) coop_layout<16, 32, 32>;
