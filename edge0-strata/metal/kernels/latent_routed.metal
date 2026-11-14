// The latent routed FFN. Reuse the quantized-row decoders from quant/units.metal.
struct LatentRoutedArgs {
    uint n_tok, groups, hidden, latent, ff, topk, plan_token_entry;
    uint u_row, gu_row, down_row, up_off, down_off, expert_bytes;
};

kernel void latent_colscale(device float* projected [[buffer(0)]],
                            device const float* inverse [[buffer(1)]],
                            constant LatentRoutedArgs& a [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < a.n_tok * a.groups * a.latent) projected[i] *= inverse[i % a.latent];
}

template <int Lanes>
static inline float latent_q3k_dot(device const uchar* row, device const float* x,
                                   uint width, ushort lane, threadgroup const uint* table) {
    float sum = 0.0f;
    for (uint unit = lane; unit < width / 32; unit += Lanes) {
        float codes[32];
        float4 scale;
        Dec<T_Q3_K>::run(row, unit, codes, scale, table);
        float low = 0.0f, high = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            // Dec leaves odd elements in the high byte of each packed ushort.
            const float unpack = (j & 1) ? 1.0f / 256.0f : 1.0f;
            low = fma(codes[j] * unpack - 4.0f, x[32 * unit + j], low);
            high = fma(codes[j + 16] * unpack - 4.0f, x[32 * unit + j + 16], high);
        }
        sum = fma(scale.x, low, fma(scale.y, high, sum));
    }
    return lane_sum<Lanes>(sum);
}

template <int Format, int Lanes = 32>
static inline float latent_32_dot(device const uchar* row, device const float* x,
                                  uint width, ushort lane, threadgroup const uint* table) {
    float sum = 0.0f;
    for (uint unit = lane; unit < width / 32; unit += Lanes) {
        float codes[32];
        const float scale = deq<Format>(row + unit * fmt<Format>::BB, 0, codes, table);
        float dot = 0.0f;
        for (uint j = 0; j < 32; ++j) dot = fma(codes[j], x[32 * unit + j], dot);
        sum = fma(scale, dot, sum);
    }
    return lane_sum<Lanes>(sum);
}

template <int Lanes>
kernel void latent_gate_up(device const uchar* experts [[buffer(0)]],
                           device const int* labels [[buffer(1)]],
                           device const int* ids [[buffer(2)]],
                           device const float* projected [[buffer(3)]],
                           device float* middle [[buffer(4)]],
                           constant LatentRoutedArgs& a [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint table[1];   // Q3_K's decoder does not use a lookup table
    const uint entry = tg.y, row = (4 * tg.x + sg) * (32 / Lanes) + lane / Lanes;
    if (row >= a.ff) return;
    const uint expert = ids[entry], group = labels[expert];
    const uint token = entry / a.topk;
    device const float* x = projected + (token * a.groups + group) * a.latent;
    device const uchar* gate = experts + ulong(expert) * a.expert_bytes + row * a.gu_row;
    const float g = latent_q3k_dot<Lanes>(gate, x, a.latent, lane % Lanes, table);
    const float u = latent_q3k_dot<Lanes>(gate + a.up_off, x, a.latent, lane % Lanes, table);
    if (lane % Lanes == 0) middle[entry * a.ff + row] = (g / (1.0f + exp(-g))) * u;
}

template <int Lanes>
kernel void latent_down(device const uchar* experts [[buffer(0)]],
                        device const int* ids [[buffer(1)]],
                        device const float* middle [[buffer(2)]],
                        device float* output [[buffer(3)]],
                        constant LatentRoutedArgs& a [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint table[fmt<T_IQ4_NL>::TAB];
    load_tab<T_IQ4_NL>(table, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint entry = tg.y, row = (4 * tg.x + sg) * (32 / Lanes) + lane / Lanes;
    if (row >= a.latent) return;
    device const uchar* w = experts + ulong(ids[entry]) * a.expert_bytes + a.down_off + row * a.down_row;
    const float v = latent_32_dot<T_IQ4_NL, Lanes>(w, middle + entry * a.ff, a.ff, lane % Lanes, table);
    if (lane % Lanes == 0) output[entry * a.latent + row] = v;
}

kernel void latent_u(device const uchar* weights [[buffer(0)]],
                     device const int* labels [[buffer(1)]],
                     device const int* ids [[buffer(2)]],
                     device const float* route_weight [[buffer(3)]],
                     device const int* plan [[buffer(4)]],
                     device const float* latent_rows [[buffer(5)]],
                     device float* output [[buffer(6)]],
                     constant LatentRoutedArgs& a [[buffer(7)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint table[1];   // Q4_0's decoder does not use a lookup table
    const uint entry = tg.y, row = 4 * tg.x + sg;
    if (row >= a.hidden) return;
    const uint group = labels[ids[entry]];
    const uint slot = plan[a.plan_token_entry + entry];
    device const uchar* w = weights + (group * a.hidden + row) * a.u_row;
    const float v = latent_32_dot<T_Q4_0>(w, latent_rows + entry * a.latent, a.latent, lane, table);
    if (lane == 0) output[slot * a.hidden + row] = v * route_weight[entry];
}

// U is linear. Pool route-weighted expert outputs within a group before
// applying that group's U, then sum the projected groups with the shared expert.
kernel void latent_group_reduce(device const int* labels [[buffer(0)]],
                                 device const int* ids [[buffer(1)]],
                                 device const float* route_weight [[buffer(2)]],
                                 device const float* latent_rows [[buffer(3)]],
                                 device float* group_rows [[buffer(4)]],
                                 device uint* active [[buffer(5)]],
                                 constant LatentRoutedArgs& a [[buffer(6)]],
                                 uint i [[thread_position_in_grid]]) {
    const uint slot = i / a.latent;
    if (slot >= a.n_tok * a.groups) return;
    const uint column = i % a.latent, token = slot / a.groups, group = slot % a.groups;
    float sum = 0.0f;
    bool used = false;
    for (uint rank = 0; rank < a.topk; ++rank) {
        const uint entry = token * a.topk + rank;
        if (uint(labels[ids[entry]]) == group) {
            sum = fma(route_weight[entry], latent_rows[entry * a.latent + column], sum);
            used = true;
        }
    }
    group_rows[i] = sum;
    if (column == 0) active[slot] = used;
}

template <int Lanes>
kernel void latent_group_u(device const uchar* weights [[buffer(0)]],
                           device const float* group_rows [[buffer(1)]],
                           device const uint* active [[buffer(2)]],
                           device float* output [[buffer(3)]],
                           constant LatentRoutedArgs& a [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint table[1];
    const uint slot = tg.y, row = (4 * tg.x + sg) * (32 / Lanes) + lane / Lanes;
    if (row >= a.hidden) return;
    float y = 0.0f;
    if (active[slot]) {
        const uint group = slot % a.groups;
        device const uchar* w = weights + (group * a.hidden + row) * a.u_row;
        y = latent_32_dot<T_Q4_0, Lanes>(w, group_rows + slot * a.latent, a.latent, lane % Lanes, table);
    }
    if (lane % Lanes == 0) output[slot * a.hidden + row] = y;
}

template [[host_name("latent_gate_up")]] kernel decltype(latent_gate_up<32>) latent_gate_up<32>;
template [[host_name("latent_down")]] kernel decltype(latent_down<32>) latent_down<32>;
template [[host_name("latent_group_u")]] kernel decltype(latent_group_u<32>) latent_group_u<32>;
// Split SIMD-groups into independent rows to avoid inactive lanes in narrow dots.
template [[host_name("latent_gate_up_8")]] kernel decltype(latent_gate_up<8>) latent_gate_up<8>;
template [[host_name("latent_gate_up_4")]] kernel decltype(latent_gate_up<4>) latent_gate_up<4>;
template [[host_name("latent_down_4")]] kernel decltype(latent_down<4>) latent_down<4>;
template [[host_name("latent_group_u_8")]] kernel decltype(latent_group_u<8>) latent_group_u<8>;
template [[host_name("latent_group_u_4")]] kernel decltype(latent_group_u<4>) latent_group_u<4>;

kernel void latent_group_combine(device const float* grouped [[buffer(0)]],
                                 device const float* shared [[buffer(1)]],
                                 device const float* shared_gate [[buffer(2)]],
                                 device float* output [[buffer(3)]],
                                 constant LatentRoutedArgs& a [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
    const uint token = i / a.hidden, row = i % a.hidden;
    if (token >= a.n_tok) return;
    float sum = 0.0f;
    for (uint group = 0; group < a.groups; ++group)
        sum += grouped[(token * a.groups + group) * a.hidden + row];
    const float g = 1.0f / (1.0f + exp(-shared_gate[token]));
    output[i] = fma(g, shared[i], sum);
}
