// D/U scratch rows are group-major. Expert rows use the prompt planner's
// padded order; each tile identifies one expert and each valid row one token.
struct LatentPromptArgs {
    uint tokens, groups, hidden, width, topk, tile_rows;
};

kernel void latent_prompt_gather(device const float* d [[buffer(0)]],
                                 device const float* inverse [[buffer(1)]],
                                 device const int* labels [[buffer(2)]],
                                 device const int2* tiles [[buffer(3)]],
                                 device const int* tokens [[buffer(4)]],
                                 device half* x [[buffer(5)]],
                                 constant LatentPromptArgs& a [[buffer(6)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]]) {
    const int token = tokens[row];
    device half* out = x + ulong(row) * a.width;
    if (token < 0) {
        for (uint k = tid; k < a.width; k += 256) out[k] = half(0.0f);
        return;
    }
    const uint group = labels[tiles[row / a.tile_rows].x];
    device const float* in = d + (ulong(group) * a.tokens + uint(token)) * a.width;
    for (uint k = tid; k < a.width; k += 256) out[k] = half(in[k] * inverse[k]);
}

kernel void latent_prompt_pool(device const float* down [[buffer(0)]],
                               device const int* positions [[buffer(1)]],
                               device const int* ids [[buffer(2)]],
                               device const float* weights [[buffer(3)]],
                               device const int* labels [[buffer(4)]],
                               device half* pooled [[buffer(5)]],
                               constant LatentPromptArgs& a [[buffer(6)]],
                               uint i [[thread_position_in_grid]]) {
    const uint row = i / a.width, k = i % a.width;
    if (row >= a.groups * a.tokens) return;
    const uint group = row / a.tokens, entry = (row % a.tokens) * a.topk;
    float sum = 0.0f;
    for (uint j = entry; j < entry + a.topk; ++j)
        if (uint(labels[ids[j]]) == group)
            sum = fma(weights[j], down[ulong(positions[j]) * a.width + k], sum);
    pooled[i] = half(sum);
}

kernel void latent_prompt_sum(device const float* projected [[buffer(0)]],
                              device float* routed [[buffer(1)]],
                              constant LatentPromptArgs& a [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    const uint stride = a.tokens * a.hidden;
    if (i >= stride) return;
    float sum = 0.0f;
    for (uint group = 0; group < a.groups; ++group) sum += projected[group * stride + i];
    routed[i] = sum;
}
