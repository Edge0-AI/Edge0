import MLX

/// Language-level Swift port of mlx-lm's vector-gated delta Metal kernel used
/// by the official Edge0 8B KDA prefill path. The Metal body, grid and
/// template parameters intentionally retain the upstream structure.
public enum OfficialGatedDelta {
    private static let vectorKernel = MLXFast.metalKernel(
        name: "edge0_gated_delta_step_vec",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            // g: [B, T, Hv, Dk]
            auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[s_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }

              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv * Dk;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """)

    public static func callAsFunction(q: MLXArray, k: MLXArray, v: MLXArray,
                                      decay: MLXArray, beta: MLXArray,
                                      state: MLXArray) throws
        -> (output: MLXArray, state: MLXArray) {
        guard q.ndim == 4, k.shape == q.shape, v.ndim == 4,
              decay.shape == q.shape,
              beta.shape == [q.dim(0), q.dim(1), v.dim(2)],
              q.dim(2) > 0, v.dim(2) % q.dim(2) == 0,
              q.dim(3) > 0, q.dim(3) % 32 == 0,
              state.shape == [q.dim(0), v.dim(2), v.dim(3), q.dim(3)],
              q.dtype == k.dtype, q.dtype == v.dtype,
              q.dtype == decay.dtype, q.dtype == state.dtype else {
            throw M1Error.invalid("Invalid official gated-delta tensor shapes")
        }
        let batch = q.dim(0)
        let tokens = q.dim(1)
        let keyHeads = q.dim(2)
        let keyDimension = q.dim(3)
        let valueHeads = v.dim(2)
        let valueDimension = v.dim(3)
        let outputs = vectorKernel(
            [q, k, v, decay, beta, state, tokens],
            template: [
                ("InT", q.dtype), ("Dk", keyDimension),
                ("Dv", valueDimension), ("Hk", keyHeads),
                ("Hv", valueHeads),
            ],
            grid: (32, valueDimension, batch * valueHeads),
            threadGroup: (32, 4, 1),
            outputShapes: [
                [batch, tokens, valueHeads, valueDimension],
                state.shape,
            ],
            outputDTypes: [q.dtype, q.dtype])
        return (outputs[0], outputs[1])
    }
}
