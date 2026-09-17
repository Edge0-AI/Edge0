import Foundation

/// Minimal configuration surface needed for the Edge0 8B port.
/// Mirrors Edge0/Edge0-8B-A1B-preview config.json.
public struct BailingConfiguration: Codable, Sendable, Equatable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float

    public let layerGroupSize: Int
    public let firstKDenseReplace: Int
    public let shortConvKernelSize: Int
    public let kdaSafeGate: Bool
    public let kdaLowerBound: Float

    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let ropeTheta: Float
    public let ropeInterleave: Bool

    public let numExperts: Int
    public let expertsPerToken: Int
    public let numSharedExperts: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let nGroup: Int
    public let topkGroup: Int
    public let normTopkProb: Bool
    public let routedScalingFactor: Float
    public let routerHasExpertBias: Bool

    public let quantization: Quantization

    public struct Quantization: Codable, Sendable, Equatable {
        public let groupSize: Int
        public let bits: Int
        public let mode: String

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case layerGroupSize = "layer_group_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case shortConvKernelSize = "short_conv_kernel_size"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTheta = "rope_theta"
        case ropeInterleave = "rope_interleave"
        case numExperts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case routerHasExpertBias = "moe_router_enable_expert_bias"
        case quantization
    }

    public static func load(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }

    public func isMLALayer(_ index: Int) -> Bool {
        let full = hiddenLayers / layerGroupSize * layerGroupSize
        return (index + 1) % layerGroupSize == 0 || index >= full
    }
}
