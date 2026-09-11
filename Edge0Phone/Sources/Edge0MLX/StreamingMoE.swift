import Edge0Core
import MLX

public struct MoEOutput {
    public let hidden: MLXArray
    public let expertIndices: [Int]
    public let expertWeights: [Float]
}

/// One decode token's feed-forward block, including the unconditional shared expert.
/// Synchronous evaluation releases each expert's lazy graph before loading the next.
/// This is a correctness baseline, not the cache/prerouter implementation.
public final class StreamingMoE {
    private let configuration: BailingConfiguration
    private let store: ExpertTensorStore
    private let layer: Int
    private let routerWeight: MLXArray
    private let expertBias: MLXArray
    private let shared: QuantizedExpert

    public init(configuration c: BailingConfiguration, store: ExpertTensorStore, layer: Int) throws {
        guard c.quantization.bits == 4, c.quantization.groupSize == 64,
              c.quantization.mode == "affine", c.numSharedExperts == 1,
              layer >= c.firstKDenseReplace, layer < c.hiddenLayers,
              c.numExperts == store.expertCount, c.nGroup > 0,
              c.numExperts % c.nGroup == 0, c.numExperts / c.nGroup >= 2,
              c.topkGroup > 0, c.topkGroup <= c.nGroup,
              c.expertsPerToken > 0,
              c.expertsPerToken <= c.topkGroup * (c.numExperts / c.nGroup) else {
            throw M1Error.invalid("M1 requires an affine 4-bit/group-64 MoE layer with one shared expert and valid grouped routing")
        }
        configuration = c; self.store = store; self.layer = layer
        routerWeight = try store.mlxArrayCopying(named: "model.layers.\(layer).mlp.gate.weight").asType(.float32)
        expertBias = c.routerHasExpertBias
            ? try store.mlxArrayCopying(named: "model.layers.\(layer).mlp.gate.expert_bias").asType(.float32)
            : MLXArray.zeros([c.numExperts])
        shared = try store.loadSharedExpert(layer: layer)
        guard routerWeight.shape == [c.numExperts, c.hiddenSize], expertBias.shape == [c.numExperts],
              shared.up.inputSize == c.hiddenSize,
              shared.up.outputSize == c.sharedExpertIntermediateSize else {
            throw M1Error.invalid("Router/shared expert dimensions disagree with config")
        }
        eval(routerWeight, expertBias)
    }

    public func callAsFunction(_ input: MLXArray) throws -> MoEOutput {
        let c = configuration
        guard input.shape == [1, c.hiddenSize],
              [.float32, .float16, .bfloat16].contains(input.dtype) else {
            throw M1Error.invalid("M1 MoE accepts exactly one floating token [1,\(c.hiddenSize)]")
        }
        let x = input.asType(.float32)
        let logits = matmul(x, routerWeight.T)
        let route = BailingGroupedRouter.select(logits: logits, expertBias: expertBias,
            topK: c.expertsPerToken, nGroup: c.nGroup, topkGroup: c.topkGroup,
            normalize: c.normTopkProb, routedScale: c.routedScalingFactor)
        eval(route.indices, route.weights)
        let indices = route.indices.asArray(Int32.self).map(Int.init)
        let weights = route.weights.asArray(Float.self)
        var result = MLXArray.zeros([1, c.hiddenSize])
        for (id, weight) in zip(indices, weights) {
            let expert = try store.loadExpert(layer: layer, expert: id)
            guard expert.up.inputSize == c.hiddenSize, expert.up.outputSize == c.moeIntermediateSize else {
                throw M1Error.invalid("Routed expert dimensions disagree with config")
            }
            result = try result + expert(x) * weight
            eval(result)
        }
        result = try result + shared(x)
        eval(result)
        return MoEOutput(hidden: result, expertIndices: indices, expertWeights: weights)
    }
}
