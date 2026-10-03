import Edge0Core
import Foundation
import MLX

public struct DecoderBlockOutput {
    public let hidden: MLXArray
    public let expertIndices: [Int]
    public let expertWeights: [Float]
}

public final class MLAKVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var count = 0

    public init() {}

    func reset() { keys = nil; values = nil; count = 0 }

    fileprivate func append(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let allKeys: MLXArray
        let allValues: MLXArray
        if let keys, let values {
            allKeys = concatenated([keys, newKeys], axis: 2)
            allValues = concatenated([values, newValues], axis: 2)
        } else {
            allKeys = newKeys
            allValues = newValues
        }
        keys = allKeys
        values = allValues
        count += newKeys.dim(2)
        return (allKeys, allValues)
    }
}

/// Correctness-first implementation of one Edge0 MLA decoder layer for a
/// single decode token. The current token attends to itself; `position`
/// exercises the checkpoint's interleaved RoPE convention. KV persistence is
/// intentionally the next milestone.
public final class StreamingMLADecoderBlock {
    private let configuration: Edge0Configuration8B
    private let layer: Int
    private let inputNorm: MLXArray
    private let postAttentionNorm: MLXArray
    private let qANorm: MLXArray
    private let kvANorm: MLXArray
    private let qA: ParallelLinear
    private let qB: ParallelLinear
    private let kvA: ParallelLinear
    private let kvB: ParallelLinear
    private let gate: ParallelLinear
    private let dense: ParallelLinear
    private let moe: StreamingMoE
    public private(set) var prerouterInput: MLXArray?
    public private(set) var prerouterIndices: MLXArray?

    public init(configuration c: Edge0Configuration8B, store: ExpertTensorStore, layer: Int,
                loraStore: ExpertTensorStore? = nil,
                bundleCache: ExpertBundleCache? = nil,
                progress: (String) -> Void = { _ in }) throws {
        let queryHeadDim = c.qkNopeHeadDim + c.qkRopeHeadDim
        guard c.isMLALayer(layer), c.ropeInterleave,
              c.qkRopeHeadDim > 0, c.qkRopeHeadDim % 2 == 0,
              c.attentionHeads == c.keyValueHeads else {
            throw M1Error.invalid("M2 requires an interleaved-RoPE MLA layer with matching Q/KV heads")
        }
        configuration = c
        self.layer = layer
        let base = "model.layers.\(layer)"
        inputNorm = try store.mlxArrayCopying(named: base + ".input_layernorm.weight")
        postAttentionNorm = try store.mlxArrayCopying(named: base + ".post_attention_layernorm.weight")
        qANorm = try store.mlxArrayCopying(named: base + ".attention.q_a_layernorm.weight")
        kvANorm = try store.mlxArrayCopying(named: base + ".attention.kv_a_layernorm.weight")
        progress("norm tensors loaded")
        qA = try Self.linear(store, base + ".attention.q_a_proj", nil)
        progress("q_a loaded")
        qB = try Self.linear(store, base + ".attention.q_b_proj", loraStore)
        progress("q_b loaded")
        kvA = try Self.linear(store, base + ".attention.kv_a_proj_with_mqa", nil)
        progress("kv_a loaded")
        kvB = try Self.linear(store, base + ".attention.kv_b_proj", loraStore)
        progress("kv_b loaded")
        gate = try Self.linear(store, base + ".attention.g_proj", loraStore)
        progress("gate loaded")
        dense = try Self.linear(store, base + ".attention.dense", loraStore)
        progress("dense loaded")
        moe = try StreamingMoE(configuration: c, store: store, layer: layer,
                               bundleCache: bundleCache)
        progress("moe loaded")
        guard qA.inputSize == c.hiddenSize, qA.outputSize == c.qLoraRank,
              qB.inputSize == c.qLoraRank, qB.outputSize == c.attentionHeads * queryHeadDim,
              kvA.inputSize == c.hiddenSize, kvA.outputSize == c.kvLoraRank + c.qkRopeHeadDim,
              kvB.inputSize == c.kvLoraRank,
              kvB.outputSize == c.attentionHeads * (c.qkNopeHeadDim + c.vHeadDim),
              gate.outputSize == c.attentionHeads,
              dense.inputSize == c.attentionHeads * c.vHeadDim,
              dense.outputSize == c.hiddenSize else {
            throw M1Error.invalid("M2 attention tensor dimensions disagree with config")
        }
        eval(inputNorm, postAttentionNorm, qANorm, kvANorm)
        progress("norm tensors evaluated")
    }

    private static func linear(_ store: ExpertTensorStore, _ prefix: String,
                               _ loraStore: ExpertTensorStore?) throws -> ParallelLinear {
        let base = try QuantizedExpertLinear(
            weight: store.mlxArrayCopying(named: prefix + ".weight"),
            scales: store.mlxArrayCopying(named: prefix + ".scales"),
            biases: store.mlxArrayCopying(named: prefix + ".biases")
        )
        return try ParallelLinear(base: base, loraStore: loraStore, prefix: prefix)
    }

    public func callAsFunction(_ input: MLXArray, position: Int,
                               cache: MLAKVCache? = nil,
                               predictedRoute: PredictedMoERoute? = nil) throws -> DecoderBlockOutput {
        let c = configuration
        guard input.shape == [1, c.hiddenSize], position >= 0,
              cache == nil || cache?.count == position else {
            throw M1Error.invalid("M2 accepts one token [1,\(c.hiddenSize)] and a nonnegative position")
        }
        let normalized = MLXFast.rmsNorm(input, weight: inputNorm, eps: c.rmsNormEps)
        let attention = try selfAttention(normalized, position: position, cache: cache)
        let residual = input + attention
        let moeInput = MLXFast.rmsNorm(residual, weight: postAttentionNorm, eps: c.rmsNormEps)
        let moeOutput = if let predictedRoute {
            try moe.predicted(moeInput, route: predictedRoute)
        } else {
            try moe(moeInput)
        }
        prerouterInput = moeInput
        prerouterIndices = moeOutput.routeIndices
        let hidden = residual + moeOutput.hidden
        return DecoderBlockOutput(hidden: hidden, expertIndices: moeOutput.expertIndices,
                                  expertWeights: moeOutput.expertWeights)
    }

    /// Official multi-token Edge0 8B MLA path used by chunked prefill.
    public func prefill(_ input: MLXArray, position: Int,
                        cache: MLAKVCache) throws -> DecoderBlockOutput {
        let c = configuration
        guard input.ndim == 2, input.dim(0) > 1, input.dim(1) == c.hiddenSize,
              position >= 0, cache.count == position else {
            throw M1Error.invalid("MLA prefill accepts [T,\(c.hiddenSize)], T > 1")
        }
        let x = input
        let normalized = MLXFast.rmsNorm(x, weight: inputNorm, eps: c.rmsNormEps)
        let attention = try selfAttentionPrefill(
            normalized, position: position, cache: cache)
        let residual = x + attention
        let moeInput = MLXFast.rmsNorm(
            residual, weight: postAttentionNorm, eps: c.rmsNormEps)
        let moeOutput = try moe.fullLayerPrefill(moeInput)
        prerouterInput = moeInput
        prerouterIndices = moeOutput.routeIndices
        let hidden = residual + moeOutput.hidden
        return DecoderBlockOutput(hidden: hidden,
                                  expertIndices: moeOutput.expertIndices,
                                  expertWeights: moeOutput.expertWeights)
    }

    public func loadFullExpertLayer() throws { try moe.loadFullLayer() }
    public func clearFullExpertLayer() { moe.clearFullLayer() }
    public func stage(experts: [Int]) throws { try moe.stage(experts: experts) }
    public func resetStaging() { moe.resetStaging() }
    public var moeStats: StreamingMoEStats { moe.stats }

    private func selfAttention(_ x: MLXArray, position: Int, cache: MLAKVCache?) throws -> MLXArray {
        let c = configuration
        let queryHeadDim = c.qkNopeHeadDim + c.qkRopeHeadDim
        let qLatent = MLXFast.rmsNorm(try qA(x), weight: qANorm, eps: c.rmsNormEps)
        let q = try qB(qLatent).reshaped([1, 1, c.attentionHeads, queryHeadDim])
            .transposed(0, 2, 1, 3)
        let qParts = split(q, indices: [c.qkNopeHeadDim], axis: -1)

        let compressedParts = split(try kvA(x), indices: [c.kvLoraRank], axis: -1)
        let kvLatent = MLXFast.rmsNorm(compressedParts[0], weight: kvANorm, eps: c.rmsNormEps)
        let kv = try kvB(kvLatent)
            .reshaped([1, 1, c.attentionHeads, c.qkNopeHeadDim + c.vHeadDim])
            .transposed(0, 2, 1, 3)
        let kvParts = split(kv, indices: [c.qkNopeHeadDim], axis: -1)
        var kRope = compressedParts[1].reshaped([1, 1, 1, c.qkRopeHeadDim])

        let qRope = interleavedRope(qParts[1], position: position)
        kRope = interleavedRope(kRope, position: position)
        kRope = broadcast(kRope, to: [1, c.attentionHeads, 1, c.qkRopeHeadDim])
        let queries = concatenated([qParts[0], qRope], axis: -1)
        var keys = concatenated([kvParts[0], kRope], axis: -1)
        var values = kvParts[1]
        if let cache {
            (keys, values) = cache.append(keys: keys, values: values)
        }
        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1.0 / Float(queryHeadDim).squareRoot(), mask: .none
        )
        output = output.transposed(0, 2, 1, 3).reshaped([1, c.attentionHeads, c.vHeadDim])
        let gates = sigmoid(try gate(x)).reshaped([1, c.attentionHeads, 1])
        output = (output * gates).reshaped([1, c.attentionHeads * c.vHeadDim])
        return try dense(output)
    }

    private func interleavedRope(_ x: MLXArray, position: Int) -> MLXArray {
        interleavedRope(x, startPosition: position, tokenCount: 1)
    }

    private func selfAttentionPrefill(_ x: MLXArray, position: Int,
                                      cache: MLAKVCache) throws -> MLXArray {
        let c = configuration
        let tokens = x.dim(0)
        let queryHeadDim = c.qkNopeHeadDim + c.qkRopeHeadDim
        let qLatent = MLXFast.rmsNorm(
            try qA(x), weight: qANorm, eps: c.rmsNormEps)
        let q = try qB(qLatent)
            .reshaped([1, tokens, c.attentionHeads, queryHeadDim])
            .transposed(0, 2, 1, 3)
        let qParts = split(q, indices: [c.qkNopeHeadDim], axis: -1)

        let compressedParts = split(
            try kvA(x), indices: [c.kvLoraRank], axis: -1)
        let kvLatent = MLXFast.rmsNorm(
            compressedParts[0], weight: kvANorm, eps: c.rmsNormEps)
        let kv = try kvB(kvLatent)
            .reshaped([1, tokens, c.attentionHeads,
                       c.qkNopeHeadDim + c.vHeadDim])
            .transposed(0, 2, 1, 3)
        let kvParts = split(kv, indices: [c.qkNopeHeadDim], axis: -1)
        var kRope = compressedParts[1]
            .reshaped([1, 1, tokens, c.qkRopeHeadDim])

        let qRope = interleavedRope(
            qParts[1], startPosition: position, tokenCount: tokens)
        kRope = interleavedRope(
            kRope, startPosition: position, tokenCount: tokens)
        kRope = broadcast(
            kRope,
            to: [1, c.attentionHeads, tokens, c.qkRopeHeadDim])
        let queries = concatenated([qParts[0], qRope], axis: -1)
        let newKeys = concatenated([kvParts[0], kRope], axis: -1)
        let (keys, values) = cache.append(keys: newKeys, values: kvParts[1])
        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1.0 / Float(queryHeadDim).squareRoot(), mask: .causal)
        output = output.transposed(0, 2, 1, 3)
            .reshaped([tokens, c.attentionHeads, c.vHeadDim])
        let gates = sigmoid(try gate(x))
            .reshaped([tokens, c.attentionHeads, 1])
        output = (output * gates)
            .reshaped([tokens, c.attentionHeads * c.vHeadDim])
        return try dense(output)
    }

    private func interleavedRope(_ x: MLXArray, startPosition: Int,
                                 tokenCount: Int) -> MLXArray {
        let pairs = configuration.qkRopeHeadDim / 2
        let angles = (0..<tokenCount).flatMap { token in
            (0..<pairs).map { i in
                Float(startPosition + token)
                    / pow(configuration.ropeTheta,
                          Float(2 * i) / Float(configuration.qkRopeHeadDim))
            }
        }
        let cosines = MLXArray(
            angles.map { cos($0) }, [1, 1, tokenCount, pairs, 1])
        let sines = MLXArray(
            angles.map { sin($0) }, [1, 1, tokenCount, pairs, 1])
        let pairParts = split(
            x.reshaped([1, x.dim(1), tokenCount, pairs, 2]),
            indices: [1], axis: -1)
        let even = pairParts[0]
        let odd = pairParts[1]
        return concatenated(
            [even * cosines - odd * sines, even * sines + odd * cosines],
            axis: -1)
            .reshaped(x.shape)
    }
}
