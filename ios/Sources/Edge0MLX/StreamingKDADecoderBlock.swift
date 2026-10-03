import Edge0Core
import MLX

public final class KDAStateCache {
    fileprivate var qConv: MLXArray?
    fileprivate var kConv: MLXArray?
    fileprivate var vConv: MLXArray?
    fileprivate var recurrent: MLXArray?
    public fileprivate(set) var count = 0

    public init() {}

    func reset() {
        qConv = nil; kConv = nil; vConv = nil; recurrent = nil; count = 0
    }
}

/// Correctness-first one-token Kimi Delta Attention decoder layer.
/// The recurrent update is expressed with MLX array operations; a fused Metal
/// kernel can replace it after numerical parity is established.
public final class StreamingKDADecoderBlock {
    private let c: Edge0Configuration8B
    private let layer: Int
    private let inputNorm: MLXArray
    private let postAttentionNorm: MLXArray
    private let qConvWeight: MLXArray
    private let kConvWeight: MLXArray
    private let vConvWeight: MLXArray
    private let aLog: MLXArray
    private let dtBias: MLXArray
    private let outputNorm: MLXArray
    private let qProj: ParallelLinear
    private let kProj: ParallelLinear
    private let vProj: ParallelLinear
    private let fProj: ParallelLinear
    private let gProj: ParallelLinear
    private let bProj: ParallelLinear
    private let outputProj: ParallelLinear
    private let moe: StreamingMoE?
    private let denseMLP: ParallelExpert?
    public private(set) var prerouterInput: MLXArray?
    public private(set) var prerouterIndices: MLXArray?

    public init(configuration c: Edge0Configuration8B, store: ExpertTensorStore, layer: Int,
                loraStore: ExpertTensorStore? = nil,
                bundleCache: ExpertBundleCache? = nil,
                progress: (String) -> Void = { _ in }) throws {
        guard !c.isMLALayer(layer), layer >= 0, layer < c.hiddenLayers,
              c.kdaSafeGate, c.shortConvKernelSize == 4,
              c.headDim * c.attentionHeads == 2048 else {
            throw M1Error.invalid("KDA decoder requires a KDA layer with the preview safe-gate configuration")
        }
        self.c = c
        self.layer = layer
        let base = "model.layers.\(layer)"
        let attention = base + ".attention"
        inputNorm = try store.mlxArrayCopying(named: base + ".input_layernorm.weight")
        postAttentionNorm = try store.mlxArrayCopying(named: base + ".post_attention_layernorm.weight")
        qConvWeight = try store.mlxArrayCopying(named: attention + ".q_conv1d.weight")
        kConvWeight = try store.mlxArrayCopying(named: attention + ".k_conv1d.weight")
        vConvWeight = try store.mlxArrayCopying(named: attention + ".v_conv1d.weight")
        aLog = try store.mlxArrayCopying(named: attention + ".A_log")
        dtBias = try store.mlxArrayCopying(named: attention + ".dt_bias")
        outputNorm = try store.mlxArrayCopying(named: attention + ".o_norm.weight")
        progress("KDA state tensors loaded")
        qProj = try Self.linear(store, attention + ".q_proj", loraStore)
        kProj = try Self.linear(store, attention + ".k_proj", loraStore)
        vProj = try Self.linear(store, attention + ".v_proj", loraStore)
        fProj = try Self.linear(store, attention + ".f_proj", loraStore)
        gProj = try Self.linear(store, attention + ".g_proj", loraStore)
        bProj = try Self.linear(store, attention + ".b_proj", loraStore)
        outputProj = try Self.linear(store, attention + ".o_proj", loraStore)
        if layer < c.firstKDenseReplace {
            denseMLP = try ParallelExpert(
                base: store.loadDenseMLP(layer: layer), loraStore: loraStore,
                prefix: base + ".mlp")
            moe = nil
            progress("dense MLP loaded")
        } else {
            moe = try StreamingMoE(configuration: c, store: store, layer: layer,
                                   bundleCache: bundleCache)
            denseMLP = nil
            progress("sparse MoE loaded")
        }
        let projection = c.attentionHeads * c.headDim
        guard [qProj, kProj, vProj, fProj, gProj].allSatisfy({
            $0.inputSize == c.hiddenSize && $0.outputSize == projection
        }), bProj.inputSize == c.hiddenSize, bProj.outputSize == c.attentionHeads,
              outputProj.inputSize == projection, outputProj.outputSize == c.hiddenSize,
              qConvWeight.shape == [projection, 1, c.shortConvKernelSize],
              kConvWeight.shape == qConvWeight.shape, vConvWeight.shape == qConvWeight.shape,
              aLog.shape == [c.attentionHeads], dtBias.shape == [projection],
              outputNorm.shape == [c.headDim] else {
            throw M1Error.invalid("M4 KDA tensor dimensions disagree with config")
        }
        eval(inputNorm, postAttentionNorm, qConvWeight, kConvWeight, vConvWeight,
             aLog, dtBias, outputNorm)
        progress("KDA block initialized")
    }

    private static func linear(_ store: ExpertTensorStore, _ prefix: String,
                               _ loraStore: ExpertTensorStore?) throws -> ParallelLinear {
        let base = try QuantizedExpertLinear(weight: store.mlxArrayCopying(named: prefix + ".weight"),
            scales: store.mlxArrayCopying(named: prefix + ".scales"),
            biases: store.mlxArrayCopying(named: prefix + ".biases"))
        return try ParallelLinear(base: base, loraStore: loraStore, prefix: prefix)
    }

    public func callAsFunction(_ input: MLXArray, cache: KDAStateCache,
                               predictedRoute: PredictedMoERoute? = nil) throws -> DecoderBlockOutput {
        guard input.shape == [1, c.hiddenSize] else {
            throw M1Error.invalid("M4 accepts one token [1,\(c.hiddenSize)]")
        }
        let x = input
        let normalized = MLXFast.rmsNorm(x, weight: inputNorm, eps: c.rmsNormEps)
        let attention = try kda(normalized, cache: cache)
        let residual = x + attention
        let moeInput = MLXFast.rmsNorm(residual, weight: postAttentionNorm, eps: c.rmsNormEps)
        let feedForward: MLXArray
        let expertIndices: [Int]
        let expertWeights: [Float]
        if let moe {
            let output = if let predictedRoute {
                try moe.predicted(moeInput, route: predictedRoute)
            } else {
                try moe(moeInput)
            }
            feedForward = output.hidden
            expertIndices = output.expertIndices
            expertWeights = output.expertWeights
            prerouterInput = moeInput
            prerouterIndices = output.routeIndices
        } else if let denseMLP {
            feedForward = try denseMLP(moeInput)
            expertIndices = []
            expertWeights = []
            prerouterInput = nil
            prerouterIndices = nil
        } else {
            throw M1Error.invalid("KDA decoder has no feed-forward block")
        }
        let hidden = residual + feedForward
        cache.count += 1
        return DecoderBlockOutput(hidden: hidden, expertIndices: expertIndices,
                                  expertWeights: expertWeights)
    }

    /// Official multi-token Edge0 8B KDA path used by chunked prefill.
    public func prefill(_ input: MLXArray, cache: KDAStateCache) throws -> DecoderBlockOutput {
        guard input.ndim == 2, input.dim(0) > 1, input.dim(1) == c.hiddenSize else {
            throw M1Error.invalid("KDA prefill accepts [T,\(c.hiddenSize)], T > 1")
        }
        let x = input
        let normalized = MLXFast.rmsNorm(x, weight: inputNorm, eps: c.rmsNormEps)
        let attention = try kdaPrefill(normalized, cache: cache)
        let residual = x + attention
        let moeInput = MLXFast.rmsNorm(residual, weight: postAttentionNorm, eps: c.rmsNormEps)
        let feedForward: MLXArray
        let expertIndices: [Int]
        let expertWeights: [Float]
        if let moe {
            let output = try moe.fullLayerPrefill(moeInput)
            feedForward = output.hidden
            expertIndices = output.expertIndices
            expertWeights = output.expertWeights
            prerouterInput = moeInput
            prerouterIndices = output.routeIndices
        } else if let denseMLP {
            feedForward = try denseMLP(moeInput)
            expertIndices = []
            expertWeights = []
        } else {
            throw M1Error.invalid("KDA decoder has no feed-forward block")
        }
        let hidden = residual + feedForward
        cache.count += input.dim(0)
        return DecoderBlockOutput(hidden: hidden, expertIndices: expertIndices,
                                  expertWeights: expertWeights)
    }

    public func loadFullExpertLayer() throws { try moe?.loadFullLayer() }
    public func clearFullExpertLayer() { moe?.clearFullLayer() }
    public func stage(experts: [Int]) throws { try moe?.stage(experts: experts) }
    public func resetStaging() { moe?.resetStaging() }
    public var moeStats: StreamingMoEStats? { moe?.stats }

    private func convStep(_ x: MLXArray, weight: MLXArray,
                          state: inout MLXArray?) -> MLXArray {
        let channels = c.attentionHeads * c.headDim
        let history = state ?? MLXArray.zeros(
            [c.shortConvKernelSize - 1, channels], dtype: x.dtype)
        let window = concatenated([history, x], axis: 0)
        let kernel = weight.reshaped([channels, c.shortConvKernelSize]).T
        let output = (window * kernel).sum(axis: 0, keepDims: true)
        state = split(window, indices: [1], axis: 0)[1]
        return output * sigmoid(output)
    }

    private func kda(_ x: MLXArray, cache: KDAStateCache) throws -> MLXArray {
        let heads = c.attentionHeads
        let dimension = c.headDim
        let qConv = convStep(try qProj(x), weight: qConvWeight, state: &cache.qConv)
        let kConv = convStep(try kProj(x), weight: kConvWeight, state: &cache.kConv)
        let vConv = convStep(try vProj(x), weight: vConvWeight, state: &cache.vConv)
        var q = qConv.reshaped([1, heads, dimension]).asType(.float32)
        var k = kConv.reshaped([1, heads, dimension]).asType(.float32)
        let v = vConv.reshaped([1, heads, dimension]).asType(.float32)
        let qNorm = sqrt((q * q).sum(axis: -1, keepDims: true)) + 1e-6
        let kNorm = sqrt((k * k).sum(axis: -1, keepDims: true)) + 1e-6
        q = q * (1.0 / Float(dimension).squareRoot()) / qNorm
        k = k / kNorm

        let f = try fProj(x).reshaped([1, heads, dimension]).asType(.float32)
        let gateInput = exp(aLog.asType(.float32)).reshaped([1, heads, 1])
            * (f + dtBias.asType(.float32).reshaped([1, heads, dimension]))
        let decay = exp(Float(c.kdaLowerBound) * sigmoid(gateInput)).expandedDimensions(axis: 2)
        let beta = sigmoid(try bProj(x).asType(.float32)).expandedDimensions(axis: -1)
        var state = cache.recurrent ?? MLXArray.zeros([1, heads, dimension, dimension])
        state = state * decay
        let kRows = k.expandedDimensions(axis: 2)
        let memory = (state * kRows).sum(axis: -1)
        let delta = (v - memory) * beta
        state = state + kRows * delta.expandedDimensions(axis: -1)
        var output = (state * q.expandedDimensions(axis: 2)).sum(axis: -1)
        cache.recurrent = state

        output = MLXFast.rmsNorm(
            output.asType(x.dtype), weight: outputNorm, eps: c.rmsNormEps)
        let outputGate = sigmoid(try gProj(x).reshaped([1, heads, dimension]))
        return try outputProj((output * outputGate).reshaped([1, heads * dimension]))
    }

    private func convPrefill(_ x: MLXArray, weight: MLXArray,
                             state: inout MLXArray?) -> MLXArray {
        let channels = c.attentionHeads * c.headDim
        let history = state ?? MLXArray.zeros(
            [c.shortConvKernelSize - 1, channels], dtype: x.dtype)
        let convInput = concatenated([history, x], axis: 0)
        let kernel = weight.transposed(0, 2, 1)
        let output = conv1d(
            convInput.expandedDimensions(axis: 0), kernel,
            groups: channels).squeezed(axis: 0)
        let stateStart = convInput.dim(0) - (c.shortConvKernelSize - 1)
        state = split(convInput, indices: [stateStart], axis: 0)[1]
        return output * sigmoid(output)
    }

    private func kdaPrefill(_ x: MLXArray, cache: KDAStateCache) throws -> MLXArray {
        let tokens = x.dim(0)
        let heads = c.attentionHeads
        let dimension = c.headDim
        let qConv = convPrefill(try qProj(x), weight: qConvWeight, state: &cache.qConv)
        let kConv = convPrefill(try kProj(x), weight: kConvWeight, state: &cache.kConv)
        let vConv = convPrefill(try vProj(x), weight: vConvWeight, state: &cache.vConv)
        var q = qConv.reshaped([1, tokens, heads, dimension]).asType(.float32)
        var k = kConv.reshaped([1, tokens, heads, dimension]).asType(.float32)
        let v = vConv.reshaped([1, tokens, heads, dimension]).asType(.float32)
        let qNorm = sqrt((q * q).sum(axis: -1, keepDims: true)) + 1e-6
        let kNorm = sqrt((k * k).sum(axis: -1, keepDims: true)) + 1e-6
        q = q * (1.0 / Float(dimension).squareRoot()) / qNorm
        k = k / kNorm

        let f = try fProj(x).reshaped([1, tokens, heads, dimension])
            .asType(.float32)
        let gateInput = exp(aLog.asType(.float32)).reshaped([1, 1, heads, 1])
            * (f + dtBias.asType(.float32)
                .reshaped([1, 1, heads, dimension]))
        let decay = exp(Float(c.kdaLowerBound) * sigmoid(gateInput))
        let beta = sigmoid(try bProj(x).asType(.float32))
            .reshaped([1, tokens, heads])
        let initialState = cache.recurrent
            ?? MLXArray.zeros([1, heads, dimension, dimension])
        let update = try OfficialGatedDelta.callAsFunction(
            q: q, k: k, v: v, decay: decay, beta: beta,
            state: initialState)
        cache.recurrent = update.state
        var output = MLXFast.rmsNorm(
            update.output.asType(x.dtype), weight: outputNorm,
            eps: c.rmsNormEps)
        let outputGate = sigmoid(
            try gProj(x).reshaped([1, tokens, heads, dimension]))
        output = output * outputGate
        return try outputProj(output.reshaped([tokens, heads * dimension]))
    }
}
