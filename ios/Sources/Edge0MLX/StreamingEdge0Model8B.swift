import Edge0Core
import Foundation
import MLX

public struct StreamingModelOutput {
    public let logits: MLXArray
    public let activeMemory: Int
    public let peakMemory: Int
    public let cacheMemory: Int
}

private enum StreamingLayer {
    case kda(StreamingKDADecoderBlock, KDAStateCache)
    case mla(StreamingMLADecoderBlock, MLAKVCache)

    func callAsFunction(_ hidden: MLXArray,
                        predictedRoute: PredictedMoERoute? = nil) throws -> MLXArray {
        switch self {
        case .kda(let block, let cache):
            return try block(hidden, cache: cache, predictedRoute: predictedRoute).hidden
        case .mla(let block, let cache):
            return try block(hidden, position: cache.count, cache: cache,
                             predictedRoute: predictedRoute).hidden
        }
    }

    var prerouterFeature: LingPrerouterFeature? {
        switch self {
        case .kda(let block, _):
            guard let input = block.prerouterInput,
                  let indices = block.prerouterIndices else { return nil }
            return LingPrerouterFeature(input: input, routeIndices: indices)
        case .mla(let block, _):
            guard let input = block.prerouterInput,
                  let indices = block.prerouterIndices else { return nil }
            return LingPrerouterFeature(input: input, routeIndices: indices)
        }
    }

    func stage(experts: [Int]) throws {
        switch self {
        case .kda(let block, _): try block.stage(experts: experts)
        case .mla(let block, _): try block.stage(experts: experts)
        }
    }

    func reset() {
        switch self {
        case .kda(let block, let cache):
            cache.reset(); block.resetStaging()
        case .mla(let block, let cache):
            cache.reset(); block.resetStaging()
        }
    }

    func prefill(_ hidden: MLXArray) throws -> MLXArray {
        switch self {
        case .kda(let block, let cache):
            return try block.prefill(hidden, cache: cache).hidden
        case .mla(let block, let cache):
            return try block.prefill(
                hidden, position: cache.count, cache: cache).hidden
        }
    }

    func loadFullExpertLayer() throws {
        switch self {
        case .kda(let block, _): try block.loadFullExpertLayer()
        case .mla(let block, _): try block.loadFullExpertLayer()
        }
    }

    func clearFullExpertLayer() {
        switch self {
        case .kda(let block, _): block.clearFullExpertLayer()
        case .mla(let block, _): block.clearFullExpertLayer()
        }
    }

    var moeStats: StreamingMoEStats? {
        switch self {
        case .kda(let block, _): block.moeStats
        case .mla(let block, _): block.moeStats
        }
    }
}

public struct StreamingRuntimeStats: Sendable {
    public let layers: [StreamingMoEStats]
    public let cacheHits: Int
    public let cacheMisses: Int
    public let cachedExperts: Int
}

/// One-token exact-streaming Edge0 model.
///
/// Resident tensors are limited to the non-routed backbone, routers, and shared
/// experts. Routed experts remain in the checkpoint mmap and are copied only for
/// the selected top-8, matching the official repository's memory contract. MLX's
/// reusable free-buffer cache is capped at 256 MiB, also matching the official
/// engine default.
public final class StreamingEdge0Model8B {
    private let configuration: Edge0Configuration8B
    private let store: ExpertTensorStore
    private let layers: [StreamingLayer]
    private let finalNorm: MLXArray
    private let lmHead: QuantizedExpertLinear
    private let prerouter: LingPrerouter?
    private let bundleCache: ExpertBundleCache
    private var predictedRoutes: [Int: PredictedMoERoute] = [:]

    public init(configuration c: Edge0Configuration8B, store: ExpertTensorStore,
                cacheLimitBytes: Int = 256 * 1024 * 1024,
                loraWeightsURL: URL? = nil,
                prerouterWeightsURL: URL? = nil,
                progress: (String) -> Void = { _ in }) throws {
        guard c.quantization.bits == 4, c.quantization.groupSize == 64,
              c.quantization.mode == "affine" else {
            throw M1Error.invalid("Full model requires affine INT4/group-64 weights")
        }
        Memory.cacheLimit = cacheLimitBytes
        configuration = c
        self.store = store
        finalNorm = try store.mlxArrayCopying(named: "model.norm.weight")
        lmHead = try QuantizedExpertLinear(
            weight: store.mlxArrayCopying(named: "lm_head.weight"),
            scales: store.mlxArrayCopying(named: "lm_head.scales"),
            biases: store.mlxArrayCopying(named: "lm_head.biases"))
        guard finalNorm.shape == [c.hiddenSize], lmHead.inputSize == c.hiddenSize,
              lmHead.outputSize == c.vocabularySize else {
            throw M1Error.invalid("Final norm or lm-head dimensions disagree with config")
        }
        progress("final norm and lm-head loaded")

        let bundleCache = ExpertBundleCache(capacity: 64)
        self.bundleCache = bundleCache
        let loraStore = try loraWeightsURL.map {
            try ExpertTensorStore(modelURL: $0, expertCount: c.numExperts)
        }
        var built: [StreamingLayer] = []
        built.reserveCapacity(c.hiddenLayers)
        for layer in 0..<c.hiddenLayers {
            if c.isMLALayer(layer) {
                let block = try StreamingMLADecoderBlock(
                    configuration: c, store: store, layer: layer,
                    loraStore: loraStore,
                    bundleCache: bundleCache)
                built.append(.mla(block, MLAKVCache()))
            } else {
                let block = try StreamingKDADecoderBlock(
                    configuration: c, store: store, layer: layer,
                    loraStore: loraStore,
                    bundleCache: bundleCache)
                built.append(.kda(block, KDAStateCache()))
            }
            progress("layer \(layer) resident backbone loaded")
        }
        layers = built
        if let prerouterWeightsURL {
            prerouter = try LingPrerouter(configuration: c,
                                         weightsURL: prerouterWeightsURL)
            progress("Ling prerouter loaded")
        } else {
            prerouter = nil
        }
        eval(finalNorm)
        progress("24-layer model initialized")
    }

    public func callAsFunction(tokenID: Int,
                               afterLayer: (Int, MLXArray) -> Void = { _, _ in }) throws -> StreamingModelOutput {
        let c = configuration
        guard tokenID >= 0, tokenID < c.vocabularySize else {
            throw M1Error.invalid("Token ID out of vocabulary range")
        }
        var hidden = try embedding(tokenID: tokenID)
        var features: [Int: LingPrerouterFeature] = [:]
        for (index, layer) in layers.enumerated() {
            hidden = try layer(hidden, predictedRoute: predictedRoutes[index])
            if (7...22).contains(index), let feature = layer.prerouterFeature {
                features[index] = feature
            }
            afterLayer(index, hidden)
        }
        hidden = MLXFast.rmsNorm(hidden, weight: finalNorm, eps: c.rmsNormEps)
        let logits = try lmHead(hidden)
        eval(logits)
        try updateStagedRoutes(features)
        let memory = Memory.snapshot()
        return StreamingModelOutput(logits: logits,
                                    activeMemory: memory.activeMemory,
                                    peakMemory: memory.peakMemory,
                                    cacheMemory: memory.cacheMemory)
    }

    /// Official chunked prefill path: each sparse layer's complete expert
    /// stack is loaded immediately before that layer and dropped when the
    /// next layer begins. KDA and MLA process the full token chunk at once.
    public func prefill(tokenIDs: [Int],
                        afterLayer: (Int, MLXArray) -> Void = { _, _ in }) throws
        -> StreamingModelOutput {
        let c = configuration
        guard tokenIDs.count > 1,
              tokenIDs.allSatisfy({ (0..<c.vocabularySize).contains($0) }) else {
            throw M1Error.invalid("Chunked prefill requires at least two valid token IDs")
        }
        defer { layers.forEach { $0.clearFullExpertLayer() } }

        var hidden = try embedding(tokenIDs: tokenIDs)
        var features: [Int: LingPrerouterFeature] = [:]
        for (index, layer) in layers.enumerated() {
            if index > 0 { layers[index - 1].clearFullExpertLayer() }
            try layer.loadFullExpertLayer()
            hidden = try layer.prefill(hidden)
            asyncEval(hidden)
            if (7...22).contains(index), let feature = layer.prerouterFeature {
                features[index] = feature
            }
            afterLayer(index, hidden)
        }
        hidden = MLXFast.rmsNorm(hidden, weight: finalNorm, eps: c.rmsNormEps)
        let last = split(hidden, indices: [hidden.dim(0) - 1], axis: 0)[1]
        let logits = try lmHead(last)
        eval(logits)
        try updateStagedRoutes(features)
        let memory = Memory.snapshot()
        return StreamingModelOutput(logits: logits,
                                    activeMemory: memory.activeMemory,
                                    peakMemory: memory.peakMemory,
                                    cacheMemory: memory.cacheMemory)
    }

    private func embedding(tokenID: Int) throws -> MLXArray {
        let prefix = "model.word_embeddings"
        let weight = try store.mlxAxisZeroSliceCopying(named: prefix + ".weight", index: tokenID)
            .expandedDimensions(axis: 0)
        let scales = try store.mlxAxisZeroSliceCopying(named: prefix + ".scales", index: tokenID)
            .expandedDimensions(axis: 0)
        let biases = try store.mlxAxisZeroSliceCopying(named: prefix + ".biases", index: tokenID)
            .expandedDimensions(axis: 0)
        let hidden = dequantized(weight, scales: scales, biases: biases,
                                 groupSize: 64, bits: 4, mode: .affine,
                                 dtype: .bfloat16)
        guard hidden.shape == [1, configuration.hiddenSize] else {
            throw M1Error.invalid("Embedding row has an unexpected shape")
        }
        return hidden
    }

    /// Official step boundary: owner layers 7...22 predict consumers 8...23,
    /// then their bundles are staged before the next token begins.
    private func updateStagedRoutes(_ features: [Int: LingPrerouterFeature]) throws {
        guard let prerouter else { return }
        let next = try prerouter.predict(features: features)
        for (consumer, route) in next {
            try layers[consumer].stage(experts: route.expertIndices)
        }
        predictedRoutes = next
    }

    private func embedding(tokenIDs: [Int]) throws -> MLXArray {
        try concatenated(tokenIDs.map { try embedding(tokenID: $0) }, axis: 0)
    }

    public func reset() {
        layers.forEach { $0.reset() }
        predictedRoutes.removeAll(keepingCapacity: true)
        prerouter?.reset()
    }

    public var stats: StreamingRuntimeStats {
        let cache = bundleCache.stats
        return StreamingRuntimeStats(
            layers: layers.compactMap(\.moeStats), cacheHits: cache.hits,
            cacheMisses: cache.misses, cachedExperts: cache.size)
    }

    /// Official Ling startup warm-up: request sequential pages, execute a
    /// tiny prompt plus one decode step, then clear only sequence state.
    ///
    /// The four-token warm-up uses the top-8 decode path. Full-layer prefill
    /// of four tokens loads every expert and is the opposite of a warm-up.
    public func prewarm(tokenID: Int = 0) throws {
        store.mappedFile.adviseWillNeed()
        store.mappedFile.adviseSequential()
        store.mappedFile.sequentialRead()
        reset()
        for _ in 0..<4 {
            _ = try callAsFunction(tokenID: tokenID)
        }
        _ = try callAsFunction(tokenID: tokenID)
        reset()
    }

}
