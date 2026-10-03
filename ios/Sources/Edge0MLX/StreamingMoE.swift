import Edge0Core
import Foundation
import MLX

public struct MoEOutput {
    public let hidden: MLXArray
    public let expertIndices: [Int]
    public let expertWeights: [Float]
    public let routeIndices: MLXArray
}

public struct PredictedMoERoute: Sendable {
    public let expertIndices: [Int]
    public let expertWeights: [Float]

    public init(expertIndices: [Int], expertWeights: [Float]) {
        self.expertIndices = expertIndices
        self.expertWeights = expertWeights
    }
}

/// Official-style whole-model expert-bundle LRU. The production profile uses
/// 64 slots globally, not per layer.
public final class ExpertBundleCache: @unchecked Sendable {
    private let capacity: Int
    private var values: [String: QuantizedExpert] = [:]
    private var order: [String] = []
    private let lock = NSLock()
    private var hitCount = 0
    private var missCount = 0

    public init(capacity: Int = 64) { self.capacity = max(0, capacity) }

    func get(layer: Int, expert: Int) -> QuantizedExpert? {
        lock.lock(); defer { lock.unlock() }
        let key = "\(layer):\(expert)"
        guard let value = values[key] else {
            missCount += 1
            return nil
        }
        hitCount += 1
        order.removeAll { $0 == key }
        order.append(key)
        return value
    }

    func put(_ value: QuantizedExpert, layer: Int, expert: Int) {
        lock.lock(); defer { lock.unlock() }
        guard capacity > 0 else { return }
        let key = "\(layer):\(expert)"
        values[key] = value
        order.removeAll { $0 == key }
        order.append(key)
        while order.count > capacity {
            values.removeValue(forKey: order.removeFirst())
        }
    }

    public var stats: (hits: Int, misses: Int, size: Int) {
        lock.lock(); defer { lock.unlock() }
        return (hitCount, missCount, values.count)
    }
}

public struct StreamingMoEStats: Sendable {
    public let calls: Int
    public let loads: Int
    public let hits: Int
    public let stagedUsed: Int
    public let stagedFallback: Int
    public let stageBuilds: Int
    public let stageCacheHits: Int
    public let stageWallSeconds: TimeInterval
}

private final class ExpertLoadSlots: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [QuantizedExpert?]
    private var firstError: Error?

    init(count: Int) {
        values = Array(repeating: nil, count: count)
    }

    func store(_ result: Result<QuantizedExpert, Error>, at index: Int) {
        lock.lock(); defer { lock.unlock() }
        switch result {
        case .success(let expert): values[index] = expert
        case .failure(let error): if firstError == nil { firstError = error }
        }
    }

    func finish() throws -> [QuantizedExpert] {
        lock.lock(); defer { lock.unlock() }
        if let firstError { throw firstError }
        guard values.allSatisfy({ $0 != nil }) else {
            throw M1Error.invalid("Parallel expert load did not fill every slot")
        }
        return values.map { $0! }
    }
}

/// One decode token's feed-forward block, including the unconditional shared expert.
/// It includes the official gathered exact path, global cache and prerouter staging paths.
public final class StreamingMoE: @unchecked Sendable {
    private let configuration: Edge0Configuration8B
    private let store: ExpertTensorStore
    private let layer: Int
    private let routerWeight: MLXArray
    private let expertBias: MLXArray
    private let shared: QuantizedExpert
    private let bundleCache: ExpertBundleCache?
    private var fullLayer: QuantizedExpertStack?
    private var stagedBundles: [Int: QuantizedExpert] = [:]
    private var stagedStack: QuantizedExpertStack?
    private var stagedOrder: [Int] = []
    private let stagingLock = NSLock()
    private var pendingOrder: [Int] = []
    private var pendingBundles: [QuantizedExpert]?
    private var pendingError: Error?
    private var calls = 0
    private var loads = 0
    private var hits = 0
    private var stagedUsed = 0
    private var stagedFallback = 0
    private var stageBuilds = 0
    private var stageCacheHits = 0
    private var stageWall: TimeInterval = 0
    private let compiledStagedMath: @Sendable ([MLXArray]) -> [MLXArray]

    public init(configuration c: Edge0Configuration8B, store: ExpertTensorStore,
                layer: Int, bundleCache: ExpertBundleCache? = nil) throws {
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
        self.bundleCache = bundleCache
        compiledStagedMath = compile { arrays in
            let x = arrays[0], local = arrays[1]
            let up = gatherQuantizedMM(
                x, arrays[2], scales: arrays[3], biases: arrays[4],
                rhsIndices: local, transpose: true,
                groupSize: 64, bits: 4, mode: .affine,
                sortedIndices: false)
            let gate = gatherQuantizedMM(
                x, arrays[5], scales: arrays[6], biases: arrays[7],
                rhsIndices: local, transpose: true,
                groupSize: 64, bits: 4, mode: .affine,
                sortedIndices: false)
            let activated = (gate * sigmoid(gate)) * up
            return [gatherQuantizedMM(
                activated, arrays[8], scales: arrays[9], biases: arrays[10],
                rhsIndices: local, transpose: true,
                groupSize: 64, bits: 4, mode: .affine,
                sortedIndices: false)]
        }
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
        stagingLock.lock(); calls += 1; stagingLock.unlock()
        let c = configuration
        guard input.shape == [1, c.hiddenSize],
              [.float32, .float16, .bfloat16].contains(input.dtype) else {
            throw M1Error.invalid("M1 MoE accepts exactly one floating token [1,\(c.hiddenSize)]")
        }
        let x = input
        let route = teacherRoute(x.asType(.float32))
        eval(route.indices, route.weights)
        let indices = route.indices.asArray(Int32.self).map(Int.init)
        let weights = route.weights.asArray(Float.self)
        let result = try executeGathered(x, indices: indices, weights: weights)
        return MoEOutput(hidden: result, expertIndices: indices,
                         expertWeights: weights, routeIndices: route.indices)
    }

    /// Consumer-layer path from the official Ling prerouter. Current
    /// production configuration feeds the executed top-k to the next head.
    public func predicted(_ input: MLXArray, route: PredictedMoERoute) throws -> MoEOutput {
        stagingLock.lock(); calls += 1; stagingLock.unlock()
        try activatePendingStage(for: route.expertIndices)
        let x = input
        let hidden: MLXArray
        if Set(route.expertIndices) == Set(stagedOrder), let stagedStack {
            stagingLock.lock(); stagedUsed += route.expertIndices.count; stagingLock.unlock()
            hidden = try executeStaged(
                x, stack: stagedStack, indices: route.expertIndices,
                weights: route.expertWeights)
        } else {
            stagingLock.lock(); stagedFallback += 1; stagingLock.unlock()
            hidden = try executeGathered(
                x, indices: route.expertIndices, weights: route.expertWeights)
        }
        // Current upstream Edge0-8B uses feature_topk="executed": the
        // prerouter's selected set feeds the next head directly. Do not
        // recompute the original gate on consumer layers.
        let executed = MLXArray(
            route.expertIndices.map(Int32.init),
            [1, route.expertIndices.count])
        return MoEOutput(hidden: hidden, expertIndices: route.expertIndices,
                         expertWeights: route.expertWeights,
                         routeIndices: executed)
    }

    /// Official `prod_k8` next-state submission (`staged_sync=True`): fill the
    /// NEXT set synchronously and promote it only at the next consume boundary.
    public func stage(experts: [Int]) throws {
        let ordered = Array(Set(experts)).sorted()
        let started = Date()
        var cacheHits = 0
        var bundles = [QuantizedExpert?](repeating: nil, count: ordered.count)
        var missingIDs: [Int] = []
        var missingPositions: [Int] = []
        store.adviseExpertsWillNeed(layer: layer, experts: ordered)
        for (position, id) in ordered.enumerated() {
            if let existing = stagedBundles[id] {
                cacheHits += 1
                bundles[position] = existing
            } else if let cached = bundleCache?.get(layer: layer, expert: id) {
                cacheHits += 1
                bundles[position] = cached
            } else {
                missingIDs.append(id)
                missingPositions.append(position)
            }
        }
        let loaded = try loadExpertsFromStore(missingIDs)
        for ((id, position), expert) in zip(
            zip(missingIDs, missingPositions), loaded) {
            bundles[position] = expert
            bundleCache?.put(expert, layer: layer, expert: id)
        }
        let resolved = bundles.map { $0! }
        let loadCount = missingIDs.count
        stagingLock.lock()
        pendingOrder = ordered
        pendingBundles = resolved
        pendingError = nil
        stageBuilds += loadCount
        stageCacheHits += cacheHits
        loads += loadCount
        hits += cacheHits
        stageWall += Date().timeIntervalSince(started)
        stagingLock.unlock()
    }

    private func activatePendingStage(for expected: [Int]) throws {
        stagingLock.lock()
        defer { stagingLock.unlock() }
        if let pendingError { throw pendingError }
        guard pendingOrder == Array(Set(expected)).sorted(),
              let bundles = pendingBundles else { return }
        if pendingOrder == stagedOrder, stagedStack != nil {
            // Upstream asm-cache behavior: an unchanged staged set reuses
            // the existing nine stacked graph nodes; only route weights vary.
            pendingBundles = nil
            pendingOrder = []
            return
        }
        stagedBundles = Dictionary(uniqueKeysWithValues: zip(pendingOrder, bundles))
        stagedStack = try QuantizedExpertStack(experts: bundles)
        stagedOrder = pendingOrder
        pendingBundles = nil
        pendingOrder = []
    }

    public func resetStaging() {
        stagingLock.lock()
        pendingBundles = nil
        pendingError = nil
        pendingOrder = []
        stagedBundles.removeAll(keepingCapacity: true)
        stagedStack = nil
        stagedOrder = []
        stagingLock.unlock()
        clearFullLayer()
    }

    private func teacherRoute(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = matmul(x, routerWeight.T)
        return Edge0GroupedRouter8B.select(
            logits: logits, expertBias: expertBias,
            topK: configuration.expertsPerToken, nGroup: configuration.nGroup,
            topkGroup: configuration.topkGroup,
            normalize: configuration.normTopkProb,
            routedScale: configuration.routedScalingFactor)
    }

    private func resolveExpert(_ id: Int) throws -> QuantizedExpert {
        if let staged = stagedBundles[id] {
            stagingLock.lock(); hits += 1; stagingLock.unlock()
            return staged
        }
        if let cached = bundleCache?.get(layer: layer, expert: id) {
            stagingLock.lock(); hits += 1; stagingLock.unlock()
            return cached
        }
        let expert = try store.loadExpert(layer: layer, expert: id)
        bundleCache?.put(expert, layer: layer, expert: id)
        stagingLock.lock(); loads += 1; stagingLock.unlock()
        return expert
    }

    /// Match the upstream loader pool: resolve independent expert bundles in
    /// parallel after issuing page-cache readahead for their tensor ranges.
    private func resolveExperts(_ ids: [Int]) throws -> [QuantizedExpert] {
        guard !ids.isEmpty else { return [] }
        store.adviseExpertsWillNeed(layer: layer, experts: ids)
        let slots = ExpertLoadSlots(count: ids.count)
        DispatchQueue.concurrentPerform(iterations: ids.count) { index in
            slots.store(Result { try self.resolveExpert(ids[index]) }, at: index)
        }
        return try slots.finish()
    }

    private func loadExpertsFromStore(_ ids: [Int]) throws -> [QuantizedExpert] {
        guard !ids.isEmpty else { return [] }
        let slots = ExpertLoadSlots(count: ids.count)
        DispatchQueue.concurrentPerform(iterations: ids.count) { index in
            slots.store(
                Result { try self.store.loadExpert(layer: self.layer,
                                                   expert: ids[index]) },
                at: index)
        }
        return try slots.finish()
    }

    public var stats: StreamingMoEStats {
        stagingLock.lock(); defer { stagingLock.unlock() }
        return StreamingMoEStats(
            calls: calls, loads: loads, hits: hits, stagedUsed: stagedUsed,
            stagedFallback: stagedFallback, stageBuilds: stageBuilds,
            stageCacheHits: stageCacheHits, stageWallSeconds: stageWall)
    }

    /// Upstream exact path: stack the unique routed bundles and execute the
    /// same compiled gathered-QMM graph used by fixed-slot decode.
    private func executeGathered(_ x: MLXArray, indices: [Int],
                                 weights: [Float]) throws -> MLXArray {
        let c = configuration
        guard indices.count == weights.count else {
            throw M1Error.invalid("Expert IDs and weights disagree")
        }
        let unique = Array(Set(indices)).sorted()
        let experts = try resolveExperts(unique).map { expert in
            guard expert.up.inputSize == c.hiddenSize,
                  expert.up.outputSize == c.moeIntermediateSize else {
                throw M1Error.invalid("Routed expert dimensions disagree with config")
            }
            return expert
        }
        let localByID = Dictionary(uniqueKeysWithValues:
            unique.enumerated().map { ($0.element, $0.offset) })
        return try executeStacked(
            x, stack: QuantizedExpertStack(experts: experts),
            localIndices: indices.map { localByID[$0]! }, weights: weights)
    }

    /// Direct Swift translation of the official fixed staged-slot math. The
    /// selected K experts map into canonical staged slots and run through three
    /// gathered INT4 matrix multiplications instead of K independent graphs.
    private func executeStaged(_ x: MLXArray, stack: QuantizedExpertStack,
                               indices: [Int],
                               weights: [Float]) throws -> MLXArray {
        let k = indices.count
        guard k == weights.count, stagedOrder.count == stack.up.expertCount else {
            throw M1Error.invalid("Staged expert stack and route disagree")
        }
        let slotByID = Dictionary(uniqueKeysWithValues:
            stagedOrder.enumerated().map { ($0.element, $0.offset) })
        return try executeStacked(
            x, stack: stack, localIndices: indices.map { slotByID[$0]! },
            weights: weights)
    }

    private func executeStacked(_ x: MLXArray, stack: QuantizedExpertStack,
                                localIndices: [Int], weights: [Float]) throws
        -> MLXArray {
        let k = localIndices.count
        guard k == weights.count, !localIndices.isEmpty,
              localIndices.allSatisfy({
                  (0..<stack.up.expertCount).contains($0)
              }) else {
            throw M1Error.invalid("Stacked expert indices and route disagree")
        }
        let local = MLXArray(localIndices.map(Int32.init))
        let routedInput = broadcast(x, to: [k, configuration.hiddenSize])
            .expandedDimensions(axis: 1)
        let outputs = compiledStagedMath([
            routedInput, local,
            stack.up.weight, stack.up.scales, stack.up.biases,
            stack.gate.weight, stack.gate.scales, stack.gate.biases,
            stack.down.weight, stack.down.scales, stack.down.biases,
        ])[0].squeezed(axis: 1)
        let routedWeights = MLXArray(weights, [k, 1]).asType(outputs.dtype)
        let routed = (outputs * routedWeights).sum(axis: 0, keepDims: true)
        return try routed + shared(x)
    }

    /// Official whole-layer prefill lifecycle (`load_full_layer`).
    public func loadFullLayer() throws {
        if fullLayer == nil {
            fullLayer = try store.loadFullExpertLayer(layer: layer)
        }
    }

    /// Official whole-layer prefill lifecycle (`clear_full_layer`).
    public func clearFullLayer() {
        fullLayer = nil
    }

    public var hasFullLayer: Bool { fullLayer != nil }

    /// Official full-layer `StreamingSwitchGLU.__call__` branch for a flat
    /// token batch `[T, H]`. The sorted gather/unsort sequence is kept in the
    /// same order as mlx-lm's `_gather_sort` and `_scatter_unsort`.
    public func fullLayerPrefill(_ input: MLXArray) throws -> MoEOutput {
        let c = configuration
        guard input.ndim == 2, input.dim(0) > 1, input.dim(1) == c.hiddenSize,
              let fullLayer else {
            throw M1Error.invalid("Full-layer prefill requires loaded weights and [T,H], T > 1")
        }
        let x = input
        let logits = matmul(x.asType(.float32), routerWeight.T)
        let route = Edge0GroupedRouter8B.select(
            logits: logits, expertBias: expertBias,
            topK: c.expertsPerToken, nGroup: c.nGroup,
            topkGroup: c.topkGroup, normalize: c.normTopkProb,
            routedScale: c.routedScalingFactor)

        let tokenCount = input.dim(0)
        let topK = c.expertsPerToken
        let flatIndices = route.indices.flattened()
        let order = argSort(flatIndices)
        let inverseOrder = argSort(order)
        let tokenRows = order.floorDivide(topK)
        let sortedInput = x.take(tokenRows, axis: 0).expandedDimensions(axis: 1)
        let sortedExperts = flatIndices.take(order, axis: 0)

        let up = try fullLayer.up(sortedInput, expertIndices: sortedExperts)
        let gate = try fullLayer.gate(sortedInput, expertIndices: sortedExperts)
        let activated = (gate * sigmoid(gate)) * up
        let sortedOutput = try fullLayer.down(activated, expertIndices: sortedExperts)
        let expertOutput = sortedOutput.take(inverseOrder, axis: 0)
            .reshaped([tokenCount, topK, c.hiddenSize])
        let routedWeights = route.weights.asType(expertOutput.dtype)
            .expandedDimensions(axis: -1)
        let routed = (expertOutput * routedWeights)
            .sum(axis: 1)
        let hidden = try routed + shared(x)
        eval(hidden, route.indices, route.weights)
        return MoEOutput(
            hidden: hidden,
            expertIndices: route.indices.asArray(Int32.self).map(Int.init),
            expertWeights: route.weights.asArray(Float.self),
            routeIndices: route.indices)
    }
}
