import Edge0Core
import Foundation
import MLX

struct LingPrerouterFeature {
    let input: MLXArray
    let routeIndices: MLXArray
}

/// Language-level port of `LingPrerouterStager` for Edge0-8B.
/// Owners 7...22 predict the routed set and weights for consumers 8...23.
final class LingPrerouter {
    private let c: Edge0Configuration8B
    private let owners = Array(7...22)
    private let fc1: MLXArray
    private let fc2: MLXArray
    private let linear: MLXArray
    private var previousIndices: [Int: MLXArray] = [:]

    init(configuration c: Edge0Configuration8B, weightsURL: URL) throws {
        self.c = c
        let store = try ExpertTensorStore(modelURL: weightsURL,
                                          expertCount: c.numExperts)
        let ownerIDs = Array(7...22)
        func weights(_ suffix: String) throws -> MLXArray {
            try stacked(ownerIDs.map {
                try store.mlxArrayCopying(named: "layers.\($0).\(suffix)")
            }, axis: 0)
        }
        fc1 = try weights("fc1.weight")
        fc2 = try weights("fc2.weight")
        linear = try weights("linear_init.weight")
        eval(fc1, fc2, linear)
    }

    func predict(features: [Int: LingPrerouterFeature]) throws
        -> [Int: PredictedMoERoute] {
        var activeOwners: [Int] = []
        var inputs: [MLXArray] = []
        var currentIndices: [Int: MLXArray] = [:]

        for owner in owners {
            guard let feature = features[owner] else { continue }
            let rows = feature.routeIndices.dim(0)
            guard rows > 0 else { continue }
            let current = row(feature.routeIndices, at: rows - 1)
            let previous = previousIndices[owner]
                ?? (rows > 1 ? row(feature.routeIndices, at: rows - 2) : nil)
            currentIndices[owner] = current

            let tokenInput = lastToken(feature.input)
            let combined = concatenated([
                tokenInput,
                oneHot(current),
                oneHot(previous),
            ], axis: -1)
            activeOwners.append(owner)
            inputs.append(combined)
        }
        guard activeOwners == owners else { return [:] }

        // Official `head_batch`: 16 owners share three batched matmuls rather
        // than issuing 48 tiny per-head GPU operations.
        let x = stacked(inputs, axis: 0).asType(fc1.dtype)
        let hidden = matmul(x, fc1.transposed(0, 2, 1))
        let gelu = 0.5 * hidden * (1 + erf(hidden / Float(2).squareRoot()))
        let logits = matmul(gelu, fc2.transposed(0, 2, 1))
            + matmul(x, linear.transposed(0, 2, 1))
        let route = Edge0GroupedRouter8B.select(
            logits: logits, expertBias: MLXArray.zeros([c.numExperts]),
            topK: c.expertsPerToken, nGroup: c.nGroup,
            topkGroup: c.topkGroup, normalize: c.normTopkProb,
            routedScale: c.routedScalingFactor)
        eval(route.indices, route.weights)
        let allIDs = route.indices.asArray(Int32.self).map(Int.init)
        let allWeights = route.weights.asArray(Float.self)

        var predictions: [Int: PredictedMoERoute] = [:]
        for (position, owner) in owners.enumerated() {
            let start = position * c.expertsPerToken
            let end = start + c.expertsPerToken
            predictions[owner + 1] = PredictedMoERoute(
                expertIndices: Array(allIDs[start..<end]),
                expertWeights: Array(allWeights[start..<end]))
        }
        previousIndices.merge(currentIndices) { _, new in new }
        return predictions
    }

    private func oneHot(_ ids: MLXArray?) -> MLXArray {
        var values = MLXArray.zeros([c.numExperts])
        guard let ids else { return values.reshaped([1, c.numExperts]) }
        values[ids.flattened()] = MLXArray.ones([c.expertsPerToken])
        return values.reshaped([1, c.numExperts])
    }

    private func row(_ values: MLXArray, at index: Int) -> MLXArray {
        let parts = split(values, indices: [index, index + 1], axis: 0)
        return parts[1]
    }

    private func lastToken(_ input: MLXArray) -> MLXArray {
        guard input.dim(0) > 1 else { return input }
        return split(input, indices: [input.dim(0) - 1], axis: 0)[1]
    }

    func reset() { previousIndices.removeAll(keepingCapacity: true) }
}
