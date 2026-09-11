import MLX

/// MLX implementation of Edge0/Ling's SIGMOID_GROUP / noaux_tc routing law.
/// Mirrors Edge0's BailingGate: expert bias affects selection only; routed
/// weights come from the unbiased sigmoid scores.
public enum BailingGroupedRouter {
    public static func select(
        logits: MLXArray,
        expertBias: MLXArray,
        topK: Int = 8,
        nGroup: Int = 8,
        topkGroup: Int = 4,
        normalize: Bool = true,
        routedScale: Float = 2.5
    ) -> (indices: MLXArray, weights: MLXArray) {
        precondition(nGroup > 0)
        precondition(logits.shape == expertBias.shape || expertBias.shape == [logits.dim(-1)])
        precondition(logits.dim(-1) / nGroup >= 2)
        precondition(topK <= topkGroup * (logits.dim(-1) / nGroup))
        let raw = sigmoid(logits.asType(.float32))
        var selection = raw + expertBias
        let experts = selection.dim(-1)
        precondition(experts % nGroup == 0)
        precondition(topK > 0 && topK <= experts)
        precondition(topkGroup > 0 && topkGroup <= nGroup)

        let groupsToDrop = nGroup - topkGroup
        if groupsToDrop > 0 {
            let grouped = unflatten(selection, axis: -1, shape: [nGroup, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let dropped = argPartition(
                groupScores,
                kth: groupsToDrop - 1,
                axis: -2
            )[.ellipsis, ..<groupsToDrop, 0...]

            selection = putAlong(
                grouped,
                stopGradient(dropped),
                values: MLXArray(-Float.infinity),
                axis: -2
            )
            selection = flattened(selection, start: -2, end: -1)
        }

        let indices = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(raw, indices, axis: -1)
        if normalize {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * routedScale)
    }
}
