import Foundation

/// CPU reference for Ling / Bailing's noaux_tc grouped router.
/// This is intentionally simple and testable; the production MLX version lives in Edge0MLX.
public enum GroupedExpertRouter {
    public struct Selection: Equatable, Sendable {
        public let indices: [Int]
        public let weights: [Float]
    }

    public static func select(
        logits: [Float],
        expertBias: [Float],
        topK: Int = 8,
        nGroup: Int = 8,
        topkGroup: Int = 4,
        normalize: Bool = true,
        routedScale: Float = 2.5
    ) -> Selection {
        precondition(nGroup > 0)
        precondition(logits.count / nGroup >= 2)
        precondition(topK <= topkGroup * (logits.count / nGroup))
        precondition(logits.count == expertBias.count)
        precondition(logits.count % nGroup == 0)
        precondition(topK > 0 && topK <= logits.count)
        precondition(topkGroup > 0 && topkGroup <= nGroup)

        let raw = logits.map { 1 / (1 + Foundation.exp(-$0)) }
        var selection = zip(raw, expertBias).map(+)
        let perGroup = logits.count / nGroup

        let groupScores: [(group: Int, score: Float)] = (0..<nGroup).map { group in
            let start = group * perGroup
            let end = start + perGroup
            let bestTwo = selection[start..<end].sorted(by: >).prefix(2)
            return (group, bestTwo.reduce(0, +))
        }
        let surviving = Set(groupScores.sorted { $0.score > $1.score }.prefix(topkGroup).map(\.group))

        for group in 0..<nGroup where !surviving.contains(group) {
            let start = group * perGroup
            for i in start..<(start + perGroup) { selection[i] = -.infinity }
        }

        let indices = selection.indices.sorted {
            if selection[$0] == selection[$1] { return $0 < $1 }
            return selection[$0] > selection[$1]
        }.prefix(topK).map { $0 }

        var weights = indices.map { raw[$0] }
        if normalize {
            let denominator = weights.reduce(0, +) + 1e-20
            weights = weights.map { $0 / denominator }
        }
        weights = weights.map { $0 * routedScale }
        return Selection(indices: indices, weights: weights)
    }
}
