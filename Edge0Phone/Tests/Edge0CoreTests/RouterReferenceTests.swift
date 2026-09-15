import Edge0Core
import Testing

@Test func groupedRouterDropsWeakGroupsAndNormalizes() {
    // 8 experts, 4 groups, 2 experts/group. Keep 2 groups, choose top 2 experts.
    let logits: [Float] = [-4, -3, 5, 4, 1, 0, 3, 2]
    let bias = Array(repeating: Float(0), count: 8)
    let result = GroupedExpertRouter.select(
        logits: logits,
        expertBias: bias,
        topK: 2,
        nGroup: 4,
        topkGroup: 2,
        normalize: true,
        routedScale: 2.5
    )
    #expect(Set(result.indices) == Set([2, 3]))
    #expect(abs(result.weights.reduce(0, +) - 2.5) < 0.0001)
}
