import Foundation

/// Tracks synchronous portions of model steps without forcing additional MLX evaluation.
/// GPU operations remain lazy, so timing them separately would change execution order.
final class StepProfile {

    private(set) var syncLinear: Double = 0
    private(set) var syncFull: Double = 0
    var routerSync: Double { syncLinear + syncFull }
    private(set) var expertRead: Double = 0
    private(set) var expertStack: Double = 0

    /// Time spent constructing MLX arrays from mapped weight bytes.
    private(set) var weightCopy: Double = 0
    private(set) var weightBytes: Int = 0
    private(set) var expertBytes: Int = 0

    /// Number of expert read batches and individual reads issued.
    private(set) var readBatches = 0
    private(set) var readsIssued = 0

    /// Number of unique experts requested compared with the naive T*K upper bound.
    private(set) var unionTotal = 0
    private(set) var pairsTotal = 0

    /// How often the "same experts as last token" guess was right.
    ///
    /// The whole prefetch scheme rests on this. It costs nothing when wrong, so any
    /// positive rate is upside — but it bounds the gain: the read can only be hidden for
    /// the fraction of blocks the guess anticipated.
    private(set) var prefetchHits = 0
    private(set) var prefetchTotal = 0

    /// Cumulative cost of the batched head forward plus its single barrier.
    private(set) var pregateStage = 0.0
    /// Core ML vocabulary projection plus its 4 KiB MLX-to-Core ML handoff.
    private(set) var vocabularyHead = 0.0

    private var steps = 0

    func reset() {
        syncLinear = 0; syncFull = 0; expertRead = 0; expertStack = 0; expertBytes = 0
        weightCopy = 0; weightBytes = 0; pregateStage = 0; vocabularyHead = 0
        prefetchHits = 0; prefetchTotal = 0
        readBatches = 0; readsIssued = 0; steps = 0
        unionTotal = 0; pairsTotal = 0
    }

    func countStep() { steps += 1 }

    /// Timing categories for synchronous portions of a model step.
    enum Phase {
        case syncLinear, syncFull, expertRead, expertStack, weightCopy
        /// Synchronous work performed by the trained routing heads.
        case pregateStage, vocabularyHead
    }

    /// Time a synchronous block. Returns the block's value so call sites stay expressions.
    @inline(__always)
    func measure<T>(_ phase: Phase, _ body: () throws -> T) rethrows -> T {
        let start = DispatchTime.now().uptimeNanoseconds
        let value = try body()
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        switch phase {
        case .syncLinear: syncLinear += elapsed
        case .syncFull: syncFull += elapsed
        case .expertRead: expertRead += elapsed
        case .expertStack: expertStack += elapsed
        case .weightCopy: weightCopy += elapsed
        case .pregateStage: pregateStage += elapsed
        case .vocabularyHead: vocabularyHead += elapsed
        }
        return value
    }

    @inline(__always)
    func measure<T>(_ phase: Phase, _ body: () async throws -> T) async rethrows -> T {
        let start = DispatchTime.now().uptimeNanoseconds
        let value = try await body()
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        switch phase {
        case .syncLinear: syncLinear += elapsed
        case .syncFull: syncFull += elapsed
        case .expertRead: expertRead += elapsed
        case .expertStack: expertStack += elapsed
        case .weightCopy: weightCopy += elapsed
        case .pregateStage: pregateStage += elapsed
        case .vocabularyHead: vocabularyHead += elapsed
        }
        return value
    }

    func recordRouting(union: Int, pairs: Int) {
        unionTotal += union
        pairsTotal += pairs
    }

    func recordPrefetch(hits: Int, total: Int) {
        prefetchHits += hits
        prefetchTotal += total
    }

    func recordWeight(bytes: Int) { weightBytes += bytes }

    func recordRead(bytes: Int, issued: Int) {
        expertBytes += bytes
        readBatches += 1
        readsIssued += issued
    }

    /// Per-step medians are not available without keeping every sample; per-step means
    /// are, and the question here is where a quarter-second goes, not its distribution.
    var report: String {
        guard steps > 0 else { return "no steps" }
        let n = Double(steps)
        func ms(_ total: Double) -> String { String(format: "%.0f", total / n) }
        let throughput = expertRead > 0
            ? String(format: "%.0f", Double(expertBytes) / (expertRead / 1000) / 1_000_000)
            : "-"
        let depth = readBatches > 0
            ? String(format: "%.2f", Double(readsIssued) / Double(readBatches))
            : "-"
        let copyRate = weightCopy > 0
            ? String(format: "%.0f", Double(weightBytes) / (weightCopy / 1000) / 1_000_000)
            : "-"
        let hitRate = prefetchTotal > 0
            ? String(format: "%.0f%%",
                     Double(prefetchHits) / Double(prefetchTotal) * 100)
            : "-"
        let sharing = pairsTotal > 0
            ? String(format: "%.2f", Double(unionTotal) / Double(pairsTotal))
            : "-"
        return "syncLinear=\(ms(syncLinear))ms syncFull=\(ms(syncFull))ms "
            + "read=\(ms(expertRead))ms "
            + "stack=\(ms(expertStack))ms readMBps=\(throughput) depth=\(depth) "
            + "union/pairs=\(sharing) "
            + "weightCopy=\(ms(weightCopy))ms wMBps=\(copyRate)"
            // Only when the heads are installed, so a baseline build's line is
            // byte-identical to the one every earlier measurement was read from.
            + (pregateStage > 0 ? " pregate=\(ms(pregateStage))ms" : "")
            + (vocabularyHead > 0 ? " coremlHead=\(ms(vocabularyHead))ms" : "")
            // `hitRate` was computed and dropped on the floor by an earlier version of
            // this line. It is the number that bounds the whole prefetch scheme — the
            // read can only be hidden for the fraction of blocks the guess anticipated
            // — so leaving it unprinted made the one decision it informs unanswerable.
            + (prefetchTotal > 0 ? " prefetch=\(hitRate)" : "")
    }
}
