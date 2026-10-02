import Darwin
import Foundation
import MLX

/// Product-facing wrapper around the streaming Edge0 35B implementation.
///
/// This engine deliberately lives beside `Edge0ChatEngine`: selecting 35B creates only
/// this model, while selecting 8B continues through the existing engine unchanged.
public final class Edge0ChatEngine35B: @unchecked Sendable {
    public static let modelFolderName = "repacked"

    private let directory: URL
    private let model: Edge0Model35B
    private let tokenizer: Tokenizer
    private let instructions: String
    private let systemPrefix: [Int32]
    private let state = Edge0Model35B.State()

    private var processed: [Int32] = []
    private var cachedPrefix: PrefixCache?

    public init(
        modelURL: URL,
        instructions: String = "You are a helpful assistant.",
        progress: (String) -> Void = { _ in }
    ) throws {
        directory = modelURL
        self.instructions = instructions

        // Configure MLX memory use for the selected 35B runtime.
        let highMem = CommandLine.arguments.contains("--high-mem-35b")
        Memory.cacheLimit = (highMem ? 1536 : 384) * 1_048_576
        print(String(
            format: "[mem] cacheLimit=%dMiB highMem=%@ available=%dMiB",
            Memory.cacheLimit / 1_048_576,
            highMem ? "true" : "false",
            os_proc_available_memory() / 1_048_576))
        progress("mapping Edge0 35B weights")
        var configuration = Edge0Model35B.Configuration()
        configuration.round6 = CommandLine.arguments.contains("--round6")
        if configuration.round6 {
            configuration.activeExperts = 2
        }
        configuration.retainAllResidentWeights = highMem
        model = try Edge0Model35B(directory: modelURL, configuration: configuration)
        guard model.availableLayers == model.configuration.layers else {
            throw Edge035BFailure.incompleteExperts(
                found: model.availableLayers, expected: model.configuration.layers)
        }
        progress("loading Edge0 35B tokenizer")
        tokenizer = try Tokenizer(url: modelURL.appendingPathComponent("tokenizer.bin"))

        if instructions.isEmpty {
            systemPrefix = []
        } else {
            let text = try ChatTemplate.render(
                [.init(role: "system", content: instructions)],
                addGenerationPrompt: false)
            // The first user role marker is fixed too. Persisting its state saves those
            // tokens on every new conversation; `PrefixCache.matches` still verifies the
            // encoded prefix against the complete prompt before it is trusted.
            systemPrefix = tokenizer.encode(text + ChatTemplate.userOpening)
        }
        cachedPrefix = PrefixCache.read(from: Self.prefixURL(
            directory: modelURL, fingerprint: model.configuration.fingerprint))
        try model.prepareForFirstTurn()
        progress("Edge0 35B ready")
    }

    public func reset() {
        state.reset()
        processed.removeAll(keepingCapacity: true)
    }

    public func reply(
        to userText: String,
        maxTokens: Int = 512,
        thinking: Bool = true,
        onText: @escaping @Sendable (String) -> Void = { _ in },
        shouldContinue: @escaping @Sendable () -> Bool = { true }
    ) async throws -> Edge0GenerationResult {
        let prompt = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { throw Edge035BFailure.invalid("Message cannot be empty") }
        guard maxTokens > 0 else { throw Edge035BFailure.invalid("maxTokens must be positive") }

        let continuing = !processed.isEmpty
        let ids: [Int32]
        if continuing {
            ids = tokenizer.encode(ChatTemplate.continuation(user: prompt, thinking: thinking))
        } else {
            var messages: [ChatTemplate.Message] = []
            if !instructions.isEmpty {
                messages.append(.init(role: "system", content: instructions))
            }
            messages.append(.init(role: "user", content: prompt))
            ids = tokenizer.encode(try ChatTemplate.render(messages, thinking: thinking))
        }
        guard !ids.isEmpty else { throw Edge035BFailure.invalid("Tokenizer returned an empty prompt") }
        print("[35b] thinking=\(thinking ? "on" : "off")")

        GPU.resetPeakMemory()
        let started = Date()
        let thermalAtStart = ProcessInfo.processInfo.thermalState
        var start = 0
        if !continuing {
            state.reset()
            processed.removeAll(keepingCapacity: true)
            if let cache = cachedPrefix, cache.matches(ids) {
                restore(cache.state, into: state)
                processed = cache.tokens
                start = cache.tokens.count
            }
        }

        var next: Int32 = 0
        // Later turns already have attention weights resident. Keep that cache on
        // for a continued prefill so those projections are not copied again.
        // The first prefill still copies each of them once; retaining earlier
        // only adds dirty pages with nothing to reuse inside that single pass.
        if continuing { model.beginDecodeCaching() }
        model.profile.reset()
        // An older cache may contain only the system message. Restore it, compute just
        // the newly-added user opening, then atomically replace it with the longer cache.
        let capturePoint = !continuing && start < systemPrefix.count
            && ids.starts(with: systemPrefix) ? systemPrefix.count : nil
        var index = start
        while index < ids.count {
            guard shouldContinue() else { return try cancelled() }
            var size = min(Edge0Model35B.maximumBatch, ids.count - index)
            if let capturePoint, index < capturePoint {
                size = min(size, capturePoint - index)
            }
            let slice = Array(ids[index ..< index + size])
            let isLast = index + size == ids.count
            let logits = try model.step(
                tokens: MLXArray(slice, [1, size]), state: state, needsLogits: isLast)
            if isLast, let logits { next = try await model.greedy(logits) }
            processed.append(contentsOf: slice)
            index += size

            if let capturePoint, index == capturePoint {
                let captured = PrefixCache(tokens: systemPrefix, state: state.snapshot())
                cachedPrefix = captured
                captured.write(to: Self.prefixURL(
                    directory: directory, fingerprint: model.configuration.fingerprint))
            }
        }
        let prefillFinished = Date()
        let prefillSeconds = prefillFinished.timeIntervalSince(started)
        let prefillTokens = ids.count - start
        print(String(
            format: "[35b prefill] tokens=%d total=%.2fs rate=%.2f tok/s %@",
            prefillTokens,
            prefillSeconds,
            Double(prefillTokens) / max(0.000_001, prefillSeconds),
            model.profile.report))
        model.beginDecodeCaching()
        model.profile.reset()

        let stop = StopCondition(maximumTokens: maxTokens)
        var raw = ""
        var shown = ""
        var produced = 0
        var firstTokenAt: Date?
        var decodeStartedAt: Date?

        while true {
            switch stop.evaluate(token: next, produced: produced) {
            case .hitEndToken, .hitLimit:
                let finished = Date()
                let decodeSeconds = decodeStartedAt.map {
                    finished.timeIntervalSince($0)
                } ?? 0
                let footprint = Self.physicalFootprintBytes()
                print(String(
                    format: "[35b decode] tokens=%d total=%.2fs rate=%.2f tok/s thermal=%@→%@ mlxPeak=%.0fMB footprint=%.0fMB available=%dMB %@",
                    max(0, produced - 1),
                    decodeSeconds,
                    Double(max(0, produced - 1)) / max(0.000_001, decodeSeconds),
                    Self.thermalName(thermalAtStart),
                    Self.thermalName(ProcessInfo.processInfo.thermalState),
                    Double(Memory.peakMemory) / 1_048_576,
                    Double(footprint) / 1_048_576,
                    os_proc_available_memory() / 1_048_576,
                    model.profile.report))
                shown = thinking
                    ? ThinkingBlock.settled(raw)
                    : ThinkingBlock.settledReply(raw)
                if !shown.isEmpty { onText(shown) }
                return Edge0GenerationResult(
                    text: shown,
                    generatedTokenCount: produced,
                    elapsedSeconds: finished.timeIntervalSince(started),
                    prefillSeconds: prefillSeconds,
                    prefillTokensPerSecond: Double(prefillTokens) /
                        max(0.000_001, prefillSeconds),
                    timeToFirstTokenSeconds: firstTokenAt?.timeIntervalSince(started)
                        ?? finished.timeIntervalSince(started),
                    decodeTokensPerSecond: decodeStartedAt.map {
                        Double(max(0, produced - 1)) /
                            max(0.000_001, finished.timeIntervalSince($0))
                    } ?? 0,
                    peakMemoryBytes: max(Memory.peakMemory, footprint))
            case .keepGoing:
                break
            }

            guard shouldContinue() else { return try cancelled() }
            if firstTokenAt == nil {
                firstTokenAt = Date()
                decodeStartedAt = firstTokenAt
            }
            raw += tokenizer.decode([next])
            let visible = thinking
                ? ThinkingBlock.visible(in: raw)
                : ThinkingBlock.visibleReply(in: raw)
            if let visible, visible != shown {
                shown = visible
                onText(shown)
            }
            produced += 1

            let logits = try model.step(tokens: MLXArray([next], [1, 1]), state: state)
            processed.append(next)
            next = try await model.greedy(logits)
        }
    }

    private func cancelled<T>() throws -> T {
        reset()
        throw CancellationError()
    }

    private func restore(_ source: Edge0Model35B.State, into target: Edge0Model35B.State) {
        let snapshot = source.snapshot()
        target.convolution = snapshot.convolution
        target.recurrent = snapshot.recurrent
        target.keys = snapshot.keys
        target.values = snapshot.values
        target.offset = snapshot.offset
    }

    private static func prefixURL(directory: URL, fingerprint: String) -> URL {
        PrefixCache.fileURL(in: directory, fingerprint: fingerprint)
    }

    private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// Process physical footprint (jetsam-relevant), not just MLX allocator peak.
    private static func physicalFootprintBytes() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.phys_footprint)
    }
}

private enum Edge035BFailure: LocalizedError {
    case incompleteExperts(found: Int, expected: Int)
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .incompleteExperts(let found, let expected):
            return "Edge0 35B expert files are incomplete (found \(found), expected \(expected))"
        case .invalid(let message):
            return message
        }
    }
}
