import Edge0Core
import Foundation
import MLX

public struct Edge0GenerationResult: Sendable {
    public let text: String
    public let generatedTokenCount: Int
    public let elapsedSeconds: TimeInterval
    public let prefillSeconds: TimeInterval
    public let prefillTokensPerSecond: Double
    public let timeToFirstTokenSeconds: TimeInterval
    public let decodeTokensPerSecond: Double
    public let peakMemoryBytes: Int
}

private enum Edge0ChatTemplate8B {
    static func firstTurn(_ text: String, thinking: Bool) -> String {
        "<role>SYSTEM</role>detailed thinking \(thinking ? "on" : "off")<|role_end|>" +
        "<role>HUMAN</role>\(text)<|role_end|>" + assistantPrefix(thinking: thinking)
    }

    static func nextTurn(_ text: String, thinking: Bool) -> String {
        "<|role_end|><role>HUMAN</role>\(text)<|role_end|>" +
        assistantPrefix(thinking: thinking)
    }

    private static func assistantPrefix(thinking: Bool) -> String {
        "<role>ASSISTANT</role>\n" + (thinking ? "<think>" : "<think></think>")
    }
}

/// Product-facing, stateful chat entry point. The model math and cache lifetime
/// remain owned by StreamingEdge0Model8B; this type only mirrors the official
/// chat template and official sampling loop.
public final class Edge0ChatEngine: @unchecked Sendable {
    private let tokenizer: Edge0Tokenizer8B
    private let model: StreamingEdge0Model8B
    private var hasConversationContext = false
    private var tokenCount = 0

    /// Short prompts do not amortize loading all 128 experts per layer.
    private static let fullLayerPrefillMinimum = 128

    private static var runtimeDevice: Device {
        #if targetEnvironment(simulator)
        .cpu
        #else
        .gpu
        #endif
    }

    public init(modelURL: URL, progress: (String) -> Void = { _ in }) throws {
        let built = try Device.withDefaultDevice(Self.runtimeDevice) {
            let configuration = try Edge0Configuration8B.load(from: modelURL.appendingPathComponent("config.json"))
            let tokenizer = try Edge0Tokenizer8B(contentsOf: modelURL.appendingPathComponent("tokenizer.json"))
            let store = try ExpertTensorStore(
                modelURL: modelURL.appendingPathComponent("model.safetensors"),
                expertCount: configuration.numExperts
            )
            let model = try StreamingEdge0Model8B(
                configuration: configuration, store: store,
                loraWeightsURL: modelURL.appendingPathComponent(
                    "lora_edge0_8b.safetensors"),
                prerouterWeightsURL: modelURL.appendingPathComponent(
                    "prerouter_edge0_8b.safetensors"),
                progress: progress)
            progress("prewarming official decode path")
            try model.prewarm()
            return (tokenizer, model)
        }
        tokenizer = built.0
        model = built.1
    }

    public func reset() {
        model.reset()
        hasConversationContext = false
        tokenCount = 0
    }

    public func reply(
        to userText: String,
        maxTokens: Int = 2048,
        thinking: Bool = false,
        seed: UInt64? = nil,
        onText: @escaping @Sendable (String) -> Void = { _ in },
        shouldContinue: @escaping @Sendable () -> Bool = { true }
    ) throws -> Edge0GenerationResult {
        try Device.withDefaultDevice(Self.runtimeDevice) {
            try generateReply(to: userText, maxTokens: maxTokens,
                              thinking: thinking, seed: seed,
                              onText: onText, shouldContinue: shouldContinue)
        }
    }

    private func generateReply(
        to userText: String, maxTokens: Int, thinking: Bool, seed: UInt64?,
        onText: @escaping @Sendable (String) -> Void,
        shouldContinue: @escaping @Sendable () -> Bool
    ) throws -> Edge0GenerationResult {
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw M1Error.invalid("Message cannot be empty") }
        guard maxTokens > 0 else { throw M1Error.invalid("maxTokens must be positive") }

        let prompt = hasConversationContext
            ? Edge0ChatTemplate8B.nextTurn(trimmed, thinking: thinking)
            : Edge0ChatTemplate8B.firstTurn(trimmed, thinking: thinking)
        let promptIDs = try tokenizer.encode(prompt)
        guard !promptIDs.isEmpty else { throw M1Error.invalid("Tokenizer returned an empty prompt") }

        let started = Date()
        guard shouldContinue() else {
            model.reset(); hasConversationContext = false; tokenCount = 0
            throw CancellationError()
        }
        var output: StreamingModelOutput?
        // Full-layer prefill loads every expert in a layer. That cost is almost
        // independent of length, so a short chat prompt pays for the whole 4 GB
        // expert set. Below this cutoff, walk the same top-8 decode path.
        let streamingPrefill = promptIDs.count < Self.fullLayerPrefillMinimum
        if streamingPrefill {
            for id in promptIDs {
                guard shouldContinue() else {
                    model.reset(); hasConversationContext = false; tokenCount = 0
                    throw CancellationError()
                }
                output = try model(tokenID: id)
            }
        } else {
            let prefillChunk = 2048
            for start in stride(from: 0, to: promptIDs.count, by: prefillChunk) {
                guard shouldContinue() else {
                    model.reset(); hasConversationContext = false; tokenCount = 0
                    throw CancellationError()
                }
                let end = min(start + prefillChunk, promptIDs.count)
                let chunk = Array(promptIDs[start..<end])
                if chunk.count > 1 {
                    output = try model.prefill(tokenIDs: chunk)
                } else if let id = chunk.first {
                    output = try model(tokenID: id)
                }
            }
        }
        let prefillFinished = Date()
        tokenCount += promptIDs.count

        var generated: [Int] = []
        var samplingHistory = promptIDs
        var peakMemory = output?.peakMemory ?? 0
        var firstTokenAt: Date?
        var decodeStartedAt: Date?
        while generated.count < maxTokens, let current = output {
            if !shouldContinue() {
                if generated.isEmpty {
                    model.reset(); hasConversationContext = false; tokenCount = 0
                    throw CancellationError()
                }
                break
            }
            let next = try sample(current.logits, history: samplingHistory,
                                  firstToken: generated.isEmpty, seed: seed.map {
                                      $0 &+ UInt64(generated.count)
                                  })
            if next == tokenizer.endOfTurnTokenID { break }
            generated.append(next)
            samplingHistory.append(next)
            onText(try tokenizer.decode(generated))
            if firstTokenAt == nil {
                firstTokenAt = Date()
                decodeStartedAt = firstTokenAt
            }
            if generated.count == maxTokens { break }
            output = try model(tokenID: next)
            peakMemory = max(peakMemory, output?.peakMemory ?? 0)
            tokenCount += 1
        }
        hasConversationContext = true

        let finished = Date()
        let firstTokenSeconds = firstTokenAt?.timeIntervalSince(started)
            ?? finished.timeIntervalSince(started)
        let decodeTokenCount = max(0, generated.count - 1)
        let decodeSeconds = decodeStartedAt.map { finished.timeIntervalSince($0) } ?? 0
        let prefillSeconds = prefillFinished.timeIntervalSince(started)
        let decodeTokensPerSecond = decodeSeconds > 0
            ? Double(decodeTokenCount) / decodeSeconds : 0
        print(String(
            format: "[8b prefill] path=%@ tokens=%d total=%.2fs rate=%.2f tok/s",
            streamingPrefill ? "streaming-top8" : "full-layer",
            promptIDs.count,
            prefillSeconds,
            Double(promptIDs.count) / max(0.000_001, prefillSeconds)))
        print(String(
            format: "[8b decode] tokens=%d total=%.2fs rate=%.2f tok/s peak=%.0f MB",
            decodeTokenCount,
            decodeSeconds,
            decodeTokensPerSecond,
            Double(peakMemory) / 1_048_576))

        return Edge0GenerationResult(
            text: try tokenizer.decode(generated),
            generatedTokenCount: generated.count,
            elapsedSeconds: finished.timeIntervalSince(started),
            prefillSeconds: prefillSeconds,
            prefillTokensPerSecond: Double(promptIDs.count) /
                max(0.000_001, prefillSeconds),
            timeToFirstTokenSeconds: firstTokenSeconds,
            decodeTokensPerSecond: decodeTokensPerSecond,
            peakMemoryBytes: peakMemory
        )
    }

    /// Official Edge0 policy: first token greedy, then temperature 0.7,
    /// top-k 64, top-p 0.95 and HF-style repetition penalty 1.1.
    private func sample(_ input: MLXArray, history: [Int],
                        firstToken: Bool, seed: UInt64?) throws -> Int {
        let logits = input.flattened().asType(.float32)
        if firstToken { return argMax(logits).item(Int32.self).intValue }

        if !history.isEmpty {
            let unique = Array(Set(history)).sorted()
            let ids = MLXArray(unique.map(Int32.init))
            let values = logits[ids]
            logits[ids] = which(values .> 0, values / 1.1, values * 1.1)
        }
        let scaled = logits / 0.7
        let k = min(64, scaled.dim(0))
        let candidateIDs = argPartition(-scaled, kth: k - 1)[..<k]
        let candidateValues = takeAlong(scaled, candidateIDs, axis: 0)
        let order = argSort(-candidateValues)
        let rankedIDs = takeAlong(candidateIDs, order, axis: 0)
        let rankedValues = takeAlong(candidateValues, order, axis: 0)
        let cumulative = cumsum(softmax(rankedValues), axis: 0)
        let retained = (cumulative .<= 0.95).asType(.int32).sum()
        let retainedCount = maximum(retained, MLXArray(Int32(1)))
        let thresholdIndex = (retainedCount - 1).reshaped([1])
        let threshold = takeAlong(rankedValues, thresholdIndex, axis: 0)
        let masked = which(rankedValues .< threshold,
                           MLXArray(-Float.infinity), rankedValues)
        let local = MLXRandom.categorical(
            masked, key: seed.map(MLXRandom.key)).item(UInt32.self)
        return rankedIDs[Int(local)].item(Int32.self).intValue
    }
}

private extension Int32 {
    var intValue: Int { Int(self) }
}
