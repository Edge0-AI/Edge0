import Foundation

public struct AppConfiguration: Codable, Equatable {
    public let systemPrompt: String
    public let suggestions: [String]
    enum CodingKeys: String, CodingKey { case systemPrompt = "system_prompt", suggestions }
    public static func load(_ url: URL) throws -> Self { try JSONDecoder().decode(Self.self, from: Data(contentsOf: url)) }
}

public struct ChatMessage: Identifiable, Equatable, Codable {
    public struct ScrollContent: Equatable {
        public let id: UUID
        public let content: String
        public let reasoning: String
        public let metrics: String
        public let stopped: Bool
    }
    public var id = UUID()
    public var role: String
    public var content: String
    public var reasoning: String
    public var metrics = ""
    public var nativeMetrics: EngineMetrics?
    public var stopped = false
    public var reasoningExpanded = true
    public var scrollContent: ScrollContent { .init(id: id, content: content, reasoning: reasoning, metrics: metrics, stopped: stopped) }
    public func showsReasoning(thinking: Bool) -> Bool { thinking && !reasoning.isEmpty }
    public var usesBubble: Bool { role == "user" }
    public var statusLabel: String? { stopped ? "Stopped" : nil }
    /// What this assistant message shows; see the shared assistant-state rule.
    public func state(isLast: Bool, generating: Bool, thinking: Bool) -> AssistantState {
        AssistantState(isLast: isLast, generating: generating, hasContent: !content.isEmpty, showsReasoning: showsReasoning(thinking: thinking))
    }
    public init(role: String, content: String, reasoning: String = "") { self.role = role; self.content = content; self.reasoning = reasoning }
    public var native: [String: String] { ["role": role, "content": content, "reasoning_content": reasoning] }
}

public struct ChatRequest: Sendable {
    public let messages: [[String: String]]
    public let thinking: Bool
}

public struct GenerationParameters: Sendable {
    public let thinking: Bool
    public let maxTokens: Int64?
    public let tokenEvents: Bool
    public init(thinking: Bool, arguments: [String], modelName: String?) {
        self.thinking = thinking
        let limit: Int64? = arguments.contains("--smoke") ? arguments.firstIndex(of: "--smoke-max-tokens").flatMap { index in
            guard index + 1 < arguments.count, let value = Int64(arguments[index + 1]), value > 0 else { return nil }
            return value
        } : nil
        maxTokens = limit
        tokenEvents = limit != nil && modelName?.hasPrefix("tiny-") == true
    }
}

public struct Conversation: Codable, Equatable {
    public private(set) var messages: [ChatMessage] = []
    public let configuration: AppConfiguration
    public private(set) var thinkingForChat: Bool?
    public var thinkingLocked: Bool { !messages.isEmpty }
    public func canChangeThinking(supported: Bool, busy: Bool) -> Bool { supported && !busy && !thinkingLocked }
    public init(configuration: AppConfiguration) { self.configuration = configuration }
    public var request: [[String: String]] { [["role": "system", "content": configuration.systemPrompt]] + messages.map(\.native) }
    public mutating func makeRequest(thinking: Bool, supported: Bool) -> ChatRequest {
        if thinkingForChat == nil { thinkingForChat = thinking && supported }
        return ChatRequest(messages: request, thinking: thinkingForChat ?? false)
    }
    public mutating func start(_ text: String) { messages.append(ChatMessage(role: "user", content: text)) }
    public mutating func appendAssistant() { messages.append(ChatMessage(role: "assistant", content: "")) }
    public mutating func fragment(_ text: String, reasoning: Bool) {
        guard messages.last?.role == "assistant" else { return }
        if reasoning { messages[messages.count - 1].reasoning += text } else { messages[messages.count - 1].content += text }
    }
    public mutating func finish(content: String, reasoning: String, metrics: String, stopped: Bool = false, nativeMetrics: EngineMetrics? = nil) {
        guard messages.last?.role == "assistant" else { return }
        messages[messages.count - 1].content = content
        messages[messages.count - 1].reasoning = reasoning
        messages[messages.count - 1].metrics = metrics
        messages[messages.count - 1].nativeMetrics = nativeMetrics
        messages[messages.count - 1].stopped = stopped
    }
    public mutating func setReasoningExpanded(id: UUID, expanded: Bool) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].reasoningExpanded = expanded
    }
    public mutating func reset() { messages.removeAll(); thinkingForChat = nil }
    public var canRegenerate: Bool { messages.last?.role == "assistant" && messages.dropLast().last?.role == "user" }
    public mutating func prepareRegeneration() -> String? {
        guard canRegenerate else { return nil }
        messages.removeLast(); return messages.removeLast().content
    }
    public mutating func prepareEdit(id: UUID) -> String? {
        guard let index = messages.firstIndex(where: { $0.id == id && $0.role == "user" }) else { return nil }
        let text = messages[index].content; messages.removeSubrange(index...); return text
    }
}

public struct ModelChoice: Identifiable, Equatable, Sendable {
    public var id: String { path }
    public let path: String
    public let name: String
    public let error: String?
    public let modelIdentifier: String?
    public let totalBytes: UInt64?
    public var displayName: String { modelIdentifier.flatMap { $0.isEmpty ? nil : $0 }.map { "\($0) · \(name)" } ?? name }
    public var selectable: Bool { error == nil }
    public init(path: String, name: String, error: String?, modelIdentifier: String? = nil, totalBytes: UInt64? = nil) { self.path = path; self.name = name; self.error = error; self.modelIdentifier = modelIdentifier; self.totalBytes = totalBytes }
}

public struct ModelPageState: Equatable {
    public private(set) var hasLoadedModel = false
    public private(set) var choosingModel = true
    public init() {}
    public var showsChat: Bool { hasLoadedModel && !choosingModel }
    public mutating func beginLoad() { hasLoadedModel = false; choosingModel = true }
    public mutating func loadSucceeded() { hasLoadedModel = true; choosingModel = false }
    public mutating func loadFailed() { hasLoadedModel = false; choosingModel = true }
    public mutating func showModels() { choosingModel = true }
    public mutating func resumeChat() { if hasLoadedModel { choosingModel = false } }
    public mutating func close() { hasLoadedModel = false; choosingModel = true }
}

public struct ThinkingControlPresentation {
    public enum TrackTone { case green, gray }
    public let isOn: Bool
    public let enabled: Bool
    public init(isOn: Bool, enabled: Bool) { self.isOn = isOn; self.enabled = enabled }
    public var opacity: Double { enabled ? 1 : 0.55 }
    public var trackTone: TrackTone { isOn ? .green : .gray }
    public var stateLabel: String { isOn ? "On" : "Off" }
}

public struct ModelChoices {
    public let loaded: [ModelChoice]
    public let available: [ModelChoice]
    public let unavailable: [ModelChoice]
    public func pushInstructions(assetsDirectory: String) -> String? {
        available.isEmpty && loaded.isEmpty ? "Use tests/platform/ios_push.py (see README) to push assets to \(assetsDirectory)." : nil
    }
    public init(_ choices: [ModelChoice], selectedPath: String? = nil) { loaded = choices.filter { $0.path == selectedPath && $0.selectable }; available = choices.filter { $0.selectable && $0.path != selectedPath }; unavailable = choices.filter { !$0.selectable } }
}

public enum GenerationPhase: Equatable {
    case idle, generating, stopping
    public var isGenerating: Bool { self != .idle }
    public var stopEnabled: Bool { self == .generating }
    public var stopTitle: String { self == .stopping ? "Stopping…" : "Stop" }
    public mutating func start() { self = .generating }
    public mutating func requestStop() { if self == .generating { self = .stopping } }
    public mutating func finish() { self = .idle }
}

public struct EngineMetrics: Codable, Equatable {
    public var generatedCount: Int?, firstTokenSeconds: Double?, inputTokens: Int?, reusedTokens: Int?, decodeTokensPerSecond: Double?, memoryPeakBytes: Double?
    enum CodingKeys: String, CodingKey {
        case generatedCount = "generated_count", firstTokenSeconds = "first_token_seconds", inputTokens = "input_tokens", reusedTokens = "reused_tokens", decodeTokensPerSecond = "decode_tokens_per_second", memoryPeakBytes = "memory_peak_bytes"
    }
    public var formatted: String {
        func number(_ value: Double, _ places: Int) -> String? {
            // Parse the shortest decimal representation before rounding: 1.005 must become 1.01.
            guard value.isFinite, var decimal = Decimal(string: String(value), locale: Locale(identifier: "en_US_POSIX")), !decimal.isNaN else { return nil }
            var rounded = Decimal()
            NSDecimalRound(&rounded, &decimal, places, .plain)
            let formatter = NumberFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.numberStyle = .decimal
            formatter.usesGroupingSeparator = false
            formatter.minimumFractionDigits = places
            formatter.maximumFractionDigits = places
            return formatter.string(from: NSDecimalNumber(decimal: rounded))
        }
        var items: [String] = []
        if let n = generatedCount { items.append("\(n) tokens") }
        if let t = firstTokenSeconds, t >= 0, let text = number(t, 2) { items.append("TTFT \(text) s") }
        if let n = inputTokens, n >= 0, let reused = reusedTokens, reused >= 0, n > reused, let t = firstTokenSeconds, t > 0, let text = number(Double(n - reused) / t, 1) { items.append("prefill \(text) tok/s") }
        if let d = decodeTokensPerSecond, d >= 0, let text = number(d, 1) { items.append("decode \(text) tok/s") }
        if let m = memoryPeakBytes, m >= 0, let text = number(m / 1_000_000, 0) { items.append("peak \(text) MB") }
        return items.joined(separator: " · ")
    }
}

public enum Markdown {
    public struct Block: Identifiable, Equatable {
        public enum Kind: Equatable { case paragraph, heading(Int), list, quote, code(String) }
        public let id: Int
        public let kind: Kind
        public let text: String
    }
    private struct Fence {
        let marker: Character
        let count: Int
        let language: String
    }
    private static func fence(_ line: String) -> Fence? {
        let leading = line.prefix(while: { $0 == " " }).count
        guard leading <= 3 else { return nil }
        let text = line.dropFirst(leading)
        guard let marker = text.first, marker == "`" || marker == "~" else { return nil }
        let count = text.prefix(while: { $0 == marker }).count
        guard count >= 3 else { return nil }
        let language = String(text.dropFirst(count)).trimmingCharacters(in: .whitespaces)
        if marker == "`" && language.contains("`") { return nil }
        return Fence(marker: marker, count: count, language: language)
    }
    private static func closes(_ line: String, _ fence: Fence) -> Bool {
        let leading = line.prefix(while: { $0 == " " }).count
        guard leading <= 3 else { return false }
        let text = line.dropFirst(leading)
        let count = text.prefix(while: { $0 == fence.marker }).count
        return count >= fence.count && text.dropFirst(count).allSatisfy(\.isWhitespace)
    }
    // Preserve existing code, dollar amounts, and unfinished streaming delimiters.
    public static func protectMath(_ input: String) -> String {
        let chars = Array(input)
        var result = "", i = 0
        func matches(_ value: String, at index: Int) -> Bool {
            let marker = Array(value)
            return index + marker.count <= chars.count && Array(chars[index..<(index + marker.count)]) == marker
        }
        func escaped(_ index: Int) -> Bool {
            var j = index
            while j > 0 && chars[j - 1] == "\\" { j -= 1 }
            return (index - j) % 2 == 1
        }
        func closing(_ marker: String, from start: Int) -> Int? {
            var j = start
            while j < chars.count {
                if matches(marker, at: j), !escaped(j) { return j }
                j += 1
            }
            return nil
        }
        func lineEnd(_ start: Int) -> Int {
            var end = start
            while end < chars.count && chars[end] != "\n" { end += 1 }
            return end
        }
        func wrap(_ raw: String, block: Bool) -> String {
            var longest = 0, run = 0
            for char in raw {
                run = char == "`" ? run + 1 : 0
                longest = max(longest, run)
            }
            let marker = String(repeating: "`", count: max(block ? 3 : 1, longest + 1))
            return block ? "\(marker)text\n\(raw)\n\(marker)" : "\(marker)\(raw)\(marker)"
        }
        while i < chars.count {
            if i == 0 || chars[i - 1] == "\n" {
                let end = lineEnd(i)
                if let opening = fence(String(chars[i..<end])) {
                    var next = end < chars.count ? end + 1 : end
                    var finish = chars.count
                    while next < chars.count {
                        let end = lineEnd(next)
                        if closes(String(chars[next..<end]), opening) { finish = end < chars.count ? end + 1 : end; break }
                        next = end < chars.count ? end + 1 : end
                    }
                    result += String(chars[i..<finish]); i = finish; continue
                }
            }
            if chars[i] == "`" && !escaped(i) {
                var count = 1
                while i + count < chars.count && chars[i + count] == "`" { count += 1 }
                var cursor = i + count, finish: Int?
                while cursor < chars.count {
                    if chars[cursor] == "`" {
                        var run = 1
                        while cursor + run < chars.count && chars[cursor + run] == "`" { run += 1 }
                        if run == count { finish = cursor + run; break }
                        cursor += run
                    } else { cursor += 1 }
                }
                let end = finish ?? chars.count
                result += String(chars[i..<end]); i = end; continue
            }
            if matches("\\$", at: i) && !escaped(i) { result += "\\$"; i += 2; continue }
            let pair: (String, String, Bool)?
            if matches("$$", at: i) && !escaped(i) { pair = ("$$", "$$", true) }
            else if matches("\\[", at: i) && !escaped(i) { pair = ("\\[", "\\]", true) }
            else if matches("\\(", at: i) && !escaped(i) { pair = ("\\(", "\\)", false) }
            else if chars[i] == "$", !escaped(i), String(chars[i..<lineEnd(i)]).range(of: #"^\$[0-9][0-9,.]*(?=\s|$)"#, options: .regularExpression) == nil { pair = ("$", "$", false) }
            else { pair = nil }
            if let (open, close, block) = pair, let end = closing(close, from: i + open.count) {
                let finish = end + close.count
                if block && !result.isEmpty && result.last != "\n" { result += "\n" }
                result += wrap(String(chars[i..<finish]), block: block)
                if block && finish < chars.count && chars[finish] != "\n" { result += "\n" }
                i = finish
            } else { result.append(chars[i]); i += 1 }
        }
        return result
    }
    public static func blocks(_ input: String) -> [Block] {
        let lines = protectMath(input).components(separatedBy: "\n")
        var output: [Block] = [], pending: [String] = [], code: [String]? = nil, activeFence: Fence?
        func emit(_ kind: Block.Kind, _ text: String) { output.append(Block(id: output.count, kind: kind, text: text)) }
        func flush() { if !pending.isEmpty { emit(.paragraph, pending.joined(separator: "\n")); pending = [] } }
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let opening = activeFence {
                if closes(line, opening) { emit(.code(opening.language), code!.joined(separator: "\n")); code = nil; activeFence = nil }
                else { code!.append(line) }
                continue
            }
            if let opening = fence(line) { flush(); activeFence = opening; code = []; continue }
            if trimmed.isEmpty { flush(); continue }
            let level = trimmed.prefix(while: { $0 == "#" }).count
            if (1...6).contains(level), trimmed.dropFirst(level).first == " " { flush(); emit(.heading(level), String(trimmed.dropFirst(level + 1))); continue }
            if trimmed.hasPrefix("> ") { flush(); emit(.quote, String(trimmed.dropFirst(2))); continue }
            if trimmed.range(of: #"^(?:[-*+] |\d+[.)] )"#, options: .regularExpression) != nil { flush(); emit(.list, trimmed); continue }
            pending.append(line)
        }
        if let code { emit(.code(activeFence?.language ?? ""), code.joined(separator: "\n")) }
        flush()
        return output
    }
}
/// The answer being generated shows dots until text appears and hides its actions until generation ends.
public struct AssistantState: Equatable {
    public let pending: Bool, actions: Bool
    public init(isLast: Bool, generating: Bool, hasContent: Bool, showsReasoning: Bool) {
        let current = isLast && generating
        pending = current && !hasContent && !showsReasoning
        actions = !current
    }
}
