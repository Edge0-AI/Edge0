import Foundation

public enum StatusFormat {
    public static func decimal(_ value: Double, places: Int) -> String? {
        guard value.isFinite, value >= 0, var number = Decimal(string: String(value), locale: Locale(identifier: "en_US_POSIX")) else { return nil }
        var rounded = Decimal()
        NSDecimalRound(&rounded, &number, places, .plain)
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.numberStyle = .decimal
        formatter.usesGroupingSeparator = false; formatter.minimumFractionDigits = places; formatter.maximumFractionDigits = places
        return formatter.string(from: NSDecimalNumber(decimal: rounded))
    }
    public static func count(_ value: UInt64?) -> String? {
        guard let value else { return nil }
        let digits = Array(String(value)); return digits.enumerated().map { index, digit in index > 0 && (digits.count - index) % 3 == 0 ? ",\(digit)" : String(digit) }.joined()
    }
    public static func bytes(_ value: UInt64?) -> String? {
        guard let value else { return nil }
        if value < 10_000_000 { return "\((value + 500_000) / 1_000_000) MB" }
        let hundredths = value / 10_000_000 + (value % 10_000_000 >= 5_000_000 ? 1 : 0)
        return "\(hundredths / 100).\(String(format: "%02llu", hundredths % 100)) GB"
    }
    public static func rate(_ value: Double?) -> String? { value.flatMap { decimal($0, places: 1).map { "\($0) tok/s" } } }
    public static func percent(_ numerator: UInt64?, _ denominator: UInt64?) -> String? {
        guard let numerator, let denominator, denominator > 0 else { return nil }
        var ratio = Decimal(numerator) * 1000 / Decimal(denominator), rounded = Decimal()
        NSDecimalRound(&rounded, &ratio, 0, .plain)
        let tenths = NSDecimalNumber(decimal: rounded).uint64Value
        return "\(tenths / 10).\(tenths % 10)%"
    }
    public static func duration(_ value: Double?) -> String? {
        guard let value, let text = decimal(value, places: 1), let tenth = Double(text) else { return nil }
        if tenth < 60 { return text + " s" }
        let seconds = UInt64(value.rounded(.toNearestOrAwayFromZero))
        if seconds < 3600 { return "\(seconds / 60) min \(seconds % 60) s" }
        return "\(seconds / 3600) h \((seconds % 3600) / 60) min"
    }
    public static func deviceMemory(_ devices: [String: NativeDeviceMemory]?) -> DeviceMemoryDisplay? {
        guard let devices, !devices.isEmpty else { return nil }
        let system = devices.values.filter { $0.memory == .system }.reduce(UInt64(0)) { $0 + $1.currentBytes }
        let dedicated = devices.values.filter { $0.memory == .dedicated }
        return .init(system: bytes(system)!, dedicated: dedicated.isEmpty ? nil : bytes(dedicated.reduce(UInt64(0)) { $0 + $1.currentBytes }))
    }
}
public struct DeviceMemoryDisplay: Equatable { public let system: String; public let dedicated: String? }
public struct ThermalDisplay: Equatable {
    public enum Tone: String { case ok, warn, error }
    public let label: String, description: String, tone: Tone
    private static let levels: [(String, String, Tone, [String])] = [
        ("Normal", "No heat slowdown", .ok, ["nominal", "none"]),
        ("Warm", "Little or no slowdown", .warn, ["fair", "light"]),
        ("Throttled", "Running slower to cool down", .warn, ["serious", "moderate", "severe"]),
        ("Too hot", "Heavily slowed; let it cool", .error, ["critical"]),
        ("Shutting down", "System is about to shut parts down", .error, ["emergency", "shutdown"]),
    ]
    public init(state: String) {
        if let level = Self.levels.first(where: { $0.3.contains(state) }) { (label, description, tone) = (level.0, level.1, level.2) }
        else { (label, description, tone) = (state, "Unrecognized state", .warn) }
    }
}
public struct ContextUsage: Equatable {
    public let text: String
    public let ringPercent: Int?
    public init(inputTokens: UInt64? = nil, generatedTokens: UInt64? = nil, contextLength: UInt64? = nil) {
        let used = (inputTokens ?? 0) + (generatedTokens ?? 0)
        guard let contextLength, contextLength > 0 else { text = "\(StatusFormat.count(used)!) tokens"; ringPercent = nil; return }
        text = "\(StatusFormat.count(used)!) / \(StatusFormat.count(contextLength)!) tokens"
        var ratio = Decimal(used) * 100 / Decimal(contextLength), rounded = Decimal()
        NSDecimalRound(&rounded, &ratio, 0, .plain)
        ringPercent = min(100, NSDecimalNumber(decimal: rounded).intValue)
    }
}
public enum NativeMemoryLocation: String, Codable { case system, dedicated }
public struct NativeDeviceMemory: Codable, Equatable {
    public var currentBytes: UInt64; public var peakBytes: UInt64; public var memory: NativeMemoryLocation
    public init(currentBytes: UInt64, peakBytes: UInt64, memory: NativeMemoryLocation) { self.currentBytes = currentBytes; self.peakBytes = peakBytes; self.memory = memory }
}
public struct NativeProgress: Codable, Equatable {
    public var phase: String
    public var inputTokens: UInt64, reusedTokens: UInt64, computedTokens: UInt64, pendingTokens: UInt64, generatedTokens: UInt64
    public var elapsedSeconds: Double?, firstTokenSeconds: Double?, prefillTokensPerSecond: Double?, decodeElapsedSeconds: Double?, decodeTokensPerSecond: Double?
    public var prefillText: String? {
        if phase == "prefill" { return ["Computing \(StatusFormat.count(inputTokens - min(inputTokens, reusedTokens))!) tokens", StatusFormat.duration(elapsedSeconds)].compactMap { $0 }.joined(separator: " · ") }
        return [StatusFormat.count(inputTokens - min(inputTokens, reusedTokens)).map { "\($0) tokens" }, StatusFormat.duration(firstTokenSeconds), StatusFormat.rate(prefillTokensPerSecond)].compactMap { $0 }.joined(separator: " · ")
    }
    public var decodeText: String? { phase == "decode" ? [StatusFormat.count(generatedTokens).map { "\($0) tokens" }, StatusFormat.rate(decodeTokensPerSecond)].compactMap { $0 }.joined(separator: " · ") : nil }
}
public struct NativeLedger: Codable, Equatable {
    public var limitBytes: UInt64, currentBytes: UInt64, peakBytes: UInt64
    public var byPurpose: [String: UInt64]; public var byDevice: [String: NativeDeviceMemory]
    public var segments: [(String, UInt64)] { [("Weights", byPurpose["resident_weights"] ?? 0), ("Experts", byPurpose["experts"] ?? 0), ("Session state", byPurpose["session_state"] ?? 0), ("Workspace", byPurpose["workspace"] ?? 0), ("Staging", byPurpose["staging"] ?? 0), ("Other", (byPurpose["metadata"] ?? 0) + (byPurpose["transfer"] ?? 0))] }
}
public struct NativeMemoryValues: Codable, Equatable { public var currentBytes: UInt64?, peakBytes: UInt64? }
public struct NativeMemoryMetric: Codable, Equatable { public var values: NativeMemoryValues?; public var reason: String; public var source: String }
public struct NativeSnapshot: Codable, Equatable { public var ledger: NativeLedger; public var resident: NativeMemoryMetric; public var physical: NativeMemoryMetric }
public struct NativeSystemValue: Codable, Equatable { public var bytes: UInt64?; public var source: String; public var reason: String; public var currentPhysicalBytes: UInt64? }
public struct NativeThermal: Codable, Equatable { public var state: String?; public var source: String; public var reason: String }
public struct NativeSystem: Codable, Equatable {
    public struct Memory: Codable, Equatable { public var total: NativeSystemValue; public var available: NativeSystemValue }
    public var memory: Memory; public var thermal: NativeThermal; public var appMemoryLimit: NativeSystemValue
}
public struct ExpertSummary: Codable, Equatable {
    public var demand: UInt64 = 0, hits: UInt64 = 0, loads: UInt64 = 0, bytes: UInt64 = 0
    public init(demand: UInt64 = 0, hits: UInt64 = 0, loads: UInt64 = 0, bytes: UInt64 = 0) { self.demand = demand; self.hits = hits; self.loads = loads; self.bytes = bytes }
}
public struct NativeGenerationSummary: Codable, Equatable {
    public var inputTokens: UInt64, reusedTokens: UInt64, computedTokens: UInt64, generatedTokens: UInt64
    public var firstTokenSeconds: Double?, decodeSeconds: Double?, decodeForwards: UInt64?, totalSeconds: Double
    public var experts: ExpertSummary?
    public var reuseMethod: String; public var checkpointRestored: Bool
    public var prefillText: String { [StatusFormat.count(inputTokens - min(inputTokens, reusedTokens)).map { "\($0) tokens" }, StatusFormat.duration(firstTokenSeconds), firstTokenSeconds.flatMap { $0 > 0 ? StatusFormat.rate(Double(inputTokens - min(inputTokens, reusedTokens)) / $0) : nil }].compactMap { $0 }.joined(separator: " · ") }
    public var decodeText: String { [StatusFormat.count(generatedTokens).map { "\($0) tokens" }, decodeSeconds.flatMap { seconds in decodeForwards.flatMap { seconds > 0 ? StatusFormat.rate(Double($0) / seconds) : nil } }].compactMap { $0 }.joined(separator: " · ") }
}
public struct UsageTotals: Codable, Equatable {
    public private(set) var requests: UInt64 = 0, inputTokens: UInt64 = 0, reusedTokens: UInt64 = 0, generatedTokens: UInt64 = 0
    public private(set) var prefillTokens: UInt64 = 0, firstTokenSeconds: Double = 0, firstTokenCount: UInt64 = 0
    public private(set) var decodeForwards: UInt64 = 0, decodeSeconds: Double = 0
    public private(set) var experts: ExpertSummary?
    public private(set) var generationIDs: Set<UUID> = []
    public init() {}
    public mutating func record(_ summary: NativeGenerationSummary, id: UUID) {
        guard generationIDs.insert(id).inserted else { return }
        requests += 1; inputTokens += summary.inputTokens; reusedTokens += summary.reusedTokens; generatedTokens += summary.generatedTokens
        if let seconds = summary.firstTokenSeconds, seconds > 0 { prefillTokens += summary.inputTokens - min(summary.inputTokens, summary.reusedTokens); firstTokenSeconds += seconds; firstTokenCount += 1 }
        if let seconds = summary.decodeSeconds, let forwards = summary.decodeForwards, seconds > 0 { decodeForwards += forwards; decodeSeconds += seconds }
        if let incoming = summary.experts { var value = experts ?? ExpertSummary(); value.demand += incoming.demand; value.hits += incoming.hits; value.loads += incoming.loads; value.bytes += incoming.bytes; experts = value }
    }
    public var reused: String? { StatusFormat.percent(reusedTokens, inputTokens) }
    public var expertHits: String? { experts.flatMap { StatusFormat.percent($0.hits, $0.demand) } }
    public var averageDecode: String? { decodeSeconds > 0 ? StatusFormat.rate(Double(decodeForwards) / decodeSeconds) : nil }
    public var averagePrefill: String? { firstTokenSeconds > 0 ? StatusFormat.rate(Double(prefillTokens) / firstTokenSeconds) : nil }
    public mutating func reset() { let seen = generationIDs; self = Self(); generationIDs = seen }
}
public struct PollSchedule {
    private var visible = false, inFlight = false
    private var next: Double = 0
    public init() {}
    public mutating func show() { visible = true }
    public mutating func hide() { visible = false }
    public mutating func begin(at time: Double) -> Bool { guard visible, !inFlight, time >= next else { return false }; inFlight = true; return true }
    public mutating func finish(at time: Double) { inFlight = false; next = time + 1 }
}
