import Foundation

public enum ThemeChoice: String, Codable, CaseIterable { case system = "System", light = "Light", dark = "Dark" }
public enum GenerationStyle: String, Codable, CaseIterable {
    case precise = "Precise", balanced = "Balanced", creative = "Creative"
    public var temperature: Double { switch self { case .precise: return 0.2; case .balanced: return 0.7; case .creative: return 1 } }
}
public struct AppSettings: Codable, Equatable {
    public var theme = ThemeChoice.system
    public var style = GenerationStyle.precise
    public var systemPrompt = ""
    public init() {}
    public func configuration(defaults: AppConfiguration) -> AppConfiguration { AppConfiguration(systemPrompt: systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? defaults.systemPrompt : systemPrompt, suggestions: defaults.suggestions) }
}
public struct SavedConversation: Codable, Identifiable, Equatable {
    public var id: UUID
    public var updatedAt: Date
    public var modelPath: String
    public var conversation: Conversation
    public var inputTokens: UInt64?, generatedTokens: UInt64?
    public init(id: UUID = UUID(), updatedAt: Date = Date(), modelPath: String, conversation: Conversation, inputTokens: UInt64? = nil, generatedTokens: UInt64? = nil) { self.id = id; self.updatedAt = updatedAt; self.modelPath = modelPath; self.conversation = conversation; self.inputTokens = inputTokens; self.generatedTokens = generatedTokens }
    public var title: String { Self.title(conversation.messages.first { $0.role == "user" }?.content ?? "New chat") }
    public static func title(_ text: String) -> String { let compact = text.split(whereSeparator: \.isWhitespace).joined(separator: " "); return compact.unicodeScalars.count > 32 ? String(String.UnicodeScalarView(compact.unicodeScalars.prefix(32))) + "…" : compact }
    public func group(now: Date, calendar: Calendar = .current) -> String {
        let today = calendar.startOfDay(for: now), day = calendar.startOfDay(for: updatedAt)
        let distance = calendar.dateComponents([.day], from: day, to: today).day ?? 0
        if distance <= 0 { return "Today" }; if distance == 1 { return "Yesterday" }; if distance <= 7 { return "Previous 7 days" }; return "Earlier"
    }
}
public struct ConversationFiles {
    public struct Listing { public let conversations: [SavedConversation]; public let damaged: [String] }
    public let directory: URL
    public init(directory: URL) { self.directory = directory }
    public func list() throws -> Listing {
        guard FileManager.default.fileExists(atPath: directory.path) else { return .init(conversations: [], damaged: []) }
        var conversations: [SavedConversation] = [], damaged: [String] = []
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).filter({ $0.pathExtension == "json" }) {
            do { let value = try JSONDecoder().decode(SavedConversation.self, from: Data(contentsOf: file)); guard file.deletingPathExtension().lastPathComponent == value.id.uuidString else { throw CocoaError(.fileReadCorruptFile) }; conversations.append(value) }
            catch { damaged.append(file.lastPathComponent) }
        }
        return .init(conversations: conversations.sorted { $0.updatedAt > $1.updatedAt }, damaged: damaged.sorted())
    }
    public func save(_ value: SavedConversation) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent(value.id.uuidString + ".json")
        if FileManager.default.fileExists(atPath: file.path) { let existing = try JSONDecoder().decode(SavedConversation.self, from: Data(contentsOf: file)); guard existing.id == value.id else { throw CocoaError(.fileReadCorruptFile) } }
        try JSONEncoder().encode(value).write(to: file, options: .atomic)
    }
    public func delete(_ id: UUID) throws { try FileManager.default.removeItem(at: directory.appendingPathComponent(id.uuidString + ".json")) }
}
public enum AppFiles {
    public static func read<T: Decodable>(_ type: T.Type, at url: URL) throws -> T { try JSONDecoder().decode(type, from: Data(contentsOf: url)) }
    public static func write<T: Encodable>(_ value: T, at url: URL) throws { try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true); try JSONEncoder().encode(value).write(to: url, options: .atomic) }
}
