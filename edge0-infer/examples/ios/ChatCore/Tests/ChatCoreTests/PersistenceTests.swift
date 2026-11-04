import XCTest
@testable import ChatCore

final class PersistenceTests: XCTestCase {
    private let config = AppConfiguration(systemPrompt: "Default", suggestions: ["hello"])
    func testSettingsTemperatureAndConversationPromptLock() {
        var settings = AppSettings(); XCTAssertEqual(settings.style.temperature, 0.2)
        for (style, temperature) in [(GenerationStyle.precise, 0.2), (.balanced, 0.7), (.creative, 1)] { settings.style = style; XCTAssertEqual(settings.style.temperature, temperature) }
        let old = Conversation(configuration: settings.configuration(defaults: config))
        settings.systemPrompt = "Custom"; let new = Conversation(configuration: settings.configuration(defaults: config))
        XCTAssertEqual(old.request.first?["content"], "Default"); XCTAssertEqual(new.request.first?["content"], "Custom")
        settings.systemPrompt = "  \n"; XCTAssertEqual(settings.configuration(defaults: config), config)
    }
    func testHistoryRoundtripDamagedFileProtectionAndDelete() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString); defer { try? FileManager.default.removeItem(at: root) }
        let files = ConversationFiles(directory: root)
        var conversation = Conversation(configuration: config); conversation.start("hello"); _ = conversation.makeRequest(thinking: true, supported: true); conversation.appendAssistant(); conversation.finish(content: "answer", reasoning: "thought", metrics: "8 tokens", stopped: true, nativeMetrics: EngineMetrics(generatedCount: 8, firstTokenSeconds: 0.2, inputTokens: 13, reusedTokens: 0, decodeTokensPerSecond: 10, memoryPeakBytes: 123456789))
        let record = SavedConversation(modelPath: "assets/model", conversation: conversation, inputTokens: 13, generatedTokens: 8)
        try files.save(record); XCTAssertEqual(try files.list().conversations, [record]); XCTAssertEqual(try files.list().conversations.first?.conversation.messages.last?.nativeMetrics?.generatedCount, 8)
        let file = root.appendingPathComponent(record.id.uuidString + ".json"); let damaged = Data("broken".utf8); try damaged.write(to: file)
        XCTAssertEqual(try files.list().damaged, [file.lastPathComponent]); XCTAssertThrowsError(try files.save(record)); XCTAssertEqual(try Data(contentsOf: file), damaged)
        try files.delete(record.id); XCTAssertTrue(try files.list().damaged.isEmpty)
    }
    func testGroupsTitlesAndUnicodeBoundaries() {
        let calendar = Calendar(identifier: .gregorian), today = Date(timeIntervalSince1970: 1735776000)
        for (days, expected) in [(0, "Today"), (1, "Yesterday"), (2, "Previous 7 days"), (7, "Previous 7 days"), (8, "Earlier")] {
            let record = SavedConversation(updatedAt: calendar.date(byAdding: .day, value: -days, to: today)!, modelPath: "model", conversation: Conversation(configuration: config))
            XCTAssertEqual(record.group(now: today, calendar: calendar), expected)
        }
        XCTAssertEqual(SavedConversation.title("  hello\n world  "), "hello world")
        XCTAssertEqual(SavedConversation.title(String(repeating: "x", count: 33)), String(repeating: "x", count: 32) + "…")
        XCTAssertEqual(SavedConversation.title(String(repeating: "😀", count: 33)).unicodeScalars.count, 33)
    }
    func testModelSectionsIncludeLoadedSizesAndUnavailableReason() {
        let loaded = ModelChoice(path: "a", name: "directory-a", error: nil, modelIdentifier: "adapter", totalBytes: 123456789)
        let available = ModelChoice(path: "b", name: "directory-b", error: nil, modelIdentifier: "adapter", totalBytes: 987654321)
        let invalid = ModelChoice(path: "c", name: "directory-c", error: "Missing weights")
        let groups = ModelChoices([loaded, available, invalid], selectedPath: "a")
        XCTAssertEqual(groups.loaded, [loaded]); XCTAssertEqual(groups.available, [available]); XCTAssertEqual(groups.unavailable, [invalid]); XCTAssertNil(groups.pushInstructions(assetsDirectory: "assets"))
        XCTAssertEqual(loaded.displayName, "adapter · directory-a"); XCTAssertEqual(StatusFormat.bytes(loaded.totalBytes), "0.12 GB"); XCTAssertEqual(groups.unavailable.first?.error, "Missing weights")
    }
    func testRegenerationAndEditRemoveOnlySuffixPreservingHistory() {
        var chat = Conversation(configuration: config); chat.start("one"); _ = chat.makeRequest(thinking: true, supported: true); chat.appendAssistant(); chat.finish(content: "first", reasoning: "thought", metrics: "native metrics")
        chat.start("two"); chat.appendAssistant(); chat.finish(content: "second", reasoning: "more", metrics: "metrics")
        XCTAssertEqual(chat.prepareRegeneration(), "two"); XCTAssertEqual(chat.messages.count, 2); XCTAssertEqual(chat.messages.last?.reasoning, "thought"); XCTAssertEqual(chat.messages.last?.metrics, "native metrics")
        chat.start("two again"); chat.appendAssistant(); XCTAssertEqual(chat.prepareEdit(id: chat.messages.first!.id), "one"); XCTAssertTrue(chat.messages.isEmpty)
        chat.start("edited"); XCTAssertTrue(chat.makeRequest(thinking: false, supported: true).thinking)
        XCTAssertNil(chat.prepareRegeneration()); XCTAssertNil(chat.prepareEdit(id: UUID()))
    }
}
