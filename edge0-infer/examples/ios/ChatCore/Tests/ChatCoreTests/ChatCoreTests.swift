import XCTest
@testable import ChatCore
final class ChatCoreTests: XCTestCase {
    private var shared: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("shared") }
    func testSharedConfigurationAndHistory() throws {
        let config = try AppConfiguration.load(shared.appendingPathComponent("app.json"))
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: shared.appendingPathComponent("app.json"))) as! [String: Any]
        XCTAssertEqual(config.systemPrompt, raw["system_prompt"] as? String)
        XCTAssertEqual(config.suggestions, raw["suggestions"] as? [String]); XCTAssertEqual(config.suggestions.count, 4)
        var history = Conversation(configuration: config)
        history.start("first"); history.appendAssistant(); history.fragment("thought", reasoning: true); history.fragment("answer", reasoning: false); history.start("second")
        XCTAssertEqual(history.request.count, 4); XCTAssertEqual(history.request[2]["reasoning_content"], "thought"); XCTAssertEqual(history.request[2]["content"], "answer")
        history.reset(); history.start("new")
        XCTAssertEqual(history.request, [["role": "system", "content": config.systemPrompt], ["role": "user", "content": "new", "reasoning_content": ""]])
    }
    func testThinkingConfigurationUsesCapabilitiesAndSwitch() throws {
        let configuration = try AppConfiguration.load(shared.appendingPathComponent("app.json"))
        var conversation = Conversation(configuration: configuration)
        XCTAssertTrue(conversation.canChangeThinking(supported: true, busy: false))
        conversation.start("first")
        XCTAssertTrue(conversation.makeRequest(thinking: true, supported: true).thinking)
        XCTAssertFalse(conversation.canChangeThinking(supported: true, busy: false))
        conversation.appendAssistant(); conversation.fragment("thought", reasoning: true); conversation.fragment("answer", reasoning: false)
        conversation.start("second")
        let locked = conversation.makeRequest(thinking: false, supported: true)
        XCTAssertTrue(locked.thinking)
        XCTAssertEqual(locked.messages.first?["content"], configuration.systemPrompt)
        XCTAssertEqual(locked.messages[2]["reasoning_content"], "thought")
        conversation.reset()
        XCTAssertTrue(conversation.canChangeThinking(supported: true, busy: false))
        XCTAssertFalse(conversation.canChangeThinking(supported: true, busy: true))
        XCTAssertFalse(conversation.makeRequest(thinking: false, supported: true).thinking)
        conversation.reset()
        XCTAssertFalse(conversation.canChangeThinking(supported: false, busy: false))
        XCTAssertFalse(conversation.makeRequest(thinking: true, supported: false).thinking)
    }
    func testSendGenerationParametersLeaveNormalLimitsUnset() {
        for thinking in [false, true] {
            for model in ["35b", "tiny-a"] {
                for arguments in [["edge0-chat"], ["edge0-chat", "--smoke-max-tokens", "16"]] {
                    let normal = GenerationParameters(thinking: thinking, arguments: arguments, modelName: model)
                    XCTAssertEqual(normal.thinking, thinking)
                    XCTAssertNil(normal.maxTokens)
                    XCTAssertFalse(normal.tokenEvents)
                }
            }
        }
        let smoke = ["edge0-chat", "--smoke", "35b", "--smoke-max-tokens", "96"]
        let real = GenerationParameters(thinking: false, arguments: smoke, modelName: "35b")
        XCTAssertEqual(real.maxTokens, 96); XCTAssertFalse(real.tokenEvents)
        let tiny = GenerationParameters(thinking: true, arguments: smoke, modelName: "tiny-a")
        XCTAssertEqual(tiny.maxTokens, 96); XCTAssertTrue(tiny.tokenEvents)
        for suffix in [[], ["--smoke-max-tokens"], ["--smoke-max-tokens", "0"], ["--smoke-max-tokens", "-1"], ["--smoke-max-tokens", "bad"]] {
            let unlimited = GenerationParameters(thinking: false, arguments: ["edge0-chat", "--smoke", "tiny-a"] + suffix, modelName: "tiny-a")
            XCTAssertNil(unlimited.maxTokens); XCTAssertFalse(unlimited.tokenEvents)
        }
    }
    #if os(macOS)
    func testActualBridgeNativeGenerationOverrideMapping() throws {
        let root = shared.deletingLastPathComponent().deletingLastPathComponent()
        let ios = root.appendingPathComponent("examples/ios")
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let executable = temporary.appendingPathComponent("generation-options-test")
        let compiler = Process(), diagnostics = Pipe()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        compiler.arguments = ["--sdk", "macosx", "clang++", "-std=c++17", "-I" + root.appendingPathComponent("include").path,
            "-I" + ios.path, ios.appendingPathComponent("Tests/GenerationOptionsTests.cpp").path, "-o", executable.path]
        compiler.standardOutput = diagnostics; compiler.standardError = diagnostics
        try compiler.run()
        let output = diagnostics.fileHandleForReading.readDataToEndOfFile()
        compiler.waitUntilExit()
        XCTAssertEqual(compiler.terminationStatus, 0, String(decoding: output, as: UTF8.self))
        guard compiler.terminationStatus == 0 else { return }
        let nativeTest = Process(); nativeTest.executableURL = executable
        try nativeTest.run(); nativeTest.waitUntilExit()
        XCTAssertEqual(nativeTest.terminationStatus, 0)
    }
    #endif
    func testStopFeedbackAndMessagePresentation() throws {
        var phase = GenerationPhase.idle
        XCTAssertFalse(phase.isGenerating)
        phase.start(); XCTAssertTrue(phase.stopEnabled); XCTAssertEqual(phase.stopTitle, "Stop")
        phase.requestStop(); XCTAssertFalse(phase.stopEnabled); XCTAssertEqual(phase.stopTitle, "Stopping…")
        phase.requestStop(); XCTAssertEqual(phase, .stopping)
        phase.finish(); XCTAssertFalse(phase.isGenerating)
        let config = try AppConfiguration.load(shared.appendingPathComponent("app.json"))
        var conversation = Conversation(configuration: config)
        conversation.start("hello"); conversation.appendAssistant()
        conversation.finish(content: "partial", reasoning: "", metrics: "3 tokens", stopped: true)
        XCTAssertEqual(conversation.messages.last?.statusLabel, "Stopped")
        XCTAssertEqual(conversation.messages.last?.metrics, "3 tokens")
        XCTAssertTrue(conversation.messages.first!.usesBubble)
        XCTAssertFalse(conversation.messages.last!.usesBubble)
        XCTAssertFalse(conversation.messages.last!.metrics.contains("state_valid"))
    }
    func testModelLabelsAndUnavailablePartition() {
        let valid = ModelChoice(path: "assets/35b", name: "35b", error: nil, modelIdentifier: "edge0_35b")
        let invalid = ModelChoice(path: "assets/incomplete", name: "incomplete", error: "Missing model.json")
        XCTAssertEqual(valid.displayName, "edge0_35b · 35b")
        let choices = ModelChoices([valid, invalid])
        XCTAssertEqual(choices.available, [valid]); XCTAssertEqual(choices.unavailable, [invalid])
        XCTAssertEqual(choices.unavailable.first?.displayName, "incomplete")
    }
    func testCleanupOnlyExactFixturesWithoutFollowingLinks() throws {
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? files.removeItem(at: root) }
        let assets = root.appendingPathComponent("assets"), outside = root.appendingPathComponent("outside")
        try files.createDirectory(at: assets, withIntermediateDirectories: true)
        try files.createDirectory(at: outside, withIntermediateDirectories: true)
        let sentinel = outside.appendingPathComponent("keep")
        try Data("keep".utf8).write(to: sentinel)
        for name in ["35b", "other", "tiny-a", "incomplete"] { try files.createDirectory(at: assets.appendingPathComponent(name), withIntermediateDirectories: true) }
        try files.createSymbolicLink(at: assets.appendingPathComponent("tiny-b"), withDestinationURL: outside)
        try files.createSymbolicLink(at: assets.appendingPathComponent("tiny-a/nested-link"), withDestinationURL: outside)
        XCTAssertEqual(try ExampleFixtureCleanup.run(in: assets), ["35b", "other"])
        XCTAssertTrue(files.fileExists(atPath: sentinel.path))
        XCTAssertEqual(try ExampleFixtureCleanup.run(in: assets), ["35b", "other"])
    }
    func testCleanupRejectsSymbolicLinkRoot() throws {
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? files.removeItem(at: root) }
        let target = root.appendingPathComponent("target")
        try files.createDirectory(at: target.appendingPathComponent("tiny-a"), withIntermediateDirectories: true)
        let link = root.appendingPathComponent("assets")
        try files.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try ExampleFixtureCleanup.run(in: link))
        XCTAssertTrue(files.fileExists(atPath: target.appendingPathComponent("tiny-a").path))
    }
    func testActualPageStateAndEmptyInstructions() {
        var page = ModelPageState()
        XCTAssertFalse(page.showsChat); XCTAssertFalse(page.hasLoadedModel)
        page.beginLoad(); XCTAssertFalse(page.showsChat)
        page.loadSucceeded(); XCTAssertTrue(page.showsChat)
        page.showModels(); XCTAssertFalse(page.showsChat); XCTAssertTrue(page.hasLoadedModel)
        page.resumeChat(); XCTAssertTrue(page.showsChat)
        page.beginLoad(); XCTAssertFalse(page.hasLoadedModel); XCTAssertFalse(page.showsChat)
        page.loadFailed(); page.resumeChat(); XCTAssertFalse(page.showsChat)
        page.loadSucceeded(); page.close(); XCTAssertFalse(page.showsChat)
        let empty = ModelChoices([])
        XCTAssertEqual(empty.pushInstructions(assetsDirectory: "Documents/assets"), "Use tests/platform/ios_push.py (see README) to push assets to Documents/assets.")
        let available = ModelChoices([ModelChoice(path: "assets/35b", name: "35b", error: nil, modelIdentifier: "edge0_35b")])
        XCTAssertNil(available.pushInstructions(assetsDirectory: "Documents/assets"))
    }
    func testReasoningDisclosureUsesMessageStateWithoutChangingHistory() throws {
        let configuration = try AppConfiguration.load(shared.appendingPathComponent("app.json"))
        var conversation = Conversation(configuration: configuration)
        conversation.start("question"); conversation.appendAssistant(); conversation.fragment("first", reasoning: true)
        let id = conversation.messages.last!.id
        XCTAssertTrue(conversation.messages.last!.reasoningExpanded)
        XCTAssertTrue(conversation.messages.last!.showsReasoning(thinking: true))
        XCTAssertFalse(conversation.messages.last!.showsReasoning(thinking: false))
        let beforeCollapse = conversation.messages.last!.scrollContent
        conversation.setReasoningExpanded(id: id, expanded: false)
        XCTAssertEqual(conversation.messages.last!.scrollContent, beforeCollapse)
        conversation.fragment(" second", reasoning: true)
        XCTAssertNotEqual(conversation.messages.last!.scrollContent, beforeCollapse)
        let beforeMetrics = conversation.messages.last!.scrollContent
        conversation.finish(content: "answer", reasoning: "first second", metrics: "2 tokens")
        XCTAssertNotEqual(conversation.messages.last!.scrollContent, beforeMetrics)
        XCTAssertFalse(conversation.messages.last!.reasoningExpanded)
        let request = conversation.makeRequest(thinking: true, supported: true)
        XCTAssertEqual(request.messages.last?["reasoning_content"], "first second")
        XCTAssertEqual(request.messages.last?["content"], "answer")
        XCTAssertEqual(Set(request.messages.last!.keys), ["role", "content", "reasoning_content"])
    }
    func testLockedThinkingPresentationRetainsOnOffColors() {
        let on = ThinkingControlPresentation(isOn: true, enabled: false)
        let off = ThinkingControlPresentation(isOn: false, enabled: false)
        XCTAssertEqual(on.trackTone, .green); XCTAssertEqual(off.trackTone, .gray)
        XCTAssertEqual(on.stateLabel, "On"); XCTAssertEqual(off.stateLabel, "Off")
        XCTAssertEqual(on.opacity, 0.55); XCTAssertEqual(off.opacity, 0.55)
        XCTAssertEqual(ThinkingControlPresentation(isOn: true, enabled: true).opacity, 1)
        XCTAssertEqual(ThinkingControlPresentation(isOn: true, enabled: true).stateLabel, "On")
        XCTAssertEqual(ThinkingControlPresentation(isOn: false, enabled: true).stateLabel, "Off")
    }
    func testMetricsVectors() throws {
        struct Case: Decodable { let name: String; let input: EngineMetrics; let expected: String }
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: shared.appendingPathComponent("metrics-cases.json")))
        for item in cases { XCTAssertEqual(item.input.formatted, item.expected, item.name) }
    }
    func testMathVectors() throws {
        struct Case: Decodable { let name: String; let input: String; let expected: String }
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: shared.appendingPathComponent("math-cases.json")))
        for item in cases { XCTAssertEqual(Markdown.protectMath(item.input), item.expected, item.name) }
    }
    func testMarkdownBlocksAndStreaming() {
        let blocks = Markdown.blocks("# Heading\n\n**bold** [link](https://example.com)\n\n- list\n> quote\n```swift\nlet x = 1\n```")
        XCTAssertEqual(blocks.map(\.kind), [.heading(1), .paragraph, .list, .quote, .code("swift")])
        XCTAssertEqual(Markdown.blocks("```swift\nunfinished").last?.text, "unfinished")
        XCTAssertEqual(Markdown.blocks("$$\nx_1\n$$").first?.text, "$$\nx_1\n$$")
        XCTAssertEqual(Markdown.blocks("````text\nfirst\n```\n$x$\n````"), [Markdown.Block(id: 0, kind: .code("text"), text: "first\n```\n$x$")])
        XCTAssertEqual(Markdown.blocks("~~~text\n$x$\n~~~").first?.text, "$x$")
        XCTAssertEqual(Markdown.blocks("```python\ns = \"```\"\n$x$\n```").first?.text, "s = \"```\"\n$x$")
    }
    func testBlockMathInsideParagraphPreservesSurroundingText() {
        for formula in ["$$x_i$$", #"\[x^2\]"#] {
            let blocks = Markdown.blocks("Before \(formula) after")
            XCTAssertEqual(blocks.map(\.kind), [.paragraph, .code("text"), .paragraph])
            XCTAssertEqual(blocks.map(\.text), ["Before ", formula, " after"])
        }
        XCTAssertEqual(Markdown.protectMath("Before\n$$x_i$$\nafter"), "Before\n```text\n$$x_i$$\n```\nafter")
    }
    func testAssetAvailability() {
        XCTAssertTrue(ModelChoice(path: "valid", name: "valid", error: nil).selectable)
        let invalid = ModelChoice(path: "missing", name: "missing", error: "missing file")
        XCTAssertFalse(invalid.selectable); XCTAssertEqual(invalid.error, "missing file")
    }
}
