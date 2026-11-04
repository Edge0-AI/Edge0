import XCTest
@testable import ChatCore

final class StatusTests: XCTestCase {
    private var shared: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("shared") }
    private func json(_ file: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: shared.appendingPathComponent(file))) }
    func testStatusVectors() throws {
        for item in try XCTUnwrap(json("status-cases.json") as? [[String: Any]]) {
            let input = try XCTUnwrap(item["input"] as? [String: Any]); let name = item["name"] as? String ?? ""
            let integer: (String) -> UInt64? = { (input[$0] as? NSNumber)?.uint64Value }
            let number = (input["value"] as? NSNumber)?.doubleValue
            let expected = item["expected"] as? String
            switch item["kind"] as? String {
            case "count": XCTAssertEqual(StatusFormat.count(integer("value")), expected, name)
            case "bytes": XCTAssertEqual(StatusFormat.bytes(integer("value")), expected, name)
            case "rate": XCTAssertEqual(StatusFormat.rate(number), expected, name)
            case "percent": XCTAssertEqual(StatusFormat.percent(integer("numerator"), integer("denominator")), expected, name)
            case "duration": XCTAssertEqual(StatusFormat.duration(number), expected, name)
            case "device_memory":
                var devices: [String: NativeDeviceMemory]?
                if let raw = input["by_device"] as? [String: [String: Any]] { devices = try raw.mapValues { value in .init(currentBytes: try XCTUnwrap(value["current_bytes"] as? NSNumber).uint64Value, peakBytes: try XCTUnwrap(value["peak_bytes"] as? NSNumber).uint64Value, memory: try XCTUnwrap(NativeMemoryLocation(rawValue: value["memory"] as? String ?? ""))) } }
                let result = StatusFormat.deviceMemory(devices)
                if let expected = item["expected"] as? [String: Any] { XCTAssertEqual(result?.system, expected["system"] as? String, name); XCTAssertEqual(result?.dedicated, expected["dedicated"] as? String, name) } else { XCTAssertNil(result, name) }
            default: XCTFail(name)
            }
        }
    }
    #if os(macOS)
    func testActualNativeStatusMapping() throws {
        let root = shared.deletingLastPathComponent().deletingLastPathComponent(), ios = shared.deletingLastPathComponent().appendingPathComponent("ios")
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true); defer { try? FileManager.default.removeItem(at: temporary) }
        let executable = temporary.appendingPathComponent("status-test"), compiler = Process(), diagnostics = Pipe()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        compiler.arguments = ["--sdk", "macosx", "clang++", "-std=c++17", "-fobjc-arc", "-framework", "Foundation", "-I" + root.appendingPathComponent("include").path, "-I" + ios.path, ios.appendingPathComponent("Tests/StatusMappingTests.mm").path, "-o", executable.path]
        compiler.standardOutput = diagnostics; compiler.standardError = diagnostics
        try compiler.run(); let output = diagnostics.fileHandleForReading.readDataToEndOfFile(); compiler.waitUntilExit()
        XCTAssertEqual(compiler.terminationStatus, 0, String(decoding: output, as: UTF8.self)); guard compiler.terminationStatus == 0 else { return }
        let test = Process(); test.executableURL = executable; test.arguments = [shared.appendingPathComponent("status-sample.json").path]
        try test.run(); test.waitUntilExit(); XCTAssertEqual(test.terminationStatus, 0)
    }
    #endif
    func testAssistantStateVectors() throws {
        for item in try XCTUnwrap(json("assistant-state-cases.json") as? [[String: Any]]) {
            let input = try XCTUnwrap(item["input"] as? [String: Bool]), expected = try XCTUnwrap(item["expected"] as? [String: Bool])
            let result = AssistantState(isLast: input["is_last"]!, generating: input["generating"]!, hasContent: input["has_content"]!, showsReasoning: input["shows_reasoning"]!)
            XCTAssertEqual(result.pending, expected["pending"], item["name"] as? String ?? "")
            XCTAssertEqual(result.actions, expected["actions"], item["name"] as? String ?? "")
        }
        var message = ChatMessage(role: "assistant", content: "")
        XCTAssertTrue(message.state(isLast: true, generating: true, thinking: true).pending)
        message.reasoning = "step"
        XCTAssertFalse(message.state(isLast: true, generating: true, thinking: true).pending)
        XCTAssertTrue(message.state(isLast: true, generating: true, thinking: false).pending)
    }
    func testThermalVectors() throws {
        for item in try XCTUnwrap(json("thermal-cases.json") as? [[String: Any]]) {
            let input = try XCTUnwrap(item["input"] as? [String: String]), expected = try XCTUnwrap(item["expected"] as? [String: String])
            let result = ThermalDisplay(state: try XCTUnwrap(input["state"])), name = item["name"] as? String ?? ""
            XCTAssertEqual(result.label, expected["label"], name)
            XCTAssertEqual(result.description, expected["description"], name)
            XCTAssertEqual(result.tone.rawValue, expected["tone"], name)
        }
    }
    func testContextVectors() throws {
        for item in try XCTUnwrap(json("context-cases.json") as? [[String: Any]]) {
            let input = try XCTUnwrap(item["input"] as? [String: NSNumber]), expected = try XCTUnwrap(item["expected"] as? [String: Any])
            let result = ContextUsage(inputTokens: input["input_tokens"]?.uint64Value, generatedTokens: input["generated_tokens"]?.uint64Value, contextLength: input["context_length"]?.uint64Value)
            XCTAssertEqual(result.text, expected["text"] as? String, item["name"] as? String ?? "")
            XCTAssertEqual(result.ringPercent, (expected["ring_percent"] as? NSNumber)?.intValue)
        }
    }
    func testNativeStructureSampleUsesStatusSampleValues() throws {
        let sample = try XCTUnwrap(json("status-sample.json") as? [String: Any])
        // These are the fields delivered by the native bridge, populated from the
        // same native GenerationProgress, MemorySnapshot and SystemStatus members.
        let progress = NativeProgress(phase: "decode", inputTokens: 13, reusedTokens: 12, computedTokens: 1, pendingTokens: 0, generatedTokens: 3, elapsedSeconds: 0.0061735, firstTokenSeconds: 0.002878875, prefillTokensPerSecond: 347.3579089053884, decodeElapsedSeconds: 0.003294625, decodeTokensPerSecond: 607.0493607011421)
        let native = NativeSnapshot(ledger: .init(limitBytes: 17179869184, currentBytes: 46837024, peakBytes: 48252608, byPurpose: ["resident_weights": 20945152, "experts": 491520, "session_state": 13328384, "workspace": 4231136, "transfer": 2176, "staging": 0, "metadata": 7838656], byDevice: ["CPU": .init(currentBytes: 38998368, peakBytes: 39993312, memory: .system), "host": .init(currentBytes: 7838656, peakBytes: 8259296, memory: .system)]), resident: .init(values: .init(currentBytes: 199081984, peakBytes: 220856320), reason: "", source: "task_info.resident_size/ru_maxrss"), physical: .init(values: .init(currentBytes: 169476720, peakBytes: 215188032), reason: "", source: "phys_footprint"))
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        XCTAssertEqual(progress, try decoder.decode(NativeProgress.self, from: JSONSerialization.data(withJSONObject: sample["current"]!)))
        XCTAssertEqual(native, try decoder.decode(NativeSnapshot.self, from: JSONSerialization.data(withJSONObject: sample["memory"]!)))
        XCTAssertEqual(progress.prefillText, "1 tokens · 0.0 s · 347.4 tok/s"); XCTAssertEqual(progress.decodeText, "3 tokens · 607.0 tok/s")
        XCTAssertEqual(StatusFormat.deviceMemory(native.ledger.byDevice), .init(system: "0.05 GB", dedicated: nil))
        XCTAssertEqual(native.ledger.segments.reduce(UInt64(0)) { $0 + $1.1 }, native.ledger.currentBytes)
        let system = NativeSystem(memory: .init(total: .init(bytes: 17179869184, source: "sysctl.hw.memsize", reason: ""), available: .init(bytes: 4924522496, source: "host_statistics64.free_count+inactive_count+purgeable_count+speculative_count", reason: "")), thermal: .init(state: "nominal", source: "NSProcessInfo.thermalState", reason: ""), appMemoryLimit: .init(bytes: nil, source: "", reason: "App memory limit is only available on iOS"))
        XCTAssertEqual(system, try decoder.decode(NativeSystem.self, from: JSONSerialization.data(withJSONObject: sample["system"]!)))
        XCTAssertNil(system.appMemoryLimit.bytes)
        var prefill = progress; prefill.phase = "prefill"; prefill.inputTokens = 3084; prefill.reusedTokens = 1262; prefill.computedTokens = 0; prefill.pendingTokens = 1822; prefill.elapsedSeconds = 2.9
        XCTAssertEqual(prefill.prefillText, "Computing 1,822 tokens · 2.9 s")
        prefill.computedTokens = 1000; prefill.pendingTokens = 822
        XCTAssertEqual(prefill.prefillText, "Computing 1,822 tokens · 2.9 s")
    }
    func testTotalsUseSumsDeduplicatePersistAndReset() throws {
        let first = NativeGenerationSummary(inputTokens: 100, reusedTokens: 10, computedTokens: 95, generatedTokens: 8, firstTokenSeconds: 2, decodeSeconds: 1, decodeForwards: 7, totalSeconds: 3, experts: .init(demand: 10, hits: 6), reuseMethod: "first", checkpointRestored: false)
        var second = first; second.decodeSeconds = 3; second.decodeForwards = 3; second.firstTokenSeconds = 1
        var totals = UsageTotals(); let id = UUID()
        totals.record(first, id: id); totals.record(first, id: id); totals.record(second, id: UUID())
        XCTAssertEqual(totals.requests, 2); XCTAssertEqual(totals.averageDecode, "2.5 tok/s"); XCTAssertEqual(totals.averagePrefill, "60.0 tok/s"); XCTAssertEqual(totals.reused, "10.0%"); XCTAssertEqual(totals.expertHits, "60.0%")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("totals.json")
        try AppFiles.write(totals, at: file)
        XCTAssertEqual(try AppFiles.read(UsageTotals.self, at: file), totals)
        let roundtrip = try JSONDecoder().decode(UsageTotals.self, from: JSONEncoder().encode(totals)); XCTAssertEqual(roundtrip, totals)
        totals.reset(); XCTAssertEqual(totals.requests, 0); XCTAssertNil(totals.averageDecode); XCTAssertNil(totals.expertHits); totals.record(first, id: id); XCTAssertEqual(totals.requests, 0)
        try AppFiles.write(totals, at: file)
        let reset = try AppFiles.read(UsageTotals.self, at: file); XCTAssertEqual(reset.requests, 0); XCTAssertEqual(reset.generationIDs, totals.generationIDs)
    }
    func testPollingVisibilityCompletionAndNonoverlap() {
        var polling = PollSchedule(); XCTAssertFalse(polling.begin(at: 0)); polling.show(); XCTAssertTrue(polling.begin(at: 0)); XCTAssertFalse(polling.begin(at: 2))
        polling.hide(); polling.show(); XCTAssertFalse(polling.begin(at: 2)); polling.finish(at: 2); XCTAssertFalse(polling.begin(at: 2.999)); XCTAssertTrue(polling.begin(at: 3)); polling.hide(); polling.finish(at: 4); XCTAssertFalse(polling.begin(at: 100))
        polling.show(); XCTAssertTrue(polling.begin(at: 100)); polling.finish(at: 101)
        polling.hide(); polling.show(); XCTAssertFalse(polling.begin(at: 101.1)); XCTAssertTrue(polling.begin(at: 102))
    }
}
