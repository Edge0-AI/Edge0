import Foundation
import SwiftUI
import UIKit
import Darwin

// Engine operations are serialized by worker; cancel alone is thread-safe.
extension Edge0Bridge: @unchecked Sendable {}

enum AppTab: String, CaseIterable { case chat = "Chat", models = "Models", status = "Status", settings = "Settings" }

@MainActor final class ChatViewModel: ObservableObject {
    @Published var tab = AppTab.chat { didSet { updatePolling() } }
    @Published var settings = AppSettings() { didSet { try? AppFiles.write(settings, at: support.appendingPathComponent("settings.json")) } }
    @Published private(set) var savedChats: [SavedConversation] = []
    @Published private(set) var damagedChats: [String] = []
    @Published private(set) var contextLength: UInt64?
    @Published private(set) var lastSummary: NativeGenerationSummary?
    @Published private(set) var progress: NativeProgress?
    @Published private(set) var snapshot: NativeSnapshot?
    @Published private(set) var systemStatus: NativeSystem?
    @Published private(set) var sessionTotals = UsageTotals()
    @Published private(set) var lifetimeTotals = UsageTotals()
    @Published var lifetimeView = false
    private var defaults: AppConfiguration?
    private var chatID = UUID()
    private var currentInput: UInt64?, currentGenerated: UInt64?
    private var pollTask: Task<Void, Never>?
    private var pollSchedule = PollSchedule()
    private var active = true
    nonisolated private let statusWorker = DispatchQueue(label: "dev.edge0.example.status", qos: .utility)
    private var support: URL { FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Edge0Chat") }
    private var historyFiles: ConversationFiles { .init(directory: support.appendingPathComponent("conversations")) }
    var contextUsage: ContextUsage { .init(inputTokens: currentInput, generatedTokens: currentGenerated, contextLength: contextLength) }
    var totals: UsageTotals { lifetimeView ? lifetimeTotals : sessionTotals }
    var chatTitle: String { SavedConversation.title(messages.first(where: { $0.role == "user" })?.content ?? "New chat") }
    var defaultPrompt: String { defaults?.systemPrompt ?? "" }
    var canRegenerate: Bool { !busy && (conversation?.canRegenerate ?? false) }
    func refreshHistory() { do { let listing = try historyFiles.list(); savedChats = listing.conversations; damagedChats = listing.damaged } catch { self.error = "Could not read conversation history" } }
    private func persistChat() {
        guard let conversation, !conversation.messages.isEmpty, let selected else { return }
        do { try historyFiles.save(.init(id: chatID, modelPath: selected, conversation: conversation, inputTokens: currentInput, generatedTokens: currentGenerated)); refreshHistory() }
        catch { self.error = "Could not save conversation; damaged files were left unchanged." }
    }
    func deleteChat(_ id: UUID) {
        guard !busy else { return }
        do { try historyFiles.delete(id); refreshHistory() } catch { self.error = "Could not delete conversation" }
        if id == chatID { conversation?.reset(); newConversation() }
    }
    func restoreChat(_ saved: SavedConversation) {
        guard !busy else { return }
        let restore = { [self] in
            chatID = saved.id; conversation = saved.conversation; currentInput = saved.inputTokens; currentGenerated = saved.generatedTokens
            thinking = saved.conversation.thinkingForChat ?? thinking; tab = .chat; page.resumeChat()
        }
        if selected == saved.modelPath { restore() }
        else if let choice = models.first(where: { $0.path == saved.modelPath && $0.selectable }) { select(choice, completion: restore) }
        else { error = "This conversation's model is unavailable." }
    }
    func regenerate() { guard canRegenerate, let text = conversation?.prepareRegeneration() else { return }; send(text) }
    func editMessage(_ id: UUID, text: String) {
        guard !busy, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, conversation?.prepareEdit(id: id) != nil else { return }; send(text)
    }
    func resetTotals() { lifetimeTotals.reset(); try? AppFiles.write(lifetimeTotals, at: support.appendingPathComponent("totals.json")) }
    private func decodeNative<T: Decodable>(_ type: T.Type, _ object: [String: Any]) -> T? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return nil }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try? decoder.decode(type, from: data)
    }
    private func updatePolling() {
        pollTask?.cancel(); pollTask = nil; pollSchedule.hide()
        guard active && tab == .status else { return }
        pollSchedule.show()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.active, self.tab == .status else { return }
                if self.pollSchedule.begin(at: ProcessInfo.processInfo.systemUptime) {
                    let rows: [[String: Any]] = await withCheckedContinuation { continuation in
                        self.statusWorker.async { [bridge = self.bridge] in continuation.resume(returning: [bridge.progress(), bridge.snapshot(), bridge.systemStatus()]) }
                    }
                    self.pollSchedule.finish(at: ProcessInfo.processInfo.systemUptime)
                    guard !Task.isCancelled, self.active, self.tab == .status else { return }
                    self.progress = self.decodeNative(NativeProgress.self, rows[0])
                    self.snapshot = self.decodeNative(NativeSnapshot.self, rows[1])
                    self.systemStatus = self.decodeNative(NativeSystem.self, rows[2])
                }
                do { try await Task.sleep(for: .seconds(1)) } catch { return }
            }
        }
    }

    @Published private(set) var conversation: Conversation?
    @Published private(set) var models: [ModelChoice] = []
    @Published private(set) var selected: String?
    @Published private(set) var page = ModelPageState()
    @Published private(set) var busy = false
    @Published private(set) var generationPhase = GenerationPhase.idle
    var generating: Bool { generationPhase.isGenerating }
    var stopping: Bool { generationPhase == .stopping }
    @Published private(set) var thinkingSupported = false
    @Published private(set) var thinking = false
    @Published var draft = ""
    @Published private(set) var validationScrollTarget = "bottom"
    @Published private(set) var status = "Preparing models"
    @Published private(set) var error: String?
    @Published private(set) var lastStop = ""
    nonisolated private let bridge = Edge0Bridge()
    nonisolated private let worker = DispatchQueue(label: "dev.edge0.example.inference", qos: .userInitiated)
    private var usable = false
    private var lastNativeMetrics: EngineMetrics?
    private let validationSession = UUID().uuidString
    private var validationPhase = "initial"
    private var foregroundFollowupComplete = false
    private var backgroundCancellationSafe = false
    private var didCaptureStreaming = false
    private var didCaptureThinking = false
    private var presentationChecksEnabled: Bool { ProcessInfo.processInfo.arguments.contains("--presentation-checks") }
    private var thinkingLabelChecksEnabled: Bool { ProcessInfo.processInfo.arguments.contains("--thinking-label-checks") }
    private var decodeComparisonEnabled: Bool { ProcessInfo.processInfo.arguments.contains("--decode-comparison") }
    private var validationEnabled: Bool { ProcessInfo.processInfo.arguments.contains("--smoke") }
    var canSend: Bool { page.showsChat && usable && !busy && conversation != nil }
    var canChangeThinking: Bool { page.showsChat && (conversation?.canChangeThinking(supported: thinkingSupported, busy: busy) ?? false) }
    var thinkingHint: String? { conversation?.thinkingLocked == true ? "Applies to a new chat" : nil }
    var thinkingPresentation: ThinkingControlPresentation { .init(isOn: thinking, enabled: canChangeThinking) }
    var assetsDirectory: String { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("assets").path }
    func showModels() { if !busy { page.showModels(); tab = .models } }
    func resumeChat() { if !busy { page.resumeChat(); tab = .chat } }
    func setReasoningExpanded(id: UUID, expanded: Bool) { conversation?.setReasoningExpanded(id: id, expanded: expanded) }
    var modelChoices: ModelChoices { ModelChoices(models, selectedPath: selected) }
    func setThinking(_ value: Bool) { if canChangeThinking { thinking = value } }
    var messages: [ChatMessage] { conversation?.messages ?? [] }
    var suggestions: [String] { conversation?.configuration.suggestions ?? [] }
    init() {
        if ProcessInfo.processInfo.arguments.contains("--cleanup-status-fixtures") {
            let args = ProcessInfo.processInfo.arguments
            let badName = args.firstIndex(of: "--bad-assets-name").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
            let includeTiny = args.contains("--remove-owned-tiny")
            worker.async {
                do {
                    let assets = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("assets")
                    try ExampleFixtureCleanup.runStatusCheck(in: assets, badName: badName, includeTiny: includeTiny)
                    NSLog("EXAMPLE_CHECK_CLEANUP complete bad_asset=\(badName != nil) tiny=\(includeTiny)")
                    fflush(nil); exit(0)
                } catch { NSLog("EXAMPLE_CHECK_CLEANUP failed"); fflush(nil); exit(1) }
            }
            return
        }
        if ProcessInfo.processInfo.arguments.contains("--cleanup-example-fixtures") {
            worker.async {
                do {
                    let assets = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("assets")
                    let remaining = try ExampleFixtureCleanup.run(in: assets)
                    NSLog("EXAMPLE_CLEANUP remaining=\(remaining.joined(separator: ","))")
                    fflush(nil); exit(0)
                } catch {
                    NSLog("EXAMPLE_CLEANUP failed: \(error.localizedDescription)")
                    fflush(nil); exit(1)
                }
            }
            return
        }
        do {
            guard let url = Bundle.main.url(forResource: "app", withExtension: "json") else { throw CocoaError(.fileNoSuchFile) }
            let configuration = try AppConfiguration.load(url); defaults = configuration
            settings = (try? AppFiles.read(AppSettings.self, at: support.appendingPathComponent("settings.json"))) ?? AppSettings()
            lifetimeTotals = (try? AppFiles.read(UsageTotals.self, at: support.appendingPathComponent("totals.json"))) ?? UsageTotals()
            conversation = Conversation(configuration: settings.configuration(defaults: configuration)); refreshHistory()
        } catch { self.error = "Could not read app.json: \(error.localizedDescription)" }
        refresh()
    }
    func refresh() {
        guard !busy else { return }
        busy = true; error = nil
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("assets")
        if validationEnabled && !decodeComparisonEnabled && !presentationChecksEnabled && !thinkingLabelChecksEnabled && !ProcessInfo.processInfo.arguments.contains("--bad-assets-name") && !ProcessInfo.processInfo.arguments.contains("--model-switch-check") { try? FileManager.default.createDirectory(at: root.appendingPathComponent("incomplete-example-check"), withIntermediateDirectories: true) }
        worker.async { [self] in
            let rows = bridge.models(at: root.path)
            let choices = rows.compactMap { row -> ModelChoice? in
                guard let path = row["path"] as? String, let name = row["name"] as? String else { return nil }
                return ModelChoice(path: path, name: name, error: row["error"] as? String, modelIdentifier: row["model_identifier"] as? String, totalBytes: (row["total_bytes"] as? NSNumber)?.uint64Value)
            }
            DispatchQueue.main.async { [self] in
                if validationEnabled { NSLog("EXAMPLE_ASSETS valid=\(choices.filter(\.selectable).map(\.name).joined(separator: ",")) invalid=\(choices.filter { !$0.selectable }.map(\.name).joined(separator: ","))") }
                models = choices; busy = false; status = choices.isEmpty ? "Push model assets to Documents/assets, then refresh." : "Choose a model"
                let args = ProcessInfo.processInfo.arguments
                if let marker = args.firstIndex(of: "--bad-assets-name"), marker + 1 < args.count {
                    let checked = choices.first { $0.name == args[marker + 1] }
                    guard let checked, !checked.selectable, let reason = checked.error, !reason.isEmpty, !modelChoices.available.contains(where: { $0.path == checked.path }) else { NSLog("EXAMPLE_BAD_ASSETS failed mapping"); return }
                    tab = .models
                    capture("bad-assets-models") { NSLog("EXAMPLE_BAD_ASSETS complete unavailable=true reason=\(reason) page=models") }
                    return
                }
                if let marker = args.firstIndex(of: "--smoke"), args.count > marker + 1, let model = models.first(where: { $0.name == args[marker + 1] && $0.selectable }) {
                    if presentationChecksEnabled {
                        capture("model-chooser") { [weak self] in self?.select(model) { self?.capture("chat-loaded") { self?.smoke() } } }
                    } else { select(model) { [weak self] in self?.capture("loaded") { self?.smoke() } } }
                }
            }
        }
    }
    func select(_ model: ModelChoice, completion: (() -> Void)? = nil) {
        guard !busy, model.selectable else { return }
        persistChat(); page.beginLoad(); selected = nil
        busy = true; usable = false; error = nil; status = "Loading \(model.name)"
        worker.async { [self] in
            let result = bridge.load(model.path)
            DispatchQueue.main.async { [self] in
                busy = false
                if let failure = result["error"] as? String { error = failure; status = "Load failed"; selected = nil; page.loadFailed(); return }
                selected = model.path; page.loadSucceeded(); usable = true; tab = .chat
                if let defaults { conversation = Conversation(configuration: settings.configuration(defaults: defaults)) }
                chatID = UUID(); currentInput = nil; currentGenerated = nil; contextLength = (result["context"] as? NSNumber)?.uint64Value; lastStop = ""
                thinkingSupported = result["thinking_supported"] as? Bool ?? false
                thinking = result["thinking_default"] as? Bool ?? false
                status = "Ready"
                completion?()
            }
        }
    }
    func send(_ suggestion: String? = nil, completion: (() -> Void)? = nil, cancelAfterFragment: Bool = false) {
        let text = (suggestion ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, !text.isEmpty else { return }
        draft = ""; error = nil; busy = true; generationPhase.start(); status = "Generating"
        conversation?.start(text)
        let request = conversation?.makeRequest(thinking: thinking, supported: thinkingSupported)
        let history = request?.messages ?? []
        conversation?.appendAssistant()
        let parameters = GenerationParameters(thinking: request?.thinking ?? false, arguments: ProcessInfo.processInfo.arguments,
            modelName: selected.map { URL(fileURLWithPath: $0).lastPathComponent })
        let temperature = settings.style.temperature
        let generationID = UUID()
        bridge.prepareGeneration()
        worker.async { [self] in
            var fragments = 0, answerFragments = 0, reasoningFragments = 0
            let result = bridge.generate(history, thinking: parameters.thinking, temperature: temperature, maxTokens: parameters.maxTokens.map { NSNumber(value: $0) }, tokenEvents: parameters.tokenEvents) { [self] kind, fragment in
                fragments += 1
                if kind == "answer" { answerFragments += 1 }
                if kind == "reasoning" { reasoningFragments += 1 }
                let capturePartial = kind == "answer" && answerFragments == 12
                let captureReasoning = kind == "reasoning" && reasoningFragments == 12
                if kind == "answer" || kind == "reasoning" { DispatchQueue.main.async { [self] in
                    conversation?.fragment(fragment, reasoning: kind == "reasoning")
                    if capturePartial && validationEnabled && !didCaptureStreaming { didCaptureStreaming = true; capture("streaming") }
                    if captureReasoning && validationEnabled && !didCaptureThinking { didCaptureThinking = true; capture("thinking") }
                } }
                if cancelAfterFragment && fragments == 3 { DispatchQueue.main.async { [self] in stop() } }
            }
            DispatchQueue.main.async { [self] in
                if let metrics = result["metrics"] as? [String: Any], let data = try? JSONSerialization.data(withJSONObject: metrics), let decoded = try? JSONDecoder().decode(EngineMetrics.self, from: data) {
                    lastNativeMetrics = decoded
                    conversation?.finish(content: result["text"] as? String ?? "", reasoning: result["reasoning"] as? String ?? "", metrics: decoded.formatted, stopped: result["stop"] as? String == "cancelled", nativeMetrics: decoded)
                }
                if let summary = result["summary"] as? [String: Any], let decoded = decodeNative(NativeGenerationSummary.self, summary) {
                    lastSummary = decoded; currentInput = decoded.inputTokens; currentGenerated = decoded.generatedTokens
                    sessionTotals.record(decoded, id: generationID); lifetimeTotals.record(decoded, id: generationID)
                    try? AppFiles.write(lifetimeTotals, at: support.appendingPathComponent("totals.json"))
                }
                persistChat()
                lastStop = "\(result["stop"] ?? "failed") · state valid: \(result["state_valid"] as? Bool == true ? "true" : "false")"
                if let failure = result["error"] as? String { error = failure }
                if result["state_valid"] as? Bool == false { usable = false; error = "Session is no longer valid. Reload the model." }
                generationPhase.finish(); busy = false; status = "Ready"; completion?()
            }
        }
    }
    func stop() {
        guard generationPhase.stopEnabled else { return }
        generationPhase.requestStop(); status = "Stopping…"; bridge.cancel()
        if validationEnabled { NSLog("EXAMPLE_UI stopping=true stop_enabled=false"); capture("stopping", delay: 0.02) }
    }
    func newConversation(completion: (() -> Void)? = nil) {
        guard !busy else { return }
        persistChat(); busy = true
        worker.async { [self] in
            let result = bridge.reset()
            DispatchQueue.main.async { [self] in
                busy = false
                if let failure = result["error"] as? String { error = failure; return }
                if let defaults { conversation = Conversation(configuration: settings.configuration(defaults: defaults)) }
                chatID = UUID(); currentInput = nil; currentGenerated = nil; usable = true; lastStop = ""; status = "Ready"; completion?()
            }
        }
    }
    func background() {
        active = false; updatePolling(); persistChat()
        validationPhase = "background"
        if validationEnabled { NSLog("EXAMPLE_LIFECYCLE background session=\(validationSession) history=\(messages.count) generating=\(generating)"); saveValidationState() }
        if generating { stop() }
    }
    func foreground() {
        active = true; updatePolling()
        let returning = validationPhase == "background"
        validationPhase = "foreground"
        if validationEnabled {
            NSLog("EXAMPLE_LIFECYCLE foreground session=\(validationSession) history=\(messages.count) generating=\(generating)")
            if returning && (ProcessInfo.processInfo.arguments.contains("--visual-checks") || ProcessInfo.processInfo.arguments.contains("--status-pages-checks") || ProcessInfo.processInfo.arguments.contains("--background-active-check")) && canSend {
                send("Reply in one short sentence: are you still ready to continue our conversation?") { [self] in
                    foregroundFollowupComplete = error == nil && usable
                    capture("foreground")
                }
            } else { capture("foreground") }
        }
    }
    private func saveValidationState() {
        guard validationEnabled else { return }
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("validation")
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let state: [String: Any] = ["session": validationSession, "lifecycle": validationPhase, "foreground_followup_complete": foregroundFollowupComplete, "background_cancellation_safe": backgroundCancellationSafe, "message_count": messages.count, "selected_model": selected.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "", "thinking": thinking, "thinking_supported": thinkingSupported, "generating": generating, "last_stop": lastStop, "error": error ?? "", "messages": messages.map { ["role": $0.role, "content": $0.content, "reasoning": $0.reasoning, "metrics": $0.metrics] }]
            try JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys]).write(to: directory.appendingPathComponent("state.json"), options: .atomic)
        } catch { NSLog("EXAMPLE_CAPTURE state failed: \(error.localizedDescription)") }
    }
    private func capture(_ name: String, delay: Double = 0.4, completion: (() -> Void)? = nil) {
        guard validationEnabled && !decodeComparisonEnabled else { completion?(); return }
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [self] in
            saveValidationState()
            guard let window = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).flatMap(\.windows).first(where: \.isKeyWindow), window.windowScene?.activationState == .foregroundActive else { completion?(); return }
            let format = UIGraphicsImageRendererFormat(); format.scale = window.screen.scale; format.opaque = true
            var rendered = false
            let image = UIGraphicsImageRenderer(bounds: window.bounds, format: format).image { _ in rendered = window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
            let target = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("validation/\(name).png")
            do {
                guard rendered, let data = image.pngData() else { throw CocoaError(.fileWriteUnknown) }
                try data.write(to: target, options: .atomic)
                NSLog("EXAMPLE_CAPTURE saved=\(name).png")
            } catch { NSLog("EXAMPLE_CAPTURE image failed: \(error.localizedDescription)") }
            completion?()
        }
    }
    func close(completion: (() -> Void)? = nil) {
        guard !busy else { return }
        persistChat(); busy = true; usable = false
        bridge.cancel()
        worker.async { [self] in
            let result = bridge.close()
            DispatchQueue.main.async { [self] in
                busy = false
                if let failure = result["error"] as? String { error = failure; return }
                selected = nil; contextLength = nil; page.close(); thinkingSupported = false; thinking = false
                conversation?.reset(); lastStop = ""; status = "Choose a model"; completion?()
            }
        }
    }
    private func validateModelSwitch(completion: @escaping () -> Void) {
        guard let original = models.first(where: { $0.path == selected }), let tiny = models.first(where: { $0.name == "tiny-a" && $0.selectable }) else {
            NSLog("EXAMPLE_SMOKE failed: visual switch requires tiny-a alongside the real model")
            return
        }
        select(tiny) { [self] in
            NSLog("EXAMPLE_SWITCH loaded=tiny-a ready=\(canSend) history=\(messages.count)")
            capture("switched-tiny") { [self] in
                select(original) { [self] in
                    NSLog("EXAMPLE_SWITCH loaded=original ready=\(canSend) history=\(messages.count)")
                    thinking = false
                    send("Reply with a short greeting.") { [self] in
                        guard usable && error == nil else { NSLog("EXAMPLE_SMOKE failed after model switch"); return }
                        capture("switched-original", completion: completion)
                    }
                }
            }
        }
    }
    private func presentationChecks() {
        guard page.showsChat else { NSLog("EXAMPLE_SMOKE failed chat routing"); return }
        setThinking(true)
        send("What is 7 times 8? Answer briefly.") { [self] in
            guard error == nil && usable, let reply = messages.last, reply.showsReasoning(thinking: thinking), reply.reasoningExpanded else { NSLog("EXAMPLE_SMOKE failed default reasoning presentation"); return }
            validationScrollTarget = "top"
            capture("thinking-expanded") { [self] in
                setReasoningExpanded(id: reply.id, expanded: false)
                capture("thinking-collapsed") { [self] in
                    let previous = messages
                    showModels()
                    guard !page.showsChat && messages == previous else { NSLog("EXAMPLE_SMOKE failed Models preserved history"); return }
                    capture("models-return") { [self] in
                        resumeChat()
                        guard page.showsChat && messages == previous else { NSLog("EXAMPLE_SMOKE failed chat resume"); return }
                        newConversation { [self] in
                            setThinking(false)
                            send("Say hello.") { [self] in
                                guard error == nil && usable && !thinking && messages.last?.reasoning.isEmpty == true && !canChangeThinking else { NSLog("EXAMPLE_SMOKE failed locked off state"); return }
                                capture("thinking-locked-off") { NSLog("EXAMPLE_PRESENTATION complete") }
                            }
                        }
                    }
                }
            }
        }
    }
    private func thinkingLabelChecks() {
        setThinking(true)
        send("Say hello.") { [self] in
            guard error == nil && usable && !canChangeThinking && thinkingPresentation.stateLabel == "On" else { NSLog("EXAMPLE_SMOKE failed locked On label"); return }
            capture("thinking-label-on") { [self] in
                newConversation { [self] in
                    setThinking(false)
                    send("Say hello.") { [self] in
                        guard error == nil && usable && !canChangeThinking && thinkingPresentation.stateLabel == "Off" else { NSLog("EXAMPLE_SMOKE failed locked Off label"); return }
                        capture("thinking-label-off") { NSLog("EXAMPLE_THINKING_LABELS complete") }
                    }
                }
            }
        }
    }
    private func decodeComparison() {
        let prompt = "Write a concise paragraph explaining how a rainbow forms."
        setThinking(false)
        NSLog("EXAMPLE_DEVICE active=\(UIApplication.shared.applicationState == .active) low_power=\(ProcessInfo.processInfo.isLowPowerModeEnabled) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)")
        NSLog("EXAMPLE_DECODE prompt=\(prompt) max_tokens=96 thinking=false")
        send(prompt) { [self] in
            guard error == nil && usable, let metrics = lastNativeMetrics else { NSLog("EXAMPLE_SMOKE failed decode comparison"); return }
            do {
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                let data = try encoder.encode(metrics)
                guard let raw = String(data: data, encoding: .utf8) else { NSLog("EXAMPLE_SMOKE failed metric encoding"); return }
                NSLog("EXAMPLE_DECODE native=\(raw)")
                NSLog("EXAMPLE_DEVICE completed active=\(UIApplication.shared.applicationState == .active) low_power=\(ProcessInfo.processInfo.isLowPowerModeEnabled) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)")
                NSLog("EXAMPLE_DECODE complete")
            } catch { NSLog("EXAMPLE_SMOKE failed metric encoding") }
        }
    }
    private func revisionChecks() {
        setThinking(false)
        NSLog("EXAMPLE_DEVICE active=\(UIApplication.shared.applicationState == .active) low_power=\(ProcessInfo.processInfo.isLowPowerModeEnabled) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)")
        NSLog("EXAMPLE_REVISION model=\(models.first(where: { $0.path == selected })?.displayName ?? "") available=\(modelChoices.available.count) unavailable=\(modelChoices.unavailable.count)")
        send("Reply with a short greeting.") { [self] in
            guard error == nil && usable && !canChangeThinking else { NSLog("EXAMPLE_REVISION failed first reply or lock"); return }
            let chosen = thinking
            setThinking(!chosen)
            guard thinking == chosen else { NSLog("EXAMPLE_REVISION failed thinking changed during chat"); return }
            NSLog("EXAMPLE_REVISION short_reply metrics=\(messages.last?.metrics ?? "") thinking_locked=true")
            capture("revision-reply") { [self] in
                newConversation { [self] in
                    guard canChangeThinking else { NSLog("EXAMPLE_REVISION failed new chat unlock"); return }
                    setThinking(true)
                    guard thinking else { NSLog("EXAMPLE_REVISION failed new chat toggle"); return }
                    setThinking(false)
                    NSLog("EXAMPLE_REVISION new_chat_unlocked=true")
                    capture("revision-new-chat") { [self] in
                        let words = Array(repeating: "memory conversation inference", count: 160).joined(separator: " ")
                        send("Summarize the following repeated word list in detail: " + words) { [self] in
                            guard messages.last?.stopped == true && !generating && error == nil && usable else { NSLog("EXAMPLE_REVISION failed stopping completion"); return }
                            NSLog("EXAMPLE_REVISION stopped_marker=Stopped send_restored=true")
                            capture("revision-stopped") { NSLog("EXAMPLE_REVISION complete") }
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [self] in stop() }
                    }
                }
            }
        }
    }
    private func statusRetakeCheck() {
        let previousTheme = settings.theme
        setThinking(false)
        send("Say hello briefly.") { [self] in
            guard usable && error == nil else { NSLog("EXAMPLE_STATUS_RETAKE failed request"); return }
            Task { [self] in
                for theme in [ThemeChoice.light, .dark] {
                    self.settings.theme = theme; self.snapshot = nil; self.tab = .status
                    try? await Task.sleep(for: .seconds(2))
                    guard self.snapshot != nil, self.systemStatus?.appMemoryLimit.bytes != nil else { NSLog("EXAMPLE_STATUS_RETAKE failed status"); return }
                    self.capture("status-retake-\(theme.rawValue.lowercased())-status")
                    try? await Task.sleep(for: .seconds(0.6))
                }
                self.settings.theme = previousTheme
                self.close { [self] in self.tab = .chat; NSLog("EXAMPLE_STATUS_RETAKE complete unloaded=true") }
            }
        }
    }
    private func modelSwitchChecks() {
        guard let original = models.first(where: { $0.path == selected }), let tiny = models.first(where: { $0.name == "tiny-a" && $0.selectable }) else { NSLog("EXAMPLE_MODEL_SWITCH failed required models"); return }
        func checkStatus(_ name: String, completion: @escaping () -> Void) {
            snapshot = nil; tab = .status
            Task { [self] in
                try? await Task.sleep(for: .seconds(2))
                guard let snapshot = self.snapshot, snapshot.ledger.currentBytes > 0, self.systemStatus != nil else { NSLog("EXAMPLE_MODEL_SWITCH failed status"); return }
                self.capture("model-switch-\(name)-status") { [self] in self.resumeChat(); completion() }
            }
        }
        select(tiny) { [self] in
            guard selected == tiny.path, messages.isEmpty, contextUsage.text.hasPrefix("0 /"), canSend else { NSLog("EXAMPLE_MODEL_SWITCH failed tiny load"); return }
            setThinking(false)
            send("Hello.") { [self] in
                guard usable && error == nil, messages.count == 2, lastSummary != nil else { NSLog("EXAMPLE_MODEL_SWITCH failed tiny turn"); return }
                capture("model-switch-tiny-chat") { [self] in
                    checkStatus("tiny") { [self] in
                        select(original) { [self] in
                            guard selected == original.path, messages.isEmpty, contextUsage.text.hasPrefix("0 /"), canSend else { NSLog("EXAMPLE_MODEL_SWITCH failed original load"); return }
                            setThinking(false)
                            send("Say hello briefly.") { [self] in
                                guard usable && error == nil, messages.count == 2 else { NSLog("EXAMPLE_MODEL_SWITCH failed original turn"); return }
                                capture("model-switch-original-chat") {
                                    checkStatus("original") { NSLog("EXAMPLE_MODEL_SWITCH complete two_way=true fresh_chats=true status=true") }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    private func focusedNativeChecks() {
        guard let original = models.first(where: { $0.path == selected }) else { return }
        thinking = false
        send("Brief Markdown example: a heading, one bullet, bold text, a quote, a link, Python code print(1), inline formula $x_i$ and block formula $$x^2$$. Keep all formula delimiters.") { [self] in
            guard usable && error == nil else { NSLog("EXAMPLE_FOCUSED failed markdown generation"); return }
            capture("focused-markdown") { [self] in
                close { [self] in
                    guard selected == nil && !page.hasLoadedModel else { NSLog("EXAMPLE_FOCUSED failed unload"); return }
                    select(original) { [self] in
                        guard canSend, messages.isEmpty else { NSLog("EXAMPLE_FOCUSED failed reload"); return }
                        setThinking(true)
                        send("What is 7 times 8? Answer briefly.") { [self] in
                            guard usable && error == nil, !canChangeThinking, thinking, let last = messages.last, last.showsReasoning(thinking: true) else { NSLog("EXAMPLE_FOCUSED failed thinking"); return }
                            validationScrollTarget = "top"
                            capture("focused-thinking-expanded") { [self] in
                                setReasoningExpanded(id: last.id, expanded: false)
                                capture("focused-thinking-collapsed") { NSLog("EXAMPLE_FOCUSED complete unload_reload=true thinking_on_off=true markdown=true") }
                            }
                        }
                    }
                }
            }
        }
    }
    private func statusPagesChecks() {
        thinking = false
        let previousSettings = settings
        settings.style = .precise
        send("Say hello in one short sentence.") { [self] in
            guard usable && error == nil, let first = savedChats.first(where: { $0.id == chatID }), messages.last?.nativeMetrics != nil else { NSLog("EXAMPLE_STATUS failed initial history"); return }
            let beforeRegenerate = messages.count
            regenerateForCheck { [self] in
                guard messages.count == beforeRegenerate else { NSLog("EXAMPLE_STATUS failed regenerate"); return }
                guard let user = messages.first(where: { $0.role == "user" }) else { return }
                let oldID = user.id
                guard conversation?.prepareEdit(id: oldID) != nil else { return }
                send("Reply briefly with the word ready.") { [self] in
                    guard messages.count == 2, messages.first?.content == "Reply briefly with the word ready." else { NSLog("EXAMPLE_STATUS failed edit"); return }
                    UIPasteboard.general.string = messages.last?.content
                    guard UIPasteboard.general.string == messages.last?.content else { NSLog("EXAMPLE_STATUS failed copy"); return }
                    persistChat()
                    newConversation { [self] in
                        guard messages.isEmpty, contextUsage.text.hasPrefix("0 /") else { NSLog("EXAMPLE_STATUS failed new chat"); return }
                        restoreChat(first)
                        guard messages == first.conversation.messages, thinking == first.conversation.thinkingForChat else { NSLog("EXAMPLE_STATUS failed history restore"); return }
                        NSLog("EXAMPLE_STATUS history=true regenerate=true edit=true copy_answer=true native_metrics=true")
                        send("Continue with one short greeting.") { [self] in
                            guard usable && error == nil else { NSLog("EXAMPLE_STATUS failed history continuation"); return }
                            Task { [self] in
                                self.tab = .status
                                try? await Task.sleep(for: .seconds(2))
                                guard let systemStatus = self.systemStatus, let limit = systemStatus.appMemoryLimit.bytes, let physical = systemStatus.appMemoryLimit.currentPhysicalBytes, limit >= physical else { NSLog("EXAMPLE_STATUS failed app memory limit"); return }
                                NSLog("EXAMPLE_STATUS app_memory_limit=\(limit) physical_same_sample=\(physical) thermal=\(systemStatus.thermal.state ?? "unavailable")")
                                for theme in [ThemeChoice.light, .dark] {
                                    self.settings.theme = theme
                                    for page in [AppTab.chat, .models, .status, .settings] {
                                        if page == .models { self.showModels() } else if page == .chat { self.resumeChat() } else { self.tab = page }
                                        try? await Task.sleep(for: .seconds(page == .status ? 2 : 0.5))
                                        self.capture("status-pages-\(theme.rawValue.lowercased())-\(page.rawValue.lowercased())")
                                        try? await Task.sleep(for: .seconds(0.6))
                                    }
                                }
                                self.settings = previousSettings; self.resumeChat()
                                NSLog("EXAMPLE_STATUS complete")
                            }
                        }
                    }
                }
            }
        }
    }
    private func regenerateForCheck(_ completion: @escaping () -> Void) {
        guard !busy, let text = conversation?.prepareRegeneration() else { return }
        send(text, completion: completion)
    }
    // Opt-in device exercise uses the same model, request/history, stream and reset paths as the UI.
    private func smoke() {
        UIApplication.shared.isIdleTimerDisabled = true
        if ProcessInfo.processInfo.arguments.contains("--status-retake-check") { statusRetakeCheck(); return }
        if ProcessInfo.processInfo.arguments.contains("--model-switch-check") { modelSwitchChecks(); return }
        if ProcessInfo.processInfo.arguments.contains("--focused-native-checks") { focusedNativeChecks(); return }
        if ProcessInfo.processInfo.arguments.contains("--background-active-check") {
            thinking = false
            send("Write a detailed story about a robot.") { [self] in
                guard lastStop.hasPrefix("cancelled"), usable, error == nil else { NSLog("EXAMPLE_ACTIVE failed cancellation"); return }
                backgroundCancellationSafe = true; saveValidationState(); NSLog("EXAMPLE_ACTIVE cancelled=true state_valid=true")
            }
            capture("background-active-start", delay: 0.1) { NSLog("EXAMPLE_ACTIVE ready") }
            return
        }
        if ProcessInfo.processInfo.arguments.contains("--history-relaunch-check") {
            refreshHistory()
            guard let saved = savedChats.first(where: { $0.modelPath == selected }), !saved.conversation.messages.isEmpty else { NSLog("EXAMPLE_STATUS failed relaunch history"); return }
            restoreChat(saved)
            guard messages == saved.conversation.messages, lifetimeTotals.requests > 0 else { NSLog("EXAMPLE_STATUS failed relaunch totals"); return }
            send("Say hello again briefly.") { [self] in
                guard error == nil && usable, messages.count == saved.conversation.messages.count + 2 else { NSLog("EXAMPLE_STATUS failed relaunch continuation"); return }
                capture("history-relaunch") { NSLog("EXAMPLE_STATUS relaunch complete") }
            }
            return
        }
        if ProcessInfo.processInfo.arguments.contains("--status-pages-checks") { statusPagesChecks(); return }
        if thinkingLabelChecksEnabled { thinkingLabelChecks(); return }
        if presentationChecksEnabled { presentationChecks(); return }
        if decodeComparisonEnabled { decodeComparison(); return }
        if ProcessInfo.processInfo.arguments.contains("--revision-checks") { revisionChecks(); return }
        let args = ProcessInfo.processInfo.arguments
        let rounds = args.firstIndex(of: "--rounds").flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil } ?? 5
        func round(_ index: Int) {
            guard index < rounds else {
                NSLog("EXAMPLE_SMOKE rounds=\(rounds) completed")
                capture("rounds-complete") { [weak self] in self?.newConversation { [weak self] in
                    guard let self else { return }
                    self.thinking = false
                    self.send("Write a long story about a robot.", completion: {
                        NSLog("EXAMPLE_SMOKE cancel=\(self.lastStop)")
                        self.capture("cancelled") {
                            self.send("Hello again.", completion: {
                                if self.usable && self.error == nil {
                                    self.capture("resumed") {
                                        if args.contains("--visual-checks") {
                                            self.newConversation {
                                                self.send("Give a compact Markdown demonstration: a heading, two bullets, bold and italic text, a link to https://example.com, a fenced Python code block, the exact inline formula $x_i * y$ and block formula $$x^2 + y^2$$, and a price of $5. Keep it short.") {
                                                    self.capture("markdown-bottom") {
                                                        self.validationScrollTarget = "top"
                                                        self.capture("markdown") {
                                                            if self.usable && self.error == nil {
                                                                self.validateModelSwitch { NSLog("EXAMPLE_SMOKE resumed=\(self.lastStop) complete") }
                                                            } else { NSLog("EXAMPLE_SMOKE failed Markdown generation") }
                                                        }
                                                    }
                                                }
                                            }
                                        } else { NSLog("EXAMPLE_SMOKE resumed=\(self.lastStop) complete") }
                                    }
                                } else { NSLog("EXAMPLE_SMOKE failed to resume") }
                            })
                        }
                    }, cancelAfterFragment: true)
                } }
                return
            }
            setThinking(thinkingSupported && index % 2 == 0)
            let prompt = index == 0 ? suggestions[0] : "Explain one more detail in a short paragraph."
            send(prompt) { [weak self] in
                guard let self else { return }
                NSLog("EXAMPLE_SMOKE round=\(index + 1) history=\(self.messages.count) metrics=\(self.messages.last?.metrics ?? "") fragments-visible=\(!self.messages.last!.content.isEmpty)")
                if self.usable && self.error == nil { round(index + 1) }
                else { NSLog("EXAMPLE_SMOKE failed before completing rounds") }
            }
        }
        round(0)
    }
}
