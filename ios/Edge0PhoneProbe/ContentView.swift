import Edge0MLX
import Darwin
import Foundation
import SwiftUI

private struct ChatMessage: Identifiable, Sendable {
    enum Role: Sendable { case user, assistant }
    let id: UUID
    let role: Role
    var text: String
    var detail: String?

    init(id: UUID = UUID(), role: Role, text: String, detail: String?) {
        self.id = id; self.role = role; self.text = text; self.detail = detail
    }
}

private enum LocalModel: String, CaseIterable, Identifiable, Sendable {
    case edge8
    case edge35b

    var id: String { rawValue }
    var title: String {
        switch self {
        case .edge8: "Edge0 8B"
        case .edge35b: "Edge0 35B"
        }
    }
    var folderName: String {
        self == .edge8 ? "Edge0-8B-A1B-preview" : Edge0ChatEngine35B.modelFolderName
    }
}

private actor ChatRuntime {
    private enum LoadedEngine {
        case edge8(Edge0ChatEngine)
        case edge35b(Edge0ChatEngine35B)
    }
    private var engine: LoadedEngine?

    func load(_ model: LocalModel, modelURL: URL) throws {
        // Do not keep both models alive while switching; 35B is memory constrained.
        engine = nil
        switch model {
        case .edge8:
            engine = .edge8(try Edge0ChatEngine(modelURL: modelURL))
        case .edge35b:
            engine = .edge35b(try Edge0ChatEngine35B(modelURL: modelURL))
        }
    }

    func reply(to text: String,
               maxTokens: Int? = nil,
               thinking: Bool,
               onText: @escaping @Sendable (String) -> Void) async throws -> Edge0GenerationResult {
        let shouldContinue = { @Sendable in
            !withUnsafeCurrentTask { $0?.isCancelled ?? false }
        }
        switch engine {
        case .edge8(let engine):
            if let maxTokens {
                return try engine.reply(
                    to: text, maxTokens: maxTokens, thinking: thinking,
                    onText: onText, shouldContinue: shouldContinue)
            }
            return try engine.reply(
                to: text, thinking: thinking, onText: onText, shouldContinue: shouldContinue)
        case .edge35b(let engine):
            if let maxTokens {
                return try await engine.reply(
                    to: text, maxTokens: maxTokens, thinking: thinking,
                    onText: onText, shouldContinue: shouldContinue)
            }
            return try await engine.reply(
                to: text, thinking: thinking, onText: onText, shouldContinue: shouldContinue)
        case nil:
            throw RuntimeFailure.modelNotLoaded
        }
    }

    func reset() {
        switch engine {
        case .edge8(let engine): engine.reset()
        case .edge35b(let engine): engine.reset()
        case nil: break
        }
    }

    func unload() { engine = nil }

    private enum RuntimeFailure: LocalizedError {
        case modelNotLoaded
        var errorDescription: String? { "Model is not loaded" }
    }
}

@MainActor
private final class ChatViewModel: ObservableObject {
    enum Phase: Equatable { case choosing, ready, preparing, generating, failed(String) }

    @Published var input = ""
    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var phase: Phase = .choosing
    @Published private(set) var selectedModel: LocalModel?

    private let runtime = ChatRuntime()
    private var generationTask: Task<Void, Never>?

    var isRunning: Bool { phase == .preparing || phase == .generating }
    /// Off unless the process is launched with `--thinking`. `--no-thinking` forces it off.
    private let thinkingEnabled: Bool = {
        let args = CommandLine.arguments
        if args.contains("--no-thinking") { return false }
        return args.contains("--thinking")
    }()
    var statusText: String? {
        switch phase {
        case .choosing: nil
        case .ready: nil
        case .preparing: "Loading \(selectedModel?.title ?? "local model")…"
        case .generating: "\(selectedModel?.title ?? "Model") is generating…"
        case .failed(let message): message
        }
    }

    func isInstalled(_ model: LocalModel) -> Bool {
        let url = resolveModelURL(model)
        switch model {
        case .edge8:
            return FileManager.default.fileExists(
                atPath: url.appendingPathComponent("model.safetensors").path)
        case .edge35b:
            return FileManager.default.fileExists(
                atPath: url.appendingPathComponent("model.safetensors.index.json").path)
                && FileManager.default.fileExists(
                    atPath: url.appendingPathComponent("tokenizer.bin").path)
                && FileManager.default.fileExists(
                    atPath: url.appendingPathComponent("experts-L39.bin").path)
        }
    }

    func choose(_ model: LocalModel) {
        guard !isRunning else { return }
        selectedModel = model
        messages.removeAll()
        input = ""
        phase = .preparing
        let url = resolveModelURL(model)
        Task {
            do {
                try await runtime.load(model, modelURL: url)
                if phase == .preparing { phase = .ready }
            } catch {
                phase = .failed(Self.friendlyMessage(for: error))
            }
        }
    }

    func chooseAnotherModel() {
        guard !isRunning else { return }
        messages.removeAll()
        input = ""
        selectedModel = nil
        phase = .choosing
        Task { await runtime.unload() }
    }

    func useSuggestion(_ text: String, maxTokens: Int? = nil) {
        input = text
        send(maxTokens: maxTokens)
    }

    func send(maxTokens: Int? = nil) {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isRunning else { return }
        input = ""
        messages.append(ChatMessage(role: .user, text: text, detail: nil))
        let assistantID = UUID()
        messages.append(ChatMessage(id: assistantID, role: .assistant,
                                    text: "", detail: nil))
        phase = .generating
        generationTask = Task {
            do {
                let result = try await runtime.reply(
                    to: text,
                    maxTokens: maxTokens,
                    thinking: thinkingEnabled,
                    onText: { [weak self] partial in
                        Task { @MainActor in
                            guard let self,
                                  let index = self.messages.firstIndex(where: { $0.id == assistantID })
                            else { return }
                            self.messages[index].text = partial
                        }
                    })
                let detail = String(
                    format: "%d tokens · TTFT %.1fs · prefill %.1f tok/s · decode %.2f tok/s · peak %.0f MB",
                    result.generatedTokenCount, result.timeToFirstTokenSeconds,
                    result.prefillTokensPerSecond,
                    result.decodeTokensPerSecond,
                    Double(result.peakMemoryBytes) / 1_048_576
                )
                if let index = messages.firstIndex(where: { $0.id == assistantID }) {
                    messages[index].text = result.text.isEmpty ? "(The model returned an empty response)" : result.text
                    messages[index].detail = detail
                }
                print("[generation] \(detail)")
                if CommandLine.arguments.contains("--bench-second"),
                   CommandLine.arguments.contains("--bench-35b") {
                    print("[bench] second turn")
                    let second = try await runtime.reply(
                        to: "Summarize it in one sentence.",
                        maxTokens: 64,
                        thinking: thinkingEnabled,
                        onText: { _ in })
                    let secondDetail = String(
                        format: "%d tokens · TTFT %.1fs · prefill %.1f tok/s · decode %.2f tok/s · peak %.0f MB",
                        second.generatedTokenCount, second.timeToFirstTokenSeconds,
                        second.prefillTokensPerSecond,
                        second.decodeTokensPerSecond,
                        Double(second.peakMemoryBytes) / 1_048_576
                    )
                    print("[generation turn2] \(secondDetail)")
                }
                if CommandLine.arguments.contains("--bench-8b")
                    || CommandLine.arguments.contains("--bench-35b") {
                    var info = task_vm_info_data_t()
                    var count = mach_msg_type_number_t(
                        MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
                    let status = withUnsafeMutablePointer(to: &info) {
                        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                        }
                    }
                    print("[bench memory] endFootprintMiB=\(status == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)")
                    print("[bench timing] ttftSeconds=\(result.timeToFirstTokenSeconds) prefillSeconds=\(result.prefillSeconds) prefillTokPerSec=\(result.prefillTokensPerSecond) decodeTokPerSec=\(result.decodeTokensPerSecond)")
                }
                phase = .ready
                if CommandLine.arguments.contains("--bench-8b")
                    || CommandLine.arguments.contains("--bench-35b")
                    || CommandLine.arguments.contains("--smoke-35b") {
                    fflush(nil)
                    exit(0)
                }
            } catch is CancellationError {
                phase = .ready
            } catch {
                phase = .failed(Self.friendlyMessage(for: error))
                if CommandLine.arguments.contains("--bench-8b")
                    || CommandLine.arguments.contains("--bench-35b")
                    || CommandLine.arguments.contains("--smoke-35b") {
                    fflush(stdout)
                    exit(1)
                }
            }
            generationTask = nil
        }
    }

    func stop() { generationTask?.cancel() }

    func newConversation() {
        guard !isRunning, selectedModel != nil else { return }
        messages.removeAll()
        input = ""
        phase = .ready
        Task { await runtime.reset() }
    }

    private func resolveModelURL(_ model: LocalModel) -> URL {
        let models = Bundle.main.resourceURL?
            .appendingPathComponent("Models", isDirectory: true)
            ?? Bundle.main.bundleURL.appendingPathComponent("Models", isDirectory: true)
        return models.appendingPathComponent(model.folderName, isDirectory: true)
    }

    private static func friendlyMessage(for error: Error) -> String {
        let description = error.localizedDescription
        if description.contains("doesn’t exist") || description.contains("No such file")
            || description.contains("incomplete") {
            return "Model not included in this build"
        }
        return "Run failed: \(description)"
    }
}

struct ContentView: View {
    @StateObject private var chat = ChatViewModel()
    @FocusState private var inputFocused: Bool

    private let suggestions = [
        "What is artificial intelligence?",
        "How do LLMs understand context?",
        "Tell me about renewable energy",
        "Write a short poem about autumn",
    ]

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if chat.selectedModel == nil {
                modelPicker
            } else {
                VStack(spacing: 0) {
                    header
                    conversation
                }
            }
        }
        .preferredColorScheme(.dark)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if chat.selectedModel != nil { composer }
        }
        .task {
            // Keeps the normal launch picker intact while allowing a repeatable
            // on-device smoke/benchmark run from `devicectl`.
            let smoke = CommandLine.arguments.contains("--smoke-35b")
            let bench8 = CommandLine.arguments.contains("--bench-8b")
            let bench35 = CommandLine.arguments.contains("--bench-35b")
            let benchmark = bench8 || bench35
            let target: LocalModel =
                bench8 || CommandLine.arguments.contains("--model=8b") ? .edge8 : .edge35b
            if chat.selectedModel == nil,
               (CommandLine.arguments.contains("--model=35b")
                || CommandLine.arguments.contains("--model=8b")
                || smoke || benchmark),
               chat.isInstalled(target) {
                print("[bench] loading \(target.title)")
                print("[bench] thermalAtLaunch=\(ProcessInfo.processInfo.thermalState.rawValue)")
                chat.choose(target)
                if smoke || benchmark {
                    while chat.phase == .preparing {
                        try? await Task.sleep(for: .milliseconds(100))
                    }
                    if case .failed(let message) = chat.phase {
                        print("[bench] load failed: \(message)")
                    } else if chat.phase == .ready {
                        let long = CommandLine.arguments.contains("--bench-long")
                        let sentence = "Streaming inference overlaps disk bandwidth with compute. Each mixture-of-experts layer activates only a few experts."
                        let prompt = smoke
                            ? "Reply with this only: 35B is running"
                            : (long
                                ? Array(repeating: sentence, count: 80).joined()
                                : "What is artificial intelligence?")
                        chat.useSuggestion(prompt, maxTokens: long ? 8 : (benchmark ? 64 : nil))
                    }
                }
            }
        }
    }

    private var modelPicker: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Spacer(minLength: 54)
                Text("Choose a local model")
                    .font(.system(size: 34, weight: .bold, design: .rounded))

                ForEach(LocalModel.allCases) { model in
                    let installed = chat.isInstalled(model)
                    Button { chat.choose(model) } label: {
                        HStack(spacing: 16) {
                            Image(systemName: model == .edge8 ? "bolt.fill" : "brain.head.profile")
                                .font(.system(size: 23, weight: .semibold))
                                .frame(width: 48, height: 48)
                                .background(.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 14))
                            VStack(alignment: .leading, spacing: 5) {
                                Text(model.title).font(.headline)
                                if !installed {
                                    Text("Not included")
                                        .font(.caption2)
                                        .foregroundStyle(.orange)
                                }
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundStyle(.secondary)
                        }
                        .padding(18)
                        .background(.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 22))
                    }
                    .buttonStyle(.plain)
                    .disabled(!installed)
                    .opacity(installed ? 1 : 0.62)
                }

                Spacer(minLength: 40)
            }
            .padding(.horizontal, 22)
            .frame(maxWidth: 620)
            .frame(maxWidth: .infinity)
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text(chat.selectedModel?.title ?? "Edge0")
                .font(.headline.weight(.semibold))
            Spacer()
            Button(action: chat.chooseAnotherModel) {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 38, height: 38)
                    .background(.white.opacity(0.07), in: Circle())
            }
            .foregroundStyle(.white)
            .disabled(chat.isRunning)
            .accessibilityLabel("Switch model")
            Button(action: chat.newConversation) {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 38, height: 38)
                    .background(.white.opacity(0.07), in: Circle())
            }
            .foregroundStyle(.white)
            .disabled(chat.isRunning)
            .accessibilityLabel("New chat")
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .padding(.bottom, 10)
    }

    @ViewBuilder
    private var conversation: some View {
        if chat.messages.isEmpty {
            welcome
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 18) {
                        ForEach(chat.messages) { message in
                            messageBubble(message).id(message.id)
                        }
                        if let status = chat.statusText { statusRow(status).id("status") }
                    }
                    .padding(.horizontal, 18)
                    .padding(.top, 16)
                    .padding(.bottom, 24)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: chat.messages.count) {
                    withAnimation { proxy.scrollTo(chat.messages.last?.id, anchor: .bottom) }
                }
                .onChange(of: chat.phase) {
                    withAnimation { proxy.scrollTo("status", anchor: .bottom) }
                }
            }
        }
    }

    private var welcome: some View {
        ScrollView {
            VStack(spacing: 18) {
                Spacer(minLength: 90)
                Text("edge0")
                    .font(.system(size: 58, weight: .medium, design: .rounded))
                    .tracking(-3)
                Text("Private AI, running locally on your iPhone")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                VStack(spacing: 10) {
                    ForEach(suggestions, id: \.self) { suggestion in
                        Button(suggestion) { chat.useSuggestion(suggestion) }
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.white.opacity(0.88))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 11)
                            .background(.white.opacity(0.07), in: Capsule())
                    }
                }
                .padding(.top, 8)
                if let status = chat.statusText { statusRow(status) }
                Spacer(minLength: 120)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 24)
        }
    }

    private func messageBubble(_ message: ChatMessage) -> some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
            Text(message.text)
                .font(.body)
                .foregroundStyle(.white.opacity(0.94))
                .textSelection(.enabled)
                .padding(.horizontal, 15)
                .padding(.vertical, 12)
                .background(
                    message.role == .user ? Color.white.opacity(0.14) : Color.white.opacity(0.07),
                    in: RoundedRectangle(cornerRadius: 19, style: .continuous)
                )
            if let detail = message.detail {
                Text(detail)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    private func statusRow(_ text: String) -> some View {
        HStack(spacing: 10) {
            if chat.isRunning { ProgressView().controlSize(.small) }
            Text(text)
                .font(.footnote)
                .foregroundStyle(chat.isRunning ? Color.secondary : Color.red.opacity(0.9))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 6)
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Ask anything…", text: $chat.input, axis: .vertical)
                .lineLimit(1...5)
                .focused($inputFocused)
                .submitLabel(.send)
                .onSubmit { chat.send() }
                .padding(.horizontal, 16)
                .padding(.vertical, 13)
                .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 22, style: .continuous))

            Button {
                if chat.isRunning, chat.phase == .generating {
                    chat.stop()
                } else {
                    chat.send()
                }
                inputFocused = false
            } label: {
                Image(systemName: chat.phase == .generating ? "stop.fill" : "arrow.up")
                    .font(.system(size: 17, weight: .bold))
                    .foregroundStyle(.black)
                    .frame(width: 45, height: 45)
                    .background(.white, in: Circle())
            }
            .disabled(chat.phase == .preparing ||
                      chat.phase == .choosing ||
                      (chat.phase != .generating &&
                       chat.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
            .opacity(chat.phase == .preparing ? 0.5 : 1)
            .accessibilityLabel(chat.phase == .generating ? "Stop" : "Send")
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .background(.ultraThinMaterial)
    }
}
