import SwiftUI
import UIKit

struct MarkdownView: View {
    let source: String
    private func inline(_ text: String) -> AttributedString { (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text) }
    var body: some View {
        VStack(alignment: .leading, spacing: Tokens.sp2) {
            ForEach(Markdown.blocks(source)) { block in
                switch block.kind {
                case .code(let language):
                    VStack(alignment: .leading, spacing: Tokens.sp2) {
                        HStack { Text(language).font(.system(size: Tokens.xs, design: .monospaced)); Spacer(); Button { UIPasteboard.general.string = block.text } label: { Image(systemName: "doc.on.doc") }.accessibilityLabel("Copy code") }.foregroundStyle(Tokens.text3)
                        ScrollView(.horizontal) { Text(block.text).font(.system(size: Tokens.sm, design: .monospaced)).textSelection(.enabled) }
                    }.padding(Tokens.sp3).background(Tokens.surface2, in: RoundedRectangle(cornerRadius: Tokens.medium)).overlay { RoundedRectangle(cornerRadius: Tokens.medium).stroke(Tokens.border) }
                case .heading(let level): Text(inline(block.text)).font(.system(size: level <= 2 ? Tokens.xl : Tokens.lg, weight: .semibold))
                case .quote: Text(inline(block.text)).foregroundStyle(Tokens.text2).padding(.leading, Tokens.sp3).overlay(alignment: .leading) { Rectangle().fill(Tokens.borderStrong).frame(width: Tokens.quoteRule) }
                case .list: Text(inline(block.text.replacingOccurrences(of: #"^[-*+] "#, with: "• ", options: .regularExpression))).padding(.leading, Tokens.small)
                case .paragraph: Text(inline(block.text))
                }
            }
        }.font(.system(size: Tokens.base)).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
    }
}
private struct ReasoningView: View {
    let source: String
    @Binding var expanded: Bool
    var body: some View {
        DisclosureGroup(isExpanded: $expanded) { MarkdownView(source: source).foregroundStyle(Tokens.text2) } label: {
            HStack { Image(systemName: "brain"); Text("Thinking"); Spacer(); Button { UIPasteboard.general.string = source } label: { Image(systemName: "doc.on.doc") }.accessibilityLabel("Copy thinking") }
        }.font(.system(size: Tokens.sm)).tint(Tokens.text2).foregroundStyle(Tokens.text2).padding(Tokens.sp3).background(Tokens.surface2, in: RoundedRectangle(cornerRadius: Tokens.medium)).overlay { RoundedRectangle(cornerRadius: Tokens.medium).stroke(Tokens.border) }
    }
}
struct ContentView: View {
    @ObservedObject var model: ChatViewModel
    @State private var history = false
    @State private var editing: ChatMessage?
    @State private var editText = ""
    private var colorScheme: ColorScheme? { switch model.settings.theme { case .system: return nil; case .light: return .light; case .dark: return .dark } }
    var body: some View {
        VStack(spacing: Tokens.none) {
            if let error = model.error { Text(error).foregroundStyle(Tokens.error).font(.system(size: Tokens.sm)).padding(Tokens.sp3).textSelection(.enabled) }
            Group { switch model.tab { case .chat: chatPage; case .models: modelsPage; case .status: statusPage; case .settings: settingsPage } }.frame(maxWidth: .infinity, maxHeight: .infinity)
            HStack(spacing: Tokens.none) { ForEach(AppTab.allCases, id: \.self) { tab in
                Button { if tab == .models { model.showModels() } else if tab == .chat { model.resumeChat(); model.tab = tab } else { model.tab = tab } } label: {
                    VStack(spacing: Tokens.sp1) { Image(systemName: icon(tab)).font(.system(size: Tokens.lg)); Text(tab.rawValue).font(.system(size: Tokens.xs)) }.frame(maxWidth: .infinity)
                }.foregroundStyle(model.tab == tab ? Tokens.accent : Tokens.text3).padding(.vertical, Tokens.sp2).accessibilityIdentifier("tab-\(tab.rawValue)")
            } }.background(Tokens.surface2).overlay(alignment: .top) { Divider().overlay(Tokens.border) }
        }.background(Tokens.surface).foregroundStyle(Tokens.text).tint(Tokens.accent).preferredColorScheme(colorScheme)
        .sheet(isPresented: $history) { historyPage }
        .sheet(item: $editing) { message in
            NavigationStack { VStack { TextEditor(text: $editText).font(.system(size: Tokens.base)).padding(Tokens.sp4); Button("Send") { model.editMessage(message.id, text: editText); editing = nil }.disabled(model.busy || editText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty).padding(Tokens.sp4) }.navigationTitle("Edit message").toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { editing = nil } } } }
        }
    }
    private func icon(_ tab: AppTab) -> String { switch tab { case .chat: return "bubble.left"; case .models: return "shippingbox"; case .status: return "waveform.path.ecg"; case .settings: return "gearshape" } }
    private func heading(_ title: String, action: (() -> Void)? = nil) -> some View {
        HStack { Text(title).font(.system(size: Tokens.xxl, weight: .semibold)); Spacer(); if let action { Button(action: action) { Image(systemName: "arrow.clockwise") }.disabled(model.busy).accessibilityLabel("Refresh models") } }.padding(.bottom, Tokens.sp2)
    }
    private var chatPage: some View {
        VStack(spacing: Tokens.none) {
            HStack(spacing: Tokens.sp2) {
                Button { model.refreshHistory(); history = true } label: { Image(systemName: "sidebar.left") }.accessibilityLabel("Conversation history").disabled(model.busy)
                VStack(alignment: .leading, spacing: Tokens.hairline) { Text(model.chatTitle).font(.system(size: Tokens.base, weight: .semibold)).lineLimit(1); if let selected = model.models.first(where: { $0.path == model.selected }) { Text("\(selected.displayName) · Metal").font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) } }.frame(maxWidth: .infinity, alignment: .leading)
                Button { model.newConversation() } label: { Image(systemName: "square.and.pencil") }.accessibilityLabel("New chat").disabled(model.busy)
            }.padding(Tokens.sp3).foregroundStyle(Tokens.text2).overlay(alignment: .bottom) { Divider().overlay(Tokens.border) }
            if !model.page.hasLoadedModel {
                Spacer(); Text("Choose a model to start chatting.").foregroundStyle(Tokens.text2); Button("Choose a model") { model.showModels() }.buttonStyle(.borderedProminent).padding(Tokens.sp4); Spacer()
            } else {
                HStack {
                    HStack(spacing: Tokens.sp2) { if let percent = model.contextUsage.ringPercent { ZStack { Circle().stroke(Tokens.surface3, lineWidth: Tokens.ringLine); Circle().trim(from: 0, to: CGFloat(percent) / 100).stroke(Tokens.accent, lineWidth: Tokens.ringLine).rotationEffect(.degrees(-90)) }.frame(width: Tokens.sp4, height: Tokens.sp4) }; Text(model.contextUsage.text).font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) }
                    Spacer(minLength: Tokens.sp2)
                    VStack(alignment: .trailing, spacing: Tokens.sp1) {
                        HStack(spacing: Tokens.sp1) { Toggle("Thinking", isOn: Binding(get: { model.thinking }, set: { model.setThinking($0) })).labelsHidden().fixedSize().tint(Tokens.accent).disabled(!model.canChangeThinking).opacity(model.thinkingPresentation.opacity); Text("Thinking \(model.thinkingPresentation.stateLabel)").font(.system(size: Tokens.xs)) }
                        if let hint = model.thinkingHint { Text(hint).font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) }
                    }
                }.padding(.horizontal, Tokens.sp3).padding(.vertical, Tokens.sp2).overlay(alignment: .bottom) { Divider().overlay(Tokens.border) }
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: Tokens.sp6) {
                            Color.clear.frame(height: Tokens.hairline).id("top")
                            if model.messages.isEmpty { VStack(alignment: .leading, spacing: Tokens.sp3) { Text("How can I help?").font(.system(size: Tokens.xxl, weight: .semibold)); ForEach(model.suggestions, id: \.self) { text in Button { model.send(text) } label: { Text(text).frame(maxWidth: .infinity, alignment: .leading).padding(Tokens.sp3).background(Tokens.surface2, in: RoundedRectangle(cornerRadius: Tokens.medium)) }.disabled(!model.canSend) } }.padding(.top, Tokens.sp8) }
                            ForEach(model.messages) { message in
                                if message.usesBubble {
                                    HStack { Spacer(minLength: Tokens.sp8); Text(message.content).font(.system(size: Tokens.base)).padding(Tokens.sp3).background(Tokens.surface3, in: RoundedRectangle(cornerRadius: Tokens.large)).textSelection(.enabled) }.contextMenu { Button("Edit and resend") { editText = message.content; editing = message }.disabled(model.busy) }.id(message.id)
                                } else {
                                    let state = message.state(isLast: message.id == model.messages.last?.id, generating: model.generating, thinking: model.thinking)
                                    VStack(alignment: .leading, spacing: Tokens.sp3) {
                                        if message.showsReasoning(thinking: model.thinking) { ReasoningView(source: message.reasoning, expanded: Binding(get: { message.reasoningExpanded }, set: { model.setReasoningExpanded(id: message.id, expanded: $0) })) }
                                        if state.pending { PendingDots() } else { MarkdownView(source: message.content) }
                                        if let marker = message.statusLabel { Text(marker).font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) }
                                        if state.actions { HStack(alignment: .top, spacing: Tokens.sp3) { Button { UIPasteboard.general.string = message.content } label: { Image(systemName: "doc.on.doc") }.accessibilityLabel("Copy answer"); if message.id == model.messages.last?.id { Button { model.regenerate() } label: { Image(systemName: "arrow.clockwise") }.disabled(!model.canRegenerate).accessibilityLabel("Regenerate") }; Text(message.metrics).font(.system(size: Tokens.xs)).textSelection(.enabled) }.foregroundStyle(Tokens.text3).font(.system(size: Tokens.sm)) }
                                    }.id(message.id)
                                }
                            }
                            Color.clear.frame(height: Tokens.hairline).id("bottom")
                        }.padding(Tokens.sp4)
                    }.onChange(of: model.messages.last?.scrollContent) { _, message in DispatchQueue.main.async { if message == nil { proxy.scrollTo("top", anchor: .top) } else if let message { proxy.scrollTo(message.id, anchor: .bottom) } } }.onChange(of: model.validationScrollTarget) { _, target in proxy.scrollTo(target, anchor: target == "top" ? .top : .bottom) }
                }
                HStack(alignment: .bottom, spacing: Tokens.sp2) {
                    TextField("Message", text: $model.draft, axis: .vertical).lineLimit(1...6).font(.system(size: Tokens.base)).disabled(!model.canSend)
                    if model.generating { Button { model.stop() } label: { HStack { if model.stopping { ProgressView().controlSize(.small) }; if model.stopping { Text("Stopping…").font(.system(size: Tokens.xs)) } else { Image(systemName: "stop.fill") } } }.disabled(!model.generationPhase.stopEnabled).accessibilityLabel(model.generationPhase.stopTitle) }
                    else { Button { model.send() } label: { Image(systemName: "arrow.up") }.disabled(!model.canSend || model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty).accessibilityLabel("Send") }
                }.padding(Tokens.sp3).background(Tokens.surface, in: RoundedRectangle(cornerRadius: Tokens.sp6)).overlay { RoundedRectangle(cornerRadius: Tokens.sp6).stroke(Tokens.borderStrong) }.padding(.horizontal, Tokens.sp3).padding(.bottom, Tokens.sp3)
            }
        }
    }
    private var modelsPage: some View {
        ScrollView { VStack(alignment: .leading, spacing: Tokens.sp4) {
            heading("Models", action: { model.refresh() })
            if model.busy && !model.generating { ProgressView(model.status) }
            if let loaded = model.models.first(where: { $0.path == model.selected }) { section("Loaded"); modelCard(loaded, loaded: true) }
            if !model.modelChoices.available.filter({ $0.path != model.selected }).isEmpty { section("Available"); ForEach(model.modelChoices.available.filter { $0.path != model.selected }) { modelCard($0, loaded: false) } }
            if !model.modelChoices.unavailable.isEmpty { section("Unavailable"); ForEach(model.modelChoices.unavailable) { modelCard($0, loaded: false) } }
            if let instructions = model.modelChoices.pushInstructions(assetsDirectory: "Documents/assets") { Text(instructions).font(.system(size: Tokens.sm)).foregroundStyle(Tokens.text3) }
        }.padding(Tokens.sp4) }.background(Tokens.background)
    }
    private func modelCard(_ choice: ModelChoice, loaded: Bool) -> some View {
        VStack(alignment: .leading, spacing: Tokens.sp3) {
            HStack(alignment: .top) { Image(systemName: choice.selectable ? "shippingbox" : "folder").foregroundStyle(choice.selectable ? Tokens.accent : Tokens.error).padding(Tokens.sp3).background(choice.selectable ? Tokens.accentSoft : Tokens.errorSoft, in: RoundedRectangle(cornerRadius: Tokens.medium)); VStack(alignment: .leading, spacing: Tokens.sp1) { Text(choice.displayName).font(.system(size: Tokens.base, weight: .semibold)); if let size = StatusFormat.bytes(choice.totalBytes) { Text(size).font(.system(size: Tokens.sm)).foregroundStyle(Tokens.text3) }; if loaded, let capacity = model.contextLength { Text("\(StatusFormat.count(capacity)!) context tokens · Metal").font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) }; if let error = choice.error { Text(error).font(.system(size: Tokens.sm)).foregroundStyle(Tokens.error) } }; Spacer(minLength: Tokens.none) }
            if loaded { HStack { Button("Open chat") { model.resumeChat() }; Spacer(); Button("Unload") { model.close() }.foregroundStyle(Tokens.error) }.disabled(model.busy) }
            else if choice.selectable { Button("Load") { model.select(choice) }.buttonStyle(.borderedProminent).disabled(model.busy) }
        }.appCard()
    }
    private func section(_ label: String) -> some View { Text(label.uppercased()).font(.system(size: Tokens.xs, weight: .semibold)).foregroundStyle(Tokens.text3).padding(.top, Tokens.sp2) }
    private var historyPage: some View {
        NavigationStack { List {
            ForEach(["Today", "Yesterday", "Previous 7 days", "Earlier"], id: \.self) { group in
                let chats = model.savedChats.filter { $0.group(now: Date()) == group }
                if !chats.isEmpty { Section(group) { ForEach(chats) { saved in Button(saved.title) { model.restoreChat(saved); history = false }.disabled(model.busy).swipeActions { Button("Delete", role: .destructive) { model.deleteChat(saved.id) }.disabled(model.busy) } } } }
            }
            if !model.damagedChats.isEmpty { Section("Unavailable conversations") { ForEach(model.damagedChats, id: \.self) { file in Text("\(file) · Could not read").font(.system(size: Tokens.sm)) } } }
        }.navigationTitle("Conversations").toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { history = false } } } }
    }
    private var statusPage: some View {
        ScrollView { VStack(alignment: .leading, spacing: Tokens.sp3) {
            heading("Status")
            if let progress = model.progress {
                VStack(alignment: .leading, spacing: Tokens.sp3) { HStack { Text("Current request").bold(); Spacer(); Text(progress.phase == "prefill" ? "Prefilling" : "Decoding").font(.system(size: Tokens.xs)).foregroundStyle(Tokens.accent).padding(Tokens.sp1).background(Tokens.accentSoft, in: Capsule()) }; if let text = progress.prefillText { pair("Prefill", text) }; if let text = progress.decodeText { pair("Decode", text) }; if progress.reusedTokens > 0 { Text("\(StatusFormat.count(progress.reusedTokens)!) tokens reused from the previous turn").font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) } }.appCard()
            } else if let last = model.lastSummary { VStack(alignment: .leading, spacing: Tokens.sp3) { HStack { Text("Last request").bold(); Spacer(); Text("Idle").font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) }; pair("Prefill", last.prefillText); pair("Decode", last.decodeText) }.appCard() }
            if let snapshot = model.snapshot {
                VStack(alignment: .leading, spacing: Tokens.sp3) {
                    HStack(alignment: .top) { Text("Memory").bold(); Spacer(); VStack(alignment: .trailing, spacing: Tokens.sp1) { if let bytes = StatusFormat.bytes(snapshot.physical.values?.currentBytes) { Text(bytes) }; if let limit = StatusFormat.bytes(model.systemStatus?.appMemoryLimit.bytes) { Text("iOS app limit \(limit)") } }.font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) }
                    GeometryReader { geometry in HStack(spacing: Tokens.none) { ForEach(Array(snapshot.ledger.segments.enumerated()), id: \.offset) { index, segment in Rectangle().fill(Tokens.segments[index]).frame(width: geometry.size.width * CGFloat(segment.1) / CGFloat(max(snapshot.ledger.limitBytes, 1))) } } }.frame(height: Tokens.medium).background(Tokens.surface3).clipShape(Capsule())
                    LazyVGrid(columns: [.init(.flexible(), alignment: .leading), .init(.flexible(), alignment: .leading), .init(.flexible(), alignment: .leading)], alignment: .leading, spacing: Tokens.sp2) { ForEach(Array(snapshot.ledger.segments.enumerated()), id: \.offset) { index, segment in HStack(spacing: Tokens.sp1) { Circle().fill(Tokens.segments[index]).frame(width: Tokens.small, height: Tokens.small); Text(segment.0) }.font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text2) } }
                }.appCard()
            }
            HStack { Picker("Totals", selection: $model.lifetimeView) { Text("Since start").tag(false); Text("All time").tag(true) }.pickerStyle(.segmented); if model.lifetimeView { Button("Reset") { model.resetTotals() }.font(.system(size: Tokens.sm)) } }
            LazyVGrid(columns: [.init(.flexible()), .init(.flexible())], spacing: Tokens.sp3) {
                if let value = model.totals.reused { stat("Reused", value, "of input tokens") }
                if let value = model.totals.expertHits { stat("Expert slot hits", value, "expert uses") }
                if let value = model.totals.averageDecode { stat("Avg decode", value) }
                if let state = model.systemStatus?.thermal.state { let heat = ThermalDisplay(state: state); stat("Thermal", heat.label, heat.description, color: heat.tone == .ok ? Tokens.ok : heat.tone == .warn ? Tokens.warn : Tokens.error) }
            }
        }.font(.system(size: Tokens.md)).padding(Tokens.sp4) }.background(Tokens.background)
    }
    private func pair(_ label: String, _ value: String) -> some View { HStack(alignment: .top, spacing: Tokens.sp3) { Text(label).foregroundStyle(Tokens.text3).frame(width: Tokens.labelWidth, alignment: .leading); Text(value).frame(maxWidth: .infinity, alignment: .leading) }.font(.system(size: Tokens.sm)) }
    private func stat(_ title: String, _ value: String, _ subtitle: String = "", color: Color = Tokens.text) -> some View { VStack(alignment: .leading, spacing: Tokens.sp1) { Text(title).font(.system(size: Tokens.sm)).foregroundStyle(Tokens.text2); Text(value).font(.system(size: Tokens.lg, weight: .semibold)).foregroundStyle(color); if !subtitle.isEmpty { Text(subtitle).font(.system(size: Tokens.xs)).foregroundStyle(Tokens.text3) } }.appCard() }
    private var settingsPage: some View {
        ScrollView { VStack(alignment: .leading, spacing: Tokens.sp4) {
            heading("Settings"); section("Appearance")
            VStack(alignment: .leading, spacing: Tokens.sp3) { Text("Theme").bold(); Picker("Theme", selection: $model.settings.theme) { ForEach(ThemeChoice.allCases, id: \.self) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented) }.appCard()
            section("Conversation")
            VStack(alignment: .leading, spacing: Tokens.sp3) { Text("Generation style").bold(); Picker("Generation style", selection: $model.settings.style) { ForEach(GenerationStyle.allCases, id: \.self) { Text("\($0.rawValue) · \(String(format: "%.1f", $0.temperature))").tag($0) } }.pickerStyle(.segmented); Text("Sets the temperature for every request. Every other sampling setting follows the model.").font(.system(size: Tokens.sm)).foregroundStyle(Tokens.text3); Divider().overlay(Tokens.border); Text("System prompt").bold(); TextField(model.defaultPrompt, text: $model.settings.systemPrompt, axis: .vertical).lineLimit(3...8).padding(Tokens.sp3).background(Tokens.surface2, in: RoundedRectangle(cornerRadius: Tokens.medium)); Text("Leave empty to use the default shown above. Applies to new chats.").font(.system(size: Tokens.sm)).foregroundStyle(Tokens.text3) }.appCard()
        }.font(.system(size: Tokens.md)).padding(Tokens.sp4) }.background(Tokens.background)
    }
}
/// Three dots bouncing in turn while the answer has no text yet.
private struct PendingDots: View {
    var body: some View {
        TimelineView(.animation) { context in
            let time = context.date.timeIntervalSinceReferenceDate
            HStack(spacing: Tokens.dotGap) {
                ForEach(0..<3, id: \.self) { index in
                    let phase = (time / 1.2 - Double(index) * 0.125).truncatingRemainder(dividingBy: 1)
                    let lift = phase < 0.5 ? sin(phase * 2 * .pi) : 0
                    Circle().fill(Tokens.text3).frame(width: Tokens.dot, height: Tokens.dot).opacity(0.45 + 0.55 * lift).offset(y: -Tokens.dotLift * lift)
                }
            }.frame(minHeight: Tokens.dotRow, alignment: .leading)
        }.accessibilityElement().accessibilityLabel("Generating")
    }
}
