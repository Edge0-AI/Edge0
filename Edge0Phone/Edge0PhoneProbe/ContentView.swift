import Edge0Core
import Edge0MLX
import Foundation
import MLX
import SwiftUI

private enum ProbeKind: Sendable {
    case m0
    case m1
}

private func executeProbe(kind: ProbeKind, modelURL: URL) throws -> String {
    try writeProbeBreadcrumb(modelURL: modelURL, "start \(kind)")
    let configuration = try BailingConfiguration.load(from: modelURL.appendingPathComponent("config.json"))
    try writeProbeBreadcrumb(modelURL: modelURL, "configuration decoded")
    let store = try ExpertTensorStore(
        modelURL: modelURL.appendingPathComponent("model.safetensors"),
        expertCount: configuration.numExperts
    )
    try writeProbeBreadcrumb(modelURL: modelURL, "safetensors indexed and file mapped")
    let fileBytes = try FileManager.default.attributesOfItem(atPath: store.index.fileURL.path)[.size] as? Int64 ?? 0
    try writeProbeBreadcrumb(modelURL: modelURL, "file size read")
    let parts = try ExpertProjection.allCases.flatMap { projection in
        try QuantPart.allCases.map { part in
            try store.slice(layer: 1, expert: 0, projection: projection, part: part).descriptor
        }
    }
    try writeProbeBreadcrumb(modelURL: modelURL, "layer 1 expert 0 addressed")
    guard kind == .m1 else {
        try writeProbeBreadcrumb(modelURL: modelURL, "M0 complete")
        return "PASS: M0 on this iPhone\n\nCheckpoint: \(fileBytes) bytes\nIndexed tensors: \(store.index.tensors.count)\nLayer 1 / expert 0 tensors: \(parts.count)\nArchitecture: \(configuration.hiddenLayers) layers, \(configuration.numExperts) experts, top-\(configuration.expertsPerToken)"
    }

    return try Device.withDefaultDevice(.gpu) {
        try writeProbeBreadcrumb(modelURL: modelURL, "starting MLX Metal MoE")
        let input = (0..<configuration.hiddenSize).map { Float(sin(Double($0) * 0.071)) * 0.5 }
        let block = try StreamingMoE(configuration: configuration, store: store, layer: 1)
        let result = try block(MLXArray(input, [1, configuration.hiddenSize]))
        try writeProbeBreadcrumb(modelURL: modelURL, "MoE graph evaluated")
        eval(result.hidden)
        let output = result.hidden.asArray(Float.self)
        guard output.allSatisfy(\.isFinite) else { throw M1Error.invalid("MoE returned non-finite values") }
        let maxMagnitude = output.map { abs($0) }.max() ?? 0
        let formattedWeights = result.expertWeights.map { String(format: "%.4f", $0) }.joined(separator: ", ")
        let formattedMagnitude = String(format: "%.6f", maxMagnitude)
        return "PASS: M1 Metal MoE on this iPhone\n\nLayer: 1\nRouted experts: \(result.expertIndices)\nWeights: \(formattedWeights)\nOutput width: \(output.count)\nMax |output|: \(formattedMagnitude)\n\nThis runs the real checkpoint's router, eight routed INT4 experts, and shared expert."
    }
}

private func writeProbeBreadcrumb(modelURL: URL, _ message: String) throws {
    let url = modelURL.appendingPathComponent("edge0phone-probe.txt")
    let line = "\(ISO8601DateFormatter().string(from: Date())) \(message)\n"
    if FileManager.default.fileExists(atPath: url.path) {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(line.utf8))
    } else {
        try Data(line.utf8).write(to: url, options: .atomic)
    }
}

@MainActor
final class ProbeViewModel: ObservableObject {
    @Published private(set) var status = "Ready. Connect a model folder through Finder or Files."
    @Published private(set) var running = false

    private let modelFolderName = "Edge0-8B-A1B-preview"

    func runM0() { run(kind: .m0) }
    func runM1() { run(kind: .m1) }

    func autorunIfRequested() {
        switch ProcessInfo.processInfo.environment["EDGE0_AUTORUN"] {
        case "M0": runM0()
        case "M1": runM1()
        default: break
        }
    }

    private func run(kind: ProbeKind) {
        guard !running else { return }
        running = true
        status = "Opening the local checkpoint…"
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let nestedModelURL = documentsURL.appendingPathComponent(modelFolderName, isDirectory: true)
        let modelURL = FileManager.default.fileExists(
            atPath: nestedModelURL.appendingPathComponent("model.safetensors").path
        ) ? nestedModelURL : documentsURL
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let message: String
            do {
                message = try executeProbe(kind: kind, modelURL: modelURL)
            } catch {
                message = "FAILED\n\(error.localizedDescription)"
            }
            DispatchQueue.main.async {
                self?.status = message
                self?.running = false
            }
        }
    }

}

struct ContentView: View {
    @StateObject private var probe = ProbeViewModel()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20) {
                Text("Edge0Phone")
                    .font(.largeTitle.bold())
                Text("Physical M0/M1 probe")
                    .foregroundStyle(.secondary)
                Text("Model files may be in Edge0Phone’s Documents folder or its Edge0-8B-A1B-preview subfolder.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                HStack {
                    Button("Run M0 storage check") { probe.runM0() }
                    Button("Run M1 Metal MoE") { probe.runM1() }
                        .buttonStyle(.borderedProminent)
                }
                .disabled(probe.running)
                if probe.running { ProgressView() }
                ScrollView {
                    Text(probe.status)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                Spacer()
            }
            .padding()
            .navigationTitle("Probe")
            .task { probe.autorunIfRequested() }
        }
    }
}
