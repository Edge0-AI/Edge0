import Edge0Core
import Edge0MLX
import Foundation
import MLX

struct Fixture: Decodable {
    let layer: Int
    let expert: Int
    let input: [Float]
    let up: [Float]
    let gate: [Float]
    let expertOutput: [Float]
    let moeOutput: [Float]
    let indices: [Int]
    let weights: [Float]
}

func compare(_ name: String, _ actual: [Float], _ expected: [Float], atol: Float = 0.0002,
             rtol: Float = 0.0002) throws {
    guard actual.count == expected.count, !actual.isEmpty else { throw M1Error.invalid("\(name): size mismatch") }
    var maxError: Float = 0
    var failures = 0
    for (a, e) in zip(actual, expected) {
        maxError = max(maxError, abs(a - e))
        if !a.isFinite || !e.isFinite || abs(a-e) > atol + rtol * abs(e) { failures += 1 }
    }
    print("\(name): max_abs=\(maxError), failures=\(failures)/\(actual.count), atol=\(atol), rtol=\(rtol)")
    guard failures == 0 else { throw M1Error.invalid("\(name): numerical parity failed") }
}

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 3 || (args.count == 4 && args[3] == "--cpu") else {
        throw M1Error.invalid("Usage: edge0-m1-validate MODEL_DIRECTORY REFERENCE.json [--cpu]")
    }
    let model = URL(fileURLWithPath: args[1])
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    let c = try BailingConfiguration.load(from: model.appendingPathComponent("config.json"))
    guard fixture.input.count == c.hiddenSize else { throw M1Error.invalid("Fixture input width mismatch") }
    let store = try ExpertTensorStore(modelURL: model.appendingPathComponent("model.safetensors"), expertCount: c.numExperts)
    let x = MLXArray(fixture.input, [1, c.hiddenSize])
    let expert = try store.loadExpert(layer: fixture.layer, expert: fixture.expert)
    try compare("up", expert.up(x).asArray(Float.self), fixture.up)
    try compare("gate", expert.gate(x).asArray(Float.self), fixture.gate)
    try compare("expert SwiGLU/down", expert(x).asArray(Float.self), fixture.expertOutput)
    let block = try StreamingMoE(configuration: c, store: store, layer: fixture.layer)
    let result = try block(x)
    guard Set(result.expertIndices) == Set(fixture.indices),
          Set(fixture.indices).count == c.expertsPerToken,
          fixture.indices.count == fixture.weights.count else { throw M1Error.invalid("Selected experts differ") }
    let expected = Dictionary(uniqueKeysWithValues: zip(fixture.indices, fixture.weights))
    try compare("router weights", result.expertWeights, result.expertIndices.map { expected[$0]! }, atol: 1e-6, rtol: 1e-6)
    try compare("complete MoE (routed + shared)", result.hidden.asArray(Float.self), fixture.moeOutput)
    print("PASS: M1 Float32 parity, layer \(fixture.layer), expert \(fixture.expert), selected \(result.expertIndices)")
}

do {
    try Device.withDefaultDevice(CommandLine.arguments.last == "--cpu" ? .cpu : .gpu) { try run() }
} catch {
    FileHandle.standardError.write(Data("FAIL: \(error.localizedDescription)\n".utf8))
    exit(1)
}
