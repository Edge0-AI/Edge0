import Edge0Core
@testable import Edge0MLX
import Foundation
import MLX
import Testing

// CPU arithmetic by default; MLX still requires an Xcode-built Metal bundle.
// EDGE0_TEST_GPU=1 is used by the documented xcodebuild verification.
private func onDevice(_ body: () throws -> Void) rethrows {
    try Device.withDefaultDevice(ProcessInfo.processInfo.environment["EDGE0_TEST_GPU"] == "1" ? .gpu : .cpu, body)
}

@Test func packedAffineLinearMatchesScalarOracle() throws {
    try onDevice {
        // Two groups, all nibble positions, nonzero offsets, non-square matrix, two tokens.
        let width = 128, rows = 64
        var packed = [UInt32](repeating: 0, count: rows * width / 8)
        var scales = [Float](), biases = [Float]()
        for o in 0..<rows {
            scales += [Float(o % 3 + 1) / 32, Float(o % 5 + 1) / 64]
            biases += [-0.25, 0.125]
            for i in 0..<width { packed[o * width / 8 + i / 8] |= UInt32((o * 3 + i) % 16) << (4 * (i % 8)) }
        }
        let x = (0..<(width * 2)).map { Float($0 % 19 - 9) / 16 }
        let linear = try QuantizedExpertLinear(weight: MLXArray(packed, [rows, width/8]),
            scales: MLXArray(scales, [rows, 2]).asType(.bfloat16),
            biases: MLXArray(biases, [rows, 2]).asType(.bfloat16))
        let y = try linear(MLXArray(x, [2, width])).asArray(Float.self)
        for t in 0..<2 {
            for o in 0..<rows {
                var expected: Float = 0
                for i in 0..<width {
                    let q = Float((o * 3 + i) % 16)
                    expected += x[t * width + i] * (q * scales[o*2+i/64] + biases[o*2+i/64])
                }
                #expect(abs(y[t*rows+o] - expected) < 1e-5)
            }
        }
        #expect(throws: M1Error.self) { try linear(MLXArray.zeros([1, 64])) }
    }
}

@Test func malformedQuantizedShapesRejected() {
    onDevice {
        #expect(throws: M1Error.self) {
            try QuantizedExpertLinear(weight: MLXArray.zeros([64, 8]),
                scales: MLXArray.zeros([64, 1]), biases: MLXArray.zeros([64, 1]))
        }
        #expect(throws: M1Error.self) {
            try QuantizedExpertLinear(weight: MLXArray.zeros([64, 8], dtype: .uint32),
                scales: MLXArray.zeros([64, 2]), biases: MLXArray.zeros([64, 2]))
        }
    }
}

@Test func groupedMLXMatchesCPUWithSelectionBias() {
    onDevice {
        let logits: [Float] = [-4, -3, 5, 4, 1, 0, 3, 2]
        let bias: [Float] = [3, 2, 0, 0, 0.2, 0, -1, -1]
        for keep in [2, 4] {
            for k in [1, 2] {
                for normalize in [false, true] {
                    let cpu = GroupedExpertRouter.select(logits: logits, expertBias: bias, topK: k,
                        nGroup: 4, topkGroup: keep, normalize: normalize)
                    let gpu = BailingGroupedRouter.select(logits: MLXArray(logits, [1, 8]),
                        expertBias: MLXArray(bias), topK: k, nGroup: 4, topkGroup: keep, normalize: normalize)
                    let ids = gpu.indices.asArray(Int32.self).map(Int.init)
                    let weights = gpu.weights.asArray(Float.self)
                    #expect(Set(ids) == Set(cpu.indices))
                    for (id, w) in zip(ids, weights) {
                        #expect(abs(w - cpu.weights[cpu.indices.firstIndex(of: id)!]) < 1e-6)
                    }
                }
            }
        }
    }
}

@Test func swigluUsesGateAndDownProjection() throws {
    try onDevice {
        // Identity INT4 projections make the expected result exactly x * sigmoid(x) * x.
        var packed = [UInt32](repeating: 0, count: 64 * 8)
        for i in 0..<64 { packed[i*8+i/8] = 1 << (4*(i%8)) }
        let linear = try QuantizedExpertLinear(weight: MLXArray(packed, [64, 8]),
            scales: MLXArray.ones([64, 1]), biases: MLXArray.zeros([64, 1]))
        let expert = try QuantizedExpert(up: linear, gate: linear, down: linear)
        let x = (0..<64).map { Float($0-32)/8 }
        let actual = try expert(MLXArray(x, [1, 64])).asArray(Float.self)
        for (v, y) in zip(x, actual) { #expect(abs(y - v*v/(1+exp(-v))) < 2e-6) }
    }
}

@Test func streamedMoEIncludesSharedAndOnlySelectedExperts() throws {
    try onDevice {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var config = try JSONSerialization.jsonObject(with: Data(contentsOf:
            Bundle.module.url(forResource: "config", withExtension: "json", subdirectory: "Fixtures")!)) as! [String: Any]
        for key in ["hidden_size", "moe_intermediate_size", "moe_shared_expert_intermediate_size"] { config[key] = 64 }
        config["num_experts"] = 8; config["n_group"] = 4
        config["topk_group"] = 2; config["num_experts_per_tok"] = 2
        let c = try JSONDecoder().decode(BailingConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
        var header = [String: Any](), payload = Data()
        func append<T>(_ name: String, _ dtype: String, _ shape: [Int], _ values: [T]) {
            let bytes = values.withUnsafeBytes { Data($0) }
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + bytes.count]]
            payload.append(bytes)
        }
        var identity = [UInt32](repeating: 0, count: 64*8)
        for i in 0..<64 { identity[i*8+i/8] = 1 << (4*(i%8)) }
        let prefix = "model.layers.1.mlp"
        let logits: [Float] = [-4, -3, -2, -1, 0, 1, 2, 3]
        let bias: [Float] = [3, 2, 0, 0, 0, 0, -1, -1]
        var router = [Float](repeating: 0, count: 8*64)
        for i in 0..<8 { router[i*64] = logits[i] }
        append(prefix + ".gate.weight", "F32", [8, 64], router)
        append(prefix + ".gate.expert_bias", "F32", [8], bias)
        let route = GroupedExpertRouter.select(logits: logits, expertBias: bias, topK: 2, nGroup: 4, topkGroup: 2)
        for projection in ExpertProjection.allCases {
            let name = prefix + ".experts." + projection.rawValue
            append(name + ".weight", "U32", [8, 64, 8], Array(repeating: identity, count: 8).flatMap { $0 })
            // Unselected experts deliberately contain NaN scales: their execution contaminates output.
            let scales: [Float] = (0..<8).flatMap { id in
                Array(repeating: route.indices.contains(id) ? (projection == .up ? Float(id+1)/8 : 1) : Float.nan, count: 64)
            }
            append(name + ".scales", "F32", [8, 64, 1], scales)
            append(name + ".biases", "F32", [8, 64, 1], [Float](repeating: 0, count: 8*64))
            let shared = prefix + ".shared_experts." + projection.rawValue
            append(shared + ".weight", "U32", [64, 8], identity)
            append(shared + ".scales", "F32", [64, 1], [Float](repeating: projection == .up ? 2 : 1, count: 64))
            append(shared + ".biases", "F32", [64, 1], [Float](repeating: 0, count: 64))
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: .sortedKeys)
        while json.count % 8 != 0 { json.append(32) }
        var length = UInt64(json.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(json); file.append(payload)
        let url = directory.appendingPathComponent("model.safetensors")
        try file.write(to: url)
        let store = try ExpertTensorStore(modelURL: url, expertCount: 8)
        let block = try StreamingMoE(configuration: c, store: store, layer: 1)
        var input = (0..<64).map { Float($0-32)/16 }; input[0] = 1
        let output = try block(MLXArray(input, [1, 64]))
        #expect(Set(output.expertIndices) == Set(route.indices))
        let multiplier = zip(route.indices, route.weights).reduce(Float(2)) { $0 + Float($1.0+1)/8 * $1.1 }
        for (x, y) in zip(input, output.hidden.asArray(Float.self)) {
            #expect(y.isFinite && abs(y - multiplier*x*x/(1+exp(-x))) < 3e-6)
        }
        #expect(throws: M1Error.self) { try block(MLXArray.zeros([2, 64])) }
        #expect(throws: M1Error.self) { try store.loadExpert(layer: 1, expert: 8) }
        #expect(throws: M1Error.self) { try StreamingMoE(configuration: c, store: store, layer: 0) }
        // A copied expert is safe after the originating mapping closes.
        let detached: QuantizedExpert = try {
            let temporary = try ExpertTensorStore(modelURL: url, expertCount: 8)
            return try temporary.loadExpert(layer: 1, expert: route.indices[0])
        }()
        let detachedY = try detached(MLXArray(input, [1,64])).asArray(Float.self)
        #expect(detachedY.allSatisfy { $0.isFinite })
    }
}
