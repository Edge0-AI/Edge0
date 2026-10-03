import CoreML
import Foundation
import MLX

/// Optional Core ML vocabulary projection used by the 35B path.
///
/// The model is discovered beside the streamed checkpoint rather than bundled with the
/// app: it is a derived 273 MiB asset and the existing MLX path remains the fallback
/// when it is absent or fails parity. Argmax stays in the graph, so only one Int32 token
/// crosses back from Core ML.
final class CoreMLVocabularyHead: @unchecked Sendable {
    static let directoryName = "lm-head.mlmodelc"
    static let width = 2_048

    private let model: MLModel

    init(directory: URL) throws {
        let url = directory.appendingPathComponent(Self.directoryName, isDirectory: true)
        let configuration = MLModelConfiguration()
        // Keep Core ML off the GPU so its projection does not compete with the MLX/Metal
        // decoder. Unsupported operations can fall back to the CPU.
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    func predict(_ hidden: MLXArray) async throws -> Int32 {
        let values = hidden.asType(.float16).asArray(Float16.self)
        guard values.count == Self.width else {
            throw Failure.badHiddenSize(values.count)
        }

        let input = try MLMultiArray(
            shape: [1, NSNumber(value: Self.width)], dataType: .float16)
        values.withUnsafeBytes { source in
            input.dataPointer.copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_state": MLFeatureValue(multiArray: input)
        ])
        let output = try await model.prediction(from: provider)
        guard let token = output.featureValue(for: "token")?.multiArrayValue,
              token.count == 1
        else { throw Failure.missingToken }
        return token[0].int32Value
    }

    private enum Failure: LocalizedError {
        case badHiddenSize(Int)
        case missingToken

        var errorDescription: String? {
            switch self {
            case .badHiddenSize(let count):
                "Core ML lm_head expected \(CoreMLVocabularyHead.width) values, got \(count)"
            case .missingToken:
                "Core ML lm_head did not return token"
            }
        }
    }
}
