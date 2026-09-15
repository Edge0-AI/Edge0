import Edge0Core
import Foundation
import MLX

public enum M1Error: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

/// Affine INT4: W[o,i] = nibble[o,i] * scale[o,i/64] + bias[o,i/64].
/// M1 deliberately computes in Float32, including BF16 coefficient promotion.
/// The packed weights remain packed; no dense expert weight matrix is created.
public struct QuantizedExpertLinear {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray
    public let inputSize: Int
    public let outputSize: Int

    public init(weight: MLXArray, scales: MLXArray, biases: MLXArray) throws {
        guard weight.ndim == 2, weight.dtype == .uint32,
              weight.dim(0) > 0, weight.dim(1) > 0, weight.dim(1) % 8 == 0,
              scales.shape == [weight.dim(0), weight.dim(1) / 8],
              biases.shape == scales.shape,
              [.float32, .float16, .bfloat16].contains(scales.dtype),
              [.float32, .float16, .bfloat16].contains(biases.dtype) else {
            throw M1Error.invalid("Expected affine INT4 [out,in/8] U32 and floating [out,in/64] coefficients")
        }
        self.weight = weight
        self.scales = scales.asType(.float32)
        self.biases = biases.asType(.float32)
        inputSize = weight.dim(1) * 8
        outputSize = weight.dim(0)
    }

    public func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) == inputSize,
              [.float32, .float16, .bfloat16].contains(input.dtype) else {
            throw M1Error.invalid("Linear input must be floating [tokens,\(inputSize)]")
        }
        return quantizedMM(input.asType(.float32), weight, scales: scales,
                           biases: biases, transpose: true, groupSize: 64, bits: 4, mode: .affine)
    }
}

public struct QuantizedExpert {
    public let up: QuantizedExpertLinear
    public let gate: QuantizedExpertLinear
    public let down: QuantizedExpertLinear

    public init(up: QuantizedExpertLinear, gate: QuantizedExpertLinear,
                down: QuantizedExpertLinear) throws {
        guard up.inputSize == gate.inputSize, up.outputSize == gate.outputSize,
              down.inputSize == up.outputSize, down.outputSize == up.inputSize else {
            throw M1Error.invalid("Inconsistent SwiGLU projection dimensions")
        }
        self.up = up; self.gate = gate; self.down = down
    }

    public func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        let g = try gate(input)
        return try down((g * sigmoid(g)) * up(input))
    }
}

public extension ExpertTensorStore {
    /// Copies one named tensor, used for the small router and shared expert.
    func mlxArrayCopying(named name: String) throws -> MLXArray {
        let descriptor = try index.tensor(named: name)
        let pointer = try mappedFile.pointer(to: descriptor.byteRange)
        return try copyTensor(descriptor: descriptor, pointer: pointer)
    }

    func loadExpert(layer: Int, expert: Int) throws -> QuantizedExpert {
        guard (0..<expertCount).contains(expert) else { throw M1Error.invalid("Expert out of range") }
        func projection(_ p: ExpertProjection) throws -> QuantizedExpertLinear {
            try QuantizedExpertLinear(
                weight: mlxArrayCopying(layer: layer, expert: expert, projection: p, part: .weight),
                scales: mlxArrayCopying(layer: layer, expert: expert, projection: p, part: .scales),
                biases: mlxArrayCopying(layer: layer, expert: expert, projection: p, part: .biases))
        }
        return try QuantizedExpert(up: projection(.up), gate: projection(.gate), down: projection(.down))
    }

    func loadSharedExpert(layer: Int) throws -> QuantizedExpert {
        func projection(_ p: ExpertProjection) throws -> QuantizedExpertLinear {
            let prefix = "model.layers.\(layer).mlp.shared_experts.\(p.rawValue)"
            return try QuantizedExpertLinear(weight: mlxArrayCopying(named: prefix + ".weight"),
                scales: mlxArrayCopying(named: prefix + ".scales"),
                biases: mlxArrayCopying(named: prefix + ".biases"))
        }
        return try QuantizedExpert(up: projection(.up), gate: projection(.gate), down: projection(.down))
    }
}

func copyTensor(descriptor: TensorDescriptor, pointer: UnsafeMutableRawPointer) throws -> MLXArray {
    let dtype: DType
    switch descriptor.dtype {
    case .u32: dtype = .uint32
    case .bf16: dtype = .bfloat16
    case .f16: dtype = .float16
    case .f32: dtype = .float32
    default: throw Edge0MLXError.unsupportedDType(descriptor.dtype)
    }
    var elements = 1
    for dimension in descriptor.shape {
        let product = elements.multipliedReportingOverflow(by: dimension)
        guard dimension > 0, !product.overflow else { throw M1Error.invalid("Invalid tensor shape") }
        elements = product.partialValue
    }
    let size = elements.multipliedReportingOverflow(by: descriptor.dtype.byteWidth)
    guard !size.overflow, size.partialValue == descriptor.byteCount else {
        throw M1Error.invalid("Tensor shape/byte count mismatch: \(descriptor.name)")
    }
    return MLXArray(Data(bytes: pointer, count: descriptor.byteCount), descriptor.shape, dtype: dtype)
}
