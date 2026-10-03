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
/// Production affine INT4 linear. Coefficients retain their checkpoint dtype
/// and the output follows the activation dtype, matching upstream MLX.
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
        self.scales = scales
        self.biases = biases
        inputSize = weight.dim(1) * 8
        outputSize = weight.dim(0)
    }

    public func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) == inputSize,
              [.float32, .float16, .bfloat16].contains(input.dtype) else {
            throw M1Error.invalid("Linear input must be floating [tokens,\(inputSize)]")
        }
        return quantizedMM(input, weight, scales: scales,
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

/// Official Edge0 whole-layer prefill representation: every routed expert for
/// one projection remains stacked on axis zero and is selected by
/// `gatherQuantizedMM`. Unlike the one-expert correctness path, checkpoint
/// BF16 scales/biases stay BF16, matching `StreamingSwitchGLU.load_full_layer`.
public struct QuantizedExpertStackedLinear {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray
    public let expertCount: Int
    public let inputSize: Int
    public let outputSize: Int

    public init(weight: MLXArray, scales: MLXArray, biases: MLXArray) throws {
        guard weight.ndim == 3, weight.dtype == .uint32,
              weight.dim(0) > 0, weight.dim(1) > 0,
              weight.dim(2) > 0, weight.dim(2) % 8 == 0,
              scales.shape == [weight.dim(0), weight.dim(1), weight.dim(2) / 8],
              biases.shape == scales.shape,
              [.float32, .float16, .bfloat16].contains(scales.dtype),
              [.float32, .float16, .bfloat16].contains(biases.dtype) else {
            throw M1Error.invalid("Expected stacked affine INT4 expert tensors")
        }
        self.weight = weight
        self.scales = scales
        self.biases = biases
        expertCount = weight.dim(0)
        inputSize = weight.dim(2) * 8
        outputSize = weight.dim(1)
    }

    public func callAsFunction(_ input: MLXArray, expertIndices: MLXArray,
                               sortedIndices: Bool = true) throws -> MLXArray {
        guard input.ndim >= 2, input.dim(-1) == inputSize,
              [.uint8, .uint16, .uint32, .uint64,
               .int8, .int16, .int32, .int64].contains(expertIndices.dtype),
              [.float32, .float16, .bfloat16].contains(input.dtype) else {
            throw M1Error.invalid("Invalid stacked expert input or indices")
        }
        return gatherQuantizedMM(
            input, weight, scales: scales, biases: biases,
            rhsIndices: expertIndices, transpose: true,
            groupSize: 64, bits: 4, mode: .affine,
            sortedIndices: sortedIndices)
    }
}

public struct QuantizedExpertStack {
    public let up: QuantizedExpertStackedLinear
    public let gate: QuantizedExpertStackedLinear
    public let down: QuantizedExpertStackedLinear

    public init(up: QuantizedExpertStackedLinear, gate: QuantizedExpertStackedLinear,
                down: QuantizedExpertStackedLinear) throws {
        guard up.expertCount == gate.expertCount,
              up.expertCount == down.expertCount,
              up.inputSize == gate.inputSize,
              up.outputSize == gate.outputSize,
              down.inputSize == up.outputSize,
              down.outputSize == up.inputSize else {
            throw M1Error.invalid("Inconsistent stacked SwiGLU dimensions")
        }
        self.up = up
        self.gate = gate
        self.down = down
    }

    /// Official staged-slot assembly: selected expert bundles are stacked in
    /// prediction order, then consumed by gathered quantized matmuls.
    public init(experts: [QuantizedExpert]) throws {
        guard !experts.isEmpty else {
            throw M1Error.invalid("Cannot stack an empty expert selection")
        }
        func projection(_ keyPath: KeyPath<QuantizedExpert, QuantizedExpertLinear>) throws
            -> QuantizedExpertStackedLinear {
            let values = experts.map { $0[keyPath: keyPath] }
            return try QuantizedExpertStackedLinear(
                weight: stacked(values.map(\.weight), axis: 0),
                scales: stacked(values.map(\.scales), axis: 0),
                biases: stacked(values.map(\.biases), axis: 0))
        }
        try self.init(up: projection(\.up), gate: projection(\.gate),
                      down: projection(\.down))
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

    func loadDenseMLP(layer: Int) throws -> QuantizedExpert {
        func projection(_ p: ExpertProjection) throws -> QuantizedExpertLinear {
            let prefix = "model.layers.\(layer).mlp.\(p.rawValue)"
            return try QuantizedExpertLinear(weight: mlxArrayCopying(named: prefix + ".weight"),
                scales: mlxArrayCopying(named: prefix + ".scales"),
                biases: mlxArrayCopying(named: prefix + ".biases"))
        }
        return try QuantizedExpert(up: projection(.up), gate: projection(.gate), down: projection(.down))
    }

    /// Direct translation of official `StreamingSwitchGLU.load_full_layer` for
    /// Edge0-8B's separate gate/up/down layout: nine whole stacked tensors.
    func loadFullExpertLayer(layer: Int) throws -> QuantizedExpertStack {
        func projection(_ p: ExpertProjection) throws -> QuantizedExpertStackedLinear {
            let prefix = "model.layers.\(layer).mlp.experts.\(p.rawValue)."
            return try QuantizedExpertStackedLinear(
                weight: mlxArrayCopying(named: prefix + "weight"),
                scales: mlxArrayCopying(named: prefix + "scales"),
                biases: mlxArrayCopying(named: prefix + "biases"))
        }
        return try QuantizedExpertStack(up: projection(.up), gate: projection(.gate),
                                        down: projection(.down))
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
