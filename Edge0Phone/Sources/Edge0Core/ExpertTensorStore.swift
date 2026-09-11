import Foundation

public enum ExpertProjection: String, CaseIterable, Sendable {
    case up = "up_proj"
    case gate = "gate_proj"
    case down = "down_proj"
}

public enum QuantPart: String, CaseIterable, Sendable {
    case weight
    case scales
    case biases
}

public struct ExpertTensorKey: Hashable, Sendable {
    public let layer: Int
    public let expert: Int
    public let projection: ExpertProjection
    public let part: QuantPart
}

public struct ExpertTensorSlice: @unchecked Sendable {
    public let descriptor: TensorDescriptor
    public let pointer: UnsafeMutableRawPointer
}

/// Index + mmap layer for the Edge0 checkpoint. No MLX dependency here.
/// The MLX adapter can turn these slices into arrays only when an expert is selected.
public final class ExpertTensorStore: @unchecked Sendable {
    public let index: SafetensorsIndex
    public let mappedFile: MappedFile
    public let expertCount: Int

    public init(modelURL: URL, expertCount: Int = 128) throws {
        self.index = try SafetensorsIndex(fileURL: modelURL)
        self.mappedFile = try MappedFile(url: modelURL)
        self.expertCount = expertCount
    }

    public func stackedTensorName(layer: Int, projection: ExpertProjection, part: QuantPart) -> String {
        "model.layers.\(layer).mlp.experts.\(projection.rawValue).\(part.rawValue)"
    }

    public func slice(
        layer: Int,
        expert: Int,
        projection: ExpertProjection,
        part: QuantPart
    ) throws -> ExpertTensorSlice {
        precondition((0..<expertCount).contains(expert))
        let name = stackedTensorName(layer: layer, projection: projection, part: part)
        let stacked = try index.tensor(named: name)
        guard stacked.shape.first == expertCount else {
            throw SafetensorsError.invalidExpertAxis(name)
        }
        let descriptor = try stacked.axisZeroSlice(expert)
        let pointer = try mappedFile.pointer(to: descriptor.byteRange)
        return ExpertTensorSlice(descriptor: descriptor, pointer: pointer)
    }
}
