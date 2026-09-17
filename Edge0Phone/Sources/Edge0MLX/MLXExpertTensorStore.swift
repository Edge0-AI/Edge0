import Edge0Core
import Foundation
import MLX

public enum Edge0MLXError: Error {
    case unsupportedDType(SafetensorsDType)
}

public extension ExpertTensorStore {
    /// Correctness-first bridge from an mmap-backed expert slice into MLX.
    ///
    /// Only the requested expert tensor is copied. The 4.5 GB checkpoint remains
    /// memory-mapped and non-resident as a whole. We intentionally avoid
    /// MLXArray(rawPointer:) here: MLX documents that Metal requires compatible
    /// backing, and a normal file mmap is not yet proven to satisfy that contract.
    func mlxArrayCopying(
        layer: Int,
        expert: Int,
        projection: ExpertProjection,
        part: QuantPart
    ) throws -> MLXArray {
        let slice = try slice(layer: layer, expert: expert, projection: projection, part: part)
        return try copyTensor(descriptor: slice.descriptor, pointer: slice.pointer)
    }
}
