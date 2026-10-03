import Edge0Core
import Foundation
import MLX

public enum Edge0MLXError: Error {
    case unsupportedDType(SafetensorsDType)
}

public extension ExpertTensorStore {
    /// Mirrors the upstream streaming loader's targeted `MADV_WILLNEED`
    /// pass. It retains no MLX arrays and is safe to issue speculatively.
    func adviseExpertsWillNeed(layer: Int, experts: [Int]) {
        for expert in Set(experts) where (0..<expertCount).contains(expert) {
            for projection in ExpertProjection.allCases {
                for part in QuantPart.allCases {
                    let name = stackedTensorName(
                        layer: layer, projection: projection, part: part)
                    guard let stacked = index[name],
                          let slice = try? stacked.axisZeroSlice(expert) else {
                        continue
                    }
                    mappedFile.adviseWillNeed(range: slice.byteRange)
                }
            }
        }
    }

    /// Correctness-first bridge from an mmap-backed expert slice into MLX.
    ///
    /// Only the requested expert tensor is copied. The 4.5 GB checkpoint remains
    /// memory-mapped and non-resident as a whole. This avoids
    /// MLXArray(rawPointer:) here: MLX documents that Metal requires compatible
    /// backing; a normal file mmap is deliberately copied into MLX-owned storage.
    func mlxArrayCopying(
        layer: Int,
        expert: Int,
        projection: ExpertProjection,
        part: QuantPart
    ) throws -> MLXArray {
        let slice = try slice(layer: layer, expert: expert, projection: projection, part: part)
        return try copyTensor(descriptor: slice.descriptor, pointer: slice.pointer)
    }

    /// Copies one row from an arbitrary axis-0-contiguous checkpoint tensor.
    /// Used by the quantized embedding so a token lookup never materializes
    /// the full 157k x 1536 embedding table.
    func mlxAxisZeroSliceCopying(named name: String, index row: Int) throws -> MLXArray {
        let tensor = try index.tensor(named: name)
        guard tensor.shape.first.map({ row >= 0 && row < $0 }) == true else {
            throw M1Error.invalid("Axis-zero index out of range for \(name)")
        }
        let slice = try tensor.axisZeroSlice(row)
        let pointer = try mappedFile.pointer(to: slice.byteRange)
        return try copyTensor(descriptor: slice, pointer: pointer)
    }
}
