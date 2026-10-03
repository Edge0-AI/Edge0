import Foundation

public enum SafetensorsError: Error, LocalizedError {
    case fileTooSmall
    case invalidHeaderLength(UInt64)
    case invalidHeader
    case malformedTensor(String)
    case unsupportedDType(String)
    case tensorNotFound(String)
    case invalidExpertAxis(String)

    public var errorDescription: String? {
        switch self {
        case .fileTooSmall: return "Safetensors file is too small."
        case .invalidHeaderLength(let n): return "Invalid safetensors header length: \(n)."
        case .invalidHeader: return "Safetensors JSON header is invalid."
        case .malformedTensor(let name): return "Malformed tensor entry: \(name)."
        case .unsupportedDType(let dtype): return "Unsupported safetensors dtype: \(dtype)."
        case .tensorNotFound(let name): return "Tensor not found: \(name)."
        case .invalidExpertAxis(let name): return "Tensor has no valid expert axis: \(name)."
        }
    }
}

public enum SafetensorsDType: String, Sendable, Codable {
    case bool = "BOOL"
    case u8 = "U8"
    case i8 = "I8"
    case i16 = "I16"
    case u16 = "U16"
    case i32 = "I32"
    case u32 = "U32"
    case i64 = "I64"
    case u64 = "U64"
    case f16 = "F16"
    case bf16 = "BF16"
    case f32 = "F32"
    case f64 = "F64"

    public var byteWidth: Int {
        switch self {
        case .bool, .u8, .i8: 1
        case .i16, .u16, .f16, .bf16: 2
        case .i32, .u32, .f32: 4
        case .i64, .u64, .f64: 8
        }
    }
}

public struct TensorDescriptor: Sendable, Equatable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    /// Absolute byte range within the safetensors file.
    public let byteRange: Range<Int>

    public var byteCount: Int { byteRange.count }

    /// Returns a contiguous slice along axis 0. Edge0 stores routed experts stacked on axis 0.
    public func axisZeroSlice(_ index: Int) throws -> TensorDescriptor {
        guard let count = shape.first, count > 0, index >= 0, index < count else {
            throw SafetensorsError.invalidExpertAxis(name)
        }
        guard byteCount % count == 0 else {
            throw SafetensorsError.invalidExpertAxis(name)
        }
        let stride = byteCount / count
        let start = byteRange.lowerBound + index * stride
        return TensorDescriptor(
            name: "\(name)[\(index)]",
            dtype: dtype,
            shape: Array(shape.dropFirst()),
            byteRange: start..<(start + stride)
        )
    }
}

public struct SafetensorsIndex: Sendable {
    public let fileURL: URL
    public let headerLength: Int
    public let dataOffset: Int
    public let tensors: [String: TensorDescriptor]

    public init(fileURL: URL) throws {
        self.fileURL = fileURL
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else { throw SafetensorsError.fileTooSmall }
        let headerLength64 = prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerLength64 <= UInt64(Int.max) else {
            throw SafetensorsError.invalidHeaderLength(headerLength64)
        }
        let headerLength = Int(headerLength64)
        guard headerLength > 0 else { throw SafetensorsError.invalidHeaderLength(headerLength64) }

        let headerData = try handle.read(upToCount: headerLength) ?? Data()
        guard headerData.count == headerLength else { throw SafetensorsError.fileTooSmall }
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SafetensorsError.invalidHeader
        }

        let base = 8 + headerLength
        var parsed: [String: TensorDescriptor] = [:]
        parsed.reserveCapacity(json.count)

        for (name, value) in json where name != "__metadata__" {
            guard
                let entry = value as? [String: Any],
                let dtypeString = entry["dtype"] as? String,
                let dtype = SafetensorsDType(rawValue: dtypeString),
                let shapeNumbers = entry["shape"] as? [NSNumber],
                let offsets = entry["data_offsets"] as? [NSNumber],
                offsets.count == 2
            else {
                if let entry = value as? [String: Any], let dtypeString = entry["dtype"] as? String,
                   SafetensorsDType(rawValue: dtypeString) == nil {
                    throw SafetensorsError.unsupportedDType(dtypeString)
                }
                throw SafetensorsError.malformedTensor(name)
            }

            let shape = shapeNumbers.map(\.intValue)
            let relativeStart = offsets[0].intValue
            let relativeEnd = offsets[1].intValue
            guard relativeStart >= 0, relativeEnd >= relativeStart else {
                throw SafetensorsError.malformedTensor(name)
            }

            parsed[name] = TensorDescriptor(
                name: name,
                dtype: dtype,
                shape: shape,
                byteRange: (base + relativeStart)..<(base + relativeEnd)
            )
        }

        self.headerLength = headerLength
        self.dataOffset = base
        self.tensors = parsed
    }

    public subscript(_ name: String) -> TensorDescriptor? { tensors[name] }

    public func tensor(named name: String) throws -> TensorDescriptor {
        guard let tensor = tensors[name] else { throw SafetensorsError.tensorNotFound(name) }
        return tensor
    }
}
