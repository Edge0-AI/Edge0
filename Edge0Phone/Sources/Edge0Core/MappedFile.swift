import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public enum MappedFileError: Error, LocalizedError {
    case openFailed(String)
    case statFailed(String)
    case mapFailed(String)
    case outOfBounds(Range<Int>)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let p): return "Could not open \(p)."
        case .statFailed(let p): return "Could not stat \(p)."
        case .mapFailed(let p): return "Could not mmap \(p)."
        case .outOfBounds(let r): return "Mapped byte range out of bounds: \(r)."
        }
    }
}

/// Read-only virtual-memory mapping of a model file.
/// mmap does not make the whole file resident; pages are faulted in on demand.
public final class MappedFile: @unchecked Sendable {
    public let url: URL
    public let count: Int
    private let fd: Int32
    private let base: UnsafeMutableRawPointer

    public init(url: URL) throws {
        self.url = url
        let path = url.path
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else { throw MappedFileError.openFailed(path) }

        var st = stat()
        guard fstat(descriptor, &st) == 0 else {
            close(descriptor)
            throw MappedFileError.statFailed(path)
        }
        let size = Int(st.st_size)
        guard size > 0 else {
            close(descriptor)
            throw MappedFileError.statFailed(path)
        }

        let mapping = mmap(nil, size, PROT_READ, MAP_PRIVATE, descriptor, 0)
        guard mapping != MAP_FAILED, let mapping else {
            close(descriptor)
            throw MappedFileError.mapFailed(path)
        }

        self.fd = descriptor
        self.count = size
        self.base = mapping
    }

    deinit {
        munmap(base, count)
        close(fd)
    }

    public func pointer(to range: Range<Int>) throws -> UnsafeMutableRawPointer {
        guard range.lowerBound >= 0, range.upperBound <= count else {
            throw MappedFileError.outOfBounds(range)
        }
        return base.advanced(by: range.lowerBound)
    }

    public func bytes(in range: Range<Int>) throws -> UnsafeRawBufferPointer {
        let ptr = try pointer(to: range)
        return UnsafeRawBufferPointer(start: UnsafeRawPointer(ptr), count: range.count)
    }

    /// Advisory only; ignored if the OS declines it.
    public func adviseSequential() {
        #if canImport(Darwin)
        _ = madvise(base, count, MADV_SEQUENTIAL)
        #endif
    }
}
