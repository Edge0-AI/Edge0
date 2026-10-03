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

    /// Official startup prewarm hint. Advisory only; iOS remains free to
    /// limit page-cache residency under memory pressure.
    public func adviseWillNeed() {
        #if canImport(Darwin)
        _ = madvise(base, count, MADV_WILLNEED)
        #endif
    }

    /// Advisory readahead for a checkpoint subrange. Streaming experts use
    /// this before copying their tensor slices so page faults can be serviced
    /// in bulk rather than serially on the decode critical path.
    public func adviseWillNeed(range: Range<Int>) {
        #if canImport(Darwin)
        guard range.lowerBound >= 0, range.upperBound <= count,
              !range.isEmpty else { return }
        let pageSize = max(4096, Int(sysconf(_SC_PAGESIZE)))
        let start = range.lowerBound / pageSize * pageSize
        let end = min(
            count,
            ((range.upperBound + pageSize - 1) / pageSize) * pageSize)
        _ = madvise(base.advanced(by: start), end - start, MADV_WILLNEED)
        #endif
    }

    /// Official `SafetensorsMmap.seq_read`: force a real sequential pass over
    /// the checkpoint. `madvise` alone does not reliably fault every page in.
    public func sequentialRead(chunkSize: Int = 1 << 24) {
        let size = max(4096, chunkSize)
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: size, alignment: MemoryLayout<UInt64>.alignment)
        defer { buffer.deallocate() }
        var offset = 0
        while offset < count {
            let requested = min(size, count - offset)
            let bytesRead = pread(fd, buffer, requested, off_t(offset))
            guard bytesRead > 0 else { break }
            offset += bytesRead
        }
    }
}
