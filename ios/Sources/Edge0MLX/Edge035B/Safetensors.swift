import Foundation

/// Read-only mmap over one safetensors shard, plus the header that says where each
/// tensor's bytes begin.
///
/// A shard is: 8 bytes of little-endian header length, that many bytes of JSON, then
/// the payload. Every tensor's `data_offsets` are relative to the start of the
/// payload, so a tensor — or, for a stacked expert tensor, one expert's slice of it
/// — is a byte range and nothing more. That is the premise the whole streaming
/// design rests on: **no parsing, no dequantisation, no copy on the read path.**
///
/// Mapped `MAP_SHARED` + read-only so the pages stay clean and file-backed. Dirtying
/// them would turn the 19 GB into the app's own memory and defeat the exercise.
final class SafetensorsShard {

    struct Entry {
        let dtype: String
        let shape: [Int]
        /// Absolute offset in the file, payload base already added.
        let offset: Int
        let byteCount: Int
    }

    let url: URL
    private let descriptor: Int32
    private let base: UnsafeRawPointer
    private let mappedSize: Int
    private(set) var entries: [String: Entry] = [:]

    enum Failure: LocalizedError {
        case cannotOpen(String)
        case cannotMap(String)
        case badHeader(String)
        case truncated(String, needs: Int, has: Int)

        var errorDescription: String? {
            switch self {
            case .cannotOpen(let p): return "cannot open \(p)"
            case .cannotMap(let p): return "cannot mmap \(p)"
            case .badHeader(let p): return "bad safetensors header in \(p)"
            case .truncated(let p, let needs, let has):
                return "\(p) is truncated: header describes \(needs) bytes, file has \(has)"
            }
        }
    }

    init(url: URL) throws {
        self.url = url

        descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw Failure.cannotOpen(url.lastPathComponent) }

        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 16 else {
            close(descriptor)
            throw Failure.cannotOpen(url.lastPathComponent)
        }
        mappedSize = Int(status.st_size)

        guard let mapped = mmap(nil, mappedSize, PROT_READ, MAP_SHARED, descriptor, 0),
              mapped != MAP_FAILED
        else {
            close(descriptor)
            throw Failure.cannotMap(url.lastPathComponent)
        }
        base = UnsafeRawPointer(mapped)

        // Read advice is applied at the call site because access patterns vary by tensor.

        let headerLength = Int(base.loadUnaligned(as: UInt64.self))
        guard headerLength > 0, headerLength + 8 <= mappedSize else {
            munmap(mapped, mappedSize)
            close(descriptor)
            throw Failure.badHeader(url.lastPathComponent)
        }

        let payloadBase = 8 + headerLength
        let headerData = Data(bytes: base.advanced(by: 8), count: headerLength)

        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            munmap(mapped, mappedSize)
            close(descriptor)
            throw Failure.badHeader(url.lastPathComponent)
        }

        for (name, value) in json {
            guard name != "__metadata__",
                  let fields = value as? [String: Any],
                  let dtype = fields["dtype"] as? String,
                  let shape = fields["shape"] as? [Int],
                  let offsets = fields["data_offsets"] as? [Int],
                  offsets.count == 2
            else { continue }

            entries[name] = Entry(
                dtype: dtype,
                shape: shape,
                offset: payloadBase + offsets[0],
                byteCount: offsets[1] - offsets[0]
            )
        }

        // Does the header describe more bytes than the file actually holds?
        //
        // The device-side size check in the transfer script compares against what
        // `devicectl` reports, which is three significant figures — enough to catch a
        // file that arrived at 133 MB of 432, useless for catching one that is short by
        // a few megabytes. This catches that case, cheaply, at the only point where it
        // can still be reported as a load failure rather than as quiet nonsense several
        // layers into inference.
        if let end = entries.values.map({ $0.offset + $0.byteCount }).max(), end > mappedSize {
            munmap(mapped, mappedSize)
            close(descriptor)
            throw Failure.truncated(url.lastPathComponent, needs: end, has: mappedSize)
        }
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), mappedSize)
        close(descriptor)
    }

    /// Raw pointer to a whole tensor. Does not touch the pages.
    func pointer(to name: String) -> UnsafeRawPointer? {
        guard let entry = entries[name] else { return nil }
        return base.advanced(by: entry.offset)
    }

    /// Raw pointer to one slice along the leading axis of a stacked tensor.
    ///
    /// Expert weights are stored as `[num_experts, out, in]`, so expert *e* is one
    /// contiguous run — `rowBytes` apart from its neighbour. This single line is why
    /// streaming is possible at all: pulling one expert is an offset, not a load.
    func pointer(to name: String, slice index: Int, of leadingDimension: Int) -> UnsafeRawPointer? {
        guard let entry = entries[name], leadingDimension > 0 else { return nil }
        let rowBytes = entry.byteCount / leadingDimension
        guard index >= 0, index < leadingDimension else { return nil }
        return base.advanced(by: entry.offset + index * rowBytes)
    }

    func sliceByteCount(of name: String, leadingDimension: Int) -> Int? {
        guard let entry = entries[name], leadingDimension > 0 else { return nil }
        return entry.byteCount / leadingDimension
    }

    /// Reset the kernel's read-ahead policy for the whole mapping.
    func advise(_ advice: Int32) {
        madvise(UnsafeMutableRawPointer(mutating: base), mappedSize, advice)
    }

    /// Ask the kernel to start pulling one expert's slice in.
    ///
    /// `MADV_WILLNEED` is advice, not a read: it returns once the I/O is queued rather
    /// than once the pages are there. Issued immediately before the copy it can only
    /// overlap within the slice; issued a step ahead — which is what pregate makes
    /// possible — it has a whole layer of compute to land in.
    func prefetch(_ name: String, slice index: Int, of leadingDimension: Int) {
        guard let entry = entries[name], leadingDimension > 0,
              index >= 0, index < leadingDimension
        else { return }
        let rowBytes = entry.byteCount / leadingDimension
        madvise(
            UnsafeMutableRawPointer(mutating: base).advanced(by: entry.offset + index * rowBytes),
            rowBytes, MADV_WILLNEED)
    }

    /// Ask the kernel to bring one whole tensor in.
    ///
    /// Request that the kernel warm the tensor's clean, file-backed pages.
    /// Returns the number of bytes requested.
    @discardableResult
    func warm(_ name: String) -> Int {
        guard let entry = entries[name] else { return 0 }
        madvise(UnsafeMutableRawPointer(mutating: base).advanced(by: entry.offset),
                entry.byteCount, MADV_WILLNEED)
        return entry.byteCount
    }
}

/// The four shards plus the index that says which shard holds which tensor.
final class SafetensorsBundle {

    private var shards: [String: SafetensorsShard] = [:]
    private var shardForTensor: [String: String] = [:]

    let directory: URL

    init(directory: URL) throws {
        self.directory = directory

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let indexData = try Data(contentsOf: indexURL)
        guard let root = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let map = root["weight_map"] as? [String: String]
        else {
            throw SafetensorsShard.Failure.badHeader("model.safetensors.index.json")
        }
        shardForTensor = map

        for file in Set(map.values) {
            shards[file] = try SafetensorsShard(url: directory.appendingPathComponent(file))
        }
    }

    var mappedShardCount: Int { shards.count }

    /// Every tensor name in the index, and which shard holds it.
    var index: [String: String] { shardForTensor }

    func entry(_ name: String) -> SafetensorsShard.Entry? {
        guard let file = shardForTensor[name] else { return nil }
        return shards[file]?.entries[name]
    }

    func pointer(to name: String) -> UnsafeRawPointer? {
        guard let file = shardForTensor[name] else { return nil }
        return shards[file]?.pointer(to: name)
    }

    func pointer(to name: String, slice index: Int, of leadingDimension: Int) -> UnsafeRawPointer? {
        guard let file = shardForTensor[name] else { return nil }
        return shards[file]?.pointer(to: name, slice: index, of: leadingDimension)
    }

    func sliceByteCount(of name: String, leadingDimension: Int) -> Int? {
        guard let file = shardForTensor[name] else { return nil }
        return shards[file]?.sliceByteCount(of: name, leadingDimension: leadingDimension)
    }

    func advise(_ advice: Int32) {
        for shard in shards.values { shard.advise(advice) }
    }

    /// Bring in every tensor the decode path reads on **every** step.
    ///
    /// Deliberately not "the whole mapping": these shards hold the 256 routed experts per
    /// layer as well, nineteen gigabytes of data that only two-of-256 of is touched per
    /// layer per token. Asking for that would evict everything worth having.
    ///
    /// Two exclusions, both by measurement rather than taste:
    ///
    /// - `switch_mlp` — the routed experts, streamed on demand and prefetched from their
    ///   own repacked files by their actual routing.
    /// - `embed_tokens` — 273 MiB of table from which `rows(_:_:)` copies one row per
    ///   token. Warming it would be 273 MiB of page cache for 1,152 bytes of need.
    ///
    /// What remains is what `weight(_:)` copies every step: 20.4 MiB per layer of
    /// attention and shared-expert projections, plus `lm_head`.
    @discardableResult
    func warmResidentWeights() -> Int {
        var total = 0
        for (name, file) in shardForTensor
        where !name.contains("switch_mlp") && !name.contains("embed_tokens") {
            total += shards[file]?.warm(name) ?? 0
        }
        return total
    }

    func prefetch(_ name: String, slice index: Int, of leadingDimension: Int) {
        guard let file = shardForTensor[name] else { return }
        shards[file]?.prefetch(name, slice: index, of: leadingDimension)
    }
}
