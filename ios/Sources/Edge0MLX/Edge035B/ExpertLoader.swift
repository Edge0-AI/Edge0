import Foundation
import MLX

/// Loads routed experts from the per-layer files and builds the stacked tensors used by
/// `gatherQuantizedMM`. Reads for the selected experts are issued concurrently.
final class ExpertLoader: @unchecked Sendable {

    enum Failure: LocalizedError {
        case noLayers(URL)
        case unknownLayer(Int, available: Int)
        case expertOutOfRange(Int)
        case wrongSlotCount(asked: Int, configured: Int)
        case readFailed(layer: Int, expert: Int)

        var errorDescription: String? {
            switch self {
            case .noLayers(let url):
                return "no intact layer files in \(url.lastPathComponent)"
            case .unknownLayer(let layer, let available):
                return "layer \(layer) is not loaded (\(available) layers available)"
            case .expertOutOfRange(let expert):
                return "expert \(expert) outside 0..<\(RepackedExperts.expertsPerLayer)"
            case .wrongSlotCount(let asked, let configured):
                return "asked for \(asked) experts, loader is built for \(configured)"
            case .readFailed(let layer, let expert):
                return "read failed for layer \(layer) expert \(expert)"
            }
        }
    }

    /// Where each tensor sits in the stacked pool handed to `gatherQuantizedMM`:
    /// weight, scales, biases for gate, then up, then down.
    static let gate = 0, up = 3, down = 6

    private let experts: RepackedExperts
    private let slots: Int
    private var buffers: [UnsafeMutableRawPointer] = []


    private let shapes: [(weight: [Int], scale: [Int])]

    /// Set by the model so reads and array construction land in the same accounting as
    /// the router barrier. Optional so the loader stays usable without it.
    var profile: StepProfile?

    var availableLayers: [Int] { experts.availableLayers }
    var rejectedFiles: [(layer: Int, bytes: Int)] { experts.rejected }

    init(directory: URL, layers: Int, slots: Int, hidden: Int,
         moeIntermediate: Int, groupSize: Int) throws
    {
        self.experts = RepackedExperts(directory: directory, expectedLayers: layers)
        guard !experts.availableLayers.isEmpty else { throw Failure.noLayers(directory) }

        self.slots = slots

        self.shapes = [
            (weight: [moeIntermediate, hidden / 8], scale: [moeIntermediate, hidden / groupSize]),
            (weight: [moeIntermediate, hidden / 8], scale: [moeIntermediate, hidden / groupSize]),
            (weight: [hidden, moeIntermediate / 8], scale: [hidden, moeIntermediate / groupSize]),
        ]
    }

    /// Nine tensors: gate weight/scales/biases, then up, then down.
    ///
    /// Throws rather than returning something plausible. A silently wrong expert
    /// produces fluent nonsense several layers later, which is far more expensive to
    /// find than a thrown error here.
    deinit {
        buffers.forEach { $0.deallocate() }
    }

    /// Ask the kernel to prefetch the selected expert blocks into the file cache.
    func prefetch(layer: Int, experts chosen: [Int]) {
        for expert in chosen where expert >= 0 && expert < RepackedExperts.expertsPerLayer {
            experts.prefetch(layer: layer, expert: expert)
        }
    }

    /// Load **1 to `slots`** distinct experts.
    ///
    /// Was fixed at exactly `slots`, which is what a single token needs at K=2. A batch
    /// of T tokens routes to the *union* of their choices — anywhere from K (all tokens
    /// agree) to T*K (none do) — so `slots` became the ceiling rather than the count.
    func load(layer: Int, experts chosen: [Int]) throws -> [MLXArray] {
        guard !chosen.isEmpty, chosen.count <= slots else {
            throw Failure.wrongSlotCount(asked: chosen.count, configured: slots)
        }
        let count = chosen.count
        guard experts.availableLayers.contains(layer) else {
            throw Failure.unknownLayer(layer, available: experts.availableLayers.count)
        }
        for expert in chosen where expert < 0 || expert >= RepackedExperts.expertsPerLayer {
            throw Failure.expertOutOfRange(expert)
        }

        // A larger prefill batch needs a larger *possible* expert union, but reserving
        // T*K blocks eagerly makes the worst case permanent. Grow to the union we have
        // actually observed and reuse that high-water mark on later layers/turns.
        while buffers.count < count {
            buffers.append(UnsafeMutableRawPointer.allocate(
                byteCount: RepackedExperts.blockBytes,
                alignment: RepackedExperts.pageSize))
        }

        // Read each expert block into reusable aligned buffers, then stack its tensors.
        let sources = Array(buffers.prefix(count))
        let missing = Array(0 ..< count)

        var failures = [Bool](repeating: false, count: missing.count)
        let buffers = self.buffers
        let experts = self.experts
        let read = {
            failures.withUnsafeMutableBufferPointer { flags in
                let flagged = flags
                let slots = missing
                DispatchQueue.concurrentPerform(iterations: slots.count) { index in
                    let slot = slots[index]
                    flagged[index] = !experts.read(
                        layer: layer, expert: chosen[slot], into: buffers[slot])
                }
            }
        }
        if let profile { profile.measure(.expertRead, read) } else { read() }
        profile?.recordRead(bytes: missing.count * RepackedExperts.blockBytes,
                            issued: missing.count)
        if let index = failures.firstIndex(of: true) {
            throw Failure.readFailed(layer: layer, expert: chosen[missing[index]])
        }

        // `stack` is where `MLXArray(raw:)` runs, and that is `malloc` + `std::copy`
        // in mlx-c — eager, so timing it needs no forced evaluation.
        var pool: [MLXArray] = []
        pool.reserveCapacity(9)
        let shapes = self.shapes
        let build: () -> Void = {
        for (projection, parts) in [
            (0, (RepackedExperts.Part.gateWeight, RepackedExperts.Part.gateScales,
                 RepackedExperts.Part.gateBiases)),
            (1, (.upWeight, .upScales, .upBiases)),
            (2, (.downWeight, .downScales, .downBiases)),
        ] {
            let shape = shapes[projection]
            pool.append(self.stack(parts.0, shape.weight, bf16: false, from: sources))
            pool.append(self.stack(parts.1, shape.scale, bf16: true, from: sources))
            pool.append(self.stack(parts.2, shape.scale, bf16: true, from: sources))
        }
        }
        if let profile { profile.measure(.expertStack, build) } else { build() }
        return pool
    }

    /// Every expert in the layer, stacked as `[256, …]` in expert-id order.
    ///
    /// The file is one sequential 432 MiB read. Parts are then packed into the
    /// nine tensors `gatherQuantizedMM` already understands, so a prefill chunk
    /// can gather by expert id without a per-expert `pread` or a CPU copy of
    /// the index list. The file buffer is released before return; MLX has
    /// copied the nine tensors. Callers must `eval` and drop those tensors
    /// before the next layer, or forty of them become 17 GB.
    func loadFullLayer(layer: Int) throws -> [MLXArray] {
        guard experts.availableLayers.contains(layer) else {
            throw Failure.unknownLayer(layer, available: experts.availableLayers.count)
        }
        let file = UnsafeMutableRawPointer.allocate(
            byteCount: RepackedExperts.layerBytes, alignment: RepackedExperts.pageSize)
        defer { file.deallocate() }

        let read: () -> Bool = { self.experts.readLayer(layer, into: file) }
        let ok = profile?.measure(.expertRead, read) ?? read()
        guard ok else { throw Failure.readFailed(layer: layer, expert: -1) }
        profile?.recordRead(bytes: RepackedExperts.layerBytes, issued: 1)

        let shapes = self.shapes
        let parts: [(RepackedExperts.Part, RepackedExperts.Part, RepackedExperts.Part)] = [
            (.gateWeight, .gateScales, .gateBiases),
            (.upWeight, .upScales, .upBiases),
            (.downWeight, .downScales, .downBiases),
        ]
        var pool: [MLXArray] = []
        pool.reserveCapacity(9)
        let build: () -> Void = {
            for (projection, triple) in parts.enumerated() {
                let shape = shapes[projection]
                for (part, bf16, partShape) in [
                    (triple.0, false, shape.weight),
                    (triple.1, true, shape.scale),
                    (triple.2, true, shape.scale),
                ] {
                    pool.append(self.stackedPart(
                        part, partShape, bf16: bf16, file: file))
                }
            }
        }
        if let profile { profile.measure(.expertStack, build) } else { build() }
        return pool
    }

    /// Expert id `e` lives at byte `e * block` in the layer file. Gather one
    /// part from all 256 blocks into a contiguous `[256, …]` tensor.
    private func stackedPart(_ part: RepackedExperts.Part, _ shape: [Int],
                             bf16: Bool, file: UnsafeMutableRawPointer) -> MLXArray {
        let count = RepackedExperts.expertsPerLayer
        let bytes = part.byteCount
        let packed = UnsafeMutableRawPointer.allocate(
            byteCount: count * bytes, alignment: 64)
        defer { packed.deallocate() }
        for expert in 0 ..< count {
            packed.advanced(by: expert * bytes).copyMemory(
                from: file.advanced(by: expert * RepackedExperts.blockBytes + part.offset),
                byteCount: bytes)
        }
        let raw = UnsafeRawBufferPointer(start: packed, count: count * bytes)
        let stackedShape = [count] + shape
        return bf16
            ? MLXArray(raw, stackedShape, type: UInt16.self).view(dtype: .bfloat16)
            : MLXArray(raw, stackedShape, type: UInt32.self)
    }

    /// One part across all K slots, stacked along a new leading axis.
    ///
    /// bf16 is reached by reinterpreting the bits, never by converting values. An early
    /// version widened each half in a Swift loop — roughly 200,000 iterations per layer
    /// — and that single artifact produced the spike's original 0.5 tok/s figure.
    private func stack(_ part: RepackedExperts.Part, _ shape: [Int], bf16: Bool,
                       from sources: [UnsafeMutableRawPointer]) -> MLXArray
    {
        // Sources are the staging buffers `load` has just filled. `MLXArray(raw:)`
        // copies eagerly, so they are free to be overwritten the moment this returns.
        let slices = sources.map { buffer -> MLXArray in
            let raw = UnsafeRawBufferPointer(
                start: buffer.advanced(by: part.offset), count: part.byteCount)
            return bf16
                ? MLXArray(raw, shape, type: UInt16.self).view(dtype: .bfloat16)
                : MLXArray(raw, shape, type: UInt32.self)
        }
        return slices.count == 1
            ? slices[0].expandedDimensions(axis: 0)
            : MLX.stacked(slices, axis: 0)
    }
}
