import Foundation

/// The repacked layout: one file per layer, one expert per contiguous block.
///
/// Each block holds an expert's nine tensors back to back, 1,769,472 bytes — exactly
/// 108 pages of 16 KB — so with data starting at offset 0 of its layer's file, **every
/// block is page-aligned**. Reading one expert is one sequential 1.69 MB read instead
/// of the nine scattered reads the original safetensors layout forces.
///
/// Each file is checked against the expected length before it is opened so truncated
/// or incomplete files cannot be used as model weights.
final class RepackedExperts {

    /// Byte layout inside one expert's block, in the order the packer wrote them.
    enum Part: Int, CaseIterable {
        case gateWeight, gateScales, gateBiases
        case upWeight, upScales, upBiases
        case downWeight, downScales, downBiases

        var byteCount: Int {
            switch self {
            case .gateWeight, .upWeight, .downWeight: return 524_288
            default: return 32_768
            }
        }

        var offset: Int {
            Part.allCases.prefix(rawValue).reduce(0) { $0 + $1.byteCount }
        }
    }

    static let blockBytes = 1_769_472
    static let expertsPerLayer = 256
    static let layerBytes = blockBytes * expertsPerLayer   // 452,984,832
    static let pageSize = 16384                            // blockBytes is 108 of these

    private struct Opened {
        let size: Int
        let descriptor: Int32
    }

    // Keep descriptors, not 40 whole-file mappings. Mapping every 432 MiB layer
    // consumes ~17 GiB of virtual address space and fails after 11 layers when a
    // development profile lacks Extended Virtual Addressing. The hot path already
    // uses pread, so those mappings were only an optional prefetch mechanism.
    private var layers: [Int: Opened] = [:]

    /// Layers whose file was present but the wrong length, with the length found.
    private(set) var rejected: [(layer: Int, bytes: Int)] = []
    private(set) var missing: [Int] = []

    var availableLayers: [Int] { layers.keys.sorted() }

    init(directory: URL, expectedLayers: Int) {
        for layer in 0 ..< expectedLayers {
            let url = directory.appendingPathComponent(String(format: "experts-L%02d.bin", layer))

            guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let size = (attributes[.size] as? NSNumber)?.intValue
            else {
                missing.append(layer)
                continue
            }
            guard size == Self.layerBytes else {
                rejected.append((layer, size))
                continue
            }

            let descriptor = open(url.path, O_RDONLY)
            guard descriptor >= 0 else {
                missing.append(layer)
                continue
            }
            layers[layer] = Opened(size: size, descriptor: descriptor)
        }
    }

    deinit {
        for opened in layers.values {
            close(opened.descriptor)
        }
    }

    /// Ask the kernel for an expert's whole block in one go.
    func prefetch(layer: Int, expert: Int) {
        guard let opened = layers[layer], expert >= 0, expert < Self.expertsPerLayer
        else { return }

        // `F_RDADVISE` is the descriptor equivalent of the old mmap + MADV_WILLNEED
        // path: it starts asynchronous readahead without reserving 17 GiB of virtual
        // address space. The eventual `pread` remains the source of truth, so ignored
        // or failed advice can affect only latency, never model output.
        var advice = radvisory(
            ra_offset: off_t(expert * Self.blockBytes),
            ra_count: Int32(Self.blockBytes))
        _ = withUnsafeMutablePointer(to: &advice) {
            fcntl(opened.descriptor, F_RDADVISE, UnsafeMutableRawPointer($0))
        }
    }

    /// Remove files that are present but have the wrong length, returning any failures.
    static func removeTruncated(in directory: URL, expectedLayers: Int)
        -> (removed: [(Int, Int)], failed: [(Int, String)])
    {
        var removed: [(Int, Int)] = []
        var failed: [(Int, String)] = []
        for layer in 0 ..< expectedLayers {
            let url = directory.appendingPathComponent(String(format: "experts-L%02d.bin", layer))
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let size = (attributes[.size] as? NSNumber)?.intValue,
                  size != layerBytes
            else { continue }
            do {
                try FileManager.default.removeItem(at: url)
                removed.append((layer, size))
            } catch {
                failed.append((layer, error.localizedDescription))
            }
        }
        return (removed, failed)
    }

    /// Read one expert's whole block with a single `pread`.
    ///
    /// The mmap path reaches the same bytes through page faults, which the kernel
    /// services in its own units; this asks for all 1,769,472 bytes in one call and
    /// lets it issue one large sequential read. Worth measuring against the fault path
    /// now that a block is contiguous — 768 MB/s is still only about a third of what
    /// the device can do.
    /// One sequential read of the whole layer file (432 MiB, 256 expert blocks).
    func readLayer(_ layer: Int, into buffer: UnsafeMutableRawPointer) -> Bool {
        guard let opened = layers[layer] else { return false }
        var moved = 0
        while moved < Self.layerBytes {
            let got = pread(
                opened.descriptor,
                buffer.advanced(by: moved),
                Self.layerBytes - moved,
                off_t(moved))
            if got <= 0 { return false }
            moved += got
        }
        return true
    }

    func read(layer: Int, expert: Int, into buffer: UnsafeMutableRawPointer) -> Bool {
        guard let opened = layers[layer], expert >= 0, expert < Self.expertsPerLayer
        else { return false }

        var moved = 0
        while moved < Self.blockBytes {
            let got = pread(
                opened.descriptor,
                buffer.advanced(by: moved),
                Self.blockBytes - moved,
                off_t(expert * Self.blockBytes + moved))
            if got <= 0 { return false }
            moved += got
        }
        return true
    }
}
