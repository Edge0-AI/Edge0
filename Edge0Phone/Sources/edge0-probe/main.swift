import Edge0Core
import Foundation

func usage() -> Never {
    print("usage: edge0-probe /path/to/Edge0-8B-A1B-preview")
    exit(2)
}

guard CommandLine.arguments.count == 2 else { usage() }
let dir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let configURL = dir.appendingPathComponent("config.json")
let modelURL = dir.appendingPathComponent("model.safetensors")

do {
    let config = try BailingConfiguration.load(from: configURL)
    let store = try ExpertTensorStore(modelURL: modelURL, expertCount: config.numExperts)

    print("Edge0Phone checkpoint probe")
    print("layers: \(config.hiddenLayers)")
    print("experts: \(config.numExperts), active/token: \(config.expertsPerToken)")
    print("hidden: \(config.hiddenSize), moe hidden: \(config.moeIntermediateSize)")
    print("model file: \(store.mappedFile.count) bytes")
    print("indexed tensors: \(store.index.tensors.count)")

    // Layer 0 is dense; layer 1 is the first sparse MoE layer.
    for projection in ExpertProjection.allCases {
        for part in QuantPart.allCases {
            let slice = try store.slice(layer: 1, expert: 0, projection: projection, part: part)
            print("L1 E0 \(projection.rawValue).\(part.rawValue): \(slice.descriptor.shape) \(slice.descriptor.dtype.rawValue) \(slice.descriptor.byteCount)B")
        }
    }

    print("PASS: parsed and addressed a single expert without loading the checkpoint into RAM.")
} catch {
    print("error: \(error.localizedDescription)")
    exit(1)
}
