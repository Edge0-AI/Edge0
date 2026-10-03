import Foundation
import MLX

/// Trained routing heads that predict expert selections for upcoming decode steps.
///
/// During decode, these heads replace the router for their configured consumer layers;
/// the model's real gate is not evaluated for those layers. The heads and their LoRA
/// adapter are a trained pair and must be used together. Layers without a prediction use
/// the normal router.
///
/// A head on layer N predicts routing for layer N+1 on the next token. This layer and
/// token offset allows expert reads to be prefetched before the corresponding forward
/// pass. Features contain the current hidden state and one-hot encodings of the current
/// and previous executed routes.
final class PregateHeads: @unchecked Sendable {

    enum Failure: LocalizedError {
        case tensorMissing(String)
        case wrongShape(String, expected: [Int], found: [Int])

        var errorDescription: String? {
            switch self {
            case .tensorMissing(let name): return "pregate file has no \(name)"
            case .wrongShape(let name, let expected, let found):
                return "pregate \(name) is \(found), expected \(expected)"
            }
        }
    }

    /// Layers whose head runs. Each predicts for `owner + 1`.
    let owners: [Int]
    /// Which layers consume a prediction, in the order their heads appear.
    let consumers: [Int]
    let hidden: Int
    let experts: Int

    /// Keep the file mapped and materialize arrays only during a pregate forward pass.
    private let shard: SafetensorsShard
    private let middle: Int

    /// Where the file lives, beside the weights. Optional by construction: a build
    /// without it, the normal router is used.
    static func fileURL(in directory: URL, round6: Bool = false) -> URL {
        if round6 {
            return directory.appendingPathComponent("pregate-round6.safetensors")
        }
        let official = directory.appendingPathComponent("prerouter-stacked.safetensors")
        if FileManager.default.fileExists(atPath: official.path) { return official }
        return directory.appendingPathComponent("pregate-round6.safetensors")
    }

    init(url: URL, hidden: Int, experts: Int) throws {
        let shard = try SafetensorsShard(url: url)
        self.shard = shard
        self.hidden = hidden
        self.experts = experts
        let inputWidth = hidden + 2 * experts

        // `nil` in the expected shape means "whatever the file says" — the hidden width
        // of the heads is the file's to decide, and everything else is pinned by the
        // model it has to plug into.
        func validate(_ name: String, expecting: [Int?]) throws -> SafetensorsShard.Entry {
            guard let entry = shard.entries[name] else {
                throw Failure.tensorMissing(name)
            }
            let matches = entry.shape.count == expecting.count
                && zip(entry.shape, expecting).allSatisfy { $1 == nil || $0 == $1 }
            guard matches else {
                throw Failure.wrongShape(
                    name, expected: expecting.map { $0 ?? -1 }, found: entry.shape)
            }
            guard entry.dtype == "F16" else {
                throw Failure.wrongShape(name, expected: expecting.map { $0 ?? -1 },
                                         found: entry.shape)
            }
            return entry
        }

        guard let ownersEntry = shard.entries["pregate.owners"],
              let ownersBase = shard.pointer(to: "pregate.owners")
        else { throw Failure.tensorMissing("pregate.owners") }
        let count = ownersEntry.shape.first ?? 0
        owners = (0 ..< count).map {
            Int(ownersBase.load(fromByteOffset: $0 * 4, as: Int32.self))
        }
        consumers = owners.map { $0 + 1 }

        let fc1 = try validate("pregate.fc1", expecting: [count, inputWidth, nil])
        middle = fc1.shape[2]
        _ = try validate("pregate.fc2", expecting: [count, middle, experts])
        _ = try validate("pregate.linear_init", expecting: [count, inputWidth, experts])
    }

    private func array(_ name: String) -> MLXArray {
        guard let entry = shard.entries[name], let base = shard.pointer(to: name) else {
            preconditionFailure("validated pregate tensor disappeared: \(name)")
        }
        return MLXArray(
            UnsafeRawBufferPointer(start: base, count: entry.byteCount),
            entry.shape, type: Float16.self)
    }

    /// One-hot rows for every owner layer at once, built on the CPU.
    ///
    /// **Deliberately not `identity[indices].sum(axis: 0)` per layer.** That is the
    /// obvious phrasing and it was measured: thirty-three lazy MLX nodes, each pinning
    /// its inputs until the step boundary evaluated them, contributed to a footprint
    /// that rose from ~600 MiB to 1535 MiB. On this device that is not merely untidy —
    /// the expert blocks live in *clean file-backed* pages, so anything the process
    /// holds dirty is taken out of the page cache those reads depend on. `read` doubled
    /// to 134 ms and the weight copy rate halved, and together they ate a 70 ms saving.
    ///
    /// The routing is already on the CPU — `moe` has just called `asArray` on it — so
    /// the whole `[n, experts]` block is a memset and `n * k` stores, then one array.
    /// No graph, nothing retained, nothing to evaluate.
    func oneHotRows(_ perLayer: [[Int32]]) -> MLXArray {
        var flat = [Float16](repeating: 0, count: perLayer.count * experts)
        for (row, chosen) in perLayer.enumerated() {
            for expert in chosen where expert >= 0 && Int(expert) < experts {
                // Accumulate so duplicate IDs retain their multiplicity.
                flat[row * experts + Int(expert)] += 1
            }
        }
        return MLXArray(flat, [perLayer.count, experts])
    }

    /// Run every head as one batch.
    ///
    /// - Parameters:
    ///   - inputs: `[n, hidden]`, the owner layers' MoE inputs this token, in owner order.
    ///   - executed: `[n, experts]`, the one-hot rows those layers actually ran, from
    ///     `oneHotRows`.
    ///   - previous: `[n, experts]`, the previous token's routing or a zero row.
    /// - Returns: `[n, experts]` float32 logits for the consuming layers.
    func logits(inputs: MLXArray, executed: MLXArray, previous: MLXArray) -> MLXArray {
        let fc1 = array("pregate.fc1")
        let fc2 = array("pregate.fc2")
        let linearInit = array("pregate.linear_init")
        let features = concatenated(
            [inputs.asType(.float16), executed, previous], axis: -1)
            .expandedDimensions(axis: 1)                          // [n, 1, in]
        let inner = matmul(features, fc1)                         // [n, 1, mid]
        // Exact erf GELU, as trained (torch's default). `geluApproximate` would be a
        // different function evaluated on 33 × 512 values a step, and there is no
        // reference to check it against; the tanh form is not what these weights saw.
        let activated = 0.5 * inner * (1 + erf(inner / Float(2).squareRoot()))
        let out = matmul(features, linearInit) + matmul(activated, fc2)
        return out.squeezed(axis: 1).asType(.float32)             // [n, experts]
    }

    /// What has to survive from one decode step to the next.
    ///
    /// Per-conversation routing history used by the trained heads.
    final class State {
        /// Owner layer to its MoE input for the current token. Stored as an MLX array so
        /// conversion to CPU does not introduce an additional synchronization point.
        var moeInput: [Int: MLXArray] = [:]
        /// Owner layer → the experts it actually ran this token.
        var executed: [Int: [Int32]] = [:]
        /// The previous token's executed experts, or absent on the first token.
        var previous: [Int: [Int32]] = [:]
        /// Consuming layer → the experts it will use next step, and their weights.
        ///
        /// Both plain Swift. They cross a step boundary, and an `MLXArray` that does
        /// that carries whatever graph produced it into the next step.
        var predicted: [Int: [Int]] = [:]
        var scores: [Int: [Float]] = [:]

        /// Advance routing history at the end of a decode step.
        func advance() {
            previous = executed
            executed = [:]
            moeInput = [:]
        }

        func reset() {
            moeInput = [:]; executed = [:]; previous = [:]
            predicted = [:]; scores = [:]
        }
    }

    // MARK: - Optional numerical self-check

    /// Compare the heads' output against an optional fixture containing inputs, expected
    /// logits, and top-2 expert IDs. Returns `nil` when the fixture is unavailable.
    func selfCheck(fixture url: URL) -> String? {
        guard let shard = try? SafetensorsShard(url: url),
              let feats = shard.entries["feats"], let featsBase = shard.pointer(to: "feats"),
              let ref = shard.entries["logits"], let refBase = shard.pointer(to: "logits"),
              let top = shard.entries["top2"], let topBase = shard.pointer(to: "top2")
        else { return nil }

        let features = MLXArray(
            UnsafeRawBufferPointer(start: featsBase, count: feats.byteCount),
            feats.shape, type: Float16.self)
        let expected = MLXArray(
            UnsafeRawBufferPointer(start: refBase, count: ref.byteCount),
            ref.shape, type: Float32.self)
        let expectedTop = MLXArray(
            UnsafeRawBufferPointer(start: topBase, count: top.byteCount),
            top.shape, type: Int32.self)

        // The fixture holds the concatenated 2560-wide features; split them back so the
        // public entry point is what gets exercised, not a shortcut around it.
        let n = feats.shape[0]
        let inputs = features[0..., 0 ..< hidden]
        let executed = features[0..., hidden ..< (hidden + experts)]
        let previous = features[0..., (hidden + experts)...]
        let got = logits(inputs: inputs, executed: executed, previous: previous)

        let difference = abs(got - expected)
        let gotTop = argSort(-got, axis: -1)[0..., 0 ..< 2]
        let agree = (gotTop .== expectedTop).all(axis: -1).sum()
        eval(difference, agree)
        return String(
            format: "[pregate] self-check: max|Δ|=%.4f mean|Δ|=%.5f top-2 agree %d/%d",
            difference.max().item(Float.self), difference.mean().item(Float.self),
            agree.item(Int32.self), Int32(n))
    }
}
