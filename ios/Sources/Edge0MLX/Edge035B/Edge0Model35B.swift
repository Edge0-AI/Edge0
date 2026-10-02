import Foundation
import MLX
import MLXFast

/// Streaming Edge0 35B model implementation used by the iOS chat engine.
///
/// This implementation is kept independent of the Python reference so parity checks can
/// compare separate code paths. It is not main-actor isolated because model steps perform
/// synchronous GPU and file I/O. The chat engine serializes access to one model instance.
final class Edge0Model35B: @unchecked Sendable {

    /// How many tokens may go through one forward pass.
    ///
    /// Sets the expert buffer ceiling: a batch of T tokens can in the worst case route to
    /// T*K distinct experts. The loader grows staging storage only to the union actually
    /// observed, so this ceiling no longer reserves the worst case up front.
    ///
    /// Storage is sized for the observed expert union rather than the worst-case T*K.
    /// Top-4 routing makes the batch size a tradeoff between repeated base-weight reads
    /// and the memory required for the active graph and expert union.
    /// Tokens per prefill forward. Decode ignores this.
    ///
    /// A full-layer read is about 8 s of flash for 40 layers, independent of
    /// how many tokens share it. 512 is the first size at which that fixed
    /// cost can land near 60 tok/s before GPU time; the Gated DeltaNet scan
    /// is what will jetsam first if 512 is too large.
    static let maximumBatch = 512
    /// Below this, prefill keeps the decode-style expert union. A full-layer
    /// read is 432 MiB × 40 and only pays for itself on a long chunk.
    static let fullLayerPrefillMinimum = 64

    struct Configuration {
        var layers = 40
        var hidden = 2048
        var moeIntermediate = 512
        var expertsPerLayer = 256
        // The public Edge0-35B-A3B-preview adapter and prerouter were trained with K=4.
        // This must stay aligned with the shipped artifacts: changing K changes both
        // the MoE output and the routing history consumed by the next-step heads.
        var activeExperts = 4
        /// Round-6 pregate + LoRA, trained at K=2. The two files are a pair.
        var round6 = false
        var groupSize = 64
        var bits = 4
        var fullAttentionInterval = 4
        var vocabulary = 248_320

        /// When true, keep every non-expert resident tensor after first materialisation.
        /// The default retains only selected tensors to limit active memory use.
        var retainAllResidentWeights = false

        /// Source used to prefetch expert reads for the next decode step.
        var routingGuess: RoutingGuess = .trainedHeads

        /// Number of recent routing steps used by `.recentRouting`.
        var routingLookback = 1

        enum RoutingGuess {
            /// Do not prefetch expert reads.
            case none
            /// Prefetch the experts selected by each layer on the previous step.
            case recentRouting
            /// The trained pregate heads, when their file is present. Predicts the
            /// *next* layer a *token* ahead, so it is the only source that can cover a
            /// layer before that layer has run — and it replaces the router to do it.
            case trainedHeads
        }

        /// Identifies this configuration in caches whose contents depend on it.
        var fingerprint: String { "full" }

        /// `(index + 1) % interval == 0` are full attention; the rest are linear.
        func isLinear(_ layer: Int) -> Bool {
            (layer + 1) % fullAttentionInterval != 0
        }
    }

    enum Failure: LocalizedError {
        case weightsMissing(URL)
        case tensorMissing(String)
        case layerFailed(Int)

        var errorDescription: String? {
            switch self {
            case .weightsMissing(let url):
                return "no model at \(url.lastPathComponent)"
            case .tensorMissing(let name):
                return "missing tensor \(name)"
            case .layerFailed(let layer):
                return "layer \(layer) could not run"
            }
        }
    }

    /// State carried between decode steps.
    ///
    /// Two kinds, and **they are not interchangeable**. Linear layers hold a fixed-size
    /// `[1, heads, dim, dim]` float32 matrix updated in place: it does not grow with
    /// context and cannot be pruned. Full-attention layers hold a KV cache that does
    /// grow. Any context policy that drops old turns shortens the second and leaves the
    /// first untouched — a model that appears to forget while still carrying everything.
    final class State {
        var convolution: [Int: MLXArray] = [:]
        var recurrent: [Int: MLXArray] = [:]
        var keys: [Int: MLXArray] = [:]
        var values: [Int: MLXArray] = [:]
        var offset = 0

        /// Routing carried between decode steps. Per-conversation, not per-model: two
        /// conversations share one set of heads but must never share the routing
        /// history that feeds them.
        let pregate = PregateHeads.State()

        /// Layer → the experts it ran, most recent step first, capped at the lookback.
        /// Previous-step router output, used by `.recentRouting`.
        var recentRouting: [Int: [[Int]]] = [:]

        /// Layer → what was asked for on its behalf. Exists so the hit rate is a
        /// measurement of what was *issued* rather than a restatement of the guess:
        /// `recentRouting` is overwritten as each layer runs, so comparing against it
        /// would compare a layer's routing with its own.
        var prefetched: [Int: [Int]] = [:]

        var byteCount: Int {
            [convolution, recurrent, keys, values].reduce(0) { total, table in
                total + table.values.reduce(0) { $0 + $1.size * $1.itemSize }
            }
        }

        func reset() {
            convolution.removeAll(); recurrent.removeAll()
            keys.removeAll(); values.removeAll()
            offset = 0
            pregate.reset()
            recentRouting.removeAll(); prefetched.removeAll()
        }
    }

    let configuration: Configuration
    private let weights: SafetensorsBundle
    private let experts: ExpertLoader
    private let lora: SafetensorsShard?
    private let directory: URL
    private let coreMLHead: CoreMLVocabularyHead?
    private var coreMLHeadEnabled: Bool
    private var coreMLHeadValidated = false

    init(directory: URL, configuration: Configuration = Configuration()) throws {
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: index.path) else {
            throw Failure.weightsMissing(directory)
        }
        self.directory = directory
        self.configuration = configuration
        self.weights = try SafetensorsBundle(directory: directory)
        self.experts = try ExpertLoader(
            directory: directory, layers: configuration.layers,
            slots: Edge0Model35B.maximumBatch * configuration.activeExperts,
            hidden: configuration.hidden,
            moeIntermediate: configuration.moeIntermediate,
            groupSize: configuration.groupSize)
        self.experts.profile = self.profile
        let round6LoRA = directory.appendingPathComponent("lora-round6.safetensors")
        let officialLoRA = directory.appendingPathComponent("lora_edge0_35b.safetensors")
        let legacyLoRA = directory.appendingPathComponent("lora.safetensors")
        let loraURL: URL
        if configuration.round6 {
            loraURL = round6LoRA
        } else if FileManager.default.fileExists(atPath: officialLoRA.path) {
            loraURL = officialLoRA
        } else {
            loraURL = legacyLoRA
        }
        self.lora = try? SafetensorsShard(url: loraURL)
        if configuration.round6 {
            print("[round6] K=\(configuration.activeExperts) lora=\(self.lora == nil ? "missing" : "loaded")")
        }
        let compiledHead = directory.appendingPathComponent(
            CoreMLVocabularyHead.directoryName, isDirectory: true)
        if CommandLine.arguments.contains("--ane-35b"),
           FileManager.default.fileExists(atPath: compiledHead.path) {
            do {
                self.coreMLHead = try CoreMLVocabularyHead(directory: directory)
                self.coreMLHeadEnabled = true
                print("[coreml-head] loaded with computeUnits=cpuAndNeuralEngine")
            } catch {
                self.coreMLHead = nil
                self.coreMLHeadEnabled = false
                print("[coreml-head] unavailable: \(error)")
            }
        } else {
            self.coreMLHead = nil
            self.coreMLHeadEnabled = false
            print("[coreml-head] disabled or absent")
        }
        let free = (try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
        print("[weights] free=\(free.map { String($0 / 1_048_576) } ?? "?")MiB")

        // Pregate weights are optional. Report an invalid present file so a fallback to
        // the normal router is visible during diagnostics.
        let pregateURL = PregateHeads.fileURL(in: directory, round6: configuration.round6)
        if configuration.routingGuess == .trainedHeads,
           FileManager.default.fileExists(atPath: pregateURL.path) {
            do {
                let heads = try PregateHeads(
                    url: pregateURL, hidden: configuration.hidden,
                    experts: RepackedExperts.expertsPerLayer)
                self.pregate = heads
                print("[pregate] heads loaded: owners \(heads.owners.first ?? -1)..\(heads.owners.last ?? -1) n=\(heads.owners.count)")
                if let report = heads.selfCheck(
                    fixture: directory.appendingPathComponent("pregate-fixture.safetensors")) {
                    print(report)
                }
            } catch {
                print("[pregate] present but unusable: \(error)")
                self.pregate = nil
            }
        } else {
            self.pregate = nil
        }
    }

    /// The trained routing heads, when the file is present. See `PregateHeads`.
    private let pregate: PregateHeads?

    /// See `StepProfile` for why only the synchronous phases are timed.
    let profile = StepProfile()

    var loraLoaded: Bool { lora != nil }
    var availableLayers: Int { experts.availableLayers.count }

    // MARK: - Weights

    /// Selected tensors retained after first use to reduce repeated materialization.
    private var retainedWeights: [String: MLXArray] = [:]
    private var retainAttentionWeights = false

    /// First-turn prefill is latency-sensitive and sees every attention weight only
    /// once, so retaining them there adds pressure without reuse. Decode immediately
    /// reuses them; enable the cache at that boundary and keep it warm for later turns.
    func beginDecodeCaching() {
        retainAttentionWeights = true
    }

    /// Prepare selected attention weights and issue a discarded token before generation
    /// so initial Metal setup is outside the first reply's latency measurement.
    func prepareForFirstTurn() throws {
        beginDecodeCaching()
        let started = Date()
        var count = 0
        for name in weights.index.keys where
            name.hasPrefix("language_model.lm_head.")
            || name.contains(".linear_attn.")
            || name.contains(".self_attn.") {
            if weight(name) != nil { count += 1 }
        }
        print(String(
            format: "[weights] first-turn cache tensors=%d %.2fs",
            count, Date().timeIntervalSince(started)))

        let gpu = Date()
        let scratch = State()
        _ = try step(tokens: MLXArray([Int32(0)], [1, 1]), state: scratch)
        profile.reset()
        print(String(format: "[weights] gpu warm %.2fs", Date().timeIntervalSince(gpu)))
    }

    /// Materialize a tensor from the mapped checkpoint. Retain selected tensors according
    /// to the configured memory policy; other tensors are released after use.
    private func weight(_ name: String) -> MLXArray? {
        if let retained = retainedWeights[name] { return retained }
        guard let entry = weights.entry(name), let pointer = weights.pointer(to: name)
        else { return nil }
        let raw = UnsafeRawBufferPointer(start: pointer, count: entry.byteCount)
        profile.recordWeight(bytes: entry.byteCount)
        let made = profile.measure(.weightCopy) { () -> MLXArray? in
            switch entry.dtype {
            case "U32", "I32": return MLXArray(raw, entry.shape, type: UInt32.self)
            case "BF16":
                return MLXArray(raw, entry.shape, type: UInt16.self).view(dtype: .bfloat16)
            case "F32": return MLXArray(raw, entry.shape, type: Float.self)
            default: return nil
            }
        }
        let shouldRetain = configuration.retainAllResidentWeights
            || (!coreMLHeadEnabled && name.hasPrefix("language_model.lm_head."))
            || (retainAttentionWeights
                && (name.contains(".linear_attn.") || name.contains(".self_attn.")))
        if shouldRetain, let made {
            retainedWeights[name] = made
        }
        return made
    }

    /// Apply the base quantized linear layer and its optional LoRA update.
    private func linear(_ x: MLXArray, _ stem: String, bits: Int) -> MLXArray? {
        guard let w = weight("\(stem).weight"),
              let s = weight("\(stem).scales"),
              let b = weight("\(stem).biases")
        else { return nil }
        let base = quantizedMatmul(
            x.asType(.bfloat16), w, scales: s, biases: b, transpose: true,
            groupSize: configuration.groupSize, bits: bits, mode: .affine)

        guard let prepared = preparedLoRA(stem) else { return base }
        let delta = matmul(matmul(x.asType(.float16), prepared.a), prepared.b)
        return base + delta.asType(base.dtype)
    }

    /// LoRA matrices are converted, transposed, scaled, and cached on first use.
    private struct PreparedLoRA {
        let a: MLXArray
        let b: MLXArray
    }
    private var loraCache: [String: PreparedLoRA] = [:]

    private func preparedLoRA(_ stem: String) -> PreparedLoRA? {
        if let cached = loraCache[stem] { return cached }
        guard let lora,
              let aEntry = lora.entries["\(stem).lora_A"],
              let bEntry = lora.entries["\(stem).lora_B"],
              let aPointer = lora.pointer(to: "\(stem).lora_A"),
              let bPointer = lora.pointer(to: "\(stem).lora_B")
        else { return nil }

        func loadAdapter(_ entry: SafetensorsShard.Entry,
                         _ pointer: UnsafeRawPointer) -> MLXArray? {
            let raw = UnsafeRawBufferPointer(start: pointer, count: entry.byteCount)
            switch entry.dtype {
            case "F16": return MLXArray(raw, entry.shape, type: Float16.self)
            case "F32": return MLXArray(raw, entry.shape, type: Float.self)
            default: return nil
            }
        }
        guard let a = loadAdapter(aEntry, aPointer),
              let bb = loadAdapter(bEntry, bPointer)
        else { return nil }
        // float16 throughout, as the Python implementation does.
        let prepared = PreparedLoRA(
            a: a.asType(.float16).transposed(),
            b: (Float16(2.0) * bb.asType(.float16)).transposed())
        // Materialised here so the graph that produced them is not replayed per token.
        eval(prepared.a, prepared.b)
        loraCache[stem] = prepared
        return prepared
    }

    private func norm(_ x: MLXArray, _ name: String) -> MLXArray? {
        guard let w = weight(name) else { return nil }
        return MLXFast.rmsNorm(x.asType(.bfloat16), weight: w, eps: 1e-6)
    }

    private func rmsNormalise(_ x: MLXArray) -> MLXArray {
        let squares = mean(x.asType(.float32) * x.asType(.float32), axis: -1, keepDims: true)
        return (x.asType(.float32) * rsqrt(squares + 1e-6)).asType(x.dtype)
    }

    // MARK: - Layers

    /// Copy just the rows named, straight out of the mapping.
    ///
    /// `weight(_:)` builds an `MLXArray` over a whole tensor, and building one **copies**
    /// — mlx-c routes it to `allocator::malloc` plus `std::copy`. For a table indexed one
    /// row at a time that is the wrong shape by five orders of magnitude.
    private func rows(_ name: String, _ indices: [Int32]) -> MLXArray? {
        guard let entry = weights.entry(name), let base = weights.pointer(to: name),
              entry.shape.count == 2, entry.shape[0] > 0
        else { return nil }
        let columns = entry.shape[1]
        let rowBytes = entry.byteCount / entry.shape[0]
        var gathered = [UInt8](repeating: 0, count: rowBytes * indices.count)
        gathered.withUnsafeMutableBytes { destination in
            for (slot, row) in indices.enumerated() {
                guard row >= 0, Int(row) < entry.shape[0] else { continue }
                memcpy(destination.baseAddress!.advanced(by: slot * rowBytes),
                       base.advanced(by: Int(row) * rowBytes), rowBytes)
            }
        }
        let shape = [indices.count, columns]
        return gathered.withUnsafeBytes { raw -> MLXArray? in
            switch entry.dtype {
            case "U32", "I32": return MLXArray(raw, shape, type: UInt32.self)
            case "BF16": return MLXArray(raw, shape, type: UInt16.self).view(dtype: .bfloat16)
            case "F32": return MLXArray(raw, shape, type: Float.self)
            default: return nil
            }
        }
    }

    /// Gather only the embedding rows needed by the current token batch.
    private func embed(_ tokens: MLXArray) -> MLXArray? {
        let stem = "language_model.model.embed_tokens"
        let ids = tokens.reshaped([-1]).asArray(Int32.self)
        guard let w = rows("\(stem).weight", ids),
              let s = rows("\(stem).scales", ids),
              let b = rows("\(stem).biases", ids)
        else { return nil }
        // Dequantize only the gathered rows rather than the full embedding table.
        let out = dequantized(w, scales: s, biases: b,
                              groupSize: configuration.groupSize, bits: configuration.bits,
                              mode: .affine)
        return out.reshaped(tokens.shape + [out.dim(-1)])
    }

    /// Grouped-query attention with partial RoPE and an output gate.
    ///
    /// The gate is the second half of `q_proj`; RoPE covers 64 of each 256-dimensional
    /// head and uses a base of 10,000,000.
    private func fullAttention(_ x: MLXArray, layer: Int, state: State) -> MLXArray? {
        let stem = "language_model.model.layers.\(layer).self_attn"
        let heads = 16, kvHeads = 2, headDim = 256
        let (batch, length) = (x.dim(0), x.dim(1))

        guard let qOut = linear(x, "\(stem).q_proj", bits: 4),
              let kOut = linear(x, "\(stem).k_proj", bits: 4),
              let vOut = linear(x, "\(stem).v_proj", bits: 4),
              let qNorm = weight("\(stem).q_norm.weight"),
              let kNorm = weight("\(stem).k_norm.weight")
        else { return nil }

        let split = qOut.reshaped([batch, length, heads, 2 * headDim])
        var queries = split[.ellipsis, 0 ..< headDim]
        let gate = split[.ellipsis, headDim ..< (2 * headDim)]
            .reshaped([batch, length, heads * headDim])

        var keys = kOut.reshaped([batch, length, kvHeads, headDim])
        var values = vOut.reshaped([batch, length, kvHeads, headDim])

        queries = MLXFast.rmsNorm(queries, weight: qNorm, eps: 1e-6).transposed(0, 2, 1, 3)
        keys = MLXFast.rmsNorm(keys, weight: kNorm, eps: 1e-6).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = state.offset
        func rope(_ a: MLXArray) -> MLXArray {
            MLXFast.RoPE(a, dimensions: 64, traditional: false,
                         base: 10_000_000, scale: 1, offset: offset)
        }
        queries = rope(queries)
        keys = rope(keys)

        if let pastKeys = state.keys[layer], let pastValues = state.values[layer] {
            keys = concatenated([pastKeys, keys], axis: 2)
            values = concatenated([pastValues, values], axis: 2)
        }
        state.keys[layer] = keys
        state.values[layer] = values

        // A single query attending to everything before it needs no mask; passing a
        // square one sized for the prompt is how a working prefill breaks decode.
        let total = keys.dim(2)
        var mask: MLXArray? = nil
        if length > 1 {
            var entries = [Float](repeating: 0, count: length * total)
            for row in 0 ..< length {
                for column in (offset + row + 1) ..< total {
                    entries[row * total + column] = -Float.greatestFiniteMagnitude
                }
            }
            mask = MLXArray(entries, [length, total]).asType(queries.dtype)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1.0 / 16.0, mask: mask)
        let merged = attended.transposed(0, 2, 1, 3)
            .reshaped([batch, length, heads * headDim])
        return linear(merged * sigmoid(gate), "\(stem).o_proj", bits: 4)
    }

    /// Gated DeltaNet: depthwise causal convolution, then a gated delta-rule recurrence
    /// over a float32 state.
    ///
    /// `q` and `k` carry 16 heads against `v`'s 32 and are matched by **consecutive**
    /// duplication — h0, h0, h1, h1 — not by tiling. Both run; one is correct.
    private func linearAttention(_ x: MLXArray, layer: Int, state: State) -> MLXArray? {
        let stem = "language_model.model.layers.\(layer).linear_attn"
        let kHeads = 16, vHeads = 32, headDim = 128
        let keyDim = kHeads * headDim, valueDim = vHeads * headDim
        let convDim = keyDim * 2 + valueDim, kernel = 4
        let (batch, length) = (x.dim(0), x.dim(1))

        guard let qkv = linear(x, "\(stem).in_proj_qkv", bits: 4),
              let zRaw = linear(x, "\(stem).in_proj_z", bits: 4),
              let aRaw = linear(x, "\(stem).in_proj_a", bits: 4),
              let bRaw = linear(x, "\(stem).in_proj_b", bits: 4),
              let convWeight = weight("\(stem).conv1d.weight"),
              let aLog = weight("\(stem).A_log"),
              let dtBias = weight("\(stem).dt_bias"),
              let normWeight = weight("\(stem).norm.weight")
        else { return nil }

        let history = state.convolution[layer]?.asType(qkv.dtype)
            ?? MLXArray.zeros([batch, kernel - 1, convDim]).asType(qkv.dtype)
        let tail = concatenated([history, qkv], axis: 1)

        // `conv1d` accumulates internally in float32. Preserve that precision for
        // numerically consistent recurrent state updates.
        let convolved = conv1d(tail, convWeight, groups: convDim)
        let activated = convolved * sigmoid(convolved)
        state.convolution[layer] = tail[0..., (tail.dim(1) - (kernel - 1))...]

        var q = activated[.ellipsis, 0 ..< keyDim].reshaped([batch, length, kHeads, headDim])
        var k = activated[.ellipsis, keyDim ..< (2 * keyDim)]
            .reshaped([batch, length, kHeads, headDim])
        let v = activated[.ellipsis, (2 * keyDim) ..< convDim]
            .reshaped([batch, length, vHeads, headDim])

        // q is scaled by Dk⁻¹ and k by Dk^-0.5 — not the same factor.
        let inverseScale = Float(pow(Double(headDim), -0.5))
        q = rmsNormalise(q) * (inverseScale * inverseScale)
        k = rmsNormalise(k) * inverseScale
        q = repeated(q, count: vHeads / kHeads, axis: -2)
        k = repeated(k, count: vHeads / kHeads, axis: -2)

        let beta = sigmoid(bRaw.asType(.float32))
        let shifted = aRaw.asType(.float32) + dtBias.asType(.float32)
        let g = MLX.exp(-MLX.exp(aLog.asType(.float32)) * MLX.log(MLX.exp(shifted) + 1))

        var recurrent = state.recurrent[layer]?.asType(.float32)
            ?? MLXArray.zeros([batch, vHeads, headDim, headDim], dtype: .float32)
        var outputs: [MLXArray] = []
        // Use matrix contractions for the recurrent updates to avoid materializing large
        // broadcast-multiply intermediates:
        //
        //     (recurrent * kt[..., None, :]).sum(-1)     2 MiB temporary, reduced to 16 KB
        //     matmul(recurrent, kt[..., None])           16 KB, straight out of a GEMV
        //
        // GEMV accumulation order can differ from elementwise reduction in the last bits.
        for t in 0 ..< length {
            let qt = q[0..., t].asType(.float32).expandedDimensions(axis: -1)
            let kt = k[0..., t].asType(.float32).expandedDimensions(axis: -1)
            let vt = v[0..., t].asType(.float32)
            let decay = g[0..., t].expandedDimensions(axis: -1).expandedDimensions(axis: -1)

            recurrent = recurrent * decay
            let memory = matmul(recurrent, kt).squeezed(axis: -1)
            let delta = (vt - memory) * beta[0..., t].expandedDimensions(axis: -1)
            // The outer product has to materialise 2 MiB whichever way it is written, so
            // this one stays as it was.
            recurrent = recurrent
                + kt.swappedAxes(-1, -2) * delta.expandedDimensions(axis: -1)
            outputs.append(matmul(recurrent, qt).squeezed(axis: -1))
        }
        state.recurrent[layer] = recurrent

        let stacked = MLX.stacked(outputs, axis: 1).asType(x.dtype)
        let z = zRaw.reshaped([batch, length, vHeads, headDim])
        // RMSNormGated, in float32 as trained.
        let normalised = MLXFast.rmsNorm(stacked, weight: normWeight, eps: 1e-6)
            .asType(.float32)
        let gated = (sigmoid(z.asType(.float32)) * z.asType(.float32) * normalised)
            .asType(x.dtype)
        return linear(gated.reshaped([batch, length, valueDim]), "\(stem).out_proj", bits: 4)
    }

    /// Router, K streamed experts, and the shared expert.
    ///
    /// Apply softmax over all expert logits before top-k selection, then renormalize the
    /// selected scores.
    /// Routed experts for **one or more** tokens.
    ///
    /// The single-token form this replaces treated the K experts as the batch axis, every
    /// one of them seeing the same input. With T tokens there are T*K (token, expert)
    /// pairs and each needs its own input row, so the batch axis becomes T*K and the
    /// expert for each row is named by `rhsIndices`.
    ///
    /// For a batch, read the union of experts selected by its tokens. The union may be
    /// smaller than T*K when tokens share expert selections.
    private func moe(_ x: MLXArray, layer: Int, state: State) -> MLXArray? {
        let prefix = "language_model.model.layers.\(layer).mlp"
        let k = configuration.activeExperts
        let tokens = x.dim(1)

        if tokens >= Self.fullLayerPrefillMinimum {
            return moeFullLayer(x, layer: layer, state: state, prefix: prefix, k: k, tokens: tokens)
        }

        let ids: [Int]
        var scores: MLXArray

        // Predictions replace the normal gate during single-token decode. Prefill uses
        // the normal gate because its multi-token routing shape differs.
        if tokens == 1, let predicted = state.pregate.predicted[layer],
           let predictedScores = state.pregate.scores[layer], predicted.count == k {
            ids = predicted
            // Rebuilt here, from k floats. The alternative — keeping the slice of the
            // staging array — would hold that array and its graph across the step.
            scores = MLXArray(predictedScores, [1, 1, k])
        } else {
            guard let logits = linear(x, "\(prefix).gate", bits: 8) else { return nil }
            let gates = softmax(logits.asType(.float32), axis: -1)
            let chosen = argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k]
            scores = takeAlong(gates, chosen, axis: -1)
            scores = scores / scores.sum(axis: -1, keepDims: true)

            // Expert IDs must reach the CPU before their file reads can be issued.
            let kind: StepProfile.Phase =
                configuration.isLinear(layer) ? .syncLinear : .syncFull
            ids = profile.measure(kind) { () -> [Int] in
                eval(chosen, scores)
                return chosen.reshaped([-1]).asArray(Int32.self).map { Int($0) }
            }
        }
        guard ids.count == tokens * k else { return nil }

        // Captured for this layer's own head, which runs at the step boundary and
        // predicts for `layer + 1`. The last position is the right one for both: a
        // prefill's final position is what the first decode step continues from.
        //
        // Store the routes actually used, including predictions, as inputs for the next
        // pregate evaluation. Tensor reshaping is deferred until after graph evaluation.
        if let pregate, pregate.owners.contains(layer) {
            state.pregate.moeInput[layer] = x
            state.pregate.executed[layer] = ids.suffix(k).map { Int32($0) }
        }

        // Distinct experts in first-appearance order, and where each pair's expert landed.
        var rowOf: [Int: Int] = [:]
        var union: [Int] = []
        for id in ids where rowOf[id] == nil {
            rowOf[id] = union.count
            union.append(id)
        }
        profile.recordRouting(union: union.count, pairs: ids.count)

        // How much of what this layer needs was already asked for. **This is the number
        // that bounds the whole scheme** — the read can only be hidden for blocks the
        // guess anticipated — and it is free to collect, because both sets are already
        // on the CPU. Recorded before the routing is overwritten below.
        if configuration.routingGuess != .none {
            let guessed = state.prefetched[layer] ?? []
            profile.recordPrefetch(
                hits: union.count { guessed.contains($0) }, total: union.count)
        }
        // What executed, for the next step to guess from. The union rather than the
        // last position's pair: during prefill that is strictly more informative, and
        // during decode they are the same thing.
        var history = state.recentRouting[layer] ?? []
        history.insert(union, at: 0)
        state.recentRouting[layer] = Array(history.prefix(configuration.routingLookback))

        guard let pool = try? experts.load(layer: layer, experts: union),
              let rows = try? ids.map({ id -> Int32 in
                  guard let row = rowOf[id] else { throw Failure.layerFailed(layer) }
                  return Int32(row)
              })
        else { return nil }

        // **The single-token form is kept separate, and it is not premature.** Generalising
        // this for batching added a `reshaped` on top of a broadcast — a stride-0 view that
        // a reshape has to materialise — and decode runs this on every token of every
        // layer. Batching is used only by prefill; making decode pay for it was a
        // regression, not a simplification.
        let expanded = x.reshaped([tokens, 1, configuration.hidden]).asType(.bfloat16)
        let inputs = tokens == 1
            ? broadcast(expanded, to: [k, 1, configuration.hidden])
            : broadcast(expanded, to: [tokens, k, configuration.hidden])
                .reshaped([tokens * k, 1, configuration.hidden])
        let indices = MLXArray(rows)

        func project(_ input: MLXArray, _ offset: Int) -> MLXArray {
            gatherQuantizedMM(
                input, pool[offset], scales: pool[offset + 1], biases: pool[offset + 2],
                rhsIndices: indices, transpose: true,
                groupSize: configuration.groupSize, bits: configuration.bits,
                // **An assertion about the input, not a request**, so it is true only
                // when it is true. With one token the union is built in first-appearance
                // order from k distinct experts, making the rows exactly `0 ..< k` and
                // ascending; with more, rows repeat and are unordered, and claiming
                // otherwise would silently gather the wrong experts.
                mode: .affine, sortedIndices: tokens == 1)
        }
        let gate = project(inputs, ExpertLoader.gate)
        let up = project(inputs, ExpertLoader.up)
        let routed = project(gate * sigmoid(gate) * up, ExpertLoader.down)
        let weighted = routed
            * scores.reshaped([tokens * k, 1, 1]).asType(routed.dtype)
        let combined = weighted
            .reshaped([tokens, k, configuration.hidden])
            .sum(axis: 1)

        // Compute the shared expert after routed expert loading to avoid competing for
        // memory bandwidth during the reads.
        guard let sharedGate = linear(x, "\(prefix).shared_expert_gate", bits: 8),
              let sg = linear(x, "\(prefix).shared_expert.gate_proj", bits: 4),
              let su = linear(x, "\(prefix).shared_expert.up_proj", bits: 4),
              let shared = linear(sg * sigmoid(sg) * su,
                                  "\(prefix).shared_expert.down_proj", bits: 4)
        else { return nil }
        return combined.reshaped(shared.shape) + sigmoid(sharedGate) * shared
    }

    /// Prefill for a long chunk: the layer file is read once, and expert ids
    /// index that stack directly. The ids still come back to the CPU afterwards
    /// so the pregate feature sees what actually ran; the read no longer waits
    /// on that sync.
    private func moeFullLayer(_ x: MLXArray, layer: Int, state: State,
                              prefix: String, k: Int, tokens: Int) -> MLXArray? {
        guard let pool = try? experts.loadFullLayer(layer: layer),
              let logits = linear(x, "\(prefix).gate", bits: 8)
        else { return nil }
        let gates = softmax(logits.asType(.float32), axis: -1)
        let chosen = argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k]
        var scores = takeAlong(gates, chosen, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)

        let kind: StepProfile.Phase =
            configuration.isLinear(layer) ? .syncLinear : .syncFull
        let ids: [Int] = profile.measure(kind) {
            eval(chosen, scores)
            return chosen.reshaped([-1]).asArray(Int32.self).map { Int($0) }
        }
        guard ids.count == tokens * k else { return nil }
        if let pregate, pregate.owners.contains(layer) {
            state.pregate.moeInput[layer] = x
            state.pregate.executed[layer] = ids.suffix(k).map { Int32($0) }
        }
        var rowOf: [Int: Int] = [:]
        var union: [Int] = []
        for id in ids where rowOf[id] == nil {
            rowOf[id] = union.count
            union.append(id)
        }
        profile.recordRouting(union: union.count, pairs: ids.count)
        var history = state.recentRouting[layer] ?? []
        history.insert(union, at: 0)
        state.recentRouting[layer] = Array(history.prefix(configuration.routingLookback))

        let expanded = x.reshaped([tokens, 1, configuration.hidden]).asType(.bfloat16)
        let inputs = broadcast(expanded, to: [tokens, k, configuration.hidden])
            .reshaped([tokens * k, 1, configuration.hidden])
        // Row `e` of the full stack is expert `e`, so the router index is the gather index.
        let indices = chosen.reshaped([tokens * k])
        func project(_ input: MLXArray, _ offset: Int) -> MLXArray {
            gatherQuantizedMM(
                input, pool[offset], scales: pool[offset + 1], biases: pool[offset + 2],
                rhsIndices: indices, transpose: true,
                groupSize: configuration.groupSize, bits: configuration.bits,
                mode: .affine, sortedIndices: false)
        }
        let gate = project(inputs, ExpertLoader.gate)
        let up = project(inputs, ExpertLoader.up)
        let routed = project(gate * sigmoid(gate) * up, ExpertLoader.down)
        let weighted = routed * scores.reshaped([tokens * k, 1, 1]).asType(routed.dtype)
        let combined = weighted.reshaped([tokens, k, configuration.hidden]).sum(axis: 1)
        guard let sharedGate = linear(x, "\(prefix).shared_expert_gate", bits: 8),
              let sg = linear(x, "\(prefix).shared_expert.gate_proj", bits: 4),
              let su = linear(x, "\(prefix).shared_expert.up_proj", bits: 4),
              let shared = linear(sg * sigmoid(sg) * su,
                                  "\(prefix).shared_expert.down_proj", bits: 4)
        else { return nil }
        return combined.reshaped(shared.shape) + sigmoid(sharedGate) * shared
    }

    /// Run all routing heads as one batch and return the expert IDs for the next step.
    ///
    /// Returns the expert set per consuming layer, which is what a prefetch needs.
    @discardableResult
    private func stagePregate(state: State) -> [Int: [Int]] {
        guard let pregate else { return [:] }
        let k = configuration.activeExperts

        var inputs: [MLXArray] = []
        var executed: [[Int32]] = [], previous: [[Int32]] = []
        inputs.reserveCapacity(pregate.owners.count)
        for owner in pregate.owners {
            // A pruned or otherwise absent owner would desynchronise the stack from
            // `owners`, so the whole staging is skipped rather than silently shifted.
            //
            // Skipping has to *clear* the predictions, not just decline to write new
            // ones. `advance()` deliberately keeps `predicted` alive across the step
            // boundary — that is how a prediction made at step N reaches step N+1 — so
            // leaving it untouched here would hand step N+1 the routing computed for
            // step N, one token stale, with nothing in any log to say so.
            guard let input = state.pregate.moeInput[owner],
                  let ran = state.pregate.executed[owner]
            else {
                state.pregate.predicted = [:]
                state.pregate.scores = [:]
                state.pregate.advance()
                return [:]
            }
            // Sliced here rather than at capture time: by now `eval(h)` has run, so this
            // reshapes a materialised array instead of adding a node to a live graph.
            // The last position is the right one for a prefill chunk too — it is what
            // the first decode step continues from.
            let vector = input.ndim == 1
                ? input
                : (input.dim(1) == 1 ? input.reshaped([-1]) : input[0, -1])
            inputs.append(vector)
            executed.append(ran)
            // Use a zero row when no previous-token routing is available.
            previous.append(state.pregate.previous[owner] ?? [])
        }

        let logits = pregate.logits(
            inputs: stacked(inputs, axis: 0),
            executed: pregate.oneHotRows(executed),
            previous: pregate.oneHotRows(previous))

        // Apply the same softmax, top-k selection, and score normalization as the router.
        let gates = softmax(logits, axis: -1)
        let chosen = argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k]
        var weights = takeAlong(gates, chosen, axis: -1)
        weights = weights / weights.sum(axis: -1, keepDims: true)

        // The one barrier, and it brings back both halves. Pulling the weights across
        // too costs 66 floats and means nothing MLX-shaped is carried into the next
        // step — a view into `weights` would have kept the array, and its graph, alive
        // for every one of the thirty-three layers that held one.
        let (flat, weighting) = profile.measure(.pregateStage) { () -> ([Int32], [Float]) in
            eval(chosen, weights)
            return (chosen.reshaped([-1]).asArray(Int32.self),
                    weights.reshaped([-1]).asArray(Float.self))
        }

        var sets: [Int: [Int]] = [:]
        for (index, consumer) in pregate.consumers.enumerated() {
            let range = (index * k) ..< ((index + 1) * k)
            let slice = flat[range].map { Int($0) }
            state.pregate.predicted[consumer] = slice
            state.pregate.scores[consumer] = Array(weighting[range])
            sets[consumer] = Array(Set(slice)).sorted()
        }

        state.pregate.advance()
        return sets
    }

    // MARK: - Forward

    /// Run one model step and evaluate the resulting graph once at the end.
    func step(tokens: MLXArray, state: State) throws -> MLXArray {
        try step(tokens: tokens, state: state, needsLogits: true)!
    }

    /// One step, with the option of **not** projecting to the vocabulary.
    ///
    /// During prefill, intermediate positions do not need vocabulary logits. This option
    /// skips that projection when the result will not be consumed.
    func step(tokens: MLXArray, state: State, needsLogits: Bool) throws -> MLXArray? {
        guard var h = embed(tokens) else { throw Failure.tensorMissing("embed_tokens") }
        // Checkpoint weights use bfloat16; keep activations in the same format.
        h = h.asType(.bfloat16)

        let batchedPrefill = tokens.dim(1) >= Self.fullLayerPrefillMinimum
        for layer in 0 ..< configuration.layers {
            if batchedPrefill && layer % 10 == 0 {
                print("[35b layer] \(layer) T=\(tokens.dim(1))")
            }
            let stem = "language_model.model.layers.\(layer)"
            guard let normed = norm(h, "\(stem).input_layernorm.weight") else {
                throw Failure.layerFailed(layer)
            }
            let attended = configuration.isLinear(layer)
                ? linearAttention(normed, layer: layer, state: state)
                : fullAttention(normed, layer: layer, state: state)
            guard let attended else { throw Failure.layerFailed(layer) }
            h = h + attended

            guard let post = norm(h, "\(stem).post_attention_layernorm.weight"),
                  let mixed = moe(post, layer: layer, state: state)
            else { throw Failure.layerFailed(layer) }
            h = h + mixed
            // A full-layer expert stack is 432 MiB. Evaluating here drops it
            // before the next layer's read; holding all forty would be 17 GB.
            if batchedPrefill {
                eval(h)
                if let held = state.pregate.moeInput[layer] {
                    let last = held[0, -1]
                    eval(last)
                    state.pregate.moeInput[layer] = last
                }
            }
        }
        eval(h)

        // Run routing predictions after graph evaluation; they consume hidden states
        // produced by this step and provide expert IDs for the next step.
        switch configuration.routingGuess {
        case .none:
            break
        case .trainedHeads:
            if tokens.dim(1) > 1 && !needsLogits {
                // Batched prefill never consumes the one-token-ahead head predictions.
                // Reuse the real routing as a read hint and avoid loading/running 132 MiB
                // of heads until the final prefill batch, whose prediction is consumed
                // by the first decode step.
                state.prefetched = state.recentRouting.mapValues { Array(Set($0.joined())) }
                for (layer, chosen) in state.prefetched {
                    experts.prefetch(layer: layer, experts: chosen)
                }
                break
            }
            state.prefetched = [:]
            for (layer, chosen) in stagePregate(state: state) {
                state.prefetched[layer] = chosen
                experts.prefetch(layer: layer, experts: chosen)
            }
            // The official heads cover consumers 7...39. Preserve their exact routing,
            // but prefetch the uncovered first layers from their last real routing.
            // This is advice only: misses still run the router and read the right files.
            for layer in 0 ..< configuration.layers where state.prefetched[layer] == nil {
                guard let chosen = state.recentRouting[layer]?.first else { continue }
                state.prefetched[layer] = chosen
                experts.prefetch(layer: layer, experts: chosen)
            }
        case .recentRouting:
            // Snapshotted, not read live: `recentRouting` is rewritten layer by layer as
            // the next step runs, so scoring the hit rate against it would compare a
            // layer's routing with its own and report a meaningless 100%.
            state.prefetched = state.recentRouting.mapValues { Array(Set($0.joined())) }
            for (layer, chosen) in state.prefetched {
                experts.prefetch(layer: layer, experts: chosen)
            }
        }

        state.offset += tokens.dim(1)
        profile.countStep()
        guard needsLogits else { return nil }
        guard let final = norm(h, "language_model.model.norm.weight") else {
            throw Failure.tensorMissing("model.norm")
        }
        if coreMLHeadEnabled {
            // Core ML consumes only the final position. This is 4 KiB instead of the
            // 248,320-element logits vector, and avoids retaining MLX's 286 MiB head.
            let last = final[0, -1].reshaped([1, configuration.hidden])
            eval(last)
            return last
        }
        guard let logits = linear(final, "language_model.lm_head", bits: 4) else {
            throw Failure.tensorMissing("lm_head")
        }
        return logits
    }

    /// Argmax over the last position's logits.
    func greedy(_ output: MLXArray) async throws -> Int32 {
        if coreMLHeadEnabled, let coreMLHead,
           output.dim(output.ndim - 1) == configuration.hidden {
            let coreMLToken = try await profile.measure(.vocabularyHead) {
                try await coreMLHead.predict(output)
            }
            if !coreMLHeadValidated {
                guard let logits = linear(output, "language_model.lm_head", bits: 4) else {
                    throw Failure.tensorMissing("lm_head")
                }
                let mlxToken = Self.mlxGreedy(logits)
                if mlxToken != coreMLToken {
                    coreMLHeadEnabled = false
                    print("[coreml-head] parity FAIL coreml=\(coreMLToken) mlx=\(mlxToken); falling back")
                    return mlxToken
                }
                coreMLHeadValidated = true
                retainedWeights = retainedWeights.filter {
                    !$0.key.hasPrefix("language_model.lm_head.")
                }
                print("[coreml-head] parity PASS token=\(coreMLToken)")
            }
            return coreMLToken
        }
        return Self.mlxGreedy(output)
    }

    private static func mlxGreedy(_ logits: MLXArray) -> Int32 {
        // `-1`, not `0`: with a batched forward the interesting logits are the last
        // position's. Identical for a single token, which is why this was never wrong.
        let final = logits.ndim == 2 ? logits[0] : logits[0, -1]
        let choice = argMax(final.asType(.float32), axis: -1)
        eval(choice)
        return choice.item(Int32.self)
    }

    /// Greedy decode. Prompt and generation take the same path — one token at a time —
    /// which costs a little on prefill and buys a single code path.
    func generate(promptIds: [Int32], stop: StopCondition,
                  state: State? = nil,
                  onToken: @escaping (Int32) -> Bool) throws -> [Int32]
    {
        let state = state ?? State()
        var produced: [Int32] = []
        var next: Int32 = 0

        for (index, token) in promptIds.enumerated() {
            let logits = try step(tokens: MLXArray([token], [1, 1]), state: state)
            if index == promptIds.count - 1 {
                let choice = argMax(logits[0, 0].asType(.float32), axis: -1)
                eval(choice)
                next = choice.item(Int32.self)
            }
        }

        while true {
            switch stop.evaluate(token: next, produced: produced.count) {
            case .hitEndToken, .hitLimit:
                return produced
            case .keepGoing:
                break
            }
            produced.append(next)
            // The callback decides whether to continue, so cancellation is the caller's
            // to express rather than something this loop has to be told about.
            guard onToken(next) else { return produced }

            let logits = try step(tokens: MLXArray([next], [1, 1]), state: state)
            let choice = argMax(logits[0, 0].asType(.float32), axis: -1)
            eval(choice)
            next = choice.item(Int32.self)
        }
    }
}
