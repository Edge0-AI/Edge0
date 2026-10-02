import Foundation
import MLX

/// Decode state captured after a fixed run of prompt tokens, so those tokens are never
/// processed twice.
///
/// **This is the single largest win available on time-to-first-character.** A measured
/// prefill runs at 258 ms/token, and of a 96-token prompt the great majority is the
/// system instructions — byte-identical on every request, in every conversation, for the
/// life of the install. Recomputing it is the bulk of a ~25 second wait.
///
/// ## Why it is unusually cheap here
///
/// Measured on device: **63 MiB**, against a 2.31 GB peak and a ~3.5 GB jetsam line.
///
/// Sixty of those MiB are the thirty DeltaNet layers' recurrent state, whose size **does
/// not depend on how many tokens produced it** — caching a 96-token prefix and a
/// 10,000-token prefix cost the same. Only the ten full-attention layers contribute a
/// per-token term, at 2 KiB each.
///
/// The property that makes those thirty layers awkward elsewhere — the state is a fixed
/// matrix updated in place, so dropping history does not shrink it and `replaceHistory`
/// cannot prune cheaply — is exactly the property that makes them nearly free to cache.
/// A plain forty-layer transformer of these dimensions would need 80 KiB per token.
///
/// ## Why the prefix is verified rather than assumed
///
/// The obvious implementation trusts that encoding the system message alone yields the
/// same tokens as encoding it as the head of the full prompt. Byte-level BPE gives no
/// such guarantee in general: merges are free to cross a boundary the renderer considers
/// structural. So the cached token run is stored alongside the state and checked against
/// the real prompt on every use. A mismatch costs an array comparison and falls back to
/// prefilling normally — it can never produce a wrong answer.
///
/// This also makes invalidation automatic. Editing the system instructions changes the
/// tokens, the stored run stops matching, and the stale file is simply ignored.
struct PrefixCache {

    /// The exact tokens this state was produced by.
    let tokens: [Int32]
    let state: Edge0Model35B.State

    /// Whether `ids` begins with this cache's tokens, and so can skip them.
    func matches(_ ids: [Int32]) -> Bool {
        // An empty run is rejected explicitly. `prefix(0)` equals `[]` for every input,
        // so without this a cache holding nothing would claim to match everything —
        // currently unreachable, since neither capture nor `read` can produce one, but
        // it is the kind of vacuous truth that stops being unreachable quietly.
        guard !tokens.isEmpty else { return false }

        // A prompt *equal* to the prefix is rejected: replaying the last token is what
        // produces the logits generation starts from, so there has to be one left.
        return ids.count > tokens.count && Array(ids.prefix(tokens.count)) == tokens
    }

    // MARK: - Persistence

    /// Where the snapshot lives between launches.
    ///
    /// **Persisting matters more than caching in memory.** An in-process cache only helps
    /// the second message onward; the first one after launch — the one that decides
    /// whether the app feels usable at all — still pays in full.
    ///
    /// **Deliberately not beside the weights.** `Documents/repacked` is created by
    /// `devicectl` when the 19 GB is pushed, and the app cannot write into it:
    ///
    ///     free=6063MiB dirExists=true
    ///     smallWrite=Code=513 "You don't have permission to save the file"
    ///
    /// Six gigabytes free and a directory that plainly exists, so this is ownership, not
    /// space. The same cause explains a puzzle from the transfer work — partial files that
    /// **neither** `devicectl` **nor** the app could delete. Anything the app needs to
    /// write belongs somewhere the app made.
    ///
    /// Application Support rather than Caches: this is derived and rebuildable, but
    /// rebuilding costs twenty-five seconds of the user's attention, which is not what
    /// Caches is for.
    /// - Parameter fingerprint: identifies the model configuration. **Required, because
    ///   `matches` compares tokens and nothing else.** A snapshot taken by the full
    ///   forty-layer model is meaningless to a pruned one — same prompt, different
    ///   function — and restoring it would produce confident nonsense with no error
    ///   anywhere. Separate files mean a configuration change simply misses the cache.
    static func fileURL(in directory: URL, fingerprint: String) -> URL {
        let manager = FileManager.default
        guard let support = manager.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
        else {
            return directory.appendingPathComponent(
                "prefix-cache-\(fingerprint).safetensors")
        }
        try? manager.createDirectory(at: support, withIntermediateDirectories: true)
        return support.appendingPathComponent("prefix-cache-\(fingerprint).safetensors")
    }

    /// Flattened for safetensors, which stores a flat name→array map.
    ///
    /// The token run is written as an array too, so the file is self-describing: nothing
    /// outside it needs to stay in sync for the verification above to work.
    private func flattened() -> [String: MLXArray] {
        // **Everything is forced contiguous before it is written.** safetensors stores a
        // plain row-major buffer and cannot represent a strided view, so saving one
        // fails. The convolution state is exactly that — `state.convolution[layer]` is a
        // tail slice of a larger array — and the failure was silent: the write was
        // wrapped in `try?`, so the only symptom was a cache that worked within a session
        // and never survived a launch.
        var out: [String: MLXArray] = ["tokens": MLXArray(tokens)]
        for (layer, value) in state.convolution { out["conv.\(layer)"] = contiguous(value) }
        for (layer, value) in state.recurrent { out["rec.\(layer)"] = contiguous(value) }
        for (layer, value) in state.keys { out["k.\(layer)"] = contiguous(value) }
        for (layer, value) in state.values { out["v.\(layer)"] = contiguous(value) }
        return out
    }

    /// Write the snapshot. Cache persistence is best-effort; a missing cache affects
    /// startup speed but not generation correctness.
    func write(to url: URL) {
        do {
            let arrays = flattened()
            eval(Array(arrays.values))
            try save(arrays: arrays, url: url)
        } catch {
            let directory = url.deletingLastPathComponent()
            let free = (try? directory.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage
            let probe = directory.appendingPathComponent(".write-probe")
            var probeResult = "ok"
            do {
                try Data([0]).write(to: probe)
                try? FileManager.default.removeItem(at: probe)
            } catch {
                probeResult = "\(error)"
            }
            print("""
                [prefix-cache] write failed: \(error)
                [prefix-cache] free=\(free.map { "\($0 / 1_048_576)MiB" } ?? "unknown") \
                dirExists=\(FileManager.default.fileExists(atPath: directory.path)) \
                smallWrite=\(probeResult)
                """)
        }
    }

    /// Read a snapshot back, or `nil` if there is nothing usable there.
    ///
    /// Every failure mode — absent file, truncated write, a format from an older build —
    /// resolves to `nil` and a normal prefill. A cache is an optimisation; it is never
    /// worth failing a request over.
    static func read(from url: URL) -> PrefixCache? {
        guard FileManager.default.fileExists(atPath: url.path),
              let arrays = try? loadArrays(url: url),
              let tokenArray = arrays["tokens"]
        else { return nil }

        let state = Edge0Model35B.State()
        for (name, value) in arrays where name != "tokens" {
            let parts = name.split(separator: ".")
            guard parts.count == 2, let layer = Int(parts[1]) else { continue }
            switch parts[0] {
            case "conv": state.convolution[layer] = value
            case "rec": state.recurrent[layer] = value
            case "k": state.keys[layer] = value
            case "v": state.values[layer] = value
            default: break
            }
        }
        let tokens = tokenArray.asArray(Int32.self)
        guard !tokens.isEmpty else { return nil }
        state.offset = tokens.count
        return PrefixCache(tokens: tokens, state: state)
    }
}

extension Edge0Model35B.State {

    /// A copy that generation can advance without disturbing the original.
    ///
    /// The four tables are Swift dictionaries — value types — so copying them is a real
    /// snapshot, and the `MLXArray`s they hold are safe to share because **every write in
    /// the model reassigns rather than mutating** (`state.recurrent[layer] = ...`, all
    /// four verified at their call sites). If that ever changes to an in-place update,
    /// this becomes silent corruption rather than a compile error, which is why it is
    /// written down here.
    func snapshot() -> Edge0Model35B.State {
        let copy = Edge0Model35B.State()
        copy.convolution = convolution
        copy.recurrent = recurrent
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }
}
