import Foundation

/// Byte-level BPE, the GPT-4 variant this checkpoint uses.
///
/// The pipeline, in order, because getting the order wrong produces a tokenizer that
/// works on English and mis-splits everything else:
///
/// 1. **NFC** normalisation
/// 2. **Split** on the pre-tokenizer regex, `Isolated` — each match becomes its own piece
/// 3. **ByteLevel** — every UTF-8 byte maps to one printable character, so BPE never sees
///    a raw byte and no `unk` token is needed
/// 4. **BPE** — repeatedly merge the adjacent pair with the lowest rank
///
/// Loaded from the compact binary produced by `tools/convert_tokenizer.py`: 5.6 MiB of
/// flat arrays rather than 19 MiB of JSON, with merges stored as **token id pairs** so
/// the merge loop never looks anything up by string.
final class Tokenizer {

    enum Failure: LocalizedError {
        case cannotOpen(String)
        case badHeader(String)
        case truncated(String, needs: Int, has: Int)

        var errorDescription: String? {
            switch self {
            case .cannotOpen(let p): return "cannot open \(p)"
            case .badHeader(let p): return "bad tokenizer header in \(p)"
            case .truncated(let p, let needs, let has):
                return "\(p) is truncated: needs \(needs) bytes, has \(has)"
            }
        }
    }

    private let tokens: [String]                  // id -> token text (byte-level encoded)
    private let ids: [String: Int32]              // token text -> id
    private let ranks: [UInt64: Int32]            // packed (left, right) -> merge rank
    private let specials: [String: Int32]
    private let specialTexts: [Int32: String]
    private let splitter: NSRegularExpression

    /// Byte ↔ printable-character tables, the GPT-2 mapping.
    ///
    /// Bytes that are already printable map to themselves; the rest are shifted into an
    /// unused Unicode range. This is what lets a BPE vocabulary of *characters* represent
    /// arbitrary bytes without an unknown token.
    private static let byteToUnicode: [Character] = {
        var mapping: [Int] = []
        var table = [Character](repeating: " ", count: 256)
        for value in UInt8(ascii: "!") ... UInt8(ascii: "~") { mapping.append(Int(value)) }
        for value in 0xA1 ... 0xAC { mapping.append(value) }
        for value in 0xAE ... 0xFF { mapping.append(value) }

        var extra = 0
        for byte in 0 ..< 256 {
            if mapping.contains(byte) {
                table[byte] = Character(UnicodeScalar(byte)!)
            } else {
                table[byte] = Character(UnicodeScalar(256 + extra)!)
                extra += 1
            }
        }
        return table
    }()

    private static let unicodeToByte: [Character: UInt8] = {
        var inverse: [Character: UInt8] = [:]
        for byte in 0 ..< 256 { inverse[byteToUnicode[byte]] = UInt8(byte) }
        return inverse
    }()

    init(url: URL) throws {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              data.count > 8
        else { throw Failure.cannotOpen(url.lastPathComponent) }

        let headerLength = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard headerLength > 0, 8 + headerLength <= data.count,
              let header = try? JSONSerialization.jsonObject(
                with: data[8 ..< (8 + headerLength)]) as? [String: Any],
              let vocabSize = header["vocab_size"] as? Int,
              let mergeCount = header["merge_count"] as? Int,
              let specialCount = header["special_count"] as? Int,
              let pattern = header["regex"] as? String
        else { throw Failure.badHeader(url.lastPathComponent) }

        // Sections are fixed-width and laid out in a known order, so each is a slice at a
        // running offset. Checked against the file length before any of it is read — a
        // truncated tokenizer that silently loses its tail would encode most text
        // correctly and fail only on rare tokens.
        var cursor = 8 + headerLength
        func take(_ count: Int) throws -> Data {
            guard cursor + count <= data.count else {
                throw Failure.truncated(url.lastPathComponent,
                                        needs: cursor + count, has: data.count)
            }
            defer { cursor += count }
            return data[cursor ..< (cursor + count)]
        }

        let offsets = try take((vocabSize + 1) * 4).toUInt32Array()
        let blob = try take(Int(offsets[vocabSize]))
        let mergeFlat = try take(mergeCount * 8).toUInt32Array()
        let specialIds = try take(specialCount * 4).toUInt32Array()
        let specialOffsets = try take((specialCount + 1) * 4).toUInt32Array()
        let specialBlob = try take(Int(specialOffsets[specialCount]))

        var tokens = [String](repeating: "", count: vocabSize)
        var ids: [String: Int32] = [:]
        ids.reserveCapacity(vocabSize)
        blob.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for index in 0 ..< vocabSize {
                let start = Int(offsets[index]), end = Int(offsets[index + 1])
                let text = String(decoding: UnsafeBufferPointer(
                    start: base + start, count: end - start), as: UTF8.self)
                tokens[index] = text
                ids[text] = Int32(index)
            }
        }
        self.tokens = tokens
        self.ids = ids

        var ranks: [UInt64: Int32] = [:]
        ranks.reserveCapacity(mergeCount)
        for rank in 0 ..< mergeCount {
            let left = UInt64(mergeFlat[rank * 2])
            let right = UInt64(mergeFlat[rank * 2 + 1])
            ranks[(left << 32) | right] = Int32(rank)
        }
        self.ranks = ranks

        var specials: [String: Int32] = [:]
        var specialTexts: [Int32: String] = [:]
        specialBlob.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for index in 0 ..< specialCount {
                let start = Int(specialOffsets[index]), end = Int(specialOffsets[index + 1])
                let text = String(decoding: UnsafeBufferPointer(
                    start: base + start, count: end - start), as: UTF8.self)
                specials[text] = Int32(specialIds[index])
                specialTexts[Int32(specialIds[index])] = text
            }
        }
        self.specials = specials
        self.specialTexts = specialTexts

        // Carried across from the source file rather than rewritten, because this pattern
        // is where CJK and emoji splitting is decided.
        self.splitter = try NSRegularExpression(pattern: pattern, options: [])
    }

    var vocabularySize: Int { tokens.count }
    var specialTokenCount: Int { specials.count }

    func id(of special: String) -> Int32? { specials[special] }

    // MARK: - Encode

    func encode(_ text: String) -> [Int32] {
        // Special tokens are matched before anything else and never split — a chat
        // template's `<|im_start|>` reaching the BPE loop would come back as a handful of
        // ordinary pieces that happen to spell it.
        var output: [Int32] = []
        for (chunk, special) in splitOnSpecials(text) {
            if let special {
                output.append(special)
            } else {
                output.append(contentsOf: encodeOrdinary(chunk))
            }
        }
        return output
    }

    private func splitOnSpecials(_ text: String) -> [(String, Int32?)] {
        guard !specials.isEmpty else { return [(text, nil)] }
        var pieces: [(String, Int32?)] = []
        var remainder = Substring(text)

        while !remainder.isEmpty {
            var earliest: (Range<Substring.Index>, Int32)? = nil
            for (token, id) in specials {
                if let range = remainder.range(of: token),
                   earliest == nil || range.lowerBound < earliest!.0.lowerBound {
                    earliest = (range, id)
                }
            }
            guard let (range, id) = earliest else { break }
            if range.lowerBound > remainder.startIndex {
                pieces.append((String(remainder[remainder.startIndex ..< range.lowerBound]), nil))
            }
            pieces.append(("", id))
            remainder = remainder[range.upperBound...]
        }
        if !remainder.isEmpty { pieces.append((String(remainder), nil)) }
        return pieces
    }

    private func encodeOrdinary(_ text: String) -> [Int32] {
        let normalised = text.precomposedStringWithCanonicalMapping   // NFC
        var output: [Int32] = []

        let full = NSRange(normalised.startIndex ..< normalised.endIndex, in: normalised)
        splitter.enumerateMatches(in: normalised, options: [], range: full) { match, _, _ in
            guard let match, let range = Range(match.range, in: normalised) else { return }
            let piece = String(normalised[range])

            // Byte level: every UTF-8 byte becomes one printable character.
            let mapped = String(piece.utf8.map { Self.byteToUnicode[Int($0)] })
            output.append(contentsOf: bpe(mapped))
        }
        return output
    }

    /// Merge the lowest-ranked adjacent pair until none remains.
    private func bpe(_ text: String) -> [Int32] {
        if let whole = ids[text] { return [whole] }

        var parts = text.map { String($0) }.compactMap { ids[$0] }
        guard parts.count > 1 else { return parts }

        while parts.count > 1 {
            var bestRank = Int32.max
            var bestIndex = -1
            for index in 0 ..< (parts.count - 1) {
                let key = (UInt64(UInt32(bitPattern: parts[index])) << 32)
                    | UInt64(UInt32(bitPattern: parts[index + 1]))
                if let rank = ranks[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }

            let merged = tokens[Int(parts[bestIndex])] + tokens[Int(parts[bestIndex + 1])]
            guard let id = ids[merged] else { break }
            parts.replaceSubrange(bestIndex ... (bestIndex + 1), with: [id])
        }
        return parts
    }

    // MARK: - Decode

    func decode(_ sequence: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in sequence {
            if let special = specialTexts[id] {
                bytes.append(contentsOf: Array(special.utf8))
                continue
            }
            guard id >= 0, Int(id) < tokens.count else { continue }
            for character in tokens[Int(id)] {
                if let byte = Self.unicodeToByte[character] { bytes.append(byte) }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}

private extension Data {
    func toUInt32Array() -> [UInt32] {
        withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(
                start: raw.bindMemory(to: UInt32.self).baseAddress!,
                count: count / 4))
        }
    }
}
