import Foundation

public enum Edge0TokenizerError8B: Error, LocalizedError {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        }
    }
}

/// Native Swift implementation of the checkpoint's NFC + regex + ByteLevel BPE pipeline.
/// It reads `tokenizer.json` from the model folder to use the checkpoint vocabulary and
/// merge table directly.
public final class Edge0Tokenizer8B: @unchecked Sendable {
    private struct FileFormat: Decodable {
        struct AddedToken: Decodable {
            let id: Int
            let content: String
        }

        struct Model: Decodable {
            let vocab: [String: Int]
            let merges: [[String]]
        }

        let added_tokens: [AddedToken]
        let model: Model
    }

    private struct Pair: Hashable {
        let left: String
        let right: String
    }

    private static let splitPattern = #"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#

    private let vocabulary: [String: Int]
    private let tokensByID: [Int: String]
    private let mergeRanks: [Pair: Int]
    private let addedTokens: [(content: String, id: Int)]
    private let byteEncoder: [UInt8: String]
    private let byteDecoder: [Character: UInt8]
    private let splitRegex: NSRegularExpression

    public let endOfTurnTokenID: Int

    public init(contentsOf url: URL) throws {
        let payload = try JSONDecoder().decode(FileFormat.self, from: Data(contentsOf: url))
        vocabulary = payload.model.vocab
        tokensByID = Dictionary(uniqueKeysWithValues: payload.model.vocab.map { ($0.value, $0.key) })
        mergeRanks = Dictionary(uniqueKeysWithValues: payload.model.merges.enumerated().compactMap { rank, merge in
            guard merge.count == 2 else { return nil }
            return (Pair(left: merge[0], right: merge[1]), rank)
        })
        addedTokens = payload.added_tokens
            .map { ($0.content, $0.id) }
            .sorted { lhs, rhs in lhs.content.count > rhs.content.count }
        guard let eos = payload.added_tokens.first(where: { $0.content == "<|role_end|>" })?.id else {
            throw Edge0TokenizerError8B.invalid("tokenizer.json is missing <|role_end|>")
        }
        endOfTurnTokenID = eos

        let maps = Self.makeByteMaps()
        byteEncoder = maps.encoder
        byteDecoder = maps.decoder
        splitRegex = try NSRegularExpression(pattern: Self.splitPattern)
    }

    public func encode(_ text: String) throws -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping
        var ids: [Int] = []
        var plain = ""
        var cursor = normalized.startIndex

        func flushPlain() throws {
            guard !plain.isEmpty else { return }
            ids.append(contentsOf: try encodePlain(plain))
            plain.removeAll(keepingCapacity: true)
        }

        while cursor < normalized.endIndex {
            let suffix = normalized[cursor...]
            if let token = addedTokens.first(where: { suffix.hasPrefix($0.content) }) {
                try flushPlain()
                ids.append(token.id)
                cursor = normalized.index(cursor, offsetBy: token.content.count)
            } else {
                let next = normalized.index(after: cursor)
                plain.append(contentsOf: normalized[cursor..<next])
                cursor = next
            }
        }
        try flushPlain()
        return ids
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) throws -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let token = tokensByID[id] ?? addedTokens.first(where: { $0.id == id })?.content else {
                throw Edge0TokenizerError8B.invalid("Unknown token ID \(id)")
            }
            if skipSpecialTokens, addedTokens.contains(where: { $0.id == id }) { continue }
            for character in token {
                if let byte = byteDecoder[character] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(character).utf8)
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodePlain(_ text: String) throws -> [Int] {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = splitRegex.matches(in: text, range: range)
        var ids: [Int] = []
        for match in matches {
            guard let swiftRange = Range(match.range, in: text) else { continue }
            let byteLevel = text[swiftRange].utf8.map { byteEncoder[$0]! }.joined()
            for piece in bpe(byteLevel) {
                guard let id = vocabulary[piece] else {
                    throw Edge0TokenizerError8B.invalid("BPE vocabulary is missing a merged token")
                }
                ids.append(id)
            }
        }
        return ids
    }

    private func bpe(_ token: String) -> [String] {
        var pieces = token.map(String.init)
        while pieces.count > 1 {
            var selected: Pair?
            var selectedRank = Int.max
            for index in 0..<(pieces.count - 1) {
                let pair = Pair(left: pieces[index], right: pieces[index + 1])
                if let rank = mergeRanks[pair], rank < selectedRank {
                    selected = pair
                    selectedRank = rank
                }
            }
            guard let selected else { break }

            var merged: [String] = []
            var index = 0
            while index < pieces.count {
                if index + 1 < pieces.count,
                   pieces[index] == selected.left,
                   pieces[index + 1] == selected.right {
                    merged.append(selected.left + selected.right)
                    index += 2
                } else {
                    merged.append(pieces[index])
                    index += 1
                }
            }
            pieces = merged
        }
        return pieces
    }

    private static func makeByteMaps() -> (encoder: [UInt8: String], decoder: [Character: UInt8]) {
        var bytes = Array(UInt8(33)...UInt8(126))
        bytes += Array(UInt8(161)...UInt8(172))
        bytes += Array(UInt8(174)...UInt8(255))
        var scalars = bytes.map(Int.init)
        var extra = 0
        for value in 0...255 where !bytes.contains(UInt8(value)) {
            bytes.append(UInt8(value))
            scalars.append(256 + extra)
            extra += 1
        }
        var encoder: [UInt8: String] = [:]
        var decoder: [Character: UInt8] = [:]
        for (byte, scalarValue) in zip(bytes, scalars) {
            let character = Character(String(UnicodeScalar(scalarValue)!))
            encoder[byte] = String(character)
            decoder[character] = byte
        }
        return (encoder, decoder)
    }
}
