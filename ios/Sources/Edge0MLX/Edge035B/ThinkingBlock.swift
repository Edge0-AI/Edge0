import Foundation

/// Separates the model's reasoning from its reply.
///
/// This checkpoint is opened with `<|im_start|>assistant\n<think>\n`, so **every answer
/// begins inside a reasoning block** and the first thing it writes is working-out, not a
/// reply. The closing `</think>` is generated; the opening tag never is, because it came
/// from the prompt.
///
/// That shape leaks straight through a stream that treats decoded text as the answer —
/// the user sees the model's notes to itself, followed by a bare `</think>`, followed by
/// the reply.
///
/// **The cost of hiding it is that nothing appears until the model stops reasoning.** At
/// a few tokens per second that is a real wait with an empty bubble at the end of it; the
/// generating indicator is what carries that interval. Showing the reasoning instead was
/// the alternative, and it is worse for the ordinary case of wanting an answer.
enum ThinkingBlock {

    static let terminator = "</think>"

    /// The part of `raw` the user should see, or `nil` while the model is still reasoning.
    ///
    /// Leading whitespace after the terminator is dropped: the model writes a blank line
    /// or two before the reply proper, which would otherwise open every answer with a gap.
    static func visible(in raw: String) -> String? {
        guard let end = raw.range(of: terminator) else { return nil }
        return String(raw[end.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The non-thinking prompt normally starts directly in the answer, but some
    /// generations still emit the checkpoint's literal thinking wrapper. Hide that
    /// block while it is incomplete, then expose only the text after its terminator.
    static func visibleReply(in raw: String) -> String? {
        guard raw.contains("<think>") || raw.contains(terminator) else { return raw }
        return visible(in: raw)
    }

    /// What to show once generation has finished.
    ///
    /// **Falls back to the whole text when the block never closed.** A reply truncated by
    /// the token limit mid-reasoning has no terminator, and showing nothing at all would
    /// present a completed generation as an empty answer. Raw reasoning is poor output;
    /// an empty bubble is a bug report.
    static func settled(_ raw: String) -> String {
        visible(in: raw) ?? raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Final text for a prompt that requested a direct answer.
    static func settledReply(_ raw: String) -> String {
        guard raw.contains("<think>") || raw.contains(terminator) else {
            return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return visible(in: raw) ?? ""
    }
}
