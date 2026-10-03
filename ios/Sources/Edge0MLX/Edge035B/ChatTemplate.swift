import Foundation

/// The text-only slice of this checkpoint's chat template.
///
/// The shipped `chat_template.jinja` is 7,764 characters of Jinja with macros, image and
/// video handling, tool calls and vision counters. None of that is reachable from a
/// text-only app, and running a Jinja interpreter on device to reach a fixed string would
/// be a large amount of machinery for no behaviour.
///
/// What the text path actually renders is regular:
///
/// ```
/// <|im_start|>{role}\n{content}<|im_end|>\n     per message
/// <|im_start|>assistant\n<think>\n              generation prompt (thinking on)
/// <|im_start|>assistant\n                       generation prompt (thinking off)
/// ```
///
/// **Scope is the point, not a shortcut.** Vision and tool-call messages are out — and
/// they fail loudly rather than rendering something that looks plausible, because a
/// silently dropped image would produce a coherent answer about nothing.
///
/// The trailing `<think>\n` is the checkpoint's default: replies open inside a reasoning
/// block. Passing `thinking: false` omits it so the model writes the answer directly —
/// fewer tokens before anything useful appears, at the cost of losing that reasoning pass.
enum ChatTemplate {

    struct Message {
        let role: String
        let content: String
    }

    enum Failure: LocalizedError {
        case unsupportedRole(String)

        var errorDescription: String? {
            switch self {
            case .unsupportedRole(let role):
                return "chat template supports system/user/assistant only, not '\(role)'"
            }
        }
    }

    static let supportedRoles: Set<String> = ["system", "user", "assistant"]
    static let userOpening = "<|im_start|>user\n"

    /// The opening tag every assistant turn is generated under.
    static func assistantOpening(thinking: Bool) -> String {
        thinking
            ? "<|im_start|>assistant\n<think>\n"
            : "<|im_start|>assistant\n"
    }

    static func render(
        _ messages: [Message],
        addGenerationPrompt: Bool = true,
        thinking: Bool = true
    ) throws -> String {
        var out = ""
        for message in messages {
            guard supportedRoles.contains(message.role) else {
                throw Failure.unsupportedRole(message.role)
            }
            // **Past assistant turns are reopened with the same thinking mode.**
            // Generation runs under `assistantOpening`, so a thinking-on reply begins
            // inside a reasoning block and contains its `</think>`. Rendering that text
            // back without the opening tag put a dangling close tag in the history —
            // well-formed-looking output describing a prompt the model never saw.
            //
            // It also makes replay exact, which is what lets a conversation's state be
            // extended instead of rebuilt.
            if message.role == "assistant" {
                out += assistantOpening(thinking: thinking) + "\(message.content)<|im_end|>\n"
            } else {
                out += message.role == "user"
                    ? userOpening + "\(message.content)<|im_end|>\n"
                    : "<|im_start|>\(message.role)\n\(message.content)<|im_end|>\n"
            }
        }
        if addGenerationPrompt {
            out += assistantOpening(thinking: thinking)
        }
        return out
    }

    /// What must be fed to continue a conversation whose state already ends inside the
    /// previous assistant turn.
    ///
    /// This is the append path: it closes the open assistant turn, adds the new question,
    /// and reopens for the answer. Nothing before it is re-rendered, so nothing before it
    /// is recomputed.
    static func continuation(user: String, thinking: Bool = true) -> String {
        "<|im_end|>\n" + userOpening + "\(user)<|im_end|>\n" + assistantOpening(thinking: thinking)
    }
}

/// When to stop generating.
///
/// Two conditions, and both are needed: `eos_token_id` is a **list** here — 248046
/// (`<|im_end|>`) and 248044 (`<|endoftext|>`) — so checking only the first lets the
/// model run past the end of its turn. The length bound is the backstop for the case
/// where neither is ever emitted, which is what a mis-specified prompt produces.
struct StopCondition {
    let endTokens: Set<Int32>
    let maximumTokens: Int

    init(endTokens: [Int32] = [248046, 248044], maximumTokens: Int = 256) {
        self.endTokens = Set(endTokens)
        self.maximumTokens = maximumTokens
    }

    enum Outcome: Equatable {
        case keepGoing
        case hitEndToken(Int32)
        case hitLimit
    }

    func evaluate(token: Int32, produced: Int) -> Outcome {
        if endTokens.contains(token) { return .hitEndToken(token) }
        if produced >= maximumTokens { return .hitLimit }
        return .keepGoing
    }
}
