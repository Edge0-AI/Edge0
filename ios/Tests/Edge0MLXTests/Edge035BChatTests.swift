@testable import Edge0MLX
import Testing

@Test func edge35bFirstTurnTemplateMatchesCheckpointContract() throws {
    let rendered = try ChatTemplate.render([
        .init(role: "system", content: "helpful"),
        .init(role: "user", content: "hello"),
    ])
    #expect(rendered == "<|im_start|>system\nhelpful<|im_end|>\n" +
        "<|im_start|>user\nhello<|im_end|>\n" +
        "<|im_start|>assistant\n<think>\n")
}

@Test func edge35bFirstTurnTemplateCanDisableThinking() throws {
    let rendered = try ChatTemplate.render([
        .init(role: "system", content: "helpful"),
        .init(role: "user", content: "hello"),
    ], thinking: false)
    #expect(rendered == "<|im_start|>system\nhelpful<|im_end|>\n" +
        "<|im_start|>user\nhello<|im_end|>\n" +
        "<|im_start|>assistant\n")
}

@Test func edge35bContinuationDoesNotReplayHistory() {
    #expect(ChatTemplate.continuation(user: "again") ==
        "<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n" +
        "<|im_start|>assistant\n<think>\n")
    #expect(ChatTemplate.continuation(user: "again", thinking: false) ==
        "<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n" +
        "<|im_start|>assistant\n")
}

@Test func edge35bStopsOnBothCheckpointEndTokens() {
    let stop = StopCondition(maximumTokens: 3)
    #expect(stop.evaluate(token: 248046, produced: 0) == .hitEndToken(248046))
    #expect(stop.evaluate(token: 248044, produced: 0) == .hitEndToken(248044))
    #expect(stop.evaluate(token: 42, produced: 3) == .hitLimit)
    #expect(stop.evaluate(token: 42, produced: 2) == .keepGoing)
}

@Test func edge35bHidesReasoningBlock() {
    #expect(ThinkingBlock.visible(in: "private notes") == nil)
    #expect(ThinkingBlock.visible(in: "private</think>\n\nanswer") == "answer")
    #expect(ThinkingBlock.settled("unfinished reasoning") == "unfinished reasoning")
}
