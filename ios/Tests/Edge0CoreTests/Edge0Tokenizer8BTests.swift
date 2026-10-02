import Edge0Core
import Foundation
import Testing

@Test func edge0Tokenizer8BMatchesOfficialHiPrompt() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let modelPath = environment["EDGE0_MODEL"] else { return }
    let tokenizer = try Edge0Tokenizer8B(contentsOf: URL(fileURLWithPath: modelPath).appendingPathComponent("tokenizer.json"))
    let prompt = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Hi<|role_end|><role>ASSISTANT</role>\n<think></think>"
    #expect(try tokenizer.encode(prompt) == [157151, 90827, 157152, 14136, 5381, 6350, 928, 156895, 157151, 39, 116171, 157152, 10754, 156895, 157151, 8469, 7342, 5468, 157152, 198, 156903, 156904])
    #expect(try tokenizer.decode([10754, 0]) == "Hi!")
}
