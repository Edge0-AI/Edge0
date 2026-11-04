#pragma once
#include "edge0/result.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace edge0 {
struct ToolCall { std::string id, name, arguments_json; };
struct ToolDefinition { std::string json; };
struct TextMessage {
    std::string role, content;
    std::vector<ToolCall> tool_calls = {};
    std::optional<std::string> reasoning_content = std::nullopt;
    std::string tool_call_id = {};
};
enum class ToolChoiceMode { automatic, none, required, function };
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::automatic;
    std::string name = {};
};
struct ChatInput {
    std::vector<TextMessage> messages;
    std::vector<ToolDefinition> tools = {};
    ToolChoice tool_choice = {};
    bool parallel_tool_calls = true;
};
struct PromptOptions {
    std::optional<bool> thinking, preserve_thinking;
};
struct TokenizedText {
    std::vector<std::int32_t> tokens;
    // Raw bytes, including control token spellings; not necessarily valid UTF-8.
    std::vector<std::string> pieces;
};
struct RenderedPrompt {
    std::string text;
    bool thinking = false;
    bool preserve_thinking = false;
    std::vector<std::int32_t> tokens;
};
struct DetokenizedText {
    // One stable UTF-8 delivery per token, followed by the final flush.
    std::vector<std::string> fragments;
    std::string text;
};
// Developer checks: reload metadata on each call, without opening a runtime,
// initializing backends, or reading tensor payloads. Also usable with a runtime open.
Result<TokenizedText> tokenize(const std::string &directory, const std::string &text,
                              bool parse_special = false);
Result<RenderedPrompt> render_prompt(const std::string &directory,
                                    const std::vector<TextMessage> &messages,
                                    const PromptOptions &options = {});
Result<RenderedPrompt> render_prompt(const std::string &directory,
                                    const ChatInput&, const PromptOptions &options = {});
// Control tokens are omitted. Invalid UTF-8 bytes use incremental replacement.
Result<DetokenizedText> detokenize(const std::string &directory,
                                 const std::vector<std::int32_t> &tokens);
} // namespace edge0
