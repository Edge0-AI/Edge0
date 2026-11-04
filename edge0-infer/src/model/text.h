#pragma once
#include "edge0/text.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace edge0::model {
struct Message {
  std::string role, content;
  std::vector<ToolCall> tool_calls = {};
  std::optional<std::string> reasoning_content = std::nullopt;
};
struct TextOptions {
  std::optional<bool> thinking, preserve_thinking;
  std::vector<ToolDefinition> tools = {};
};
struct SamplingRecommendation {
  double temperature = 0, top_p = 1, repetition_penalty = 1;
  std::int64_t top_k = 0, max_tokens = 256;
  double presence_penalty = 0, frequency_penalty = 0;
};
struct TextFragment {
  std::string text;
  bool protocol = false;
};
struct TextDescription {
  std::string template_sha256, model, pre;
  bool default_thinking = false, tools = false;
  bool default_preserve_thinking = false;
  bool thinking = false, thinking_enabled = false;
  std::optional<SamplingRecommendation> sampling_thinking_off = std::nullopt;
  std::optional<SamplingRecommendation> sampling_thinking_on = std::nullopt;
  std::optional<std::int64_t> recommended_context = std::nullopt;
};
struct TextResult {
  std::vector<TextFragment> fragments;
  bool thinking = false;
};
enum class ParsedKind { answer, reasoning, tool_call_begin, tool_call_arguments, tool_call_end, tool_call_rejected };
struct ParsedOutput {
  ParsedKind kind = ParsedKind::answer;
  std::string text;
  std::size_t index = 0;
  ToolCall call;
  std::string reason;
};
// Returning false halts parsing at the event, including the rest of the input fragment.
using ParsedReceiver = std::function<bool(const ParsedOutput&)>;
class OutputParser {
public:
  virtual ~OutputParser() = default;
  virtual ParsedKind section() const { return ParsedKind::answer; }
  // GGUF may classify format markers as control tokens. Preserve only markers
  // understood by this parser; unrelated control tokens remain hidden.
  virtual bool preserve_control_piece(const std::string&) const { return false; }
  virtual bool feed(const std::string& text, const ParsedReceiver& receive) {
    return text.empty() || receive({ParsedKind::answer, text, 0, {}, {}});
  }
  virtual bool finish(const ParsedReceiver&) { return true; }
};
} // namespace edge0::model
