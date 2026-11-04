#include "edge0/text.h"
#include "app/text/rules.h"
#include "app/text/decode.h"
#include "app/text/tool_results.h"
namespace edge0 {
namespace {
Error text_error(const text::TextError &e) {
    return {e.code == text::Error::invalid_argument ? ErrorCategory::invalid_argument
                                                  : ErrorCategory::incompatible, e.what()};
}
}
Result<TokenizedText> tokenize(const std::string &directory, const std::string &input, bool parse_special) {
    auto rules = text::load_rules(directory);
    if (!rules) return rules.error();
    try {
        TokenizedText out;
        const auto &bpe = rules.value().tokenizer;
        out.tokens = bpe.encode(input, parse_special);
        for (auto token : out.tokens) out.pieces.push_back(bpe.token_to_piece(token, true));
        return out;
    } catch (const text::TextError &e) { return text_error(e); }
}
Result<RenderedPrompt> render_prompt(const std::string &directory,
                                    const std::vector<TextMessage> &messages,
                                    const PromptOptions &options) {
    return render_prompt(directory, ChatInput{messages}, options);
}
Result<RenderedPrompt> render_prompt(const std::string &directory,
                                    const ChatInput& chat, const PromptOptions &options) {
    auto rules = text::load_rules(directory);
    if (!rules) return rules.error();
    return text::render_input(rules.value(), chat, options);
}
Result<RenderedPrompt> text::render_input(const text::Rules& r,
    const std::vector<TextMessage>& messages, const PromptOptions& options) {
    return render_input(r,ChatInput{messages},options);
}
Result<RenderedPrompt> text::render_input(const text::Rules& r,
    const ChatInput& chat, const PromptOptions& options) {
    try {
        if(auto error=accept_tool_results(chat)) return *error;
        if(!chat.tools.empty() && !r.description.tools)
            return edge0::Error{ErrorCategory::unsupported,"Tool formatting is not supported"};
        if(options.thinking.value_or(r.description.default_thinking) && !r.description.thinking_enabled)
            return edge0::Error{ErrorCategory::unsupported,"Thinking is not supported"};
        std::vector<model::Message> input;
        for (const auto &m : chat.messages) {
            input.push_back({m.role, m.content, m.tool_calls, m.reasoning_content});
        }
        const bool preserve = options.preserve_thinking.value_or(r.description.default_preserve_thinking);
        auto formatted = r.adapter->text_format(input, {options.thinking.value_or(r.description.default_thinking), preserve,
            chat.tool_choice.mode==ToolChoiceMode::none?std::vector<ToolDefinition>{}:chat.tools});
        RenderedPrompt out;
        out.thinking = formatted.thinking;
        out.preserve_thinking = preserve;
        std::vector<text::Fragment> fragments;
        for (const auto &f : formatted.fragments) {
            out.text += f.text;
            fragments.push_back({f.text, f.protocol});
        }
        out.tokens = r.tokenizer.encode_input(fragments);
        return out;
    } catch (const text::TextError &e) { return text_error(e); }
      catch (const std::invalid_argument &e) { return edge0::Error{ErrorCategory::invalid_argument, e.what()}; }
}
Result<DetokenizedText> detokenize(const std::string &directory,
                                 const std::vector<std::int32_t> &tokens) {
    auto rules = text::load_rules(directory);
    if (!rules) return rules.error();
    try {
        DetokenizedText out;
        text::Decoder decoder;
        for (auto token : tokens) {
            out.fragments.push_back(decoder.decode_increment(rules.value().tokenizer.token_to_piece(token, false)));
            out.text += out.fragments.back();
        }
        out.fragments.push_back(decoder.finish());
        out.text += out.fragments.back();
        return out;
    } catch (const text::TextError &e) { return text_error(e); }
}
} // namespace edge0
