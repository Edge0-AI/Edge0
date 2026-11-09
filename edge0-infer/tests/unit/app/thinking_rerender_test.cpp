#include "models/edge0_35b/output.h"
#include "edge0/text.h"
#include "tests/support/check.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
template<class T> T take(const edge0::Result<T>& r) {CHECK(r);return r.value();}
int main() {
    const char* root=std::getenv("MOE_TINY");if(!root) return 77;
    const auto directory=std::string(root)+"/tiny-a";
    const edge0::ToolDefinition tool{R"({"type":"function","function":{"name":"f","parameters":{"properties":{"city":{"type":"string"}}}}})"};
    const std::string call="<tool_call>\n<function=f>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n</tool_call>";
    for(int kind=0;kind<3;++kind) {
        const bool thinking=kind!=1,tools=kind!=0;
        edge0::ChatInput chat{{{"user","Hello"}}};if(tools) chat.tools={tool};
        const auto first=take(edge0::render_prompt(directory,chat,{thinking,true}));
        const auto output=(thinking?std::string("Reason\n</think>\n\n"):std::string{})+(tools?call:"Answer");
        auto parser=edge0::models::edge0_35b::output_parser({thinking,true,chat.tools});
        edge0::TextMessage assistant{"assistant","",{},std::string{}};
        auto receive=[&](const edge0::model::ParsedOutput& event) {
            if(event.kind==edge0::model::ParsedKind::answer) assistant.content+=event.text;
            if(event.kind==edge0::model::ParsedKind::reasoning) *assistant.reasoning_content+=event.text;
            if(event.kind==edge0::model::ParsedKind::tool_call_end) {
                auto call=event.call;call.id="fixture";assistant.tool_calls.push_back(call);
            }
            CHECK(event.kind!=edge0::model::ParsedKind::tool_call_rejected);return true;
        };
        CHECK(parser->feed(output,receive));CHECK(parser->finish(receive));
        chat.messages.push_back(assistant);
        if(tools) chat.messages.push_back({"tool","sunny",{},std::nullopt,"fixture"});
        chat.messages.push_back({"user","Next"});
        const auto next=take(edge0::render_prompt(directory,chat,{thinking,true}));
        auto computed=first.tokens;const auto generated=take(edge0::tokenize(directory,output,true)).tokens;
        computed.insert(computed.end(),generated.begin(),generated.end());
        std::size_t i=0;while(i<computed.size() && i<next.tokens.size() && computed[i]==next.tokens[i]) ++i;
        if(i!=computed.size()) std::cerr<<"kind="<<kind<<" first divergence="<<i<<" computed="<<computed[i]<<" next="<<next.tokens[i]<<'\n';
        CHECK(i==computed.size());
    }
    const auto prompt=take(edge0::render_prompt(directory,edge0::ChatInput{{{"user","<tool_call>"}},{tool}}));
    const auto without=take(edge0::render_prompt(directory,edge0::ChatInput{{{"user","ordinary"}},{tool}}));
    CHECK(std::count(prompt.tokens.begin(),prompt.tokens.end(),248058)==std::count(without.tokens.begin(),without.tokens.end(),248058));
    CHECK(std::count(prompt.tokens.begin(),prompt.tokens.end(),248058)>0);
    std::cout<<"Thinking/tool rerender prefixes and protected user markers passed\n";
}
