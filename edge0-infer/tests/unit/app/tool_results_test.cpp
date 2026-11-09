#include "app/text/tool_results.h"
#include "tests/support/check.h"
using namespace edge0;
int main() {
    const TextMessage user{"user","question"};
    const TextMessage assistant{"assistant","",{{"one","f","{}"},{"two","f","{}"}},std::string("reason")};
    const TextMessage one{"tool","result",{},std::nullopt,"one"};
    const TextMessage two{"tool","result",{},std::nullopt,"two"};
    const auto check=[](std::vector<TextMessage> messages,bool valid) {
        auto error=text::accept_tool_results(ChatInput{messages});
        CHECK(!error==valid);if(error) CHECK(error->category==ErrorCategory::invalid_argument);
    };
    check({user},true);check({user,assistant,one,two},true);check({user,assistant,two,one,user},true);
    check({user,TextMessage{"assistant","answer",{},std::string("")},user},true);
    check({one},false);check({user,one},false);check({user,assistant,one,one},false);
    check({user,assistant,one,user},false);check({user,assistant,one},false);
    check({user,assistant,one,two,one},false);
    auto wrong=one;wrong.tool_call_id="unknown";check({user,assistant,wrong,two},false);
    wrong=one;wrong.tool_call_id.clear();check({user,assistant,wrong,two},false);
    auto bad=assistant;bad.tool_calls[1].id="one";check({user,bad,one},false);
    bad=assistant;bad.tool_calls[0].id.clear();check({user,bad,one,two},false);
    bad=assistant;bad.tool_calls[0].name.clear();check({user,bad,one,two},false);
    bad=assistant;bad.role="user";check({bad},false);
    wrong=user;wrong.tool_call_id="one";check({wrong},false);
    wrong=user;wrong.reasoning_content="";check({wrong},false);
    wrong=user;wrong.role="unknown";check({wrong},false);
    bad=assistant;bad.tool_calls[0].arguments_json="[]";check({user,bad,one,two},false);
    CHECK(text::accept_tool_results(ChatInput{{user},{{"[]"}}}));
    ChatInput input{{user},{{"{}"}}};input.tool_choice.mode=ToolChoiceMode::none;
    CHECK(!text::accept_tool_results(input));
    for(auto choice:{ToolChoiceMode::required,ToolChoiceMode::function}) {
        input.tool_choice.mode=choice;
        CHECK(text::accept_tool_results(input)->category==ErrorCategory::unsupported);
    }
}
