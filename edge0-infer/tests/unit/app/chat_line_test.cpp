#include "edge0/generation.h"
namespace cli { std::vector<edge0::TextMessage> completed_history(const std::vector<edge0::TextMessage>&, const std::vector<edge0::TextMessage>&, const edge0::GenerationResult&); }
#include "tests/support/check.h"
#include <string>
#include <utility>
namespace cli { std::pair<std::string, std::string> classify_chat_line(std::string); }
void test_chat_lines();
int main() {
    test_chat_lines();
    CHECK(cli::classify_chat_line("/think on")==std::make_pair(std::string("/think"),std::string("on")));
    CHECK(cli::classify_chat_line(" /think off \r")==std::make_pair(std::string("/think"),std::string("off")));
    for(const auto* line:{"/think","/think yes","/think on now"}) CHECK(cli::classify_chat_line(line).first=="unknown");
    const std::vector<edge0::TextMessage> previous{{"user","old"},{"assistant","answer"}};
    for (bool regenerate : {false,true}) for (bool valid : {false,true}) for (bool text : {false,true}) {
        auto candidate=previous;
        if (regenerate) candidate.pop_back(); else candidate.push_back({"user","new"});
        edge0::GenerationResult result;result.stop_reason=edge0::StopReason::cancelled;result.state_valid=valid;result.text=text?"partial":"";
        const auto history=cli::completed_history(candidate,previous,result);
        auto expected=previous;
        if(valid && text) {expected=candidate;expected.push_back({"assistant","partial"});}
        CHECK(history.size()==expected.size());
        for(std::size_t i=0;i<history.size();++i) CHECK(history[i].role==expected[i].role && history[i].content==expected[i].content);
    }

    for (const auto& line : {"", " ", "\t\r", "\r", " \t\n"})
        CHECK(cli::classify_chat_line(line).first == "empty");
    for (const std::string command : {"/regen", "/new", "/quit"}) {
        CHECK(cli::classify_chat_line(command).first == command);
        CHECK(cli::classify_chat_line(" \t" + command + " \r").first == command);
    }
    for (const auto& line : {"/other", "/", "/new please", " /ordinary text "})
        CHECK(cli::classify_chat_line(line).first == "unknown");
    for (const std::string line : {"hello", " hello\t ", "a/b", "\xc3\xa9"}) {
        CHECK(cli::classify_chat_line(line).first == "message");
        CHECK(cli::classify_chat_line(line).second == line);
        CHECK(cli::classify_chat_line(line + "\r").second == line);
    }
}
