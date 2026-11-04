#include "app/text/stop.h"
#include "tests/support/check.h"
#include <tuple>
using Result=std::pair<std::string,std::optional<std::string>>;
Result run(const std::vector<std::string>& stops,const std::vector<std::string>& chunks) {
    edge0::text::StopMatcher match(stops);std::string text;
    for(const auto& chunk:chunks) text+=match.feed(chunk);
    text+=match.finish();return {text,match.matched()};
}
int main() {
    using Case=std::tuple<std::vector<std::string>,std::string,std::string,std::optional<std::string>>;
    for(const auto& c:std::vector<Case>{
        {{"STOP"},"prefixSTOPdiscard","prefix","STOP"},
        {{"STOP"},"prefixST","prefixST",{}},
        {{"ab","a"},"ab","","a"},
        {{"bc","abc"},"zabc","z","abc"},
        {{"zz","bc"},"abczz","a","bc"},
        {{"STOP"},"STxSTOP","STx","STOP"},
        {{"\xe4\xb8\xad"},"a\xe4\xb8\xad""b","a","\xe4\xb8\xad"},
        {{},"abc","abc",{}},
    }) {
        const auto& stops=std::get<0>(c);const auto& text=std::get<1>(c);
        const Result expected{std::get<2>(c),std::get<3>(c)};
        CHECK(run(stops,{text})==expected);
        std::vector<std::string> chars;
        for(std::size_t i=0;i<text.size();) {
            const auto b=static_cast<unsigned char>(text[i]);const std::size_t n=b<128?1:b<224?2:b<240?3:4;
            chars.push_back(text.substr(i,n));i+=n;
            CHECK(run(stops,{text.substr(0,i),text.substr(i)})==expected);
        }
        CHECK(run(stops,chars)==expected);
    }
    edge0::text::StopMatcher hold({"STOP"});
    CHECK(hold.feed("aST")=="a");CHECK(!hold.matched());
    CHECK(hold.feed("x")=="STx");CHECK(hold.finish().empty());
}
