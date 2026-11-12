#include "app/text/stop.h"
#include "tests/support/check.h"
#include <tuple>
#include <random>
#include <limits>
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
    // The complete-string oracle is independent of the streaming prefix state.
    std::mt19937 random(351104);
    for (int trial = 0; trial < 500; ++trial) {
        std::vector<std::string> stops;
        for (unsigned n = 1 + random() % 4; n; --n) {
            std::string stop;
            for (unsigned size = 1 + random() % 8; size; --size) stop += char('a' + random() % 3);
            stops.push_back(stop);
        }
        std::string input;
        for (unsigned size = random() % 128; size; --size) input += char('a' + random() % 4);
        std::size_t end = std::numeric_limits<std::size_t>::max(), start = end;
        std::optional<std::string> matched;
        for (const auto& stop : stops) {
            const auto found = input.find(stop);
            if (found != std::string::npos &&
                (found + stop.size() < end || (found + stop.size() == end && found < start))) {
                start = found; end = found + stop.size(); matched = stop;
            }
        }
        const Result expected{matched ? input.substr(0, start) : input, matched};
        std::vector<std::string> chunks;
        for (std::size_t at = 0; at < input.size();) {
            const auto size = std::min(input.size() - at, std::size_t(1 + random() % 7));
            chunks.push_back(input.substr(at, size)); at += size;
        }
        CHECK(run(stops, chunks) == expected);
        CHECK(run(stops, {input}) == expected);
    }
    const std::string repeated(4096, 'a');
    std::vector<std::string> bytes(4096, "a");
    bytes.push_back("c");
    CHECK(run({repeated + "b"}, bytes) == Result(repeated + "c", {}));
    edge0::text::StopMatcher hold({"STOP"});
    CHECK(hold.feed("aST")=="a");CHECK(!hold.matched());
    CHECK(hold.feed("x")=="STx");CHECK(hold.finish().empty());
    CHECK(hold.feed("ST").empty());CHECK(hold.finish()=="ST");
    CHECK(hold.feed("OP")=="OP");CHECK(!hold.matched());
}
