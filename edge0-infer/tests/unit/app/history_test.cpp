#include "app/text/history.h"
#include "tests/support/check.h"
using namespace edge0;
int main() {
    const session::Identity id{"a","b","c"};
    session::Snapshot cached;cached.identity=id;cached.tokens={1,2,3};cached.e=3;
    CHECK(text::match_history({},cached,id)==0);
    CHECK(text::match_history({1,2,3},cached,id)==3);
    CHECK(text::match_history({9,2,3},cached,id)==0);
    CHECK(text::match_history({1,9,3},cached,id)==1);
    CHECK(text::match_history({1,2,9},cached,id)==2);
    CHECK(text::match_history({1,2,3},cached,{"a","b","different"})==0);
    cached.tokens.clear();cached.e=0;CHECK(text::match_history({1},cached,id)==0);
    CHECK(!text::prepare_suffix({1,2},2));CHECK(!text::prepare_suffix({1},2));
    CHECK(text::prepare_suffix({1,2,3},1).value()==std::vector<std::int32_t>({2,3}));
    text::Decoder decoder;CHECK(decoder.decode_increment("\xe4").empty());
    auto finished=text::finish_turn(decoder,3,{7,8},4,true);
    CHECK(finished.text=="\xef\xbf\xbd" && finished.tail==8 && finished.tail_is_end);
    CHECK(!text::finish_turn(decoder,3,{7,8},5,false).tail);
    CHECK(!text::finish_turn(decoder,3,{},2,false).tail);
    CHECK(text::finish_turn(decoder,0,{},0,false).text.empty());
}
