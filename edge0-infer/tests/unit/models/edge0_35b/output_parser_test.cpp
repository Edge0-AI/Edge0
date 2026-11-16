#include "models/edge0_35b/output.h"
#include "model/json.h"
#include "tests/support/check.h"
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <new>
namespace allocation_probe {
bool active=false;
std::size_t maximum=0,large=0;
constexpr std::size_t threshold=32768;
// Keep the deallocator out of line so GCC does not diagnose an inlined free
// against the matching replacement operator new used by this test.
#if defined(__GNUC__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
void release(void* p) noexcept {std::free(p);}
}
void* operator new(std::size_t bytes) {
    if (allocation_probe::active) {
        allocation_probe::maximum=std::max(allocation_probe::maximum,bytes);
        if (bytes>=allocation_probe::threshold) ++allocation_probe::large;
    }
    if (auto* p=std::malloc(bytes?bytes:1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {allocation_probe::release(p);}
void operator delete(void* p,std::size_t) noexcept {allocation_probe::release(p);}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete[](void* p) noexcept {allocation_probe::release(p);}
void operator delete[](void* p,std::size_t) noexcept {allocation_probe::release(p);}

using namespace edge0;
using model::ParsedKind;
using Event = std::tuple<ParsedKind,std::string,std::size_t,std::string,std::string,std::string>;
const ToolDefinition tool{R"({"function":{"name":"f","parameters":{"properties":{"s":{"type":"string"},"a":{"type":["string"]},"n":{"type":"number"}}}}})"};
std::vector<std::string> characters(const std::string& s) {
    std::vector<std::string> out;
    for(std::size_t i=0;i<s.size();) {
        const auto c=static_cast<unsigned char>(s[i]);
        const std::size_t n=c<128?1:c<224?2:c<240?3:4;
        out.push_back(s.substr(i,n));i+=n;
    }
    return out;
}
std::vector<Event> parse(const std::vector<std::string>& chunks,bool thinking,bool tools) {
    auto parser=models::edge0_35b::output_parser({thinking,{},tools?std::vector<ToolDefinition>{tool}:std::vector<ToolDefinition>{}});
    std::vector<Event> events;
    std::string arguments;
    bool begun=false;
    auto receive=[&](const model::ParsedOutput& e) {
        if(e.kind==ParsedKind::tool_call_begin) {CHECK(!begun);begun=true;arguments.clear();}
        if(e.kind==ParsedKind::tool_call_arguments) {CHECK(begun);arguments+=e.text;}
        if(e.kind==ParsedKind::tool_call_end) {
            CHECK(begun);begun=false;CHECK(arguments==e.call.arguments_json);
            CHECK(model::Json::parse(arguments).kind==model::Json::Kind::object);
        }
        if(e.kind==ParsedKind::tool_call_rejected) begun=false;
        const bool merge=e.kind==ParsedKind::answer||e.kind==ParsedKind::reasoning||e.kind==ParsedKind::tool_call_arguments;
        if(merge && !events.empty() && std::get<0>(events.back())==e.kind && std::get<2>(events.back())==e.index)
            std::get<1>(events.back())+=e.text;
        else events.emplace_back(e.kind,e.text,e.index,e.call.name,e.call.arguments_json,e.reason);
        return true;
    };
    for(const auto& chunk:chunks) CHECK(parser->feed(chunk,receive));
    CHECK(parser->finish(receive));return events;
}
std::string joined(const std::vector<Event>& events,ParsedKind kind) {
    std::string out;for(const auto& e:events) if(std::get<0>(e)==kind) out+=std::get<1>(e);return out;
}
std::string block(const std::string& body) {return "<tool_call>\n<function=f>\n"+body+"</function>\n</tool_call>";}
int main() {
    {
        auto parser=models::edge0_35b::output_parser({false,{}, {tool}});
        const std::string name(65536,'f');
        const auto input="<tool_call><function="+name+"></function></tool_call>";
        std::size_t begins=0,ends=0;
        const auto receive=[&](const model::ParsedOutput& event) {
            if (event.kind==ParsedKind::tool_call_begin) {++begins;CHECK(event.call.name==name);}
            if (event.kind==ParsedKind::tool_call_end) {++ends;CHECK(event.call.arguments_json=="{}");}
            return true;
        };
        allocation_probe::large=0;allocation_probe::active=true;
        CHECK(parser->feed(input,receive));CHECK(parser->finish(receive));
        allocation_probe::active=false;
        std::cout << "large_parser_allocations=" << allocation_probe::large << "\n";
        CHECK(begins==1 && ends==1 && allocation_probe::large<=32);
    }
    const auto complete=block("<parameter=s>\nhello\n</parameter>\n<parameter=n>\n3\n</parameter>\n");
    const std::vector<std::string> cases={
        " \nreason \t</think>\n\nanswer  ", "reason"+complete, "<tool_call>literal",
        "<think>literal</think> text", "answer  ", "answer \n"+complete,
        complete, complete+"\n"+complete+"\n text", block(""),
        block("<parameter=s>\n\"\\\n\xe4\xb8\xad\n</parameteX\n</parameter>\n"),
        block("<parameter=a>\nx\n</parameter>\n"),
        block("<parameter=x>\n{\"z\": 1}\n</parameter>\n"),
        block("<parameter=x>\nnot json\n</parameter>\n"),
        "<tool_call><function= ></function></tool_call>",
        "<tool_call>bad</tool_call>",
        "<tool_call><function=f>bad</function></tool_call>",
        "<tool_call><function=f></tool_call>",
        block("<parameter=s>\nx\n</parameter><parameter=s>\ny\n</parameter>"),
        "<tool_call><function=f><parameter=s>\npartial",
        "<tool_call><fun", "reason \n", "abc<tool_cal", "x</thin"
    };
    for(bool thinking:{false,true}) for(bool tools:{false,true}) for(const auto& text:cases) {
        const auto expected=parse({text},thinking,tools);
        const auto chars=characters(text);
        CHECK(parse(chars,thinking,tools)==expected);
        std::size_t offset=0;
        for(const auto& c:chars) {
            offset+=c.size();
            CHECK(parse({text.substr(0,offset),text.substr(offset)},thinking,tools)==expected);
        }
    }
    auto events=parse({" \nreason \t</think>\n\nanswer  "},true,false);
    CHECK(joined(events,ParsedKind::reasoning)=="reason");
    CHECK(joined(events,ParsedKind::answer)=="answer  ");
    events=parse({"answer \n"+complete+"\n"+complete+"\n"},false,true);
    CHECK(joined(events,ParsedKind::answer)=="answer");
    CHECK(std::get<0>(events.back())==ParsedKind::tool_call_end);
    CHECK(std::get<2>(events.back())==1);
    CHECK(std::get<4>(events.back())==R"({"s": "hello", "n": 3})");
    events=parse({"<tool_call><function=f><parameter=s>\npartial"},false,true);
    CHECK(std::get<5>(events.back())=="incomplete");
    CHECK(joined(events,ParsedKind::tool_call_arguments)==R"({"s": "partial)");
    for(const auto& text:std::vector<std::string>{
        "<tool_call><function= ></function></tool_call>","<tool_call>bad</tool_call>",
        "<tool_call><function=f>bad</function></tool_call>","<tool_call><function=f></tool_call>",
        block("<parameter=s>\nx\n</parameter><parameter=s>\ny\n</parameter>")}) {
        events=parse({text},false,true);CHECK(std::get<5>(events.back())=="malformed");
        CHECK(std::get<1>(events.back())==text.substr(11,text.size()-23));
    }
    auto parser=models::edge0_35b::output_parser({false,{}, {tool}});
    std::string args;
    auto receive=[&](const model::ParsedOutput& e) {
        if(e.kind==ParsedKind::tool_call_arguments) args+=e.text;
        return true;
    };
    CHECK(parser->feed("<tool_call><function=f><parameter=s>\n",receive));
    CHECK(args==R"({"s": ")");
    for(const auto& c:characters("q\"\\\t\xe4\xb8\xad")) {
        const auto before=args;CHECK(parser->feed(c,receive));
        const auto quoted=model::json_quote(c);CHECK(args==before+quoted.substr(1,quoted.size()-2));
    }
    CHECK(parser->feed("\n</parameter><parameter=n>\n12",receive));
    const auto before=args;
    CHECK(parser->feed("3",receive));CHECK(args==before);
    CHECK(parser->feed("\n</parameter></function></tool_call>",receive));
    CHECK(args.find(": 123}")!=std::string::npos);
    std::cout<<"Output parsing, split invariance and streaming assertions passed\n";
}
