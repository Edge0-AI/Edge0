#include "edge0/generation.h"
#include "edge0/diagnostics.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {
int failures=0;
void require(bool okay,const std::string& message) {
    if(!okay) {++failures;std::cout<<"FAIL: "<<message<<'\n';}
}
template<class T> T take(const edge0::Result<T>& r) {
    if(!r){throw std::runtime_error(r.error().message);}return r.value();
}
void report(const std::string& label,const edge0::GenerationResult& r,const edge0::RuntimeReference& runtime) {
    const auto memory=take(edge0::memory_snapshot(runtime));
    std::cout<<label<<" reuse="<<static_cast<int>(r.reuse_method)<<" reused="<<r.reused_tokens
        <<" e="<<r.e<<" stop="<<static_cast<int>(r.stop_reason)<<" reasoning_bytes="<<r.reasoning.size()
        <<" answer_bytes="<<r.text.size()<<" calls="<<r.tool_calls.size()<<" rejected="<<r.rejected_tool_calls.size()
        <<" physical_peak="<<(memory.physical.values && memory.physical.values->peak_bytes?std::to_string(*memory.physical.values->peak_bytes):"unavailable")<<'\n'
        <<"reasoning="<<std::quoted(r.reasoning)<<"\nanswer="<<std::quoted(r.text)<<'\n';
    for(const auto& call:r.tool_calls) std::cout<<"call "<<call.name<<' '<<call.arguments_json<<'\n';
    for(const auto& call:r.rejected_tool_calls) std::cout<<"rejected "<<call.reason<<' '<<std::quoted(call.text)<<'\n';
    if(r.error) std::cout<<"error="<<r.error->message<<'\n';
    std::cout<<"generated=";for(auto id:r.generated)std::cout<<id<<',';std::cout<<'\n'<<std::flush;
    require(memory.physical.values && memory.physical.values->peak_bytes && *memory.physical.values->peak_bytes<=4ULL*1024*1024*1024,label+" physical peak <= 4 GiB");
}
void reuse(const std::string& directory,const edge0::GenerationResult& first,const edge0::GenerationResult& second) {
    auto prior=first.input;prior.insert(prior.end(),first.generated.begin(),first.generated.end());
    if(first.e<0 || static_cast<std::size_t>(first.e)>prior.size()) {
        require(false,"valid first-turn computed boundary");return;
    }
    prior.resize(static_cast<std::size_t>(first.e));
    std::size_t i=0;while(i<prior.size() && i<second.input.size() && prior[i]==second.input[i]) ++i;
    if(second.reuse_method==edge0::ReuseMethod::keep) {
        std::cout<<"reuse_case=keep computed_boundary="<<first.e<<'\n';
        require(second.reused_tokens==prior.size(),"keep reuses complete computed boundary");
        require(i==prior.size(),"keep computed tokens are an exact input prefix");
        return;
    }
    require(second.reuse_method==edge0::ReuseMethod::restore,"non-keep reuse must restore, never cold");
    if(second.reuse_method!=edge0::ReuseMethod::restore) return;
    require(!first.input.empty(),"first-turn checkpoint exists");
    if(first.input.empty()) return;
    const auto checkpoint=first.input.size()-1;
    std::cout<<"reuse_case=restore checkpoint_boundary="<<checkpoint<<'\n';
    require(second.reused_tokens==checkpoint,"restore reuses first-turn input checkpoint");
    require(i<prior.size() && i<second.input.size(),"restore has a token divergence");
    require(i>=first.input.size(),"divergence is inside the generated assistant turn");
    if(i>=prior.size() || i>=second.input.size()) return;
    std::cout<<"first_divergence="<<i<<" prior="<<prior[i]<<" next="
        <<(i<second.input.size()?second.input[i]:-1)<<'\n';
    const auto end_tokens=take(edge0::tokenize(directory,"<|im_end|>",true)).tokens;
    require(end_tokens.size()==1,"assistant end marker is one token");
    if(end_tokens.size()!=1) return;
    const auto next_begin=second.input.begin()+static_cast<std::ptrdiff_t>(i);
    const auto next_end=std::find(next_begin,second.input.end(),end_tokens.front());
    require(next_end!=second.input.end(),"second input contains closing assistant marker");
    require(std::find(prior.begin()+static_cast<std::ptrdiff_t>(i),prior.end(),end_tokens.front())==prior.end(),
            "computed span ends before assistant end marker");
    if(next_end==second.input.end()) return;
    const std::vector<std::int32_t> previous_span(prior.begin()+static_cast<std::ptrdiff_t>(i),prior.end());
    const std::vector<std::int32_t> next_span(next_begin,next_end);
    const auto previous_text=take(edge0::detokenize(directory,previous_span)).text;
    const auto next_text=take(edge0::detokenize(directory,next_span)).text;
    const bool equal=previous_text==next_text;
    std::cout<<"divergent_text_bytes="<<previous_text.size()<<" rerendered_text_bytes="<<next_text.size()
        <<" byte_identical="<<equal<<'\n';
    require(equal,"noncanonical tokenization preserves decoded bytes through assistant turn end");
    if(!equal) std::cout<<"computed_text="<<std::quoted(previous_text)<<"\nrerendered_text="<<std::quoted(next_text)<<'\n';
}
}
int main() {
    try {
        const char* directory=std::getenv("MOE_ASSETS_35B");
        if(!directory) {std::cout<<"SKIP: missing MOE_ASSETS_35B\n";return 77;}
        std::string device;
        for(const auto& d:edge0::query_capabilities(false).devices) if(d.backend=="metal") device=d.name;
        if(device.empty()) {std::cout<<"SKIP: Metal unavailable\n";return 77;}
        auto runtime=take(edge0::open_runtime({{device},{}}));
        struct Close {edge0::RuntimeReference runtime;~Close(){edge0::finish_runtime_use(runtime);}} close{runtime};
        auto model=take(edge0::load_model(runtime,take(edge0::resolve_load_configuration(runtime,directory))));
        edge0::GenerationOverrides options;options.temperature=0;options.max_tokens=1024;options.thinking=true;
        auto configuration=take(edge0::resolve_generation_configuration(model,options));
        std::ifstream fixture(std::string(MOE_SOURCE_ROOT)+"/tests/fixtures/multi_turn/conversations.json");
        if(!fixture) throw std::runtime_error("Missing conversations fixture");
        std::string line;std::vector<std::vector<std::string>> conversations;
        while(std::getline(fixture,line)) {
            if(line.find("\"messages\"")!=std::string::npos) conversations.emplace_back();
            else if(!conversations.empty() && line.find('"')!=std::string::npos && line.find("\"name\"")==std::string::npos) {
                std::istringstream input(line);std::string message;input>>std::quoted(message);
                if(input) conversations.back().push_back(message);
            }
        }
        require(conversations.size()==3,"three conversation fixtures");
        for(std::size_t index=0;index<conversations.size();++index) {
            auto session=take(edge0::create_session(model));edge0::ChatInput chat{{{"user",conversations[index].at(0)}}};
            auto first=take(edge0::generate(session,chat,configuration));report("conversation "+std::to_string(index)+" first",first,runtime);
            require(first.stop_reason==edge0::StopReason::end_of_generation && !first.reasoning.empty() && !first.text.empty(),"thinking first turn completes with reasoning and answer");
            chat.messages.push_back({"assistant",first.text,first.tool_calls,first.reasoning});
            chat.messages.push_back({"user",conversations[index].at(1)});
            auto second=take(edge0::generate(session,chat,configuration));report("conversation second",second,runtime);reuse(directory,first,second);
            take(edge0::release_session(session));
        }
        for(bool thinking:{false,true}) {
            auto session=take(edge0::create_session(model));
            edge0::ChatInput chat{{{"user","\xe5\x8c\x97\xe4\xba\xac\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa4\xa9\xe6\xb0\x94\xe6\x80\x8e\xe4\xb9\x88\xe6\xa0\xb7\xef\xbc\x9f\xe8\xaf\xb7\xe8\xb0\x83\xe7\x94\xa8\xe5\xb7\xa5\xe5\x85\xb7\xe6\x9f\xa5\xe8\xaf\xa2\xe3\x80\x82"}},
                {{R"({"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}})"}}};
            edge0::GenerationOverrides request;request.thinking=thinking;
            std::vector<edge0::OutputKind> kinds;std::string arguments;
            auto first=take(edge0::generate(session,chat,configuration,request,[&](const edge0::OutputEvent& event)->std::optional<edge0::Error> {
                if(event.kind==edge0::OutputKind::tool_call_begin || event.kind==edge0::OutputKind::tool_call_arguments || event.kind==edge0::OutputKind::tool_call_end) kinds.push_back(event.kind);
                if(event.kind==edge0::OutputKind::tool_call_arguments){arguments+=event.text;}return {};
            }));
            report(thinking?"tool thinking first":"tool nonthinking first",first,runtime);
            require(first.tool_calls.size()==1 && first.rejected_tool_calls.empty(),"one complete weather call");
            if(thinking) require(!first.reasoning.empty(),"tool thinking is nonempty");
            if(first.tool_calls.size()==1) {
                const auto& call=first.tool_calls.front();
                require(call.name=="get_weather","weather function name");
                std::smatch match;
                require(std::regex_search(call.arguments_json,match,std::regex(R"rx("city":\s*"([^"]*)")rx")) &&
                    (match[1].str().find("Beijing")!=std::string::npos || match[1].str().find("\xe5\x8c\x97\xe4\xba\xac")!=std::string::npos),"city string contains Beijing");
                require(arguments==call.arguments_json,"argument event concatenation");
                require(kinds.size()>=3 && kinds.front()==edge0::OutputKind::tool_call_begin && kinds.back()==edge0::OutputKind::tool_call_end,"begin arguments end order");
                for(std::size_t i=1;i+1<kinds.size();++i) require(kinds[i]==edge0::OutputKind::tool_call_arguments,"only argument deltas between begin and end");
                chat.messages.push_back({"assistant",first.text,first.tool_calls,first.reasoning});
                chat.messages.push_back({"tool",R"({"condition": "sunny", "temperature_c": 22})",{},std::nullopt,call.id});
                auto second=take(edge0::generate(session,chat,configuration,request));report("tool second",second,runtime);reuse(directory,first,second);
                require(second.stop_reason==edge0::StopReason::end_of_generation && !second.text.empty(),"tool result produces completed answer");
            }
            take(edge0::release_session(session));
        }
        std::cout<<"prediction_failures="<<failures<<'\n';return failures?1:0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
