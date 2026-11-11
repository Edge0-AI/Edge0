#include "tools/cli/serve_protocol.h"
#include "tests/support/check.h"
#include <fstream>
#include <iostream>
#include <set>
using portal::Json;
using portal::Protocol;
void rejects(const Json& j,Protocol p,int status=400) {
    try {portal::parse(j,p,"test");CHECK(false);}catch(const portal::Failure& f){CHECK(f.status==status);CHECK(f.body().contains("error"));}
}
portal::Failure failure_of(const Json& j,Protocol p) {
    try {portal::parse(j,p,"test");}catch(const portal::Failure& f){return f;}
    CHECK(false);return portal::Failure(0,"");
}
bool ends_with(const std::string& s,const std::string& tail) {return s.size()>=tail.size() && s.compare(s.size()-tail.size(),tail.size(),tail)==0;}
void names_field(const Json& j,Protocol p,const std::string& param) {
    const auto f=failure_of(j,p);const auto error=f.body()["error"];
    CHECK(f.status==400 && error["code"]=="unsupported_parameter" && error["param"]==param);
    CHECK(error["message"]==f.message() && ends_with(error["message"].get<std::string>(),": "+param));
    portal::Encoder encoder(portal::parse(p==Protocol::chat?Json{{"model","test"},{"messages",Json::array({{{"role","user"},{"content","hello"}}})}}:Json{{"model","test"},{"input","hello"}},p,"test"),"test","1");
    const auto streamed=encoder.fail(f);CHECK(streamed.size()==1);
    const auto payload=Json::parse(streamed[0].substr(streamed[0].find("data: ")+6));
    CHECK((p==Protocol::chat?payload["error"]["message"]:payload["response"]["error"]["message"])==f.message());
}
void metrics_contract() {
    edge0::GenerationResult result;result.stop_reason=edge0::StopReason::end_of_generation;
    result.computed_tokens=12;result.request_summary.first_token_seconds=0.25;
    result.request_summary.decode_tokens_per_second=7.5;
    edge0::MemorySnapshot memory;
    memory.physical.values=edge0::MemoryValues{};memory.physical.values->peak_bytes=2000;
    memory.rss.values=edge0::MemoryValues{};memory.rss.values->peak_bytes=1000;
    for(auto protocol:{Protocol::chat,Protocol::responses})for(bool include_usage:{false,true}) {
        portal::Request request;request.protocol=protocol;request.include_usage=include_usage;
        portal::Encoder encoder(request,"test","1");
        const auto full=encoder.complete(result,memory);
        const Json expected={{"first_token_seconds",0.25},{"decode_tokens_per_second",7.5},{"memory_peak_bytes",2000},{"memory_peak_metric","physical"}};
        CHECK(full["metrics"]==expected);
        for(const auto& record:encoder.begin())CHECK(record.find("metrics")==std::string::npos);
        auto records=encoder.finish(result,memory);bool found=false;
        for(const auto& record:records) {
            if(record.find("metrics")==std::string::npos)continue;
            const auto payload=Json::parse(record.substr(record.find("data: ")+6));
            CHECK((protocol==Protocol::chat?payload["metrics"]:payload["response"]["metrics"])==expected);found=true;
        }
        CHECK(found==(protocol==Protocol::responses || include_usage));
        auto missing=result;missing.request_summary={};
        CHECK(encoder.complete(missing)["metrics"]==Json::object());
        auto resident=memory;resident.physical.values->peak_bytes.reset();
        CHECK(encoder.complete(result,resident)["metrics"]["memory_peak_metric"]=="resident");
        CHECK(encoder.complete(result,resident)["metrics"]["memory_peak_bytes"]==1000);
        for(auto stop:{edge0::StopReason::cancelled,edge0::StopReason::failed}) {
            auto failed=result;failed.stop_reason=stop;
            for(const auto& record:encoder.finish(failed,memory))CHECK(record.find("metrics")==std::string::npos);
        }
    }
}
void response_truncated_item_status() {
    const Json request={{"model","test"},{"input","hello"}};
    for (const auto reason : {edge0::StopReason::length, edge0::StopReason::capacity,
                             edge0::StopReason::end_of_generation, edge0::StopReason::stop_string}) {
        for (const auto kind : {edge0::OutputKind::answer, edge0::OutputKind::reasoning}) {
            portal::Encoder encoder(portal::parse(request,Protocol::responses,"test"),"test","1");
            encoder.begin();
            edge0::OutputEvent event;
            event.kind=kind==edge0::OutputKind::answer?edge0::OutputKind::reasoning:edge0::OutputKind::answer;
            event.text="earlier";encoder.accept(event);
            event.kind=kind;event.text="partial";encoder.accept(event);
            edge0::GenerationResult result;result.stop_reason=reason;
            const auto records=encoder.finish(result);
            const auto full=encoder.complete(result);
            const bool truncated=reason==edge0::StopReason::length || reason==edge0::StopReason::capacity;
            const std::string status=truncated?"incomplete":"completed";
            CHECK(full["output"][0]["status"]=="completed");
            CHECK(full["output"][1]["status"]==status);
            bool found=false;
            for (const auto& record : records) {
                const auto value=Json::parse(record.substr(record.find("data: ")+6));
                if(value["type"]!="response.output_item.done")continue;
                found=true;CHECK(value["item"]["status"]==status);
            }
            CHECK(found);
        }
    }
}
int main(int argc,char** argv) {
    metrics_contract();
    response_truncated_item_status();
    CHECK(argc==2);
    std::ifstream input(argv[1]);Json fixtures;input>>fixtures;
    for(const auto& fixture:fixtures) {
        const auto protocol=fixture["protocol"]=="chat"?Protocol::chat:Protocol::responses;
        bool accepted=true;try {portal::parse(fixture["request"],protocol,"test");}catch(const portal::Failure&){accepted=false;}
        if(accepted!=fixture["accept"].get<bool>())std::cerr<<fixture["name"]<<'\n';
        CHECK(accepted==fixture["accept"].get<bool>());
    }
    const Json chat={{"model","test"},{"messages",Json::array({{{"role","user"},{"content","hello"}}})}};
    const Json responses={{"model","test"},{"input","hello"}};
    for(auto protocol:{Protocol::chat,Protocol::responses}) {
        const auto base=protocol==Protocol::chat?chat:responses;
        CHECK(portal::parse(base,protocol,"test").input.messages[0].content=="hello");
        for(const auto* field:{"unknown","previous_response_id","conversation"}) {auto j=base;j[field]="x";rejects(j,protocol);}
        auto j=base;j["model"]="wrong";rejects(j,protocol,404);
        j=base;j["tool_choice"]="required";rejects(j,protocol);
        j=base;j["tool_choice"]="none";CHECK(portal::parse(j,protocol,"test").input.tool_choice.mode==edge0::ToolChoiceMode::none);
        for(const auto* field:{"temperature","top_p","top_logprobs","stream","parallel_tool_calls","store","metadata","user","service_tier","prompt_cache_key","promptCacheKey","safety_identifier"}) {j=base;j[field]=Json::array();rejects(j,protocol);}
        j=base;j["promptCacheKey"]=Json::array();CHECK(failure_of(j,protocol).param=="promptCacheKey");
        j=base;j["promptcachekey"]="x";names_field(j,protocol,"promptcachekey");
        j=base;if(protocol==Protocol::chat)j["messages"][0]["name"]="x";else j["input"]=Json::array({{{"role","user"},{"content","hello"},{"name","x"}}});
        names_field(j,protocol,protocol==Protocol::chat?"messages.name":"input.name");
    }
    auto j=chat;j["n"]=2;rejects(j,Protocol::chat);j["n"]=1;portal::parse(j,Protocol::chat,"test");
    j=chat;j["max_tokens"]=1;j["max_completion_tokens"]=2;rejects(j,Protocol::chat);j["max_completion_tokens"]=1;portal::parse(j,Protocol::chat,"test");
    j=chat;j["reasoning_effort"]="medium";j["chat_template_kwargs"]={{"enable_thinking",false}};rejects(j,Protocol::chat);
    j["chat_template_kwargs"]["enable_thinking"]=true;CHECK(*portal::parse(j,Protocol::chat,"test").overrides.thinking);
    j=chat;j["logit_bias"]={{"bad",1}};rejects(j,Protocol::chat);j["logit_bias"]={{"42",1}};CHECK(portal::parse(j,Protocol::chat,"test").overrides.logit_bias->count(42)==1);
    j=chat;j["top_logprobs"]=2;rejects(j,Protocol::chat);j["logprobs"]=true;portal::parse(j,Protocol::chat,"test");
    j=chat;j["messages"][0]["content"]=Json::array({{{"type","image_url"},{"text","x"}}});rejects(j,Protocol::chat);
    j=chat;j["tools"]=Json::array({{{"type","function"},{"function",{{"name","weather"},{"strict",true}}}}});rejects(j,Protocol::chat);
    auto tool_history=chat;
    tool_history["messages"]=Json::array({{{"role","assistant"},{"tool_calls",Json::array({{{"id","call_a"},{"type","function"},{"function",{{"name","f"},{"arguments","{}"}}}}})}}});
    CHECK(portal::parse(tool_history,Protocol::chat,"test").input.messages[0].tool_calls.size()==1);
    j["tools"][0]["function"]["strict"]=false;portal::parse(j,Protocol::chat,"test");
    j=responses;j["tools"]=Json::array({{{"type","function"},{"parameters",Json::object()},{"name","weather"}}});
    CHECK(portal::parse(j,Protocol::responses,"test").input.tools[0].json=="{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"parameters\":{}}}");
    j=responses;j["input"]=Json::array({{{"type","reasoning"},{"content",Json::array({{{"type","reasoning_text"},{"text","think"}}})}},{{"role","assistant"},{"content","answer"}},{{"type","function_call"},{"call_id","call_1"},{"name","weather"},{"arguments","{}"}},{{"type","function_call_output"},{"call_id","call_1"},{"output","sunny"}}});
    auto parsed=portal::parse(j,Protocol::responses,"test");CHECK(parsed.input.messages.size()==2);CHECK(parsed.input.messages[0].tool_calls.size()==1);CHECK(*parsed.input.messages[0].reasoning_content=="think");
    for(auto protocol:{Protocol::chat,Protocol::responses}) {
        auto request=portal::parse(protocol==Protocol::chat?chat:responses,protocol,"test");request.include_usage=true;
        portal::Encoder encoder(request,"test","1");auto start=encoder.begin();CHECK(!start.empty());
        edge0::OutputEvent e;e.kind=edge0::OutputKind::tool_call_rejected;e.index=0;e.rejected={0,"bad","malformed"};encoder.accept(e);
        e.kind=edge0::OutputKind::reasoning;e.text="think";encoder.accept(e);
        e.kind=edge0::OutputKind::token_logprob;e.logprob.bytes="\xff";e.logprob.logprob=-1;encoder.accept(e);
        e.kind=edge0::OutputKind::answer;e.text="answer";auto answer=encoder.accept(e);CHECK(!answer.empty());
        e.kind=edge0::OutputKind::tool_call_begin;e.index=2;e.tool_call={"call_2","weather",""};auto records=encoder.accept(e);
        if(protocol==Protocol::chat) {auto data=Json::parse(records[0].substr(6));CHECK(data["choices"][0]["delta"]["tool_calls"][0]["index"]==0);}
        e.kind=edge0::OutputKind::tool_call_arguments;e.text="{}";encoder.accept(e);e.kind=edge0::OutputKind::tool_call_end;encoder.accept(e);
        edge0::GenerationResult result;result.stop_reason=edge0::StopReason::tool_calls;result.text="answer";result.reasoning="think";result.tool_calls={{"call_2","weather","{}"}};result.input={1,2};result.reused_tokens=1;result.generated_count=3;
        auto tail=encoder.finish(result);auto full=encoder.complete(result);CHECK(full["usage"]["total_tokens"]==5);
        if(protocol==Protocol::chat) {CHECK(tail.back()=="data: [DONE]\n\n");CHECK(full["choices"][0]["finish_reason"]=="tool_calls");}
        else {CHECK(full["output"].size()==3);CHECK(full["output"][1]["content"][0]["text"]=="answer");CHECK(full["status"]=="completed");}
        for(auto stop:{edge0::StopReason::length,edge0::StopReason::capacity,edge0::StopReason::end_of_generation,edge0::StopReason::stop_string}) {
            result.stop_reason=stop;auto value=encoder.complete(result);bool length=stop==edge0::StopReason::length || stop==edge0::StopReason::capacity;
            if(protocol==Protocol::responses)CHECK(value["status"]==(length?"incomplete":"completed"));
            else CHECK(value["choices"][0]["finish_reason"]==(length?"length":stop==edge0::StopReason::end_of_generation?"tool_calls":"stop"));
        }
        result.stop_reason=edge0::StopReason::failed;auto error=encoder.finish(result);CHECK(error.size()==1 && error[0].find("[DONE]")==std::string::npos);
    }
    for(auto protocol:{Protocol::chat,Protocol::responses}) {
        auto request=portal::parse(protocol==Protocol::chat?chat:responses,protocol,"test");
        request.include_usage=true;
        for(bool truncated:{false,true}) {
            portal::Encoder encoder(request,"test","1");
            auto wire=encoder.begin();
            auto send=[&](edge0::OutputEvent e){auto records=encoder.accept(e);wire.insert(wire.end(),records.begin(),records.end());};
            edge0::OutputEvent event;event.kind=edge0::OutputKind::reasoning;event.text="reason";send(event);
            event.kind=edge0::OutputKind::token_logprob;event.logprob.bytes="x";event.logprob.logprob=-0.5;send(event);
            event.kind=edge0::OutputKind::answer;event.text="x";send(event);
            event.kind=edge0::OutputKind::tool_call_begin;event.index=3;event.tool_call={"call_3","f",""};send(event);
            event.kind=edge0::OutputKind::tool_call_arguments;event.text="{\"a\":";send(event);
            edge0::GenerationResult result;result.text="x";result.reasoning="reason";result.configuration.thinking=true;
            if(truncated) {
                event.kind=edge0::OutputKind::tool_call_rejected;event.rejected={3,"{\"a\":","truncated"};send(event);
                result.rejected_tool_calls={event.rejected};result.stop_reason=edge0::StopReason::length;
            } else {
                event.kind=edge0::OutputKind::tool_call_arguments;event.text="1}";send(event);
                event.kind=edge0::OutputKind::tool_call_end;send(event);
                result.tool_calls={{"call_3","f","{\"a\":1}"}};result.stop_reason=edge0::StopReason::end_of_generation;
            }
            auto tail=encoder.finish(result);wire.insert(wire.end(),tail.begin(),tail.end());
            std::string answer,reasoning,arguments;std::size_t sequence=0;std::set<int> added;
            bool saw_probs=false;
            for(const auto& record:wire) {
                auto offset=record.find("data: ");CHECK(offset!=std::string::npos);
                auto payload=record.substr(offset+6);if(payload=="[DONE]\n\n")continue;
                const auto value=Json::parse(payload);
                if(protocol==Protocol::chat) {
                    if(value["choices"].empty())continue;
                    const auto& choice=value["choices"][0];const auto& delta=choice["delta"];
                    answer+=delta.value("content",std::string{});reasoning+=delta.value("reasoning_content",std::string{});
                    if(delta.contains("tool_calls"))for(const auto& call:delta["tool_calls"]) {CHECK(call["index"]==0);arguments+=call["function"].value("arguments",std::string{});}
                    if(!choice["logprobs"].is_null()) {CHECK(choice["logprobs"]["content"][0]["token"]=="x");saw_probs=true;}
                } else {
                    CHECK(value["sequence_number"]==sequence++);
                    const auto type=value["type"].get<std::string>();
                    if(type=="response.output_item.added")added.insert(value["output_index"].get<int>());
                    if(value.contains("output_index") && type!="response.output_item.added")CHECK(added.count(value["output_index"].get<int>())==1);
                    if(type=="response.output_text.delta") {answer+=value["delta"].get<std::string>();CHECK(value["logprobs"][0]["token"]=="x");saw_probs=true;}
                    if(type=="response.reasoning_text.delta")reasoning+=value["delta"].get<std::string>();
                    if(type=="response.function_call_arguments.delta")arguments+=value["delta"].get<std::string>();
                    if(type=="response.completed" || type=="response.incomplete")CHECK(value["response"]==encoder.complete(result));
                }
            }
            CHECK(answer==result.text && reasoning==result.reasoning && saw_probs);
            CHECK(arguments==(truncated?"{\"a\":" : result.tool_calls[0].arguments_json));
            auto full=encoder.complete(result);
            if(protocol==Protocol::responses)CHECK(full["output"][2]["status"]==(truncated?"incomplete":"completed"));
            else if(truncated) {CHECK(!full["choices"][0]["message"].contains("tool_calls"));CHECK(full["choices"][0]["message"]["rejected_tool_calls"].size()==1);}
        }
    }
    CHECK(portal::failure({edge0::ErrorCategory::busy,""}).status==429);
    CHECK(portal::failure({edge0::ErrorCategory::busy,""}).message()=="Server busy");
    CHECK(portal::failure({edge0::ErrorCategory::busy,""}).body()["error"]["message"]=="Server busy");
    CHECK(portal::failure({edge0::ErrorCategory::capacity_exceeded,""}).status==400);
    CHECK(portal::failure({edge0::ErrorCategory::backend_error,""},true).status==503);
}
