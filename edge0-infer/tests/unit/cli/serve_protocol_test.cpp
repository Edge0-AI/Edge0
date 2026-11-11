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

void messages_parse_contract() {
    const Json base={{"model","test"},{"max_tokens",32},{"messages",Json::array({{{"role","user"},{"content","hello"}}})}};
    auto parse=[](const Json& j,bool count=false){return portal::parse(j,Protocol::messages,"test",count);};
    auto invalid=[&](const Json& j,const std::string& path,bool count=false,int status=400) {
        try {parse(j,count);CHECK(false);}catch(const portal::Failure& f) {
            CHECK(f.status==status && f.param==path);CHECK(ends_with(f.message(),": "+path));
        }
    };
    auto j=base;auto r=parse(j);CHECK(r.overrides.max_tokens==32 && !r.stream);
    CHECK(r.input.messages.size()==1 && r.input.messages[0].content=="hello");
    j.erase("max_tokens");invalid(j,"max_tokens");CHECK(!parse(j,true).overrides.max_tokens);
    j["stream"]=false;invalid(j,"stream",true);
    j=base;j.erase("model");invalid(j,"model");j["model"]="other";invalid(j,"model",false,404);
    j=base;j["messages"]=Json::array();invalid(j,"messages");
    j=base;j["messages"][0]["role"]="assistant";invalid(j,"messages");
    j["messages"][0]["role"]="system";invalid(j,"messages.role");
    j=base;j["temperature"]=0.25;j["top_p"]=0.9;j["top_k"]=4;j["stream"]=true;j["stop_sequences"]=Json::array({"END","STOP"});
    j["metadata"]={{"user_id","u"}};j["service_tier"]="auto";j["cache_control"]={{"type","ephemeral"}};
    r=parse(j);CHECK(r.overrides.temperature==0.25 && r.overrides.top_p==0.9 && r.overrides.top_k==4 && r.stream);
    CHECK(*r.overrides.stop==std::vector<std::string>({"END","STOP"}));
    for(const auto* field:{"temperature","top_p","top_k","stop_sequences","max_tokens","service_tier","metadata"}) {j=base;j[field]=false;invalid(j,field);}
    j=base;j["stream"]=Json::array();invalid(j,"stream");
    for(const auto* field:{"output_config","container","inference_geo","speed","diagnostics","user_profile_id","workspace_id","mcp_servers","unknown"}) {j=base;j[field]=Json::object();invalid(j,field);}
    j=base;j["metadata"]={{"other",1}};invalid(j,"metadata.other");
    for(const auto* mode:{"enabled","adaptive","disabled"}) {
        j=base;j["thinking"]={{"type",mode}};r=parse(j);CHECK(r.overrides.thinking==(std::string(mode)!="disabled"));
    }
    j=base;j["thinking"]={{"type","enabled"},{"budget_tokens",10}};CHECK(*parse(j).overrides.thinking);
    for(const auto& bad:Json::array({0,-1,1.5,"10",true})) {j["thinking"]["budget_tokens"]=bad;invalid(j,"thinking.budget_tokens");}
    for(const auto* field:{"between_tools","display"}) {j=base;j["thinking"]={{"type","enabled"},{field,true}};invalid(j,std::string("thinking.")+field);}
    j=base;j["thinking"]={{"type","other"}};invalid(j,"thinking.type");
    for(const auto* mode:{"auto","none"}) {
        j=base;j["tool_choice"]={{"type",mode},{"disable_parallel_tool_use",true}};r=parse(j);
        CHECK(!r.input.parallel_tool_calls);CHECK(r.input.tool_choice.mode==(std::string(mode)=="none"?edge0::ToolChoiceMode::none:edge0::ToolChoiceMode::automatic));
    }
    for(const auto* mode:{"any","tool"}) {j=base;j["tool_choice"]={{"type",mode}};invalid(j,"tool_choice.type");}
    j=base;j["tools"]=Json::array({{{"name","f"},{"description","desc"},{"input_schema",{{"type","object"}}},{"type","custom"},{"strict",false},{"cache_control",Json::object()}}});
    CHECK(Json::parse(parse(j).input.tools[0].json)==Json({{"type","function"},{"function",{{"name","f"},{"description","desc"},{"parameters",{{"type","object"}}}}}}));
    auto tools=j;j["tools"][0]["strict"]=true;invalid(j,"tools.strict");
    j=tools;j["tools"][0]["type"]="web_search";invalid(j,"tools.type");
    j=tools;j["tools"][0]["input_schema"]=Json::array();invalid(j,"tools.input_schema");
    j=tools;j["tools"][0].erase("input_schema");invalid(j,"tools.input_schema");
    j=base;j["system"]="system";CHECK(parse(j).input.messages[0].content=="system");
    j["system"]=Json::array({{{"type","text"},{"text","a"},{"cache_control",Json::object()}},{{"type","text"},{"text","b"}}});
    CHECK(parse(j).input.messages[0].content=="ab");
    j["messages"]=Json::array({
        {{"role","assistant"},{"content",Json::array({{{"type","thinking"},{"thinking","think"},{"signature","ignored"}},{{"type","text"},{"text","answer"}},
          {{"type","tool_use"},{"id","one"},{"name","f"},{"input",{{"z",1},{"a",2}}}}})}},
        {{"role","user"},{"content",Json::array({{{"type","text"},{"text","after"}},{{"type","tool_result"},{"tool_use_id","one"},{"content",Json::array({{{"type","text"},{"text","result"}}})},{"is_error",true}},
          {{"type","tool_result"},{"tool_use_id","two"},{"content","second"}}})}}});
    r=parse(j);CHECK(r.input.messages.size()==5);CHECK(r.input.messages[1].reasoning_content=="think");
    CHECK(r.input.messages[1].content=="answer" && r.input.messages[1].tool_calls[0].arguments_json=="{\"z\":1,\"a\":2}");
    CHECK(r.input.messages[2].role=="tool" && r.input.messages[2].tool_call_id=="one" && r.input.messages[2].content=="result");
    CHECK(r.input.messages[3].tool_call_id=="two" && r.input.messages[4].content=="after");
    j["messages"][1]["content"].erase(0);CHECK(parse(j).input.messages.size()==4);
    j["messages"][0]["content"][2]["input"]=Json::array();invalid(j,"messages.content.input");
    for(const auto* type:{"image","document","search_result","redacted_thinking","server_tool_use","thinking","tool_use"}) {
        j=base;j["messages"][0]["content"]=Json::array({{{"type",type}}});invalid(j,"messages.content");
    }
    j=base;j["messages"][0]["content"]=Json::array({{{"type","text"},{"text","x"},{"unknown",true}}});invalid(j,"messages.content.unknown");
}

void messages_encoding_contract() {
    portal::Request request;request.protocol=Protocol::messages;
    auto payload=[](const std::string& wire){return Json::parse(wire.substr(wire.find("data: ")+6));};
    for(bool rejected:{false,true}) {
        portal::Encoder encoder(request,"test","1");
        auto wire=encoder.begin(12);auto start=payload(wire[0]);
        CHECK(start["type"]=="message_start" && start["message"]["content"].empty());
        CHECK(start["message"]["usage"]["input_tokens"]==12 && start["message"]["usage"]["cache_read_input_tokens"].is_null());
        CHECK(start["message"]["usage"]["output_tokens"]==0 && start["message"]["stop_reason"].is_null());
        edge0::GenerationProgress progress;progress.input_tokens=12;progress.reused_tokens=5;
        portal::Encoder counted(request,"test","1");auto counted_start=payload(counted.begin(99,progress)[0]);
        CHECK(counted_start["message"]["usage"]["input_tokens"]==7 && counted_start["message"]["usage"]["cache_read_input_tokens"]==5);
        auto send=[&](const edge0::OutputEvent& e){auto records=encoder.accept(e);wire.insert(wire.end(),records.begin(),records.end());};
        edge0::OutputEvent e;e.kind=edge0::OutputKind::reasoning;e.text="think";send(e);e.text=" more";send(e);
        e.kind=edge0::OutputKind::answer;e.text="answer";send(e);e.text="!";send(e);
        edge0::GenerationResult r;r.configuration.thinking=true;r.stop_reason=edge0::StopReason::tool_calls;
        r.input.resize(12);r.reused_tokens=5;r.generated_count=8;r.reasoning_tokens=3;r.request_summary.first_token_seconds=0.2;
        for(std::size_t i=0;i<2;++i) {
            e.kind=edge0::OutputKind::tool_call_begin;e.index=i+7;e.tool_call={"call_"+std::to_string(i),"f",""};send(e);
            e.kind=edge0::OutputKind::tool_call_arguments;e.text="{\"a\":";send(e);
            if(rejected && i==1) {e.kind=edge0::OutputKind::tool_call_rejected;e.rejected={e.index,e.text,"truncated"};send(e);r.rejected_tool_calls.push_back(e.rejected);}
            else {e.text="1}";send(e);e.kind=edge0::OutputKind::tool_call_end;send(e);r.tool_calls.push_back({e.tool_call.id,"f","{\"a\":1}"});}
        }
        const auto tail=encoder.finish(r);wire.insert(wire.end(),tail.begin(),tail.end());const auto full=encoder.complete(r);
        CHECK(full["type"]=="message" && full["role"]=="assistant" && full["model"]=="test");
        CHECK(full["id"].get<std::string>().find("msg_")==0 && full["stop_reason"]=="tool_use" && full["stop_sequence"].is_null());
        CHECK(full["content"].size()==(rejected?3:4));
        CHECK(full["usage"]==Json({{"input_tokens",7},{"cache_read_input_tokens",5},{"cache_creation_input_tokens",nullptr},{"output_tokens",8},{"output_tokens_details",{{"thinking_tokens",3}}}}));
        CHECK(full["metrics"]["first_token_seconds"]==0.2);
        Json blocks=Json::array();std::vector<std::string> types;int active=-1;std::size_t stops=0;
        for(const auto& record:wire) {
            const auto v=payload(record);const auto type=v["type"].get<std::string>();types.push_back(type);
            CHECK(record.find("event: "+type+"\n")==0 && !v.contains("sequence_number"));
            if(type=="content_block_start") {CHECK(active==-1 && v["index"]==blocks.size());active=v["index"].get<int>();blocks.push_back(v["content_block"]);}
            if(type=="content_block_delta") {
                CHECK(active>=0 && v["index"]==active);const auto& d=v["delta"];auto& b=blocks[active];
                if(d["type"]=="text_delta")b["text"]=b["text"].get<std::string>()+d["text"].get<std::string>();
                else if(d["type"]=="thinking_delta")b["thinking"]=b["thinking"].get<std::string>()+d["thinking"].get<std::string>();
                else {CHECK(d["type"]=="input_json_delta");b["arguments"]=b.value("arguments",std::string{})+d["partial_json"].get<std::string>();}
            }
            if(type=="content_block_stop") {CHECK(active>=0 && v["index"]==active);active=-1;++stops;}
            if(type=="message_delta") {
                CHECK(active==-1 && v["usage"]==full["usage"] && v["metrics"]==full["metrics"]);
                CHECK(v["delta"]["stop_reason"]==full["stop_reason"] && v["delta"]["stop_sequence"].is_null());
                if(rejected)CHECK(v["rejected_tool_calls"]==full["rejected_tool_calls"]);
            }
        }
        CHECK(stops==4 && types.front()=="message_start" && types[types.size()-2]=="message_delta" && types.back()=="message_stop");
        if(rejected) {CHECK(full["rejected_tool_calls"].size()==1);blocks.erase(3);}
        for(auto& block:blocks)if(block["type"]=="tool_use") {block["input"]=Json::parse(block["arguments"].get<std::string>());block.erase("arguments");}
        CHECK(blocks==full["content"]);
        r.configuration.thinking=false;CHECK(encoder.complete(r)["content"][0]["type"]=="text");
        r.tool_calls.clear();
        for(const auto& pair:std::vector<std::pair<edge0::StopReason,std::string>>{{edge0::StopReason::end_of_generation,"end_turn"},{edge0::StopReason::length,"max_tokens"},{edge0::StopReason::capacity,"model_context_window_exceeded"},{edge0::StopReason::stop_string,"stop_sequence"},{edge0::StopReason::tool_calls,"tool_use"}}) {
            r.stop_reason=pair.first;r.matched_stop="END";const auto value=encoder.complete(r);CHECK(value["stop_reason"]==pair.second);
            CHECK(value["stop_sequence"]==(pair.first==edge0::StopReason::stop_string?Json("END"):Json(nullptr)));
        }
        for(auto stop:{edge0::StopReason::cancelled,edge0::StopReason::failed}) {
            r.stop_reason=stop;auto fail=encoder.finish(r);CHECK(fail.size()==1 && payload(fail[0])["type"]=="error");
        }
    }
    for(const auto& pair:std::vector<std::pair<int,std::string>>{{400,"invalid_request_error"},{401,"authentication_error"},{404,"not_found_error"},{413,"request_too_large"},{429,"rate_limit_error"},{500,"api_error"},{503,"overloaded_error"}}) {
        portal::Failure f(pair.first,"bad","code","field");auto b=f.body(Protocol::messages);
        CHECK(b==Json({{"type","error"},{"error",{{"type",pair.second},{"message","bad: field"}}}}));
        portal::Encoder encoder(request,"test","1");CHECK(payload(encoder.fail(f)[0])==b);
    }
}
int main(int argc,char** argv) {
    metrics_contract();
    messages_parse_contract();
    messages_encoding_contract();
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
