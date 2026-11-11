#include "serve_protocol.h"
#include <chrono>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
namespace portal {
Failure::Failure(int s, std::string message, std::string c, std::string p)
    : std::runtime_error(std::move(message)), status(s), code(std::move(c)), param(std::move(p)) {}
std::string Failure::message() const { return param.empty() ? std::string(what()) : std::string(what()) + ": " + param; }
Json Failure::body() const {
    return {{"error", {{"message", message()}, {"type", status == 429 ? "server_busy" : status >= 500 ? "server_error" : "invalid_request_error"},
                       {"param", param.empty() ? Json(nullptr) : Json(param)}, {"code", code}}}};
}
namespace {
[[noreturn]] void bad(const std::string& p, const std::string& message = "Invalid parameter", const std::string& code = "invalid_parameter") {
    throw Failure(400, message, code, p);
}
void keys(const Json& j, const std::set<std::string>& allowed, const std::string& path) {
    if (!j.is_object()) bad(path, "Expected an object");
    for (auto it = j.begin(); it != j.end(); ++it) if (!allowed.count(it.key())) bad(path.empty() ? it.key() : path + "." + it.key(), "Unknown field", "unsupported_parameter");
}
std::string str(const Json& j, const std::string& p) { if (!j.is_string()) bad(p, "Expected a string"); return j.get<std::string>(); }
bool boolean(const Json& j, const std::string& p) { if (!j.is_boolean()) bad(p, "Expected a boolean"); return j.get<bool>(); }
std::int64_t integer(const Json& j, const std::string& p) {
    if (!j.is_number_integer() || (j.is_number_unsigned() && j.get<std::uint64_t>() > std::uint64_t(INT64_MAX))) bad(p, "Expected a signed integer");
    return j.get<std::int64_t>();
}
double number(const Json& j, const std::string& p) { if (!j.is_number()) bad(p, "Expected a number"); return j.get<double>(); }
const Json& required(const Json& j, const std::string& key) { if (!j.contains(key)) bad(key, "Missing field"); return j.at(key); }
std::string content(const Json& j, const std::set<std::string>& types, const std::string& p) {
    if (j.is_string()) return j.get<std::string>();
    if (!j.is_array()) bad(p, "Expected text or text parts");
    std::string result;
    for (const auto& part : j) {
        keys(part, {"type", "text", "annotations", "logprobs"}, p);
        auto type = str(required(part,"type"), p);
        if (!types.count(type)) bad(p, "Unsupported content type", "unsupported_value");
        if (type != "output_text" && (part.contains("annotations") || part.contains("logprobs"))) bad(p);
        result += str(required(part,"text"), p);
    }
    return result;
}
bool thinking(const Json& j, const std::string& p) {
    const auto value = str(j,p);
    if (value == "none") return false;
    if (value == "minimal" || value == "low" || value == "medium" || value == "high") return true;
    bad(p,"Unsupported reasoning effort","unsupported_value");
}
void tool_function(const Json& j, const std::string& p) {
    keys(j, {"name", "description", "parameters", "strict"}, p);
    str(required(j,"name"),p + ".name");
    if (j.contains("description")) str(j["description"],p);
    if (j.contains("parameters") && !j["parameters"].is_object()) bad(p);
    if (j.contains("strict") && boolean(j["strict"],p)) bad(p + ".strict", "Strict tools require constrained decoding", "unsupported_value");
}
Json usage(const edge0::GenerationResult& r, bool chat) {
    return {{chat ? "prompt_tokens" : "input_tokens", r.input.size()}, {chat ? "completion_tokens" : "output_tokens", r.generated_count},
            {"total_tokens", r.input.size()+r.generated_count},
            {chat ? "prompt_tokens_details" : "input_tokens_details", {{"cached_tokens",r.reused_tokens}}},
            {chat ? "completion_tokens_details" : "output_tokens_details", {{"reasoning_tokens",r.reasoning_tokens}}}};
}
Json metrics(const edge0::GenerationResult& r, const edge0::MemorySnapshot& memory) {
    Json value=Json::object();
    if(r.request_summary.first_token_seconds)value["first_token_seconds"]=*r.request_summary.first_token_seconds;
    if(r.request_summary.decode_tokens_per_second)value["decode_tokens_per_second"]=*r.request_summary.decode_tokens_per_second;
    const auto peak=[](const edge0::MemoryMetric& metric) -> std::optional<std::uint64_t> {
        return metric.values?metric.values->peak_bytes:std::nullopt;
    };
    if(auto bytes=peak(memory.physical)) {value["memory_peak_bytes"]=*bytes;value["memory_peak_metric"]="physical";}
    else if(auto bytes=peak(memory.rss)) {value["memory_peak_bytes"]=*bytes;value["memory_peak_metric"]="resident";}
    return value;
}
std::string reason(const edge0::GenerationResult& r) {
    if (r.stop_reason == edge0::StopReason::length || r.stop_reason == edge0::StopReason::capacity) return "length";
    if (r.stop_reason == edge0::StopReason::tool_calls || (r.stop_reason == edge0::StopReason::end_of_generation && !r.tool_calls.empty())) return "tool_calls";
    return "stop";
}
Json rejected(const edge0::GenerationResult& r) {
    Json result=Json::array();
    for(const auto& c:r.rejected_tool_calls) result.push_back({{"index",c.index},{"reason",c.reason},{"text",c.text}});
    return result;
}
// JSON's replacement decoder preserves the original bytes separately below.
Json candidate(const edge0::LogprobCandidate& p) {
    Json bytes=Json::array(); for(unsigned char b:p.bytes) bytes.push_back(b);
    auto token=Json::parse(Json(p.bytes).dump(-1,' ',false,Json::error_handler_t::replace));
    return {{"token",token},{"logprob",p.logprob},{"bytes",bytes}};
}
Json probability(const edge0::TokenLogprob& p) {
    auto value=candidate({p.token,p.bytes,p.logprob});value["top_logprobs"]=Json::array();
    for(const auto& top:p.top) value["top_logprobs"].push_back(candidate(top));
    return value;
}
}
Request parse(const Json& j, Protocol protocol, const std::string& alias) {
    const bool chat=protocol==Protocol::chat;
    std::set<std::string> allowed={"model","tools","tool_choice","parallel_tool_calls","stream","temperature","top_p","top_logprobs","store","metadata","user","service_tier","prompt_cache_key","promptCacheKey","safety_identifier"};
    const std::set<std::string> extra=chat ? std::set<std::string>{"messages","stream_options","seed","stop","presence_penalty","frequency_penalty","max_tokens","max_completion_tokens","logit_bias","logprobs","n","response_format","reasoning_effort","chat_template_kwargs","top_k","repetition_penalty"}
        : std::set<std::string>{"instructions","input","max_output_tokens","include","text","reasoning","truncation","background"};
    allowed.insert(extra.begin(),extra.end()); keys(j,allowed,"");
    if(str(required(j,"model"),"model")!=alias) throw Failure(404,"Unknown model","model_not_found","model");
    Request r; r.protocol=protocol;
    if(j.contains("stream")) r.stream=boolean(j["stream"],"stream");
    if(j.contains("parallel_tool_calls")) r.input.parallel_tool_calls=boolean(j["parallel_tool_calls"],"parallel_tool_calls");
    if(j.contains("tool_choice")) {
        const auto choice=str(j["tool_choice"],"tool_choice");
        if(choice!="auto" && choice!="none") bad("tool_choice","Unsupported tool choice","unsupported_value");
        r.input.tool_choice.mode=choice=="none" ? edge0::ToolChoiceMode::none : edge0::ToolChoiceMode::automatic;
    }
    if(j.contains("temperature")) r.overrides.temperature=number(j["temperature"],"temperature");
    if(j.contains("top_p")) r.overrides.top_p=number(j["top_p"],"top_p");
    if(j.contains("top_logprobs")) r.overrides.top_logprobs=integer(j["top_logprobs"],"top_logprobs");
    if(j.contains("store")) boolean(j["store"],"store");
    for(const auto* field:{"user","service_tier","prompt_cache_key","promptCacheKey","safety_identifier"}) if(j.contains(field)) str(j[field],field);
    if(j.contains("metadata") && !j["metadata"].is_null() && !j["metadata"].is_object()) bad("metadata");
    if(j.contains("tools")) {
        if(!j["tools"].is_array()) bad("tools");
        for(const auto& tool:j["tools"]) {
            if(chat) {
                keys(tool,{"type","function"},"tools");
                if(str(required(tool,"type"),"tools.type")!="function") bad("tools.type");
                tool_function(required(tool,"function"),"tools.function"); r.input.tools.push_back({tool.dump()});
            } else {
                keys(tool,{"type","name","description","parameters","strict"},"tools");
                if(str(required(tool,"type"),"tools.type")!="function") bad("tools.type");
                Json f=tool;f.erase("type");tool_function(f,"tools");
                Json ordered={{"name",f["name"]}};
                for(const auto* key:{"description","parameters"}) if(f.contains(key)) ordered[key]=f[key];
                r.input.tools.push_back({Json{{"type","function"},{"function",ordered}}.dump()});
            }
        }
    }
    if(chat) {
        const auto& messages=required(j,"messages"); if(!messages.is_array() || messages.empty()) bad("messages");
        for(const auto& m:messages) {
            keys(m,{"role","content","tool_calls","tool_call_id","reasoning_content"},"messages");
            edge0::TextMessage msg;msg.role=str(required(m,"role"),"messages.role");
            if(msg.role=="developer") msg.role="system";
            if(msg.role!="system" && msg.role!="user" && msg.role!="assistant" && msg.role!="tool") bad("messages.role");
            if(m.contains("content")) {
                const auto& c=m["content"];if(!(msg.role=="assistant" && c.is_null()))msg.content=content(c,{"text"},"messages.content");
            } else if(msg.role!="assistant" || !m.contains("tool_calls"))bad("messages.content","Missing content");
            if(m.contains("reasoning_content")) {if(msg.role!="assistant") bad("messages.reasoning_content");msg.reasoning_content=str(m["reasoning_content"],"messages.reasoning_content");}
            if(m.contains("tool_call_id")) {if(msg.role!="tool") bad("messages.tool_call_id");msg.tool_call_id=str(m["tool_call_id"],"messages.tool_call_id");}
            if(msg.role=="tool" && msg.tool_call_id.empty()) bad("messages.tool_call_id");
            if(m.contains("tool_calls")) {
                if(msg.role!="assistant" || !m["tool_calls"].is_array()) bad("messages.tool_calls");
                for(const auto& call:m["tool_calls"]) {
                    keys(call,{"id","type","function"},"messages.tool_calls");
                    if(str(required(call,"type"),"tool_calls.type")!="function") bad("tool_calls.type");
                    const auto& f=required(call,"function");keys(f,{"name","arguments"},"tool_calls.function");
                    msg.tool_calls.push_back({str(required(call,"id"),"tool_calls.id"),str(required(f,"name"),"tool_calls.name"),str(required(f,"arguments"),"tool_calls.arguments")});
                }
            }
            r.input.messages.push_back(std::move(msg));
        }
        if(j.contains("stream_options")) {keys(j["stream_options"],{"include_usage"},"stream_options");if(j["stream_options"].contains("include_usage"))r.include_usage=boolean(j["stream_options"]["include_usage"],"stream_options.include_usage");}
        if(j.contains("n") && integer(j["n"],"n")!=1) bad("n");
        if(j.contains("response_format")) {keys(j["response_format"],{"type"},"response_format");if(str(required(j["response_format"],"type"),"response_format.type")!="text")bad("response_format");}
        if(j.contains("max_tokens"))r.overrides.max_tokens=integer(j["max_tokens"],"max_tokens");
        if(j.contains("max_completion_tokens")) {auto n=integer(j["max_completion_tokens"],"max_completion_tokens");if(r.overrides.max_tokens && *r.overrides.max_tokens!=n)bad("max_completion_tokens");r.overrides.max_tokens=n;}
        if(j.contains("seed")) {if(!j["seed"].is_number_integer() || (!j["seed"].is_number_unsigned() && j["seed"].get<std::int64_t>()<0))bad("seed");r.overrides.seed=j["seed"].get<std::uint64_t>();}
        if(j.contains("stop")) {r.overrides.stop=std::vector<std::string>{};if(j["stop"].is_string()) r.overrides.stop->push_back(str(j["stop"],"stop"));else {if(!j["stop"].is_array())bad("stop");for(const auto& v:j["stop"])r.overrides.stop->push_back(str(v,"stop"));}}
        if(j.contains("presence_penalty"))r.overrides.presence_penalty=number(j["presence_penalty"],"presence_penalty");
        if(j.contains("frequency_penalty"))r.overrides.frequency_penalty=number(j["frequency_penalty"],"frequency_penalty");
        if(j.contains("repetition_penalty"))r.overrides.repetition_penalty=number(j["repetition_penalty"],"repetition_penalty");
        if(j.contains("top_k"))r.overrides.top_k=integer(j["top_k"],"top_k");
        if(j.contains("logprobs"))r.overrides.logprobs=boolean(j["logprobs"],"logprobs");
        if(j.contains("logit_bias")) {
            if(!j["logit_bias"].is_object())bad("logit_bias");r.overrides.logit_bias=std::map<std::int32_t,double>{};
            for(auto it=j["logit_bias"].begin();it!=j["logit_bias"].end();++it) {
                const auto& key=it.key(); if(key.empty() || key.find_first_not_of("0123456789")!=std::string::npos)bad("logit_bias");
                std::uint64_t n=0;for(char c:key) {n=n*10+unsigned(c-'0');if(n>INT32_MAX)bad("logit_bias");}
                (*r.overrides.logit_bias)[static_cast<std::int32_t>(n)]=number(it.value(),"logit_bias");
            }
        }
        if(j.contains("reasoning_effort"))r.overrides.thinking=thinking(j["reasoning_effort"],"reasoning_effort");
        if(j.contains("chat_template_kwargs")) {
            const auto& kw=j["chat_template_kwargs"];keys(kw,{"enable_thinking"},"chat_template_kwargs");
            if(kw.contains("enable_thinking")) {bool on=boolean(kw["enable_thinking"],"chat_template_kwargs.enable_thinking");if(r.overrides.thinking && *r.overrides.thinking!=on)bad("chat_template_kwargs","Conflicting thinking controls");r.overrides.thinking=on;}
        }
    } else {
        if(j.contains("background") && boolean(j["background"],"background"))bad("background");
        if(j.contains("truncation") && str(j["truncation"],"truncation")!="disabled")bad("truncation");
        if(j.contains("max_output_tokens"))r.overrides.max_tokens=integer(j["max_output_tokens"],"max_output_tokens");
        if(j.contains("include")) {if(!j["include"].is_array())bad("include");for(const auto& v:j["include"]) {if(str(v,"include")!="message.output_text.logprobs")bad("include");r.overrides.logprobs=true;}}
        if(j.contains("text")) {keys(j["text"],{"format"},"text");const auto& f=required(j["text"],"format");keys(f,{"type"},"text.format");if(str(required(f,"type"),"text.format.type")!="text")bad("text.format");}
        if(j.contains("reasoning")) {const auto& v=j["reasoning"];keys(v,{"effort","summary"},"reasoning");if(v.contains("effort"))r.overrides.thinking=thinking(v["effort"],"reasoning.effort");if(v.contains("summary") && !v["summary"].is_null() && str(v["summary"],"reasoning.summary")!="auto")bad("reasoning.summary");}
        if(j.contains("instructions")) {auto s=str(j["instructions"],"instructions");if(!s.empty())r.input.messages.push_back({"system",s});}
        const auto& input=required(j,"input");
        if(input.is_string()) r.input.messages.push_back({"user",input.get<std::string>()});
        else {
            if(!input.is_array())bad("input");
            bool group=false;
            auto assistant=[&]() -> edge0::TextMessage& {if(!group) {r.input.messages.push_back({"assistant",""});group=true;}return r.input.messages.back();};
            for(const auto& item:input) {
                if(!item.is_object())bad("input");const auto type=item.contains("type") ? str(item["type"],"input.type") : "message";
                if(type=="message") {
                    keys(item,{"type","role","content","id","status"},"input");auto role=str(required(item,"role"),"input.role");if(role=="developer")role="system";
                    auto text=content(required(item,"content"),{"input_text","output_text"},"input.content");
                    if(role=="assistant")assistant().content+=text;
                    else {if(role!="user" && role!="system")bad("input.role");group=false;r.input.messages.push_back({role,text});}
                } else if(type=="function_call") {
                    keys(item,{"type","id","call_id","name","arguments","status"},"input");
                    assistant().tool_calls.push_back({str(required(item,"call_id"),"input.call_id"),str(required(item,"name"),"input.name"),str(required(item,"arguments"),"input.arguments")});
                } else if(type=="function_call_output") {
                    keys(item,{"type","id","call_id","output","status"},"input");group=false;
                    edge0::TextMessage msg{"tool",content(required(item,"output"),{"input_text"},"input.output")};msg.tool_call_id=str(required(item,"call_id"),"input.call_id");r.input.messages.push_back(std::move(msg));
                } else if(type=="reasoning") {
                    keys(item,{"type","id","content","summary","status"},"input");
                    auto text=content(required(item,"content"),{"reasoning_text"},"input.content");auto& msg=assistant();if(!msg.reasoning_content)msg.reasoning_content="";*msg.reasoning_content+=text;
                } else bad("input.type","Unsupported input item","unsupported_value");
            }
        }
    }
    if(r.overrides.top_logprobs && !r.overrides.logprobs.value_or(false))bad("top_logprobs","Enable logprobs first");
    r.echo=j;return r;
}
Failure failure(const edge0::Error& e, bool shutdown) {
    if(shutdown)return Failure(503,"Server shutting down","shutting_down");
    switch(e.category) {
        case edge0::ErrorCategory::busy:return Failure(429,"Server busy","busy");
        case edge0::ErrorCategory::invalid_argument:return Failure(400,"Invalid generation input","invalid_argument","input");
        case edge0::ErrorCategory::unsupported:return Failure(400,"Unsupported input","unsupported_value","input");
        case edge0::ErrorCategory::capacity_exceeded:return Failure(400,"Input exceeds context capacity","context_length_exceeded","input");
        default:break;
    }
    const char* code="backend_error";
    switch(e.category) {
        case edge0::ErrorCategory::conflict:code="conflict";break;
        case edge0::ErrorCategory::runtime_ended:code="runtime_ended";break;
        case edge0::ErrorCategory::asset_invalid:code="asset_invalid";break;
        case edge0::ErrorCategory::incompatible:code="incompatible";break;
        case edge0::ErrorCategory::budget_exceeded:code="budget_exceeded";break;
        case edge0::ErrorCategory::session_invalid:code="session_invalid";break;
        case edge0::ErrorCategory::unloading:code="unloading";break;
        default:break;
    }
    return Failure(500,"Generation failed",code);
}
std::string identifier(const std::string& prefix) {
    std::random_device random;std::ostringstream out;out<<prefix<<std::hex<<std::setfill('0');
    for(int i=0;i<3;++i)out<<std::setw(8)<<static_cast<std::uint32_t>(random());return out.str();
}
Encoder::Encoder(const Request& r,std::string alias,std::string version)
    : request_(r),alias_(std::move(alias)),fingerprint_("edge0-infer-"+version),id_(identifier(r.protocol==Protocol::chat?"chatcmpl-":"resp_")),
      created_(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()) {}
Json Encoder::chunk(Json delta,Json finish,Json probs) const {
    return {{"id",id_},{"object","chat.completion.chunk"},{"created",created_},{"model",alias_},{"system_fingerprint",fingerprint_},
            {"choices",Json::array({{{"index",0},{"delta",delta},{"finish_reason",finish},{"logprobs",probs}}})}};
}
std::string Encoder::event(const std::string& type,Json value) {
    value["type"]=type;value["sequence_number"]=sequence_++;
    return "event: "+type+"\ndata: "+value.dump()+"\n\n";
}
Json Encoder::response(const edge0::GenerationResult* r,const std::string& status,const edge0::MemorySnapshot& memory) const {
    Json value={{"id",id_},{"object","response"},{"created_at",created_},{"model",alias_},{"status",status},
        {"output",output_},{"error",nullptr},{"incomplete_details",nullptr},{"usage",nullptr},{"store",false},
        {"parallel_tool_calls",request_.input.parallel_tool_calls},{"tools",request_.echo.value("tools",Json::array())},
        {"tool_choice",request_.echo.value("tool_choice",Json("auto"))},{"reasoning",request_.echo.value("reasoning",Json{{"effort",nullptr},{"summary",nullptr}})},
        {"text",{{"format",{{"type","text"}}}}}};
    if(r) {
        value["usage"]=usage(*r,false);value["metrics"]=metrics(*r,memory);value["temperature"]=r->configuration.temperature;value["top_p"]=r->configuration.top_p;value["max_output_tokens"]=r->configuration.max_tokens;
        if(status=="incomplete")value["incomplete_details"]={{"reason","max_output_tokens"}};
        if(!r->rejected_tool_calls.empty())value["rejected_tool_calls"]=rejected(*r);
    }
    return value;
}
std::vector<std::string> Encoder::begin() {
    if(request_.protocol==Protocol::chat)return {"data: "+chunk({{"role","assistant"},{"content",""}}).dump()+"\n\n"};
    return {event("response.created",{{"response",response(nullptr,"in_progress")}}),event("response.in_progress",{{"response",response(nullptr,"in_progress")}})};
}
void Encoder::close(std::vector<std::string>& records) {
    if(active_<0)return;
    auto& item=output_[active_];const auto type=item["type"].get<std::string>();
    Json common={{"output_index",active_},{"item_id",item["id"]}};
    if(type=="function_call") {
        common["arguments"]=item["arguments"];common["name"]=item["name"];
        records.push_back(event("response.function_call_arguments.done",common));
    } else {
        common["content_index"]=0;common["text"]=item["content"][0]["text"];
        if(type=="message") {common["logprobs"]=pending_probs_;for(const auto& p:pending_probs_)item["content"][0]["logprobs"].push_back(p);pending_probs_=Json::array();}
        records.push_back(event(type=="message"?"response.output_text.done":"response.reasoning_text.done",common));
        common.erase("text");common.erase("logprobs");common["part"]=item["content"][0];records.push_back(event("response.content_part.done",common));
    }
    if(item["status"]=="in_progress")item["status"]="completed";
    records.push_back(event("response.output_item.done",{{"output_index",active_},{"item",item}}));active_=-1;
}
void Encoder::open(std::vector<std::string>& records,const edge0::OutputEvent& e) {
    close(records);
    Json item;
    if(e.kind==edge0::OutputKind::tool_call_begin) item={{"type","function_call"},{"id",identifier("fc_")},{"call_id",e.tool_call.id},{"name",e.tool_call.name},{"arguments",""},{"status","in_progress"}};
    else if(e.kind==edge0::OutputKind::reasoning) item={{"type","reasoning"},{"id",identifier("rs_")},{"summary",Json::array()},{"content",Json::array()},{"status","in_progress"}};
    else item={{"type","message"},{"id",identifier("msg_")},{"role","assistant"},{"content",Json::array()},{"status","in_progress"}};
    active_=static_cast<int>(output_.size());output_.push_back(item);
    records.push_back(event("response.output_item.added",{{"output_index",active_},{"item",item}}));
    if(e.kind!=edge0::OutputKind::tool_call_begin) {
        Json part={{"type",e.kind==edge0::OutputKind::reasoning?"reasoning_text":"output_text"},{"text",""}};
        if(e.kind==edge0::OutputKind::answer) {part["annotations"]=Json::array();part["logprobs"]=Json::array();}
        output_[active_]["content"].push_back(part);
        records.push_back(event("response.content_part.added",{{"output_index",active_},{"item_id",item["id"]},{"content_index",0},{"part",part}}));
    }
}
std::vector<std::string> Encoder::accept(const edge0::OutputEvent& e) {
    using K=edge0::OutputKind;
    std::vector<std::string> records;
    if(e.kind==K::token_logprob) {if(e.logprob.section==K::answer)pending_probs_.push_back(probability(e.logprob));return records;}
    if(request_.protocol==Protocol::chat) {
        Json delta=Json::object(),probs=nullptr;
        if(e.kind==K::answer || e.kind==K::reasoning) {delta[e.kind==K::answer?"content":"reasoning_content"]=e.text;if(e.kind==K::answer && !pending_probs_.empty()){probs={{"content",pending_probs_}};pending_probs_=Json::array();}}
        else if(e.kind==K::tool_call_begin) {
            const auto index=call_indices_.size();call_indices_[e.index]=index;
            delta["tool_calls"]=Json::array({{{"index",index},{"id",e.tool_call.id},{"type","function"},{"function",{{"name",e.tool_call.name},{"arguments",""}}}}});
        } else if(e.kind==K::tool_call_arguments) {
            auto it=call_indices_.find(e.index);if(it==call_indices_.end())throw std::logic_error("Tool arguments without begin");
            delta["tool_calls"]=Json::array({{{"index",it->second},{"function",{{"arguments",e.text}}}}});
        } else return records;
        records.push_back("data: "+chunk(delta,nullptr,probs).dump()+"\n\n");return records;
    }
    if(e.kind==K::tool_call_rejected) {
        auto it=call_indices_.find(e.index);if(it!=call_indices_.end())output_[it->second]["status"]="incomplete";
        if(active_>=0 && it!=call_indices_.end() && std::size_t(active_)==it->second)close(records);
        return records;
    }
    if(e.kind==K::tool_call_end) {close(records);return records;}
    if(e.kind==K::tool_call_begin) {open(records,e);call_indices_[e.index]=std::size_t(active_);return records;}
    if(e.kind==K::tool_call_arguments) {
        auto it=call_indices_.find(e.index);if(it==call_indices_.end())throw std::logic_error("Tool arguments without begin");
        auto& item=output_[it->second];item["arguments"].get_ref<std::string&>().append(e.text);
        records.push_back(event("response.function_call_arguments.delta",{{"output_index",it->second},{"item_id",item["id"]},{"delta",e.text}}));return records;
    }
    if(e.kind==K::answer || e.kind==K::reasoning) {
        const std::string type=e.kind==K::answer?"message":"reasoning";
        if(active_<0 || output_[active_]["type"]!=type)open(records,e);
        auto& item=output_[active_];auto& part=item["content"][0];part["text"].get_ref<std::string&>().append(e.text);
        Json value={{"output_index",active_},{"item_id",item["id"]},{"content_index",0},{"delta",e.text}};
        if(e.kind==K::answer) {value["logprobs"]=pending_probs_;for(const auto& p:pending_probs_)part["logprobs"].push_back(p);pending_probs_=Json::array();}
        records.push_back(event(e.kind==K::answer?"response.output_text.delta":"response.reasoning_text.delta",value));
    }
    return records;
}
Json Encoder::complete(const edge0::GenerationResult& r,const edge0::MemorySnapshot& memory) const {
    if(request_.protocol==Protocol::responses)return response(&r,reason(r)=="length"?"incomplete":"completed",memory);
    Json message={{"role","assistant"},{"content",r.text.empty() && !r.tool_calls.empty()?Json(nullptr):Json(r.text)}};
    if(r.configuration.thinking)message["reasoning_content"]=r.reasoning;
    if(!r.tool_calls.empty()) {message["tool_calls"]=Json::array();for(const auto& c:r.tool_calls)message["tool_calls"].push_back({{"id",c.id},{"type","function"},{"function",{{"name",c.name},{"arguments",c.arguments_json}}}});}
    if(!r.rejected_tool_calls.empty())message["rejected_tool_calls"]=rejected(r);
    Json probs=nullptr;if(r.configuration.logprobs) {probs={{"content",Json::array()}};for(const auto& p:r.logprobs)if(p.section==edge0::OutputKind::answer)probs["content"].push_back(probability(p));}
    return {{"id",id_},{"object","chat.completion"},{"created",created_},{"model",alias_},{"system_fingerprint",fingerprint_},{"choices",Json::array({{{"index",0},{"message",message},{"finish_reason",reason(r)},{"logprobs",probs}}})},{"usage",usage(r,true)},{"metrics",metrics(r,memory)}};
}
std::vector<std::string> Encoder::finish(const edge0::GenerationResult& r,const edge0::MemorySnapshot& memory) {
    if(r.stop_reason==edge0::StopReason::failed || r.stop_reason==edge0::StopReason::cancelled)return fail(r.error?failure(*r.error):Failure(500,"Generation cancelled","cancelled"));
    if(request_.protocol==Protocol::chat) {
        auto last=chunk(Json::object(),reason(r),pending_probs_.empty()?Json(nullptr):Json{{"content",pending_probs_}});
        if(!r.rejected_tool_calls.empty())last["choices"][0]["rejected_tool_calls"]=rejected(r);
        std::vector<std::string> records={"data: "+last.dump()+"\n\n"};
        if(request_.include_usage) {auto u=chunk(Json::object());u["choices"]=Json::array();u["usage"]=usage(r,true);u["metrics"]=metrics(r,memory);records.push_back("data: "+u.dump()+"\n\n");}
        records.push_back("data: [DONE]\n\n");return records;
    }
    std::vector<std::string> records;close(records);
    records.push_back(event(reason(r)=="length"?"response.incomplete":"response.completed",{{"response",complete(r,memory)}}));return records;
}
std::vector<std::string> Encoder::fail(const Failure& error) {
    if(request_.protocol==Protocol::chat)return {"data: "+error.body().dump()+"\n\n"};
    auto value=response(nullptr,"failed");value["error"]={{"code",error.code},{"message",error.message()}};
    return {event("response.failed",{{"response",value}})};
}
}
