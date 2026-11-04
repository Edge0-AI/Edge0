#include "edge0/weight_activity.h"
#include "edge0/diagnostics.h"
#include "edge0/generation.h"
#include <map>
#include <ostream>
#include <string>
namespace cli {
void memory_snapshot(std::ostream&,const edge0::MemorySnapshot&);
void request_summary(std::ostream&,const edge0::RequestSummary&);
void load_configuration(std::ostream&,const edge0::LoadConfiguration&);
void model_option(std::map<std::string,std::string>&,const std::string&);
void weight_activity(std::ostream&,const edge0::WeightActivity&);
void json_string(std::ostream&,const std::string&);
}
#include <iomanip>
#include <sstream>
namespace cli {
const char* stop(edge0::StopReason value) {
    switch(value) {
    case edge0::StopReason::length:return "length";
    case edge0::StopReason::capacity:return "capacity";
    case edge0::StopReason::cancelled:return "cancelled";
    case edge0::StopReason::end_of_generation:return "end_of_generation";
    case edge0::StopReason::failed:return "failed";
    case edge0::StopReason::stop_string:return "stop_string";
    case edge0::StopReason::tool_calls:return "tool_calls";
    }
    return "failed";
}
const char* source(edge0::ConfigurationSource value) {
    switch(value) {
    case edge0::ConfigurationSource::platform:return "platform";
    case edge0::ConfigurationSource::model:return "model";
    case edge0::ConfigurationSource::application:return "application";
    case edge0::ConfigurationSource::request:return "request";
    }
    return "platform";
}
std::string summary(const edge0::GenerationResult& r,const edge0::MemorySnapshot& memory,const edge0::LoadConfiguration& loading) {
    std::ostringstream out; out<<std::setprecision(17)<<std::boolalpha;
    out<<"{\"stop_reason\":\""<<stop(r.stop_reason)<<"\",\"input_count\":"<<r.input.size()<<",\"generated_count\":"<<r.generated_count
       <<",\"cancel_requested\":"<<r.cancel_requested<<",\"rolled_back_tokens\":"<<r.rolled_back_tokens
       <<",\"e\":"<<r.e<<",\"tail_token\":"<<(r.tail_token?std::to_string(*r.tail_token):"null")<<",\"tail_is_end\":"<<r.tail_is_end
       <<",\"checkpoint_saved\":"<<r.checkpoint_saved<<",\"state_valid\":"<<r.state_valid<<",\"input_complete\":"<<r.input_complete<<",\"cold\":"<<r.cold
       <<",\"reset_healthy_cache\":"<<r.reset_healthy_cache<<",\"reused_tokens\":"<<r.reused_tokens<<",\"computed_tokens\":"<<r.computed_tokens
       <<",\"decode_plan_builds\":"<<r.decode_plan_builds<<",\"decode_plan_reuses\":"<<r.decode_plan_reuses
       <<",\"budget_peak\":"<<r.budget_peak<<",\"budget_current\":"<<r.budget_current;
    out<<",\"reasoning\":";json_string(out,r.reasoning);
    out<<",\"reasoning_tokens\":"<<r.reasoning_tokens<<",\"matched_stop\":";
    if(r.matched_stop) json_string(out,*r.matched_stop);else out<<"null";
    auto candidate=[&](std::int32_t token,const std::string& bytes,double logprob) {
        out<<"\"token\":"<<token<<",\"bytes\":[";
        for(std::size_t i=0;i<bytes.size();++i) out<<(i?",":"")<<static_cast<unsigned>(static_cast<unsigned char>(bytes[i]));
        out<<"],\"logprob\":"<<logprob;
    };
    out<<",\"logprobs\":[";
    for(std::size_t i=0;i<r.logprobs.size();++i) {
        const auto& p=r.logprobs[i];out<<(i?",{":"{");candidate(p.token,p.bytes,p.logprob);
        const char* section=p.section==edge0::OutputKind::answer?"answer":p.section==edge0::OutputKind::reasoning?"reasoning":"tool_call";
        out<<",\"section\":\""<<section<<"\",\"top\":[";
        for(std::size_t j=0;j<p.top.size();++j) {out<<(j?",{":"{");candidate(p.top[j].token,p.top[j].bytes,p.top[j].logprob);out<<'}';}
        out<<"]}";
    }
    out<<']';
    const char* methods[]={"first","keep","truncate","restore","cold"};
    out<<",\"reuse_method\":\""<<methods[static_cast<unsigned>(r.reuse_method)]<<"\",\"common_prefix\":"<<r.common_prefix
       <<",\"checkpoint_restored\":"<<r.checkpoint_restored<<",\"checkpoint_boundary\":"<<(r.checkpoint_boundary?std::to_string(*r.checkpoint_boundary):"null")
       <<",\"checkpoint_save_seconds\":"<<r.checkpoint_save_seconds<<",\"configuration\":{";
    const auto& c=r.configuration;
    out<<"\"temperature\":"<<c.temperature<<",\"top_k\":"<<c.top_k<<",\"top_p\":"<<c.top_p<<",\"repetition_penalty\":"<<c.repetition_penalty
       <<",\"max_tokens\":"<<c.max_tokens<<",\"thinking\":"<<c.thinking<<",\"seed\":"<<(c.seed?std::to_string(*c.seed):"null")
       <<",\"presence_penalty\":"<<c.presence_penalty<<",\"frequency_penalty\":"<<c.frequency_penalty
       <<",\"logprobs\":"<<c.logprobs<<",\"top_logprobs\":"<<c.top_logprobs<<",\"stop\":[";
    for(std::size_t i=0;i<c.stop.size();++i) {if(i) out<<',';json_string(out,c.stop[i]);}
    out<<"]"
       <<",\"seed_used\":"<<(c.temperature > 0 && c.seed.has_value())<<"},\"sources\":{";
    bool first=true; for(const auto& item:c.sources) { out<<(first?"":",")<<'"'<<item.first<<"\":\""<<source(item.second)<<'"'; first=false; }
    out<<"},\"request_summary\":"; cli::request_summary(out,r.request_summary); out<<",\"weight_activity\":"; cli::weight_activity(out,r.weight_activity); out<<",\"memory_snapshot\":";cli::memory_snapshot(out,memory);out<<",\"loading\":";cli::load_configuration(out,loading);out<<"}\n"; return out.str();
}
}
