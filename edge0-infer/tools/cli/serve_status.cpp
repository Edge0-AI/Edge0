#include "serve_status.h"
#include <algorithm>
#include <array>
namespace portal {
namespace {
template<class T> Json optional(const std::optional<T>& value) { return value ? Json(*value) : Json(nullptr); }
const char* stop_name(edge0::StopReason value) {
    switch (value) {
    case edge0::StopReason::end_of_generation: return "end_of_generation";
    case edge0::StopReason::length: return "length";
    case edge0::StopReason::capacity: return "capacity";
    case edge0::StopReason::cancelled: return "cancelled";
    case edge0::StopReason::failed: return "failed";
    case edge0::StopReason::stop_string: return "stop_string";
    case edge0::StopReason::tool_calls: return "tool_calls";
    }
    return "unknown";
}
const char* reuse_name(edge0::ReuseMethod value) {
    switch (value) {
    case edge0::ReuseMethod::first: return "first";
    case edge0::ReuseMethod::keep: return "keep";
    case edge0::ReuseMethod::truncate: return "truncate";
    case edge0::ReuseMethod::restore: return "restore";
    case edge0::ReuseMethod::cold: return "cold";
    }
    return "unknown";
}
Json expert_json(const edge0::ExpertActivity& e) { return {{"demand",e.demand},{"hits",e.hits},{"loads",e.loads},{"bytes",e.bytes}}; }
double seconds(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<double>(to-from).count();
}
}
RequestRecord completed_record(RequestRecord record, const edge0::GenerationResult& r) {
    record.entered = true; record.result_available = true; record.total_seconds = r.request_summary.total_seconds;
    record.stop_reason = stop_name(r.stop_reason); record.error_stage = static_cast<int>(r.error_stage);
    record.reuse = r.reuse_method; record.input_tokens = r.input.size(); record.reused_tokens = r.reused_tokens;
    record.computed_tokens = r.computed_tokens; record.generated_tokens = r.generated_count;
    record.checkpoint_restored = r.checkpoint_restored; record.summary = r.request_summary; record.experts = r.weight_activity;
    return record;
}
RequestRecord failed_record(RequestRecord record, int status, std::string code, std::string message, std::string param, bool stream_opened) {
    record.http_status = stream_opened ? 200 : status;
    record.code = std::move(code); record.message = std::move(message); record.param = std::move(param);
    return record;
}
Json record_json(const RequestRecord& r) {
    Json result{{"time",r.time},{"endpoint",r.endpoint},{"source",r.source},{"http_status",r.http_status},
        {"stream",r.stream},{"total_seconds",r.total_seconds}};
    if (!r.code.empty()) result["code"] = r.code;
    if (!r.message.empty()) result["message"] = r.message;
    if (!r.param.empty()) result["param"] = r.param;
    if (r.entered) {
        result["stop_reason"] = r.stop_reason; result["error_stage"] = r.error_stage;
        result["reuse_method"] = r.result_available ? Json(reuse_name(r.reuse)) : Json(nullptr); result["input_tokens"] = r.input_tokens;
        result["reused_tokens"] = r.result_available ? Json(r.reused_tokens) : Json(nullptr); result["computed_tokens"] = r.result_available ? Json(r.computed_tokens) : Json(nullptr);
        result["generated_tokens"] = r.result_available ? Json(r.generated_tokens) : Json(nullptr);
        result["first_token_seconds"] = optional(r.summary.first_token_seconds);
        result["decode_tokens_per_second"] = optional(r.summary.decode_tokens_per_second);
        result["prefill_tokens_per_second"] = r.summary.first_token_seconds && *r.summary.first_token_seconds > 0 ?
            Json(double(r.input_tokens-r.reused_tokens) / *r.summary.first_token_seconds) : Json(nullptr);
        result["decode_seconds"] = optional(r.summary.decode_seconds);
        result["decode_forwards"] = optional(r.summary.decode_forwards);
        result["checkpoint_restored"] = r.result_available ? Json(r.checkpoint_restored) : Json(nullptr);
        if (r.experts.applicable) { auto experts = r.experts.prefill; experts += r.experts.decode; result["experts"] = expert_json(experts); }
    }
    return result;
}
void accumulate(Totals& t, const RequestRecord& r) {
    if (!r.entered) {
        if (r.http_status == 401) ++t.unauthorized;
        else if (r.http_status == 429) ++t.rejected_busy;
        else ++t.rejected_invalid;
        return;
    }
    ++t.requests; t.input_tokens += r.input_tokens;
    if (!r.result_available) return;
    t.reused_tokens += r.reused_tokens;
    t.computed_tokens += r.computed_tokens; t.generated_tokens += r.generated_tokens;
    t.checkpoint_restores += r.checkpoint_restored;
    t.continued_turns += r.reuse == edge0::ReuseMethod::keep || r.reuse == edge0::ReuseMethod::truncate || r.reuse == edge0::ReuseMethod::restore;
    if (r.summary.first_token_seconds) {
        ++t.first_count; t.first_seconds += *r.summary.first_token_seconds;
        t.prefill_tokens += r.input_tokens-r.reused_tokens;
    }
    if (r.summary.decode_seconds && r.summary.decode_forwards) {
        t.decode_seconds += *r.summary.decode_seconds; t.decode_forwards += *r.summary.decode_forwards;
    }
    if (r.experts.applicable) { t.experts_applicable = true; t.experts += r.experts.prefill; t.experts += r.experts.decode; }
}
Json totals_json(const Totals& t) {
    Json result{{"requests",t.requests},{"rejected_busy",t.rejected_busy},{"rejected_invalid",t.rejected_invalid},{"unauthorized",t.unauthorized},
        {"input_tokens",t.input_tokens},{"reused_tokens",t.reused_tokens},{"computed_tokens",t.computed_tokens},{"generated_tokens",t.generated_tokens},
        {"checkpoint_restores",t.checkpoint_restores},{"continued_turns",t.continued_turns},{"decode_seconds",t.decode_seconds}};
    if (t.first_count) result["first_token_seconds_mean"] = t.first_seconds/t.first_count;
    if (t.first_seconds > 0) result["prefill_tokens_per_second"] = t.prefill_tokens/t.first_seconds;
    if (t.decode_seconds > 0) result["decode_tokens_per_second"] = t.decode_forwards/t.decode_seconds;
    if (t.experts_applicable) result["experts"] = expert_json(t.experts);
    return result;
}
void History::add(const RequestRecord& r) {
    accumulate(totals,r); recent.push_front(r); if (recent.size() > 100) recent.pop_back();
    if (r.entered) last = r;
}
bool loopback(const std::string& source) {
    if (source == "::1" || source == "0:0:0:0:0:0:0:1") return true;
    std::string address = source;
    if (address.compare(0,7,"::ffff:") == 0 || address.compare(0,7,"::FFFF:") == 0) address.erase(0,7);
    std::array<unsigned,4> octets{}; std::size_t begin = 0;
    for (std::size_t i=0;i<4;++i) {
        const auto end = address.find('.',begin);
        const auto finish = end == std::string::npos ? address.size() : end;
        if ((i<3) != (end != std::string::npos) || finish == begin) return false;
        if (finish-begin > 3) return false;
        for (auto j=begin;j<finish;++j) {
            if (address[j] < '0' || address[j] > '9') return false;
            octets[i] = octets[i]*10 + unsigned(address[j]-'0');
        }
        if (octets[i] > 255) return false;
        begin = finish+1;
    }
    return octets[0] == 127;
}
bool authorized(const std::string& source, const std::string& token, const std::string& authorization, const std::string& api_key) {
    if (loopback(source)) return true;
    if (token.empty()) return false;
    auto equal = [](const std::string& expected, const std::string& actual) {
        std::size_t difference = expected.size() ^ actual.size();
        for (std::size_t i=0;i<std::max(expected.size(),actual.size());++i)
            difference |= (i<expected.size()?static_cast<unsigned char>(expected[i]):0) ^
                          (i<actual.size()?static_cast<unsigned char>(actual[i]):0);
        return difference == 0;
    };
    const bool bearer = equal("Bearer " + token, authorization);
    const bool key = equal(token, api_key);
    return bearer | key;
}

Json memory_json(const edge0::MemorySnapshot& m) {
    Json purposes = Json::object(), devices = Json::object();
    for (const auto& entry : m.ledger.by_purpose) purposes[edge0::memory_purpose_name(entry.first)] = entry.second;
    for (const auto& entry : m.ledger.by_device) devices[entry.first] = {{"current_bytes",entry.second.current_bytes},{"peak_bytes",entry.second.peak_bytes},
        {"memory",entry.second.memory == edge0::MemoryLocation::system ? "system" : "dedicated"}};
    auto metric = [](const edge0::MemoryMetric& item) {
        Json values = nullptr;
        if (item.values) values = {{"current_bytes",optional(item.values->current_bytes)},{"peak_bytes",optional(item.values->peak_bytes)},
            {"current_reason",item.values->current_reason},{"peak_reason",item.values->peak_reason}};
        return Json{{"scope","process_lifetime"},{"values",values},{"reason",item.reason},{"source",item.source}};
    };
    return {{"ledger",{{"limit_bytes",m.ledger.limit_bytes},{"current_bytes",m.ledger.current_bytes},{"peak_bytes",m.ledger.peak_bytes},
        {"scope","runtime_lifetime"},{"by_purpose",purposes},{"by_device",devices}}},{"resident",metric(m.rss)},{"physical",metric(m.physical)}};
}
Json system_json(const edge0::SystemStatus& s) {
    auto value = [](const edge0::SystemValue& v) {return Json{{"bytes",optional(v.bytes)},{"source",v.source},{"reason",v.reason}};};
    auto limit = value(s.app_memory_limit);
    limit["current_physical_bytes"] = optional(s.app_memory_limit.current_physical_bytes);
    return {{"memory",{{"total",value(s.memory.total)},{"available",value(s.memory.available)}}},
        {"thermal",{{"state",optional(s.thermal.state)},{"source",s.thermal.source},{"reason",s.thermal.reason}}},
        {"app_memory_limit",limit}};
}
Json engine_json(const edge0::Capabilities& c) {
    auto state = [](edge0::TriState s) {return s==edge0::TriState::yes?"yes":s==edge0::TriState::no?"no":"unknown";};
    Json backends = Json::array(), devices = Json::array();
    for (const auto& b : c.backends) backends.push_back({{"name",b.name},{"compiled",state(b.compiled)},{"initializable",state(b.initializable)},{"verified",state(b.verified)},{"reason",b.reason}});
    for (const auto& d : c.devices) devices.push_back({{"name",d.name},{"backend",d.backend},{"description",d.description},
        {"memory_total_bytes",d.memory_total_bytes},{"memory_free_bytes",d.memory_free_bytes}});
    return {{"version",c.engine_version},{"source_commit",c.source_commit},{"source_modified",state(c.source_modified)},
        {"source_reason",c.source_reason},{"ggml_origin",c.ggml_origin},{"ggml_version",c.ggml_version},{"backends",backends},{"devices",devices}};
}
Json model_json(const std::string& alias, const edge0::ModelDescription& m) {
    return {{"alias",alias},{"directory",m.directory},{"context_length",m.context},{"device",m.loading.device},{"prefill_device",m.loading.prefill_device}};
}
Json status_json(const StatusInput& s) {
    Json current = nullptr;
    const auto& p = s.progress;
    if (p.phase != edge0::GenerationPhase::idle) {
        current = {{"endpoint",s.endpoint},{"source",s.source},{"phase",p.phase==edge0::GenerationPhase::prefill?"prefill":"decode"},
            {"started_steady_seconds",std::chrono::duration<double>(p.started.time_since_epoch()).count()},
            {"elapsed_seconds",seconds(p.started,s.now)},{"input_tokens",p.input_tokens},{"reused_tokens",p.reused_tokens},
            {"computed_tokens",p.computed_tokens},{"pending_tokens",p.pending_tokens},{"generated_tokens",p.generated_tokens}};
        if (p.first_token) {
            const auto first = seconds(p.started,*p.first_token), decode = seconds(*p.first_token,s.now);
            current["first_token_seconds"] = first;
            current["prefill_tokens_per_second"] = first>0?Json(p.computed_tokens/first):Json(nullptr);
            current["decode_elapsed_seconds"] = decode;
            current["decode_tokens_per_second"] = decode>0?Json((p.generated_tokens? p.generated_tokens-1:0)/decode):Json(nullptr);
        }
    }
    Json recent = Json::array(), last = nullptr;
    for (const auto& r : s.history.recent) recent.push_back(record_json(r));
    if (s.history.last) {last = record_json(*s.history.last);last["ended_at"] = s.history.last->time;}
    return {{"instance",s.instance},{"listen",s.listen},{"engine",s.engine},{"model",s.model},{"current",current},
        {"last",last},{"totals",totals_json(s.history.totals)},{"memory",memory_json(s.memory)},{"system",system_json(s.system)},{"recent_requests",recent}};
}
}
