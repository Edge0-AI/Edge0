#include "edge0/diagnostics.h"
#include "edge0/engine.h"
#include "edge0/generation.h"
#include <iomanip>
#include <ostream>
namespace cli {
void json_string(std::ostream& out,const std::string& text) {
    out<<'"';
    for(unsigned char c:text) {
        if(c=='"'||c=='\\') out<<'\\'<<char(c);
        else if(c<32) { const char* hex="0123456789abcdef"; out<<"\\u00"<<hex[c>>4]<<hex[c&15]; }
        else out<<char(c);
    }
    out<<'"';
}
const char* configuration_source(edge0::ConfigurationSource source) {
    switch(source) {
    case edge0::ConfigurationSource::platform:return "platform";
    case edge0::ConfigurationSource::model:return "model";
    case edge0::ConfigurationSource::application:return "application";
    case edge0::ConfigurationSource::request:return "request";
    }
    return "unknown";
}
void load_configuration(std::ostream& out,const edge0::LoadConfiguration& c) {
    out<<"{\"directory\":";json_string(out,c.directory);
    out<<",\"device\":";json_string(out,c.device);
    out<<",\"prefill_device\":";json_string(out,c.prefill_device);
    out<<",\"stage_policy\":";json_string(out,c.stage_policy);
    out<<",\"context\":"<<c.context<<",\"prefill_chunk\":"<<c.prefill_chunk<<",\"model_options\":{";
    bool first=true;
    for(const auto& item:c.model_options) {
        out<<(first?"":",");first=false;json_string(out,item.first);out<<':';json_string(out,item.second);
    }
    out<<"},\"sources\":{";first=true;
    for(const auto& item:c.sources) {
        out<<(first?"":",");first=false;json_string(out,item.first);out<<':';json_string(out,configuration_source(item.second));
    }
    out<<"}}";
}
void memory_snapshot(std::ostream& out,const edge0::MemorySnapshot& m) {
    out<<"{\"ledger\":{\"limit_bytes\":"<<m.ledger.limit_bytes<<",\"current_bytes\":"<<m.ledger.current_bytes
       <<",\"peak_bytes\":"<<m.ledger.peak_bytes<<",\"scope\":\"runtime_lifetime\",\"by_purpose\":{";
    bool first = true;
    for (const auto& entry : m.ledger.by_purpose) {
        out << (first ? "" : ","); first = false;
        json_string(out, edge0::memory_purpose_name(entry.first)); out << ':' << entry.second;
    }
    out << "},\"by_device\":{"; first = true;
    for (const auto& entry : m.ledger.by_device) {
        out << (first ? "" : ","); first = false;
        json_string(out, entry.first);
        out << ":{\"current_bytes\":" << entry.second.current_bytes << ",\"peak_bytes\":" << entry.second.peak_bytes
            << ",\"memory\":\"" << (entry.second.memory == edge0::MemoryLocation::system ? "system" : "dedicated") << "\"}";
    }
    out << "}}";
    auto metric=[&](const char* name,const edge0::MemoryMetric& item) {
        out<<",\""<<name<<"\":{\"scope\":\"process_lifetime\",\"values\":";
        if(item.values) {
            auto number=[&](const auto& value) { if(value) out<<*value; else out<<"null"; };
            out<<"{\"current_bytes\":";number(item.values->current_bytes);
            out<<",\"peak_bytes\":";number(item.values->peak_bytes);
            out<<",\"current_reason\":";json_string(out,item.values->current_reason);
            out<<",\"peak_reason\":";json_string(out,item.values->peak_reason);out<<'}';
        }
        else out<<"null";
        out<<",\"reason\":";json_string(out,item.reason);
        out<<",\"source\":";json_string(out,item.source);out<<'}';
    };
    metric("rss",m.rss);metric("physical",m.physical);out<<'}';
}
void model_description(std::ostream& out,const edge0::ModelDescription& model) {
    out<<std::boolalpha<<std::setprecision(17)<<"{\"adapter\":";json_string(out,model.adapter);
    out<<",\"directory\":";json_string(out,model.directory);
    out<<",\"context\":"<<model.context<<",\"context_limit\":"<<model.context_limit
       <<",\"thinking\":{\"supported\":"<<model.thinking_supported<<",\"default\":"<<model.thinking_default
       <<",\"enabled_supported\":"<<model.thinking_enabled_supported<<"},\"tools_supported\":"<<model.tools_supported;
    auto recommendation=[&](const edge0::GenerationConfiguration& c) {
    out<<"{\"temperature\":"<<c.temperature<<",\"top_k\":"<<c.top_k<<",\"top_p\":"<<c.top_p
       <<",\"repetition_penalty\":"<<c.repetition_penalty<<",\"max_tokens\":"<<c.max_tokens<<",\"thinking\":"<<c.thinking
       <<",\"presence_penalty\":"<<c.presence_penalty<<",\"frequency_penalty\":"<<c.frequency_penalty
       <<",\"seed\":"<<(c.seed?std::to_string(*c.seed):"null")<<",\"sources\":{";
    bool first=true;
    for(const auto& item:c.sources) {
        out<<(first?"":",");first=false;json_string(out,item.first);out<<':';json_string(out,configuration_source(item.second));
    }
    out<<"}}";
    };
    out<<",\"recommended\":";recommendation(model.recommended);
    out<<",\"recommended_thinking\":";
    if(model.recommended_thinking) recommendation(*model.recommended_thinking);else out<<"null";
    out<<",\"loading\":";load_configuration(out,model.loading);
    out<<",\"assets\":[";
    bool first_asset=true;
    for(const auto& asset:model.assets) {
        out<<(first_asset?"":",");first_asset=false;
        out<<"{\"role\":";json_string(out,asset.role);out<<",\"filename\":";json_string(out,asset.filename);
        out<<",\"bytes\":"<<asset.bytes<<",\"sha256\":";json_string(out,asset.sha256);out<<'}';
    }
    out<<"],\"assets_verified\":"<<model.assets_verified<<'}';
}
void engine_identity(std::ostream& out,const edge0::Capabilities& caps) {
    const char* modified=caps.source_modified==edge0::TriState::yes?"yes":caps.source_modified==edge0::TriState::no?"no":"unknown";
    out<<"{\"version\":";json_string(out,caps.engine_version);
    out<<",\"source_commit\":";json_string(out,caps.source_commit);
    out<<",\"source_modified\":";json_string(out,modified);
    out<<",\"source_reason\":";json_string(out,caps.source_reason);
    out<<",\"ggml_origin\":";json_string(out,caps.ggml_origin);
    out<<",\"ggml_version\":";json_string(out,caps.ggml_version);out<<'}';
}
}

namespace cli {
void request_summary(std::ostream& out,const edge0::RequestSummary& s) {
    auto number=[&](const auto& value) {if(value) out<<*value; else out<<"null";};
    out<<"{\"first_token_seconds\":";number(s.first_token_seconds);
    out<<",\"first_token_reason\":";json_string(out,s.first_token_reason);
    out<<",\"decode_seconds\":";number(s.decode_seconds);
    out<<",\"decode_forwards\":";number(s.decode_forwards);
    out<<",\"decode_tokens_per_second\":";number(s.decode_tokens_per_second);
    out<<",\"decode_reason\":";json_string(out,s.decode_reason);
    out<<",\"prefill_device\":";json_string(out,s.prefill_device);
    out<<",\"handoffs\":"<<s.handoffs<<",\"handoff_bytes\":"<<s.handoff_bytes<<",\"handoff_seconds\":"<<s.handoff_seconds;
    out<<",\"total_seconds\":"<<s.total_seconds<<'}';
}
}
