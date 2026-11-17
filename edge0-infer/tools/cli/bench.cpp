#include "runtime_options.h"
#include "edge0/generation.h"
#include "edge0/generation_check.h"
#include "edge0/diagnostics.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "json_value.h"
namespace cli {
const char* sampling_help();
bool sampling_option(const std::string&, const std::string&, edge0::GenerationOverrides&);
std::vector<std::int32_t> tokens(const std::string&,const std::string&);
void weight_activity(std::ostream&,const edge0::WeightActivity&);
void memory_snapshot(std::ostream&,const edge0::MemorySnapshot&);
void request_summary(std::ostream&,const edge0::RequestSummary&);
void model_description(std::ostream&,const edge0::ModelDescription&);
void engine_identity(std::ostream&,const edge0::Capabilities&);
void json_string(std::ostream&,const std::string&);
const char* stop(edge0::StopReason);
std::vector<edge0::TextMessage> completed_history(const std::vector<edge0::TextMessage>&, const std::vector<edge0::TextMessage>&, const edge0::GenerationResult&);
}
edge0::LogReceiver cli_log_receiver(bool);
namespace {
template<class T> T take(const edge0::Result<T>& value) {
    if(!value) throw std::runtime_error(value.error().message);
    return value.value();
}
std::string read(const std::string& path) {
    // Binary mode: keep prompt bytes verbatim (no CRLF translation) so a UTF-8
    // file tokenizes identically across platforms.
    std::ifstream stream(path,std::ios::binary);
    if(!stream) throw std::runtime_error("Cannot read: "+path);
    return {std::istreambuf_iterator<char>(stream),{}};
}
void distribution(std::ostream& out,std::vector<double> values) {
    if(values.empty()) throw std::runtime_error("Empty timing window");
    std::sort(values.begin(),values.end()); const auto n=values.size();
    out<<"{\"median\":"<<(values[(n-1)/2]+values[n/2])/2<<",\"p95\":"<<values[std::size_t(std::ceil(.95*static_cast<double>(n)))-1]
       <<",\"maximum\":"<<values.back()<<",\"samples\":"<<n<<"}";
}
void breakdown(std::ostream& out,const edge0::GenerationCheckResult& result,std::size_t warmup,std::size_t count,const edge0::TimingDistribution& fixed) {
    if(result.decode_timing.size()!=warmup+count) throw std::runtime_error("Incomplete decode timing window");
    out<<"{\"window_first_acceptance\":"<<warmup+1<<",\"window_last_acceptance\":"<<warmup+count+1<<",\"fields\":{";
    const std::vector<std::pair<const char*,double edge0::DecodeTiming::*>> fields={
        {"binding_seconds",&edge0::DecodeTiming::binding_seconds},{"prepare_seconds",&edge0::DecodeTiming::prepare_seconds},{"wait_seconds",&edge0::DecodeTiming::wait_seconds},
        {"submit_seconds",&edge0::DecodeTiming::submit_seconds},{"route_seconds",&edge0::DecodeTiming::route_seconds},
        {"total_seconds",&edge0::DecodeTiming::total_seconds}};
    bool first=true;
    for(const auto& field:fields) {
        if(!first){out<<",";} first=false; out<<"\""<<field.first<<"\":";
        std::vector<double> values;for(std::size_t i=warmup;i<warmup+count;++i) values.push_back(result.decode_timing[i].*field.second);
        distribution(out,values);
    }
    if (!result.sampling_seconds.empty()) {
        if (result.sampling_seconds.size() != warmup + count + 1) throw std::runtime_error("Incomplete sampling timing window");
        out << ",\"sampling_seconds\":";
        distribution(out, {result.sampling_seconds.begin() + warmup + 1, result.sampling_seconds.end()});
    }
    std::vector<double> rest,segments,lower,readback;
    for(std::size_t i=warmup;i<warmup+count;++i) {
        const auto& t=result.decode_timing[i];
        readback.push_back(static_cast<double>(result.decode_readback_bytes.at(i)));
        rest.push_back(t.total_seconds-t.binding_seconds-t.prepare_seconds-t.wait_seconds-t.submit_seconds-t.route_seconds);
        segments.push_back(static_cast<double>(t.segments)); lower.push_back(static_cast<double>(t.segments)*fixed.median_seconds);
    }
    out<<",\"other_seconds\":";distribution(out,rest);
    out<<",\"segments\":";distribution(out,segments);
    out<<",\"readback_bytes\":";distribution(out,readback);
    out<<"},\"prefill_block_seconds\":";distribution(out,result.prefill_seconds);
    out<<",\"submission_overhead\":{\"median_seconds\":"<<fixed.median_seconds<<",\"p95_seconds\":"<<fixed.p95_seconds
       <<",\"maximum_seconds\":"<<fixed.maximum_seconds<<",\"samples\":"<<fixed.samples<<"},\"estimated_lower_bound_seconds\":";
    distribution(out,lower);out<<",\"note\":\"Segments times minimal submission overhead is a lower bound, not stable tail latency.\"}";
}
std::int64_t integer(const std::string& text) {
    std::size_t used=0;const auto value=std::stoll(text,&used);
    if(used!=text.size()||value<0) throw std::runtime_error("Expected nonnegative integer");
    return value;
}

struct Turn {std::string user; std::optional<bool> thinking;};
std::vector<Turn> parse_turns(const std::string& text, std::optional<std::string>& system) {
    const auto document=cli::JsonParser(text).parse();
    if(!std::holds_alternative<cli::Json::Object>(document.value)||!document.has("turns")||
       !std::holds_alternative<cli::Json::Array>(document.at("turns").value)||document.at("turns").array().empty())
        throw std::runtime_error("Turns file needs a nonempty turns array");
    for(const auto& item:document.object()) if(item.first!="turns"&&item.first!="system") throw std::runtime_error("Unknown turns file field: "+item.first);
    if(document.has("system")) {
        if(!std::holds_alternative<std::string>(document.at("system").value)) throw std::runtime_error("system must be a string");
        system=document.at("system").string();
    }
    std::vector<Turn> turns;
    for(const auto& item:document.at("turns").array()) {
        if(!std::holds_alternative<cli::Json::Object>(item.value)||!item.has("user")||!std::holds_alternative<std::string>(item.at("user").value))
            throw std::runtime_error("Each turn needs a user string");
        Turn turn{item.at("user").string(),std::nullopt};
        for(const auto& field:item.object()) {
            if(field.first=="thinking") {
                if(!std::holds_alternative<bool>(field.second.value)) throw std::runtime_error("thinking must be boolean");
                turn.thinking=field.second.boolean();
            }
            else if(field.first!="user") throw std::runtime_error("Unknown turn field: "+field.first);
        }
        turns.push_back(std::move(turn));
    }
    return turns;
}
const char* reuse_name(edge0::ReuseMethod method) {
    switch(method) {
    case edge0::ReuseMethod::first:return "first";
    case edge0::ReuseMethod::keep:return "keep";
    case edge0::ReuseMethod::truncate:return "truncate";
    case edge0::ReuseMethod::restore:return "restore";
    case edge0::ReuseMethod::cold:return "cold";
    }
    return "unknown";
}
void token_array(std::ostream& out,const std::vector<std::int32_t>& tokens) {
    out<<'[';
    for(std::size_t i=0;i<tokens.size();++i) out<<(i?",":"")<<tokens[i];
    out<<']';
}
void optional_seconds(std::ostream& out,const std::optional<double>& value) {
    if(value) out<<*value; else out<<"null";
}
// Session-reuse cost per turn of one conversation, each next to a cold computation of the same input
// on a separate, short-lived session. The cold session never touches the conversation's own session.
void bench_turns(const edge0::RuntimeReference& runtime,const edge0::ModelReference& model,const edge0::Capabilities& capabilities,
                 const std::string& turns_text,std::int64_t max_tokens,edge0::GenerationOverrides overrides) {
    std::optional<std::string> system;
    const auto turns=parse_turns(turns_text,system);
    overrides.max_tokens=max_tokens;
    const auto configuration=take(edge0::resolve_generation_configuration(model,overrides));
    auto session=take(edge0::create_session(model));
    edge0::ChatInput chat;
    if(system) chat.messages.push_back({"system",*system});
    std::cout<<std::setprecision(17);
    for(std::size_t index=0;index<turns.size();++index) {
        auto turn_overrides=overrides;
        if(turns[index].thinking) turn_overrides.thinking=turns[index].thinking;
        const auto previous=chat.messages;
        chat.messages.push_back({"user",turns[index].user});
        auto generated=take(edge0::generate_chat_check(session,chat,configuration,turn_overrides,{}));
        const auto& result=generated.result;
        chat.messages=cli::completed_history(chat.messages,previous,result);
        std::cout<<"{\"type\":\"turn\",\"turn\":"<<index+1<<",\"thinking\":"<<std::boolalpha<<result.configuration.thinking
                 <<",\"input_count\":"<<result.input.size()<<",\"reuse_method\":\""<<reuse_name(result.reuse_method)
                 <<"\",\"reused_tokens\":"<<result.reused_tokens<<",\"computed_tokens\":"<<result.computed_tokens
                 <<",\"checkpoint_restored\":"<<result.checkpoint_restored<<",\"checkpoint_boundary\":";
        if(result.checkpoint_boundary) std::cout<<*result.checkpoint_boundary; else std::cout<<"null";
        std::cout<<",\"first_token_seconds\":";optional_seconds(std::cout,result.request_summary.first_token_seconds);
        std::cout<<",\"generated_count\":"<<result.generated_count<<",\"stop_reason\":\""<<cli::stop(result.stop_reason)
                 <<"\",\"request_summary\":";cli::request_summary(std::cout,result.request_summary);
        std::cout<<",\"input\":";token_array(std::cout,result.input);
        std::cout<<",\"generated\":";token_array(std::cout,result.generated);
        if(result.error) {std::cout<<"}\n"<<std::flush;throw std::runtime_error(result.error->message);}
        // Cold comparison: a new session, the complete input of this turn, one generated token.
        auto cold_session=take(edge0::create_session(model));
        edge0::GenerationOverrides cold_overrides=turn_overrides;cold_overrides.max_tokens=1;
        auto cold=take(edge0::generate_tokens_check(cold_session,result.input,configuration,cold_overrides,{}));
        const auto& cold_result=cold.result;
        std::cout<<",\"cold\":{\"reuse_method\":\""<<reuse_name(cold_result.reuse_method)<<"\",\"reused_tokens\":"<<cold_result.reused_tokens
                 <<",\"computed_tokens\":"<<cold_result.computed_tokens<<",\"first_token_seconds\":";
        optional_seconds(std::cout,cold_result.request_summary.first_token_seconds);
        std::cout<<",\"stop_reason\":\""<<cli::stop(cold_result.stop_reason)<<"\"}}\n"<<std::flush;
        take(edge0::release_session(cold_session));
        if(cold_result.error) throw std::runtime_error(cold_result.error->message);
    }
    const auto memory=take(edge0::memory_snapshot(runtime));
    const auto description=take(edge0::query_model(model));
    std::cout<<"{\"type\":\"summary\",\"turns\":"<<turns.size()<<",\"max_tokens\":"<<max_tokens<<",\"engine\":";
    cli::engine_identity(std::cout,capabilities);std::cout<<",\"runtime\":";cli::runtime_json(std::cout,runtime);
            std::cout<<",\"memory_snapshot\":";cli::memory_snapshot(std::cout,memory);
    std::cout<<",\"model\":";cli::model_description(std::cout,description);std::cout<<"}\n";
    take(edge0::release_session(session));
}
}
int bench_command(int argc,char** argv) {
    try {
        edge0::RuntimeOptions runtime_options;
        std::string directory,backend="cpu",user_file,token_file,route_trace,turns_file;
        std::optional<std::int64_t> max_tokens;
        std::int64_t warmup=10,ntok=200,runs=2;
        bool detailed=false,warmup_given=false,ntok_given=false,runs_given=false;
        edge0::LoadOverrides loading; loading.context=4096;
        edge0::GenerationOverrides overrides; overrides.temperature=0; overrides.presence_penalty=0;
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") { std::cout<<"Usage: edge0-infer bench --model <directory> --user-file <file> [--route-trace <file>] [--breakdown] [--warmup <n>] [--ntok <n>] [--runs <n>] [--device <cpu|metal|vulkan>] [--prefill-device <cpu|metal|vulkan>] [--stage-policy <name>] [--threads <n>] [--prefill-threads <n>] [--memory-budget <bytes>] [--model-option <name=value>] | edge0-infer bench --model <directory> --turns <json> [--max-tokens <n>] [--context <n>] [--device <cpu|metal|vulkan>] [--prefill-device <cpu|metal|vulkan>] [--stage-policy <name>] [--threads <n>] [--prefill-threads <n>] [--memory-budget <bytes>] [--model-option <name=value>]" << cli::sampling_help() << "\nDefaults to greedy generation.\n"; return 0; }
            if(arg=="--breakdown") { detailed=true; continue; }
            if(i+1>=argc) throw std::runtime_error("Missing bench argument: "+arg);
            const std::string value=argv[++i];
            if(cli::runtime_option(arg,value,runtime_options)) {}
            else if(cli::stage_option(arg,value,loading)) {}
            else if(arg=="--model") directory=value;
            else if(arg=="--device") backend=value;
            else if(arg=="--user-file") user_file=value;
            else if(arg=="--route-trace") route_trace=value;
            else if(arg=="--tokens") token_file=value;
            else if(arg=="--turns") turns_file=value;
            else if(arg=="--max-tokens") {max_tokens=integer(value); if(*max_tokens==0) throw std::runtime_error("max-tokens must be positive");}
            else if(arg=="--warmup") {warmup=integer(value);warmup_given=true;}
            else if(arg=="--ntok") {ntok=integer(value);ntok_given=true;}
            else if(arg=="--runs") {runs=integer(value);runs_given=true;}
            else if(arg=="--prefill-chunk") loading.prefill_chunk=integer(value);
            else if(arg=="--context") loading.context=integer(value);
            else if(cli::sampling_option(arg,value,overrides)) {}
            else if(arg=="--model-option") {
                const auto equals=value.find('=');
                if(equals==std::string::npos || equals==0 || equals+1==value.size()) throw std::runtime_error("Expected name=value");
                if(!loading.model_options.emplace(value.substr(0,equals),value.substr(equals+1)).second) throw std::runtime_error("Duplicate model option");
            }
            else throw std::runtime_error("Invalid bench argument: "+arg);
        }
        const bool turn_mode=!turns_file.empty();
        if(turn_mode&&(!user_file.empty()||!token_file.empty()||detailed||!route_trace.empty()||warmup_given||ntok_given||runs_given))
            throw std::runtime_error("--turns excludes --user-file, --tokens, --route-trace, --breakdown, --warmup, --ntok and --runs");
        if(!turn_mode&&max_tokens) throw std::runtime_error("--max-tokens is only valid with --turns");
        if(directory.empty()||(!turn_mode&&user_file.empty()==token_file.empty())||!cli::device_family(backend)||ntok==0||runs==0||
           warmup>=std::numeric_limits<std::int64_t>::max()-ntok)
            throw std::runtime_error("Usage: bench --model <directory> [--device <cpu|metal|vulkan>] [--prefill-device <cpu|metal|vulkan>] [--stage-policy <name>] [--threads <n>] [--prefill-threads <n>] [--memory-budget <bytes>] (--user-file <file> | --tokens <json>) [--warmup <n>] [--ntok <n>] [--runs <n>] [--breakdown] [--prefill-chunk <n>] [--model-option <name=value>]");
        const auto count=1+warmup+ntok;
        // Rendering is deliberately outside all measured generation windows.
        const auto input=turn_mode?std::vector<std::int32_t>{}:token_file.empty()?take(edge0::render_prompt(directory,{{"user",read(user_file)}},{false,{}})).tokens:cli::tokens(read(token_file),"input");
        const auto logger=cli_log_receiver(false);
        std::string device;
        const auto capabilities=edge0::query_capabilities(false,logger);
        for(const auto& item:capabilities.devices) if(item.backend==backend) {device=item.name;break;}
        if(device.empty()) {std::cerr<<"SKIP: requested device unavailable\n";return 77;}
        auto runtime=take(cli::open_runtime(runtime_options,device,logger,&loading));
        struct Close {edge0::RuntimeReference runtime;~Close(){edge0::finish_runtime_use(runtime);}} close{runtime};
        auto model=take(edge0::load_model(runtime,take(edge0::resolve_load_configuration(runtime,directory,loading))));
        if(turn_mode) {
            bench_turns(runtime,model,capabilities,read(turns_file),max_tokens.value_or(256),overrides);
            take(edge0::unload_model(model));return 0;
        }
        const auto description=take(edge0::query_model(model));
        if(input.empty()||input.size()>std::uint64_t(description.context)||std::uint64_t(count-1)>std::uint64_t(description.context)-input.size())
            throw std::runtime_error("Benchmark workload exceeds context capacity");
        overrides.max_tokens=count;
        const auto configuration=take(edge0::resolve_generation_configuration(model,overrides));
        edge0::GenerationCheckOptions options;options.ignore_end_of_generation=true;options.timing_marker=std::size_t(1+warmup);
        options.breakdown=detailed; options.route_trace=route_trace;
        if(!route_trace.empty()) { std::ofstream trace(route_trace); if(!trace) throw std::runtime_error("Cannot open route trace"); }
        const auto fixed=detailed?take(edge0::measure_submission_overhead(runtime,device)):edge0::TimingDistribution{};
        std::cout<<std::setprecision(17);
        double prefill_total=0,decode_total=0;
        for(std::int64_t run=1;run<=runs;++run) {
            auto session=take(edge0::create_session(model));
            auto checked=take(edge0::generate_tokens_check(session,input,configuration,{},options));
            const auto memory=take(edge0::memory_snapshot(runtime));
            const auto& result=checked.result;const auto& timing=result.request_summary;
            if(result.stop_reason!=edge0::StopReason::length||result.generated.size()!=std::uint64_t(count)||
               !timing.first_token_seconds||!timing.decode_seconds||!checked.marker_seconds)
                throw std::runtime_error(result.error?result.error->message:"Incomplete benchmark window");
            const double window=*timing.first_token_seconds+*timing.decode_seconds-*checked.marker_seconds;
            if(window<=0||*timing.first_token_seconds<=0) throw std::runtime_error("Clock resolution cannot measure benchmark window");
            prefill_total+=*timing.first_token_seconds;decode_total+=window;
            std::cout<<"{\"type\":\"run\",\"run\":"<<run<<",\"input_count\":"<<input.size()
                     <<",\"accepted_tokens\":"<<result.generated.size()<<",\"generated_count\":"<<result.generated_count
                     <<",\"stop_reason\":\"length\",\"warmup\":"<<warmup<<",\"decode_forwards\":"<<ntok
                     <<",\"marker_seconds\":"<<*checked.marker_seconds<<",\"decode_seconds\":"<<window
                     <<",\"decode_tokens_per_second\":"<<ntok/window<<",\"prefill_seconds\":"<<*timing.first_token_seconds
                     <<",\"prefill_tokens_per_second\":"<<input.size()/ *timing.first_token_seconds
                     <<",\"end_token_count\":"<<checked.end_token_count<<",\"first_end_token_position\":";
            if(checked.first_end_token_position) std::cout<<*checked.first_end_token_position;else std::cout<<"null";
            std::cout<<",\"request_summary\":";cli::request_summary(std::cout,timing);
            std::cout<<",\"runtime\":";cli::runtime_json(std::cout,runtime);
            std::cout<<",\"memory_snapshot\":";cli::memory_snapshot(std::cout,memory);
            std::cout<<",\"weight_activity\":";cli::weight_activity(std::cout,result.weight_activity);
            if(detailed) { std::cout<<",\"breakdown\":";breakdown(std::cout,checked,std::size_t(warmup),std::size_t(ntok),fixed); }
            std::cout<<"}\n"<<std::flush;
            take(edge0::release_session(session));
        }
        std::cout<<"{\"type\":\"summary\",\"runs\":"<<runs<<",\"input_count\":"<<input.size()
                 <<",\"prefill_seconds\":"<<prefill_total<<",\"decode_seconds\":"<<decode_total
                 <<",\"decode_forwards\":"<<ntok*static_cast<double>(runs)
                 <<",\"decode_tokens_per_second\":"<<ntok*static_cast<double>(runs)/decode_total<<",\"engine\":";
        cli::engine_identity(std::cout,capabilities);std::cout<<",\"runtime\":";cli::runtime_json(std::cout,runtime);std::cout<<",\"model\":";
        cli::model_description(std::cout,description);std::cout<<"}\n";
        take(edge0::unload_model(model));return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
