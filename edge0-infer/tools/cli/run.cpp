#include "runtime_options.h"
#include "edge0/generation.h"
#include "edge0/weight_activity.h"
#include "edge0/diagnostics.h"
#include <map>
#include <ostream>
#include <string>
#include "edge0/generation_check.h"
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <iomanip>

namespace cli {
void finish_thinking(bool&);
std::optional<edge0::Error> output_event(const edge0::OutputEvent&,bool&,bool);
const char* sampling_help();
bool sampling_option(const std::string&, const std::string&, edge0::GenerationOverrides&);
std::vector<edge0::TextMessage> completed_history(const std::vector<edge0::TextMessage>&, const std::vector<edge0::TextMessage>&, const edge0::GenerationResult&);
const char* stop(edge0::StopReason);
std::string summary(const edge0::GenerationResult&, const edge0::MemorySnapshot&, const edge0::LoadConfiguration&);
void memory_snapshot(std::ostream&,const edge0::MemorySnapshot&);
void request_summary(std::ostream&,const edge0::RequestSummary&);
void load_configuration(std::ostream&,const edge0::LoadConfiguration&);
void json_string(std::ostream&,const std::string&);
void model_option(std::map<std::string,std::string>&,const std::string&);
void weight_activity(std::ostream&,const edge0::WeightActivity&);
}
#include "json_value.h"

namespace cli {
std::vector<std::int32_t> tokens(const std::string& json,const std::string& name) {
    const auto key=json.find('"'+name+'"');
    if(key==std::string::npos)throw std::runtime_error("Missing token array: "+name);
    auto index=json.find(':',key+name.size()+2);
    auto spaces=[&] { while(index<json.size()&&std::isspace(static_cast<unsigned char>(json[index])))++index; };
    if(index==std::string::npos)throw std::runtime_error("Invalid token JSON");
    ++index; spaces(); if(index==json.size()||json[index++]!='[')throw std::runtime_error("Expected token array");
    std::vector<std::int32_t> result; spaces();
    if(index<json.size()&&json[index]==']')return result;
    while(index<json.size()) {
        spaces(); auto start=index;
        while(index<json.size()&&std::isdigit(static_cast<unsigned char>(json[index])))++index;
        if(start==index)throw std::runtime_error("Expected nonnegative token ID");
        const auto value=std::stoull(json.substr(start,index-start));
        if(value>std::uint64_t(std::numeric_limits<std::int32_t>::max()))throw std::runtime_error("Token ID out of range");
        result.push_back(std::int32_t(value)); spaces();
        if(index<json.size()&&json[index]==']')return result;
        if(index==json.size()||json[index++]!=',')throw std::runtime_error("Expected comma in token array");
    }
    throw std::runtime_error("Unterminated token array");
}
inline void write(const std::filesystem::path& path,const void* data,std::size_t size) {
    std::ofstream stream(path,std::ios::binary); stream.write(static_cast<const char*>(data),std::streamsize(size));
    if(!stream)throw std::runtime_error("Cannot write output: "+path.string());
}
inline void text(const std::filesystem::path& path,const std::string& content) {write(path,content.data(),content.size());}
inline void array(std::ostream& stream,const std::vector<std::int32_t>& values) {
    stream<<'['; for(std::size_t i=0;i<values.size();++i)stream<<(i?",":"")<<values[i]; stream<<']';
}
}

edge0::LogReceiver cli_log_receiver(bool verbose);
namespace {
template<class T> T take(const edge0::Result<T>& value) {
    if(!value) throw std::runtime_error((value.error().category == edge0::ErrorCategory::budget_exceeded ? "budget_exceeded: " : value.error().category == edge0::ErrorCategory::invalid_argument ? "invalid_argument: " : "") + value.error().message);
    return value.value();
}
std::string read(const std::filesystem::path& path) {
    // Binary mode: keep prompt bytes verbatim so a UTF-8 file tokenizes the same
    // on every platform (no CRLF translation).
    std::ifstream file(path,std::ios::binary); if(!file) throw std::runtime_error("Cannot read: "+path.string());
    return {std::istreambuf_iterator<char>(file),{}};
}
using cli::stop;
using cli::summary;
std::string checked_summary(const edge0::GenerationCheckResult& checked, const edge0::MemorySnapshot& memory, const edge0::LoadConfiguration& loading) {
    auto report = summary(checked.result, memory, loading);
    report.resize(report.size() - 2);
    return report + ",\"random_draws\":" + std::to_string(checked.random_draws) +
        ",\"host_sampled_steps\":" + std::to_string(checked.host_sampled_steps) +
        ",\"sampling_readback_bytes\":" + std::to_string(checked.sampling_readback_bytes) +
        ",\"vocabulary\":" + std::to_string(checked.vocabulary) + "}\n";
}
}
namespace {
std::vector<std::int32_t> json_tokens(const cli::Json& value) {
    std::vector<std::int32_t> result;
    for (const auto& item : value.array()) {
        const auto n = item.integer();
        if (n < 0 || n > std::numeric_limits<std::int32_t>::max()) throw std::runtime_error("Invalid token ID");
        result.push_back(static_cast<std::int32_t>(n));
    }
    return result;
}
std::string replay_json(const edge0::ReplayInput& input) {
    std::ostringstream out; out << "{\"tokens\":"; cli::array(out,input.tokens); out << ",\"blocks\":[";
    bool first = true;
    for (const auto& block : input.blocks) {
        out << (first ? "" : ",") << "{\"start\":" << block.start << ",\"length\":" << block.length
            << ",\"phase\":\"" << (block.phase == edge0::BlockPhase::prefill ? "prefill" : "decode")
            << "\",\"sample\":" << (block.sample ? "true" : "false") << ",\"device\":";
        cli::json_string(out,block.device); out << '}'; first = false;
    }
    out << "]}\n"; return out.str();
}
void write_turn(const std::filesystem::path& out, const edge0::GenerationCheckResult& generated, const std::string& report) {
    std::filesystem::create_directories(out);
    std::ostringstream tokens; tokens << "{\"input\":"; cli::array(tokens, generated.result.input);
    tokens << ",\"generated\":"; cli::array(tokens, generated.result.generated);
    tokens << ",\"logits_shape\":[" << (generated.vocabulary ? generated.logits.size()/generated.vocabulary : 0)
           << ',' << generated.vocabulary << "]}\n";
    cli::text(out/"tokens.json", tokens.str()); cli::text(out/"result.json", report);
    cli::text(out/"blocks.json", replay_json(generated.replay));
    cli::text(out/"answer.txt", generated.result.text);
    cli::write(out/"logits.f32", generated.logits.data(), generated.logits.size()*sizeof(float));
}
int run_turns(const std::string& turns_file, const std::string& replay_file, const std::string& output,
    const edge0::SessionReference& session, const edge0::ModelReference& model, const edge0::RuntimeReference& runtime,
    const edge0::GenerationConfiguration& configuration, const edge0::GenerationOverrides& overrides,
    const edge0::GenerationCheckOptions& checking) {
    if (!replay_file.empty()) {
        const auto json = cli::JsonParser(read(replay_file)).parse();
        edge0::ReplayInput input; input.tokens = json_tokens(json.at("tokens"));
        for (const auto& item : json.at("blocks").array()) {
            const auto start=item.at("start").integer(), length=item.at("length").integer();
            const auto phase=item.at("phase").string();
            if (start < 0 || length <= 0 || (phase!="prefill" && phase!="decode")) throw std::runtime_error("Invalid replay block");
            input.blocks.push_back({static_cast<std::size_t>(start),static_cast<std::size_t>(length),
                phase=="prefill"?edge0::BlockPhase::prefill:edge0::BlockPhase::decode,item.at("sample").boolean(),item.has("device")?item.at("device").string():std::string{}});
        }
        const auto generated=take(edge0::replay_check(session,input));
        const auto report=cli::with_runtime(checked_summary(generated,take(edge0::memory_snapshot(runtime)),take(edge0::query_model(model)).loading),runtime);
        if (!output.empty()) write_turn(output,generated,report);
        std::cerr << report; return 0;
    }
    const auto document=cli::JsonParser(read(turns_file)).parse();
    edge0::ChatInput chat;
    if(document.has("system")) chat.messages.push_back({"system",document.at("system").string()});
    std::vector<std::int32_t> last_input, last_all;
    edge0::ReplayInput latest;
    bool message_mode=false, token_mode=false;
    std::size_t turn_index=0;
    for(const auto& turn:document.at("turns").array()) {
        if (turn.object().size()==1 && turn.has("reset") && turn.at("reset").boolean()) {
            take(edge0::reset_conversation(session)); chat.messages.clear();
            if(document.has("system"))chat.messages.push_back({"system",document.at("system").string()});
            last_input.clear();last_all.clear();latest={}; continue;
        }
        auto options = checking;
        std::optional<edge0::CancelToken> token;
        std::optional<std::size_t> after_fragments;
        std::size_t fragments = 0;
        auto turn_overrides=overrides;
        if(turn.has("thinking")) turn_overrides.thinking=turn.at("thinking").boolean();
        if(turn.has("stop")) {turn_overrides.stop=std::vector<std::string>{};for(const auto& value:turn.at("stop").array()) turn_overrides.stop->push_back(value.string());}
        const auto fields = turn.object().size() - (turn.has("cancel") ? 1 : 0) - (turn.has("thinking")?1:0) - (turn.has("stop")?1:0);
        if (turn.has("cancel")) {
            const auto& cancel = turn.at("cancel");
            auto number = [&](const char* key) {
                const auto value = cancel.at(key).integer();
                if (value < 0) throw std::runtime_error("Negative cancel coordinate");
                return static_cast<std::size_t>(value);
            };
            if (cancel.has("after_fragments")) {
                if (cancel.object().size() != 1) throw std::runtime_error("Invalid fragment cancellation");
                after_fragments = number("after_fragments");
                token = take(edge0::create_cancel_token(runtime));
                if (*after_fragments == 0) edge0::request_cancel(*token);
            } else {
                const auto at = cancel.at("at").string();
                edge0::CancelPoint point{edge0::CancelPointKind::before_compute};
                std::size_t expected = 1;
                if (at == "after_reuse") point.kind = edge0::CancelPointKind::after_reuse;
                else if (at == "before_block" || at == "in_block") {
                    point.kind = at == "in_block" ? edge0::CancelPointKind::in_block : edge0::CancelPointKind::before_block;
                    point.block = number("block"); expected = 2;
                    if (at == "in_block") { point.segment = number("segment"); expected = 3; if (!point.segment) throw std::runtime_error("Invalid segment"); }
                    else if (!point.block) throw std::runtime_error("Invalid block");
                } else if (at == "before_step") {
                    point.kind = edge0::CancelPointKind::before_step; point.token = number("token"); expected = 2;
                    if (!point.token) throw std::runtime_error("Invalid token boundary");
                } else if (at != "before_compute") throw std::runtime_error("Unknown cancel point");
                if (cancel.object().size() != expected) throw std::runtime_error("Invalid cancel fields");
                options.cancel_probe = [point](const edge0::CancelPoint& seen) {
                    return seen.kind == point.kind &&
                        (seen.kind != edge0::CancelPointKind::before_block || seen.block == point.block) &&
                        (seen.kind != edge0::CancelPointKind::in_block || (seen.block == point.block && seen.segment == point.segment)) &&
                        (seen.kind != edge0::CancelPointKind::before_step || seen.token == point.token);
                };
            }
        }
        bool thinking_open=false;
        edge0::OutputReceiver receiver = [&](const edge0::OutputEvent& event) -> std::optional<edge0::Error> {
            if(auto error=cli::output_event(event,thinking_open,false)) return error;
            if ((event.kind==edge0::OutputKind::answer || event.kind==edge0::OutputKind::reasoning) &&
                !event.text.empty() && after_fragments && ++fragments == *after_fragments) edge0::request_cancel(*token);
            return {};
        };
        edge0::GenerationCheckResult generated;
        if(turn.has("user") || turn.has("regenerate")) {
            if(token_mode || fields!=1)throw std::runtime_error("Mixed or invalid turn modes");
            message_mode=true;
            const auto previous = chat.messages;
            if(turn.has("user"))chat.messages.push_back({"user",turn.at("user").string()});
            else {
                if(!turn.at("regenerate").boolean() || chat.messages.empty() || chat.messages.back().role!="assistant")throw std::runtime_error("No answer to regenerate");
                chat.messages.pop_back();
            }
            generated=take(edge0::generate_chat_check(session,chat,configuration,turn_overrides,options,receiver,token ? &*token : nullptr));
            chat.messages = cli::completed_history(chat.messages, previous, generated.result);
        } else {
            if(message_mode){throw std::runtime_error("Mixed turn modes");} token_mode=true;
            std::vector<std::int32_t> input;
            if(turn.has("input") && fields==1)input=json_tokens(turn.at("input"));
            else if(turn.has("same") && fields==1 && turn.at("same").boolean())input=last_input;
            else if(turn.has("continue") && fields==1) {
                input=last_all;const auto tail=json_tokens(turn.at("continue"));input.insert(input.end(),tail.begin(),tail.end());
            } else if(turn.has("prefix") && turn.has("append") && fields==2) {
                const auto prefix=turn.at("prefix").integer();
                if(prefix<0 || static_cast<std::size_t>(prefix)>last_all.size())throw std::runtime_error("Invalid prefix");
                input.assign(last_all.begin(),last_all.begin()+prefix);const auto tail=json_tokens(turn.at("append"));input.insert(input.end(),tail.begin(),tail.end());
            } else throw std::runtime_error("Invalid turn object");
            generated=take(edge0::generate_tokens_check(session,input,configuration,turn_overrides,options,receiver,token ? &*token : nullptr));
        }
        cli::finish_thinking(thinking_open);
        const auto& result=generated.result;
        last_input=result.input;last_all=last_input;last_all.insert(last_all.end(),result.generated.begin(),result.generated.end());
        latest=generated.replay;
        auto report=cli::with_runtime(checked_summary(generated,take(edge0::memory_snapshot(runtime)),take(edge0::query_model(model)).loading),runtime);
        report.resize(report.size() - 2); // Replace the closing object delimiter and newline.
        report += ",\"cancel_point\":";
        if (generated.cancel_point) {
            const auto& point = *generated.cancel_point;
            const char* names[] = {"before_compute", "after_reuse", "before_block", "in_block", "before_step"};
            report += "{\"at\":\"" + std::string(names[static_cast<unsigned>(point.kind)]) + "\",\"block\":" + std::to_string(point.block) +
                ",\"segment\":" + std::to_string(point.segment) + ",\"token\":" + std::to_string(point.token) + "}";
        } else report += "null";
        report += ",\"block_snapshot_seconds\":[";
        for (std::size_t i=0;i<generated.block_snapshot_seconds.size();++i) report += (i ? "," : "") + std::to_string(generated.block_snapshot_seconds[i]);
        report += "]}\n";
        std::cerr<<report;std::cout<<result.text<<'\n';
        if(!output.empty())write_turn(std::filesystem::path(output)/std::to_string(++turn_index),generated,report);
        if(result.error)throw std::runtime_error(result.error->message);
    }
    if(!output.empty()) {std::filesystem::create_directories(output);cli::text(std::filesystem::path(output)/"blocks.json",replay_json(latest));}
    return 0;
}
}

// One text for --help and for the usage error, so the two cannot drift apart.
const char* const run_usage="Usage: edge0-infer run --model <directory> (--user <text> [--system <text>] | --user-file <file> [--system <text>] | --tokens <json> [--fresh-graphs] | --turns <json> | --replay <blocks.json>) [--route-trace <file>] [--skip-checkpoint-restore] [--skip-block-rollback (development)] [--logprobs <top-count>] [--out <directory>] [--check <directory>] [--max-tokens <n>] [--context <n>] [--prefill-chunk <n>] [--device <cpu|metal|vulkan>] [--prefill-device <cpu|metal|vulkan>] [--stage-policy <name>] [--threads <n>] [--prefill-threads <n>] [--memory-budget <bytes>] [--model-option <name=value>] [--verbose]";

int run_command(int argc,char** argv) {
    try {
        edge0::RuntimeOptions runtime_options;
        bool skip_rollback=false;
        std::string directory,backend="cpu",token_file,user,system,out,check,route_trace,turns_file,replay_file;
        bool has_user=false,fresh=false,verbose=false,skip_restore=false;
        edge0::LoadOverrides load; edge0::GenerationOverrides overrides;
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") { std::cout<<run_usage<<cli::sampling_help()<<"\nDefaults use model recommendations. A supplied seed restarts for every turn.\n"; return 0; }
            if(arg=="--skip-block-rollback") {skip_rollback=true;continue;}
            if(arg=="--skip-checkpoint-restore") {skip_restore=true;continue;}
            if(arg=="--verbose") { verbose=true; continue; }
            if(arg=="--fresh-graphs") { fresh=true; continue; }
            if(i+1>=argc) throw std::runtime_error("Missing run argument: "+arg);
            const std::string value=argv[++i];
            if(cli::runtime_option(arg,value,runtime_options)) {}
            else if(cli::stage_option(arg,value,load)) {}
            else if(arg=="--model")directory=value;
            else if(arg=="--device")backend=value;
            else if(arg=="--model-option")cli::model_option(load.model_options,value);
            else if(arg=="--turns")turns_file=value;
            else if(arg=="--replay")replay_file=value;
            else if(arg=="--tokens")token_file=value;
            else if(arg=="--route-trace")route_trace=value;
            else if(arg=="--user-file") {user=read(value);has_user=true;}
            else if(arg=="--user") {user=value;has_user=true;}
            else if(arg=="--system")system=value;
            else if(arg=="--out")out=value;
            else if(arg=="--check")check=value;
            else if(arg=="--logprobs") {std::size_t used=0;const auto n=std::stoll(value,&used);if(used!=value.size()||n<0||n>20)throw std::runtime_error("Invalid logprobs count");overrides.logprobs=true;overrides.top_logprobs=n;}
            else if(cli::sampling_option(arg,value,overrides)) {}
            else if(arg=="--max-tokens"||arg=="--context"||arg=="--prefill-chunk") {
                std::size_t n=0; const auto number=std::stoll(value,&n); if(n!=value.size())throw std::runtime_error("Invalid integer");
                if(arg=="--context")load.context=number; else if(arg=="--prefill-chunk")load.prefill_chunk=number; else overrides.max_tokens=number;
            } else throw std::runtime_error("Invalid run argument: "+arg);
        }
        if(directory.empty()||(int(has_user)+int(!token_file.empty())+int(!turns_file.empty())+int(!replay_file.empty())!=1)||!cli::device_family(backend)||(!system.empty()&&!has_user))
            throw std::runtime_error(std::string(run_usage)+cli::sampling_help());
        if(fresh&&has_user)throw std::runtime_error("--fresh-graphs requires --tokens");
        if(!out.empty()&&std::filesystem::exists(out))throw std::runtime_error("Output directory already exists");
        std::string device;
        const auto log_receiver=cli_log_receiver(verbose);
        for(const auto& item:edge0::query_capabilities(false,log_receiver).devices)if(item.backend==backend){device=item.name;break;}
        if(device.empty()){std::cerr<<"SKIP: requested device unavailable\n";return 77;}
        auto runtime=take(cli::open_runtime(runtime_options,device,log_receiver,&load));
        struct Close {edge0::RuntimeReference runtime; ~Close(){edge0::finish_runtime_use(runtime);} } close{runtime};
        auto model=take(edge0::load_model(runtime,take(edge0::resolve_load_configuration(runtime,directory,load))));
        auto session=take(edge0::create_session(model));
        auto configuration=take(edge0::resolve_generation_configuration(model));
        edge0::GenerationCheckResult generated;
        bool thinking_open=false;
        auto receive=[&](const edge0::OutputEvent& event) {return cli::output_event(event,thinking_open,true);};
        edge0::GenerationCheckOptions checking;checking.fresh_graphs=fresh;checking.retain_logits=!out.empty();checking.route_trace=route_trace;checking.skip_checkpoint_restore=skip_restore;checking.skip_block_rollback=skip_rollback;
        if(!route_trace.empty()) { std::ofstream trace(route_trace); if(!trace) throw std::runtime_error("Cannot open route trace"); }
        if(!turns_file.empty() || !replay_file.empty())
            return run_turns(turns_file,replay_file,out,session,model,runtime,configuration,overrides,checking);
        if(has_user) {
            edge0::ChatInput chat;
            if(!system.empty())chat.messages.push_back({"system",system});
            chat.messages.push_back({"user",user});
            generated=take(edge0::generate_chat_check(session,chat,configuration,overrides,checking,receive));
        } else {
            generated=take(edge0::generate_tokens_check(session,cli::tokens(read(token_file),"input"),configuration,overrides,checking,receive));
        }
        cli::finish_thinking(thinking_open);
        const auto& result = generated.result;
        const auto& logits = generated.logits;
        const auto vocabulary = generated.vocabulary;
        auto report=cli::with_runtime(checked_summary(generated,take(edge0::memory_snapshot(runtime)),take(edge0::query_model(model)).loading),runtime);
        std::cerr<<report;
        if(!out.empty()) {
            std::filesystem::create_directories(out);
            std::ostringstream metadata;metadata<<"{\"input\":";cli::array(metadata,result.input);
            metadata<<",\"generated\":";cli::array(metadata,result.generated);
            metadata<<",\"stop_reason\":\""<<stop(result.stop_reason)<<"\",\"chunking\":\"split-last\",\"logits_shape\":["
                    <<(vocabulary?logits.size()/vocabulary:0)<<','<<vocabulary<<"]}\n";
            cli::text(std::filesystem::path(out)/"tokens.json",metadata.str());
            cli::text(std::filesystem::path(out)/"result.json",report);
            cli::write(std::filesystem::path(out)/"logits.f32",logits.data(),logits.size()*sizeof(float));
        }
        int status=result.stop_reason==edge0::StopReason::failed?1:0;
        if(!check.empty()) {
            const auto expected=read(std::filesystem::path(check)/"tokens.json");
            if(cli::tokens(expected,"input")!=result.input) {std::cerr<<"check: input differs\n";status=1;}
            auto generated=cli::tokens(expected,"generated");std::size_t matched=0;
            while(matched<generated.size()&&matched<result.generated.size()&&generated[matched]==result.generated[matched])++matched;
            std::cerr<<"check: "<<matched<<'/'<<generated.size()<<" tokens match\n";
            if(matched!=generated.size()||matched!=result.generated.size()) {std::cerr<<"check: first difference at "<<matched<<'\n';status=1;}
        }
        if(result.error)std::cerr<<result.error->message<<'\n';
        take(edge0::release_session(session));take(edge0::unload_model(model));take(edge0::finish_runtime_use(runtime));
        return status;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
