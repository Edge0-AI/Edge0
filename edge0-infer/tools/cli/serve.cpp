#include "runtime_options.h"
#include "serve_engine.h"
#include "serve_status.h"
#include "edge0/diagnostics.h"
#include <httplib.h>
#include <csignal>
#include <filesystem>
#include <optional>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <functional>
#include <thread>
edge0::LogReceiver cli_log_receiver(bool);
namespace cli {
bool sampling_option(const std::string&,const std::string&,edge0::GenerationOverrides&);
const char* sampling_help();
void model_option(std::map<std::string,std::string>&,const std::string&);
std::string summary(const edge0::GenerationResult&,const edge0::MemorySnapshot&,const edge0::LoadConfiguration&);
const char* stop(edge0::StopReason);
}
namespace {
using portal::Json;
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void signal_handler(int) {interrupted=true;}
template<class T>T take(const edge0::Result<T>& result) {if(!result){auto error=portal::failure(result.error());error.native_detail=result.error().message;throw error;}return result.value();}
std::mutex log_mutex;
void log(const std::string& text) {std::lock_guard<std::mutex> lock(log_mutex);std::cerr<<text<<'\n'<<std::flush;}
struct Options {
    edge0::RuntimeOptions runtime;
    std::string directory,device="cpu",host="127.0.0.1",alias,out,hold,key_file;
    int port=8000,write_timeout=30;
    std::size_t max_body=16*1024*1024;
    double idle=0;
    bool warmup=true,verbose=false;
    edge0::LoadOverrides load;
    edge0::GenerationOverrides generation;
};
std::int64_t integer(const std::string& text) {std::size_t used=0;auto n=std::stoll(text,&used);if(used!=text.size() || n<0)throw std::runtime_error("Expected a nonnegative integer");return n;}
Options options(int argc,char** argv) {
    Options o;
    for(int i=2;i<argc;++i) {
        const std::string key=argv[i];if(key=="--verbose") {o.verbose=true;continue;}if(key=="--no-warmup"){o.warmup=false;continue;}
        if(i+1>=argc){throw std::runtime_error("Missing option value");}const std::string value=argv[++i];
        if(cli::runtime_option(key,value,o.runtime)) {}
        else if(cli::stage_option(key,value,o.load)) {}
        else if(key=="--model")o.directory=value;else if(key=="--device")o.device=value;else if(key=="--host")o.host=value;
        else if(key=="--api-key-file")o.key_file=value;
        else if(key=="--alias")o.alias=value;else if(key=="--out")o.out=value;else if(key=="--hold-file")o.hold=value;
        else if(key=="--port") {auto n=integer(value);if(n>65535)throw std::runtime_error("Invalid port");o.port=static_cast<int>(n);}
        else if(key=="--write-timeout-seconds") {auto n=integer(value);if(n<1 || n>INT32_MAX)throw std::runtime_error("Invalid write timeout");o.write_timeout=static_cast<int>(n);}
        else if(key=="--max-body-bytes")o.max_body=static_cast<std::size_t>(integer(value));
        else if(key=="--session-idle-seconds")o.idle=static_cast<double>(integer(value));
        else if(key=="--context")o.load.context=integer(value);else if(key=="--prefill-chunk")o.load.prefill_chunk=integer(value);
        else if(key=="--max-tokens")o.generation.max_tokens=integer(value);
        else if(key=="--model-option")cli::model_option(o.load.model_options,value);
        else if(!cli::sampling_option(key,value,o.generation))throw std::runtime_error("Unknown serve option: "+key);
    }
    if(o.directory.empty() || !cli::device_family(o.device))throw std::runtime_error("Expected model and a cpu, metal, or vulkan device");
    return o;
}
edge0::GenerationOverrides merge(edge0::GenerationOverrides a,const edge0::GenerationOverrides& b) {
#define COPY(name) if(b.name)a.name=b.name
    COPY(temperature);COPY(top_p);COPY(top_k);COPY(max_tokens);COPY(repetition_penalty);COPY(seed);COPY(thinking);COPY(stop);COPY(presence_penalty);COPY(frequency_penalty);COPY(logit_bias);COPY(logprobs);COPY(top_logprobs);
#undef COPY
    return a;
}
struct Watch {
    std::atomic<bool> done{false},disconnected{false};
    std::thread thread;
    Watch(const httplib::Request& request,const edge0::CancelToken& token) :thread([this,&request,token]{
        while(!done) {if(request.is_connection_closed()){disconnected=true;edge0::request_cancel(token);break;}std::this_thread::sleep_for(std::chrono::milliseconds(20));}
    }) {}
    ~Watch(){done=true;thread.join();}
};
}
int serve_command(int argc,char** argv) {
    for(int i=2;i<argc;++i)if(std::string(argv[i])=="--help") {
        std::cout<<"Usage: edge0-infer serve --model <directory> [--device cpu|metal|vulkan] [--prefill-device cpu|metal|vulkan] [--stage-policy <name>] [--alias <name>] [--host <address>] [--api-key-file <path>] [--port <0..65535>] [--context <n>] [--prefill-chunk <n>] [--model-option <name=value>] [--max-tokens <n>] [--no-warmup] [--session-idle-seconds <s>] [--max-body-bytes <n>] [--write-timeout-seconds <s>] [--verbose] [--out <directory> (development)] [--hold-file <path> (development)]"<<cli::sampling_help()<<"\nFor a delegated model (such as Edge0 145B or Edge0 120B, run on edge0-strata): [--strata-dir <directory>] (the edge0-strata directory) [--strata-python <path>] (its Python interpreter) [--strata-ready-timeout <s>] (seconds to wait until it is ready) [--strata-arg <argument>]... (one engine command-line word, repeatable)\n";return 0;
    }
    try {
        auto o=options(argc,argv);interrupted=false;
        std::string api_token;
        if (!o.key_file.empty()) {
            std::ifstream input(o.key_file, std::ios::binary);
            if (!input) throw std::runtime_error("Cannot read API key file");
            std::ostringstream content; content << input.rdbuf();
            api_token = content.str();
            const auto first = api_token.find_first_not_of(" \t\r\n\f\v");
            if (first == std::string::npos) throw std::runtime_error("API key file is empty");
            api_token = api_token.substr(first, api_token.find_last_not_of(" \t\r\n\f\v")-first+1);
        }
        if (!portal::loopback(o.host) && o.host != "localhost" && api_token.empty())
            throw std::runtime_error("Non-loopback listening requires --api-key-file");
        if(!o.out.empty() && std::filesystem::exists(o.out))throw std::runtime_error("Output directory already exists");
        const auto logger=cli_log_receiver(o.verbose);
        std::string device;
        const auto capabilities = edge0::query_capabilities(false,logger);
        for(const auto& item:capabilities.devices)if(item.backend==o.device){device=item.name;break;}
        if(device.empty())throw std::runtime_error("Requested device unavailable");
        o.load.device=device;
        auto runtime=take(cli::open_runtime(o.runtime,device,logger,&o.load));
        cli::runtime_line(std::cerr,runtime);
        edge0::ModelReference model;edge0::SessionReference session;
        struct Cleanup {
            edge0::RuntimeReference& runtime;edge0::ModelReference& model;edge0::SessionReference& session;bool done=false;
            bool close() {
                if(done)return true;
                if(session){if(!edge0::release_session(session))return false;session.reset();}
                if(model){if(!edge0::unload_model(model))return false;model.reset();}
                if(!edge0::finish_runtime_use(runtime)){return false;}done=true;return true;
            }
            ~Cleanup(){if(!done && !close())log("Cleanup failed");}
        } cleanup{runtime,model,session};
        auto loading=take(edge0::resolve_load_configuration(runtime,o.directory,o.load));model=take(edge0::load_model(runtime,loading));session=take(edge0::create_session(model));
        const auto description=take(edge0::query_model(model));
        if(o.alias.empty())o.alias=description.adapter;
        auto configuration=take(edge0::resolve_generation_configuration(model,o.generation));
        if(o.warmup) {
            edge0::GenerationOverrides warm;warm.max_tokens=1;warm.temperature=0;
            auto r=take(edge0::generate(session,std::vector<edge0::TextMessage>{{"user","Hello"}},configuration,warm));
            if(r.error || r.stop_reason==edge0::StopReason::failed || r.stop_reason==edge0::StopReason::cancelled){throw std::runtime_error("Warmup failed");}take(edge0::reset_conversation(session));
        }
        const auto version=capabilities.engine_version;
        const auto started = std::chrono::steady_clock::now();
        const auto wall_start = std::chrono::system_clock::now();
        const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(wall_start.time_since_epoch()).count();
#ifdef _WIN32
        const auto pid = GetCurrentProcessId();
#else
        const auto pid = getpid();
#endif
        auto timestamp = [] {
            const auto now = std::chrono::system_clock::now();
            const auto value = std::chrono::system_clock::to_time_t(now);
            std::tm utc{};
#ifdef _WIN32
            gmtime_s(&utc, &value);
#else
            gmtime_r(&value, &utc);
#endif
            std::ostringstream out;
            out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
                << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()%1000 << 'Z';
            return out.str();
        };
        const auto started_at = timestamp();
        std::mutex status_mutex;
        portal::History history;
        std::string active_endpoint, active_source;
        struct TrackedRequest {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            bool recorded = false, entered = false;
            std::optional<portal::Failure> failure;
        };
        std::map<const httplib::Request*, std::shared_ptr<TrackedRequest>> tracked;
        auto save_record = [&](const portal::RequestRecord& record, const std::shared_ptr<TrackedRequest>& state) {
            std::lock_guard<std::mutex> lock(status_mutex);
            if (state->recorded) return;
            history.add(record); state->recorded = true;
            if (record.entered) { active_endpoint.clear(); active_source.clear(); }
            log(portal::record_json(record).dump());
        };
        std::mutex token_mutex;std::optional<edge0::CancelToken> active_token;
        std::atomic<std::size_t> turn{0};
        portal::EngineHooks hooks;
        hooks.create=[&]{session=take(edge0::create_session(model));log("session created");};
        hooks.release=[&]{take(edge0::release_session(session));session.reset();log("session released");};
        hooks.unload=[&]{take(edge0::unload_model(model));model.reset();log("model unloaded");};
        hooks.finish=[&]{take(edge0::finish_runtime_use(runtime));cleanup.done=true;log("runtime safely finished");};
        hooks.rebuilt=[] {log("session rebuilt");};
        hooks.cancel=[&]{std::lock_guard<std::mutex> lock(token_mutex);if(active_token)edge0::request_cancel(*active_token);};
        hooks.generate=[&](const portal::Request& request,edge0::OutputReceiver receiver){
            return edge0::generate(session,request.input,configuration,request.overrides,std::move(receiver),&*active_token);
        };
        portal::Engine engine(std::move(hooks));httplib::Server server;server.set_tcp_nodelay(true);
        server.set_socket_options([](socket_t socket) {
            // A second server must not share an existing listener's port.
#ifdef _WIN32
            const auto option=SO_EXCLUSIVEADDRUSE;
#else
            const auto option=SO_REUSEADDR;
#endif
            if(!httplib::set_socket_opt(socket,SOL_SOCKET,option,1))
                throw std::runtime_error("Cannot configure exclusive server binding");
        });
        server.new_task_queue=[] {return new httplib::ThreadPool(8);};
        server.set_payload_max_length(o.max_body);server.set_write_timeout(o.write_timeout);
        auto is_inference=[](const std::string& path) {return path=="/v1/chat/completions" || path=="/v1/responses" || path=="/v1/messages";};
        auto error_protocol=[](const std::string& path) {return path=="/v1/messages" || path.rfind("/v1/messages/",0)==0?portal::Protocol::messages:portal::Protocol::chat;};
        auto send_error=[&](const httplib::Request& req,httplib::Response& response,const portal::Failure& error){
            {
                std::lock_guard<std::mutex> lock(status_mutex);
                auto found=tracked.find(&req);
                if(found==tracked.end() && is_inference(req.path))found=tracked.emplace(&req,std::make_shared<TrackedRequest>()).first;
                if(found!=tracked.end())found->second->failure=error;
            }
            response.status=error.status;response.set_content(error.body(error_protocol(req.path)).dump(),"application/json");
        };
        server.set_error_handler([&](const httplib::Request& req,httplib::Response& res){
            if(res.status==413)send_error(req,res,portal::Failure(413,"Request body too large","request_too_large"));
            else if(res.body.empty())send_error(req,res,portal::http_failure(res.status));
        });
        server.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
            const bool allowed = portal::authorized(req.remote_addr, api_token, req.get_header_value("Authorization"), req.get_header_value("x-api-key"));
            const bool inference = is_inference(req.path);
            if (inference || !allowed) {
                std::lock_guard<std::mutex> lock(status_mutex);
                tracked[&req] = std::make_shared<TrackedRequest>();
            }
            if (!allowed) {
                send_error(req,res, portal::Failure(401,"Invalid API key","invalid_api_key"));
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_logger([&](const httplib::Request& req, const httplib::Response& res) {
            std::shared_ptr<TrackedRequest> state;
            {
                std::lock_guard<std::mutex> lock(status_mutex);
                const auto found = tracked.find(&req);
                if (found != tracked.end()) { state = found->second; tracked.erase(found); if (state->entered) return; }
            }
            if (!state && is_inference(req.path)) state = std::make_shared<TrackedRequest>();
            if (!state) return;
            portal::RequestRecord record; record.time = timestamp(); record.endpoint = req.path; record.source = req.remote_addr;
            record.http_status = res.status;
            record.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-state->start).count();
            if (state->failure) {
                record.code=state->failure->code;record.message=state->failure->message();record.param=state->failure->param;
            }
            save_record(record,state);
        });
        int bound_port = 0;
        server.Get("/status",[&](const httplib::Request&,httplib::Response& res) {
            portal::StatusInput input;
            input.listen = {{"host",o.host},{"port",bound_port},{"auth",!api_token.empty()}};
            input.engine = portal::engine_json(capabilities); input.model = portal::model_json(o.alias,description);
            {
                std::lock_guard<std::mutex> lock(status_mutex);
                input.history = history; input.endpoint = active_endpoint; input.source = active_source;
                input.progress = take(edge0::query_generation_progress(runtime));
            }
            input.now = std::chrono::steady_clock::now();
            input.instance = {{"id",std::to_string(pid)+"-"+std::to_string(unix_ms)},{"pid",pid},{"started_at",started_at},
                {"uptime_seconds",std::chrono::duration<double>(input.now-started).count()}};
            input.memory = take(edge0::memory_snapshot(runtime)); input.system = edge0::query_system_status();
            res.set_content(portal::status_json(input).dump(),"application/json");
        });
        auto model_object=[&]{return Json{{"id",o.alias},{"object","model"},{"created",0},{"owned_by","local"},{"context_length",description.context},{"capabilities",{{"thinking",description.thinking_enabled_supported},{"thinking_default",description.thinking_default}}}};};
        server.Get("/health",[&](const httplib::Request& req,httplib::Response& res){if(engine.stopping())send_error(req,res,portal::Failure(503,"Server shutting down","shutting_down"));else res.set_content("{\"status\":\"ok\"}","application/json");});
        server.Get("/v1/models",[&](const httplib::Request&,httplib::Response& res){res.set_content(Json{{"object","list"},{"data",Json::array({model_object()})}}.dump(),"application/json");});
        server.Get(R"(/v1/models/(.+))",[&](const httplib::Request& req,httplib::Response& res){if(req.matches[1].str()!=o.alias)send_error(req,res,portal::Failure(404,"Unknown model","model_not_found","model"));else res.set_content(model_object().dump(),"application/json");});
        auto handler=[&](const httplib::Request& http,httplib::Response& response) {
            try {
                const bool count_only=http.path=="/v1/messages/count_tokens";
                auto protocol=http.path=="/v1/responses"?portal::Protocol::responses:error_protocol(http.path);
                auto request=portal::parse(Json::parse(http.body),protocol,o.alias,count_only);
                const auto effective=take(edge0::resolve_generation_configuration(model,merge(o.generation,request.overrides)));
                const auto count=take(edge0::count_tokens(model,request.input,effective));
                if(count.input_tokens>std::size_t(count.capacity))throw portal::Failure(400,"Input exceeds context capacity","context_length_exceeded","input");
                if(count_only) {response.set_content(Json{{"input_tokens",count.input_tokens}}.dump(),"application/json");return;}
                auto lease=engine.acquire();
                std::shared_ptr<TrackedRequest> request_state;
                {
                    std::lock_guard<std::mutex> lock(status_mutex);
                    request_state = tracked.at(&http); request_state->entered = true; active_endpoint = http.path; active_source = http.remote_addr;
                }
                auto token=take(edge0::create_cancel_token(runtime));
                {std::lock_guard<std::mutex> lock(token_mutex);active_token=token;if(engine.stopping())edge0::request_cancel(token);}
                const auto sequence=++turn;
                auto held_lease=std::make_shared<std::shared_ptr<portal::Engine::Lease>>(std::move(lease));
                // MSVC 19.44 (/std:c++17 /permissive-) gives the nested lambdas in this
                // function one closure type (C2011/C2440/C7746). Local classes keep a
                // single inner lambda. Destruction still records an unrecorded request,
                // and streaming still writes the same bytes.
                auto run=[&,request,token,sequence,held_lease,request_state,count](const std::function<httplib::DataSink*()>& sink_provider)->Json {
                    std::optional<portal::Failure> sent_failure; bool sent_opened=false;
                    struct FinishRecord {
                        decltype(save_record)& save;
                        decltype(timestamp)& clock;
                        std::mutex& status_mutex;
                        std::shared_ptr<TrackedRequest> state;
                        const httplib::Request& http;
                        const portal::Request& request;
                        const edge0::TokenCount& count;
                        std::optional<portal::Failure>& sent_failure;
                        bool& sent_opened;
                        std::mutex& token_mutex;
                        std::optional<edge0::CancelToken>& active_token;
                        FinishRecord(decltype(save_record)& save,decltype(timestamp)& clock,std::mutex& status_mutex,std::shared_ptr<TrackedRequest> state,const httplib::Request& http,const portal::Request& request,const edge0::TokenCount& count,std::optional<portal::Failure>& sent_failure,bool& sent_opened,std::mutex& token_mutex,std::optional<edge0::CancelToken>& active_token)
                            : save(save),clock(clock),status_mutex(status_mutex),state(std::move(state)),http(http),request(request),count(count),sent_failure(sent_failure),sent_opened(sent_opened),token_mutex(token_mutex),active_token(active_token) {}
                        ~FinishRecord() {
                            bool recorded;
                            { std::lock_guard<std::mutex> lock(status_mutex); recorded = state->recorded; }
                            if (recorded) return;
                            portal::RequestRecord record; record.time = clock(); record.endpoint = http.path; record.source = http.remote_addr;
                            record.http_status = 500; record.stream = request.stream; record.entered = true; record.result_available = false;
                            record.input_tokens = count.input_tokens; record.stop_reason = "failed";
                            if (sent_failure) record = portal::failed_record(record, sent_failure->status, sent_failure->code, sent_failure->message(), sent_failure->param, sent_opened);
                            record.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-state->start).count();
                            save(record,state);
                            { std::lock_guard<std::mutex> lock(token_mutex); active_token.reset(); }
                        }
                    } fallback(save_record,timestamp,status_mutex,request_state,http,request,count,sent_failure,sent_opened,token_mutex,active_token);
                    httplib::DataSink* sink=nullptr;
                    portal::Encoder encoder(request,o.alias,version);Watch watch(http,token);bool opened=false,held=false;
                    struct StreamIO {
                        httplib::DataSink*& sink;
                        bool& opened;
                        const portal::Request& request;
                        const std::function<httplib::DataSink*()>& sink_provider;
                        portal::Encoder& encoder;
                        const edge0::RuntimeReference& runtime;
                        std::size_t input_tokens;
                        bool write(const std::vector<std::string>& records) {for(const auto& record:records)if(sink && !sink->write(record.data(),record.size()))return false;return true;}
                        bool ensure_open() {
                            if(opened || !request.stream)return true;
                            sink=sink_provider();if(!sink)return false;
                            opened=true;
                            std::optional<edge0::GenerationProgress> progress;
                            if(request.protocol==portal::Protocol::messages) {auto value=edge0::query_generation_progress(runtime);if(value && value.value().phase!=edge0::GenerationPhase::idle)progress=value.value();}
                            return write(encoder.begin(input_tokens,progress));
                        }
                    } io{sink,opened,request,sink_provider,encoder,runtime,count.input_tokens};
                    auto result=engine.generate(request,[&](const edge0::OutputEvent& event)->std::optional<edge0::Error>{
                        if(!io.ensure_open() || !io.write(encoder.accept(event)))return edge0::Error{edge0::ErrorCategory::backend_error,"Stream delivery failed"};
                        if(!held && !event.text.empty() && event.kind!=edge0::OutputKind::token_logprob) {
                            held=true;while(!o.hold.empty() && std::filesystem::exists(o.hold) && !watch.disconnected && !engine.stopping())std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                        return {};
                    });
                    {std::lock_guard<std::mutex> lock(token_mutex);active_token.reset();}
                    if(!result) {held_lease->reset();auto error=portal::failure(result.error(),engine.stopping());sent_failure=error;sent_opened=opened;if(opened){io.write(encoder.fail(error));sink->done();}throw error;}
                    const auto& r=result.value();
                    const auto memory=take(edge0::memory_snapshot(runtime));
                    if(!o.out.empty()) {
                        auto folder=std::filesystem::path(o.out)/std::to_string(sequence);std::filesystem::create_directories(folder);
                        std::ofstream tokens(folder/"tokens.json");tokens<<Json{{"input",r.input},{"generated",r.generated}}.dump()<<'\n';
                        std::ofstream summary(folder/"result.json");summary<<cli::with_runtime(cli::summary(r,memory,loading),runtime);
                        std::ofstream answer(folder/"answer.txt");answer<<r.text;
                        if(!tokens || !summary || !answer)throw std::runtime_error("Cannot write request evidence");
                    }
                    const bool failed=r.stop_reason==edge0::StopReason::failed || r.stop_reason==edge0::StopReason::cancelled;
                    auto error=r.error?portal::failure(*r.error,engine.stopping()):portal::Failure(engine.stopping()?503:500,"Generation cancelled",engine.stopping()?"shutting_down":"cancelled");
                    if(failed)error.status=engine.stopping()?503:500;
                    portal::RequestRecord record;
                    record.time = timestamp(); record.endpoint = http.path; record.source = http.remote_addr;
                    record.stream = request.stream; record.http_status = 200;
                    record = portal::completed_record(record,r);
                    if(failed) record = portal::failed_record(record,error.status,error.code,error.message(),error.param,opened);
                    save_record(record,request_state);
                    held_lease->reset();
                    if(request.stream) {
                        if(failed && !opened)throw error;
                        if(io.ensure_open()) {io.write(failed?encoder.fail(error):encoder.finish(r,memory));sink->done();}
                    } else {encoder.finish(r,memory);if(failed)throw error;}
                    return encoder.complete(r,memory);
                };
                if(request.stream) {
                    auto handoff=std::make_shared<portal::StreamHandoff<httplib::DataSink>>();
                    handoff->worker=std::thread([run,handoff] {
                        try {run([&]{return handoff->await_sink();});handoff->finish(nullptr);}
                        catch(...) {handoff->finish(std::current_exception());}
                    });
                    try {handoff->wait_ready();}
                    catch(...) {handoff->abort();throw;}
                    response.set_chunked_content_provider("text/event-stream",[handoff](std::size_t,httplib::DataSink& sink) {
                        handoff->attach(sink);
                        if(handoff->worker.joinable())handoff->worker.join();
                        return !handoff->error;
                    },[handoff,token](bool) {
                        edge0::request_cancel(token);handoff->abort();
                    });
                } else response.set_content(run({}).dump(),"application/json");
            } catch(const portal::Failure& f) {send_error(http,response,f);}
            catch(const Json::exception&) {send_error(http,response,portal::Failure(400,"Invalid JSON body","invalid_json"));}
            catch(const std::exception&) {send_error(http,response,portal::Failure(500,"Request failed","backend_error"));}
        };
        server.Post("/v1/chat/completions",handler);server.Post("/v1/responses",handler);
        server.Post("/v1/messages",handler);server.Post("/v1/messages/count_tokens",handler);
        const int port=o.port==0?server.bind_to_any_port(o.host):(server.bind_to_port(o.host,o.port)?o.port:-1);
        if(port<0)throw std::runtime_error("Cannot bind server");
        bound_port = port;
        auto old_int=std::signal(SIGINT,signal_handler),old_term=std::signal(SIGTERM,signal_handler);
#ifdef _WIN32
        // TerminateProcess (what a parent's terminate() uses) cannot be caught, so a
        // graceful stop arrives as a console Ctrl+Break (SIGBREAK) delivered to this
        // process group; handle it the same way as SIGINT/SIGTERM.
        auto old_break=std::signal(SIGBREAK,signal_handler);
#endif
        std::atomic<bool> done{false};
        std::thread watcher([&]{
            while(!done) {
                if(interrupted) {log("stopping acceptance");engine.stop();server.stop();break;}
                try {engine.expire(o.idle);}catch(...){log("Session expiry failed");interrupted=true;}
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        log("Listening: http://"+o.host+":"+std::to_string(port)+" model="+o.alias);
        const bool listened=server.listen_after_bind();done=true;watcher.join();
        std::signal(SIGINT,old_int);std::signal(SIGTERM,old_term);
#ifdef _WIN32
        std::signal(SIGBREAK,old_break);
#endif
        // listen_after_bind joins HTTP workers, so providers and their leases have exited.
        log("active requests finished");const bool closed=engine.close();
        if(!closed){cleanup.done=true;log("Cleanup failed or completion unknown");}
        return listened && closed?0:1;
    } catch(const portal::Failure& error) {log(std::string(error.what())+": "+error.code+(error.native_detail.empty()?"":": "+error.native_detail));return 1;}
    catch(const std::exception& error) {log(error.what());return 1;}
}
