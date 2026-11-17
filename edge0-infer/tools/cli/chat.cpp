#include "runtime_options.h"
#include "chat_line.h"
#include "console_input.h"
#include "edge0/generation.h"
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>
#include "edge0/diagnostics.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

edge0::LogReceiver cli_log_receiver(bool);
namespace cli {
void finish_thinking(bool&);
std::optional<edge0::Error> output_event(const edge0::OutputEvent&,bool&,bool);
const char* sampling_help();
bool sampling_option(const std::string&, const std::string&, edge0::GenerationOverrides&);
std::vector<edge0::TextMessage> completed_history(const std::vector<edge0::TextMessage>&, const std::vector<edge0::TextMessage>&, const edge0::GenerationResult&);

void model_option(std::map<std::string, std::string>&, const std::string&);
const char* stop(edge0::StopReason);
std::string summary(const edge0::GenerationResult&, const edge0::MemorySnapshot&, const edge0::LoadConfiguration&);
}
namespace {
template<class T> T take(const edge0::Result<T>& result) {
    if (!result) throw std::runtime_error(result.error().message);
    return result.value();
}
std::atomic<bool> interrupted{false};
static_assert(decltype(interrupted)::is_always_lock_free);
void interrupt_handler(int) { interrupted.store(true, std::memory_order_relaxed); }
struct CancelTimer {
    std::mutex mutex;
    std::condition_variable condition;
    bool done = false;
    std::thread thread;
    explicit CancelTimer(std::optional<double> delay) {
        if (delay) thread = std::thread([this, delay] {
            std::unique_lock<std::mutex> lock(mutex);
            if (!condition.wait_for(lock, std::chrono::duration<double>(*delay), [&] { return done; })) std::raise(SIGINT);
        });
    }
    ~CancelTimer() {
        { std::lock_guard<std::mutex> lock(mutex); done = true; }
        condition.notify_one(); if (thread.joinable()) thread.join();
    }
};
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
bool normal(const edge0::GenerationResult& result) {
    return !result.error && (result.stop_reason == edge0::StopReason::end_of_generation || result.stop_reason == edge0::StopReason::length || result.stop_reason == edge0::StopReason::stop_string);
}
struct Resources {
    edge0::RuntimeReference runtime;
    edge0::ModelReference model;
    edge0::SessionReference session;
    bool closed = false;
    bool close() {
        if (closed) return true;
        closed = true;
        bool ok = true;
        auto check = [&](const edge0::Result<std::string>& result) {
            if (!result) { std::cerr << "Cleanup: " << result.error().message << '\n'; ok = false; }
        };
        if (session) check(edge0::release_session(session));
        if (model) check(edge0::unload_model(model));
        check(edge0::finish_runtime_use(runtime));
        return ok;
    }
    ~Resources() { close(); }
};
struct Signals {
    Resources& resources;
    std::mutex mutex;
    std::condition_variable delivered;
    bool done = false, generating = false;
    const edge0::CancelToken* token = nullptr;
    std::optional<Clock::time_point> requested;
    void (*previous)(int);
    std::thread watcher;
    explicit Signals(Resources& value) : resources(value), previous(std::signal(SIGINT, interrupt_handler)) {
        if (previous == SIG_ERR) throw std::runtime_error("Cannot install SIGINT handler");
        watcher = std::thread([this] {
            std::unique_lock<std::mutex> lock(mutex);
            while (!done) {
                if (interrupted.load(std::memory_order_relaxed)) {
                    if (!generating) {
                        resources.close();
                        std::cerr.flush();
                        std::_Exit(130);
                    }
                    if (token) {
                        interrupted.store(false, std::memory_order_relaxed);
                        if (!requested) requested = Clock::now();
                        edge0::request_cancel(*token);
                        delivered.notify_all();
                    }
                }
                delivered.wait_for(lock, std::chrono::milliseconds(2), [&] { return done; });
            }
        });
    }
    ~Signals() {
        { std::lock_guard<std::mutex> lock(mutex); done = true; }
        delivered.notify_all(); watcher.join(); std::signal(SIGINT, previous);
    }
    struct Turn {
        Signals& signals;
        explicit Turn(Signals& value) : signals(value) {
            std::lock_guard<std::mutex> lock(signals.mutex); signals.generating = true;
        }
        ~Turn() {
            std::lock_guard<std::mutex> lock(signals.mutex);
            signals.token = nullptr; signals.generating = false;
        }
    };
    void raise_and_wait() {
        std::raise(SIGINT);
        std::unique_lock<std::mutex> lock(mutex);
        delivered.wait(lock, [&] { return requested.has_value(); });
    }
    void unpublish() {
        std::lock_guard<std::mutex> lock(mutex); token = nullptr;
        const auto returned = Clock::now();
        if (requested) std::cerr << "Cancellation latency: " << std::chrono::duration<double>(returned-*requested).count() << " s\n";
    }
};
void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write output: " + path.string());
}
void write_turn(const std::filesystem::path& path, const edge0::GenerationResult& result, const std::string& report) {
    std::filesystem::create_directories(path);
    std::ostringstream tokens;
    auto array = [&](const std::vector<std::int32_t>& values) {
        tokens << '[';
        for (std::size_t i = 0; i < values.size(); ++i) tokens << (i ? "," : "") << values[i];
        tokens << ']';
    };
    tokens << "{\"input\":"; array(result.input);
    tokens << ",\"generated\":"; array(result.generated); tokens << "}\n";
    write_text(path / "tokens.json", tokens.str());
    write_text(path / "answer.txt", result.text);
    write_text(path / "result.json", report);
}
void statistics(const edge0::GenerationResult& result) {
    const char* methods[] = {"first", "keep", "truncate", "restore", "cold"};
    std::cerr << '[' << methods[static_cast<unsigned>(result.reuse_method)] << "] reused " << result.reused_tokens
              << ", computed " << result.computed_tokens << ", first token ";
    if (result.request_summary.first_token_seconds) std::cerr << *result.request_summary.first_token_seconds << " s";
    else std::cerr << "n/a";
    std::cerr << ", ";
    if (result.request_summary.decode_tokens_per_second) std::cerr << *result.request_summary.decode_tokens_per_second << " tok/s";
    else std::cerr << "n/a tok/s";
    if (result.configuration.temperature > 0 && result.configuration.seed) std::cerr << ", seed=" << *result.configuration.seed;
    std::cerr << ", " << cli::stop(result.stop_reason) << '\n';
}
}
#ifdef _WIN32
// The console reader accumulates a whole line.
constexpr const char* terminal_line_note = "";
#else
// A terminal in canonical mode stops accepting a line at about 1024 bytes.
constexpr const char* terminal_line_note = " Terminal lines are limited to 1024 bytes; use /file for long paragraphs.";
#endif
int chat_command(int argc, char** argv) {
    try {
        edge0::RuntimeOptions runtime_options;
        std::string directory, backend = "cpu", system, output;
        std::optional<std::size_t> sigint_fragments;
        std::optional<double> sigint_seconds;
        bool warmup = true, verbose = false, has_system = false;
        edge0::LoadOverrides load;
        edge0::GenerationOverrides overrides;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: edge0-infer chat --model <directory> [--device <cpu|metal|vulkan>] [--prefill-device <cpu|metal|vulkan>] [--stage-policy <name>] [--threads <n>] [--prefill-threads <n>] [--memory-budget <bytes>] [--system <text>] [--max-tokens <n>] [--context <n>] [--prefill-chunk <n>] [--model-option <name=value>] [--no-warmup] [--out <directory>] [--verbose] [--sigint-after-fragments <n> (development)] [--sigint-after-seconds <s> (development)]" << cli::sampling_help() << "\nDefaults use model recommendations. A supplied seed restarts each turn; /regen repeats the answer. Use /file <path> for long text or a line containing \"\"\" to open/close a multiline block." << terminal_line_note << " Use /think on or /think off to switch thinking for the next turn.\n"
                          << "For a delegated model (such as Edge0 145B or Edge0 120B, run on edge0-strata): [--strata-dir <directory>] (the edge0-strata directory) [--strata-python <path>] (its Python interpreter) [--strata-ready-timeout <s>] (seconds to wait until it is ready) [--strata-arg <argument>]... (one engine command-line word, repeatable)\n";
                return 0;
            }
            if (arg == "--no-warmup") { warmup = false; continue; }
            if (arg == "--verbose") { verbose = true; continue; }
            if (i + 1 >= argc) throw std::runtime_error("Missing chat argument: " + arg);
            const std::string value = argv[++i];
            if (cli::runtime_option(arg,value,runtime_options)) {}
            else if(cli::stage_option(arg,value,load)) {}
            else if (arg == "--model") directory = value;
            else if (arg == "--device") backend = value;
            else if (arg == "--system") { system = value; has_system = true; }
            else if (arg == "--sigint-after-fragments") {
                std::size_t used=0; const auto n=std::stoll(value,&used);
                if (used!=value.size() || n<0) throw std::runtime_error("Invalid fragment count");
                sigint_fragments=static_cast<std::size_t>(n);
            } else if (arg == "--sigint-after-seconds") {
                std::size_t used=0; const auto n=std::stod(value,&used);
                if (used!=value.size() || !std::isfinite(n) || n<0) throw std::runtime_error("Invalid cancellation delay");
                sigint_seconds=n;
            }
            else if (cli::sampling_option(arg, value, overrides)) {}
            else if (arg == "--out") output = value;
            else if (arg == "--model-option") cli::model_option(load.model_options, value);
            else if (arg == "--max-tokens" || arg == "--context" || arg == "--prefill-chunk") {
                std::size_t used = 0;
                const auto number = std::stoll(value, &used);
                if (used != value.size()) throw std::runtime_error("Invalid integer: " + value);
                if (arg == "--max-tokens") overrides.max_tokens = number;
                else if (arg == "--context") load.context = number;
                else load.prefill_chunk = number;
            } else throw std::runtime_error("Invalid chat argument: " + arg);
        }
        if (directory.empty() || !cli::device_family(backend))
            throw std::runtime_error("Expected --model <directory> and --device cpu, metal, or vulkan; see chat --help");
        if (!output.empty() && std::filesystem::exists(output)) throw std::runtime_error("Output directory already exists");
        const auto started = Clock::now();
        const auto logger = cli_log_receiver(verbose);
        std::string device;
        for (const auto& item : edge0::query_capabilities(false, logger).devices)
            if (item.backend == backend) { device = item.name; break; }
        if (device.empty()) { std::cerr << "SKIP: requested device unavailable\n"; return 77; }
        Resources resources{take(cli::open_runtime(runtime_options,device,logger,&load)), {}, {}};
        cli::runtime_line(std::cerr,resources.runtime);
        resources.model = take(edge0::load_model(resources.runtime, take(edge0::resolve_load_configuration(resources.runtime, directory, load))));
        resources.session = take(edge0::create_session(resources.model));
        auto configuration = take(edge0::resolve_generation_configuration(resources.model, overrides));
        const auto loading = take(edge0::query_model(resources.model)).loading;
        const double loading_seconds = elapsed(started);
        std::cerr << "Generation: temperature=" << configuration.temperature << ", max_tokens=" << configuration.max_tokens
                  << ", top_k=" << configuration.top_k << ", top_p=" << configuration.top_p
                  << ", presence_penalty=" << configuration.presence_penalty << ", frequency_penalty=" << configuration.frequency_penalty
                  << ", seed=" << (configuration.seed ? std::to_string(*configuration.seed) : "per-turn")
                  << ", repetition_penalty=" << configuration.repetition_penalty << ", thinking=" << std::boolalpha << configuration.thinking << '\n';
        double warmup_seconds = 0;
        if (warmup) {
            const auto start = Clock::now();
            edge0::GenerationOverrides warming; warming.max_tokens = 2;
            const auto result = take(edge0::generate(resources.session, std::vector<edge0::TextMessage>{{"user", "Hi"}}, configuration, warming));
            if (!normal(result)) throw std::runtime_error("Warmup failed: " + (result.error ? result.error->message : cli::stop(result.stop_reason)));
            take(edge0::reset_conversation(resources.session));
            warmup_seconds = elapsed(start);
        }
        Signals signals(resources);
        std::cerr << std::setprecision(6) << "Ready: load " << loading_seconds << " s, warmup " << warmup_seconds
                  << " s" << (warmup ? "" : " (disabled)") << "; /regen /new /quit\n";
        std::vector<edge0::TextMessage> history;
        auto clear_history = [&] { history.clear(); if (has_system) history.push_back({"system", system}); };
        clear_history();
        bool state_valid = true;
        std::size_t turn = 0;
        bool first_message = true;
        std::string line;
        cli::ChatLines lines;
        while (std::cerr << (lines.collecting() ? "... " : "> ") << std::flush, cli::read_input_line(line)) {
            Signals::Turn active(signals);
            auto classified = lines.feed(line);
            if (classified.first == "/file") {
                std::ifstream input(classified.second, std::ios::binary);
                if (!input) { std::cerr << "Cannot read file\n"; continue; }
                std::ostringstream contents; contents << input.rdbuf();
                if (input.bad() || contents.str().empty()) { std::cerr << "Empty or unreadable file\n"; continue; }
                classified = {"message", contents.str()};
            }
            const auto& action = classified.first;
            if (action == "empty") continue;
            if (action == "/quit") break;
            if(action=="/think") {overrides.thinking=classified.second=="on";configuration=take(edge0::resolve_generation_configuration(resources.model,overrides));std::cerr<<"Thinking: "<<classified.second<<'\n';continue;}
            if (action == "unknown") { std::cerr << "Unknown command: " << classified.second << '\n'; continue; }
            if (action == "/new") {
                bool rebuild = !state_valid;
                if (!rebuild) {
                    const auto reset = edge0::reset_conversation(resources.session);
                    if (!reset && reset.error().category == edge0::ErrorCategory::session_invalid) rebuild = true;
                    else take(reset);
                }
                if (rebuild) {
                    take(edge0::release_session(resources.session)); resources.session.reset();
                    resources.session = take(edge0::create_session(resources.model));
                }
                clear_history(); state_valid = true;
                continue;
            }
            // Commit the candidate only after normal completion, including length stops.
            auto candidate = history;
            if (action == "/regen") {
                if (candidate.empty() || candidate.back().role != "assistant") { std::cerr << "No answer to regenerate\n"; continue; }
                candidate.pop_back();
            } else candidate.push_back({"user", classified.second});
            std::optional<edge0::CancelToken> token;
            const auto fragment_limit = first_message ? sigint_fragments : std::nullopt;
            const auto delay = first_message ? sigint_seconds : std::nullopt;
            first_message = false;
            std::size_t fragments = 0;
            {
                std::lock_guard<std::mutex> lock(signals.mutex);
                token = take(edge0::create_cancel_token(resources.runtime));
                signals.token = &*token; signals.requested.reset();
            }
            struct Unpublish { Signals& signals; ~Unpublish() { std::lock_guard<std::mutex> lock(signals.mutex); signals.token = nullptr; } } unpublish{signals};
            if (fragment_limit && !*fragment_limit) signals.raise_and_wait();
            bool thinking_open=false;
            const auto generated = [&] {
                CancelTimer timer(delay);
                return edge0::generate(resources.session, candidate, configuration, {},
                [&](const edge0::OutputEvent& event) -> std::optional<edge0::Error> {
                    if(auto error=cli::output_event(event,thinking_open,true)) return error;
                    if(fragment_limit && (event.kind==edge0::OutputKind::answer || event.kind==edge0::OutputKind::reasoning) &&
                        !event.text.empty() && ++fragments==*fragment_limit) signals.raise_and_wait();
                    return {};
                }, &*token);
            }();
            cli::finish_thinking(thinking_open);
            signals.unpublish();
            if (!generated) {
                if (generated.error().category == edge0::ErrorCategory::session_invalid) state_valid = false;
                std::cerr << generated.error().message << "; use /new to start a new conversation\n";
                continue;
            }
            const auto& result = generated.value();
            std::cout << '\n';
            state_valid = result.state_valid;
            statistics(result);
            if (!output.empty()) write_turn(std::filesystem::path(output) / std::to_string(++turn), result,
                cli::with_runtime(cli::summary(result, take(edge0::memory_snapshot(resources.runtime)), loading),resources.runtime));
            history = cli::completed_history(candidate, history, result);
            if (!normal(result) && !(result.stop_reason == edge0::StopReason::cancelled && result.state_valid)) std::cerr << (result.error ? result.error->message : cli::stop(result.stop_reason)) << "; use /new to start a new conversation\n";
        }
        if (std::cin.bad()) throw std::runtime_error("Cannot read chat input");
        std::lock_guard<std::mutex> lock(signals.mutex);
        return resources.close() ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
