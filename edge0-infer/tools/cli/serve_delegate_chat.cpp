#include "serve_delegate.h"
#include "serve_delegate_status.h"
#include "chat_line.h"
#include "console_input.h"
#include "edge0/generation.h"
#include <httplib.h>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

namespace cli {
std::vector<edge0::TextMessage> completed_history(const std::vector<edge0::TextMessage>&, const std::vector<edge0::TextMessage>&, const edge0::GenerationResult&);
}

// chat in delegate mode: edge0-strata runs on a loopback port and this process is
// its HTTP client. Commands and history rules are those of the native chat.
namespace delegate {
namespace {
using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
std::atomic<bool> interrupted{false};
void on_sigint(int) {
#ifdef _WIN32
    std::signal(SIGINT, on_sigint);
#endif
    interrupted = true;
}

struct ChatArgs {
    CommonArgs common;
    std::string system;
    bool has_system = false;
    std::optional<std::size_t> sigint_fragments;
    std::optional<double> sigint_seconds;
};
ChatArgs parse(int argc, char** argv) {
    ChatArgs args;
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--verbose") continue;
        if (key == "--no-warmup") unsupported_option(key);
        if (i + 1 >= argc) throw ConfigError("invalid_argument: missing value for " + key);
        const std::string value = argv[++i];
        if (common_argument(key, value, args.common)) {}
        else if (key == "--system") { args.system = value; args.has_system = true; }
        else if (key == "--sigint-after-fragments") {
            if (value.empty() || value.size() > 9 || value.find_first_not_of("0123456789") != std::string::npos)
                throw ConfigError("invalid_argument: invalid fragment count");
            args.sigint_fragments = static_cast<std::size_t>(std::stoull(value));
        } else if (key == "--sigint-after-seconds") {
            std::size_t used = 0;
            const auto seconds = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(seconds) || seconds < 0) throw ConfigError("invalid_argument: invalid cancellation delay");
            args.sigint_seconds = seconds;
        } else unsupported_option(key);
    }
    return args;
}

struct Turn {
    std::string text, reasoning, error;
    int status = 0;
    bool cancelled = false, transport_failed = false;
    Accounting accounting;
};
}

int chat(int argc, char** argv, const Match& match) {
    try {
        const auto args = parse(argc, argv);
        const char* cuda_path = std::getenv("CUDA_PATH");
        const auto options = resolve(args.common, match, executable_path(argv[0]), host_platform(), cuda_path ? cuda_path : "");
        for (const auto& warning : options.warnings) std::cerr << "warning: " << warning << '\n';
        interrupted = false;
        auto previous = std::signal(SIGINT, on_sigint);
        struct Restore { void (*handler)(int); ~Restore() { std::signal(SIGINT, handler); } } restore{previous};
        TempDir logs;
        const auto log_file = logs.path / "edge0-strata.log";
        const auto started = Clock::now();
        std::cerr << "edge0-strata: starting " << match.registration->id << " on " << options.device
                  << " (log: " << log_file.string() << ")\n" << std::flush;
        // When edge0-strata fails (to start, or later), the last lines of its log are printed and the log is kept.
        auto report_log = [&] {
            std::ifstream input(log_file);
            std::string line;
            std::deque<std::string> tail;
            while (std::getline(input, line)) { tail.push_back(line); if (tail.size() > 20) tail.pop_front(); }
            for (const auto& item : tail) std::cerr << "edge0-strata: " << item << '\n';
            std::cerr << "edge0-strata: log kept at " << log_file.string() << '\n';
            logs.keep();
        };
        std::unique_ptr<Launch> launched;
        try {
            launched = launch(*match.registration, options, log_file, &interrupted);
        } catch (const ConfigError&) {
            throw;  // raised before edge0-strata starts: its log is empty, and the directory goes with `logs`
        } catch (...) {
            // Preparation fails before the engine log exists. Do not name that file or keep the directory.
            std::error_code missing;
            if (std::filesystem::is_regular_file(log_file, missing)) report_log();
            throw;
        }
        auto& engine = *launched->engine;
        const int port = engine.port();
        interrupted = false;

        std::mutex mutex;
        bool generating = false, done = false;
        httplib::Client* current = nullptr;
        std::condition_variable wake;
        std::thread watcher([&] {
            std::unique_lock<std::mutex> lock(mutex);
            while (!done) {
                if (interrupted) {
                    if (!generating) {
                        lock.unlock();
                        engine.stop(60);
                        // _Exit skips destructors: remove the private directories here.
                        launched->temp.remove();
                        logs.remove();
                        std::cerr.flush();
                        std::_Exit(130);
                    }
                    if (current) current->stop();
                }
                wake.wait_for(lock, std::chrono::milliseconds(10));
            }
        });
        // Ends the watcher. Called by the main thread only. If the watcher has already begun handling a Ctrl-C at
        // the prompt, joining it waits for that to finish (it exits the process with 130), so the main thread never
        // stops the engine or removes the directories at the same time as the watcher.
        auto end_watcher = [&] {
            { std::lock_guard<std::mutex> lock(mutex); done = true; }
            wake.notify_all();
            if (watcher.joinable()) watcher.join();
        };
        struct Join {
            std::function<void()> end;
            ~Join() { end(); }
        } join{end_watcher};

        std::cerr << "Ready: load " << std::chrono::duration<double>(Clock::now() - started).count()
                  << " s (edge0-strata, pid " << engine.pid() << "); /regen /new /quit\n";
        std::vector<edge0::TextMessage> history;
        auto clear_history = [&] { history.clear(); if (args.has_system) history.push_back({"system", args.system}); };
        clear_history();
        std::optional<bool> thinking;
        bool first_message = true, engine_exited = false;
        cli::ChatLines lines;
        std::string line;
        while (std::cerr << (lines.collecting() ? "... " : "> ") << std::flush, cli::read_input_line(line)) {
            auto classified = lines.feed(line);
            if (classified.first == "/file") {
                std::ifstream input(classified.second, std::ios::binary);
                if (!input) { std::cerr << "Cannot read file\n"; continue; }
                std::ostringstream contents;
                contents << input.rdbuf();
                if (input.bad() || contents.str().empty()) { std::cerr << "Empty or unreadable file\n"; continue; }
                classified = {"message", contents.str()};
            }
            const auto& action = classified.first;
            if (action == "empty") continue;
            if (action == "/quit") break;
            if (action == "/think") { thinking = classified.second == "on"; std::cerr << "Thinking: " << classified.second << '\n'; continue; }
            if (action == "unknown") { std::cerr << "Unknown command: " << classified.second << '\n'; continue; }
            if (action == "/new") { clear_history(); continue; }
            auto candidate = history;
            if (action == "/regen") {
                if (candidate.empty() || candidate.back().role != "assistant") { std::cerr << "No answer to regenerate\n"; continue; }
                candidate.pop_back();
            } else candidate.push_back({"user", classified.second});
            if (!engine.running()) {
                engine_exited = true;
                break;
            }
            Json messages = Json::array();
            for (const auto& message : candidate) messages.push_back({{"role", message.role}, {"content", message.content}});
            Json body{{"model", options.alias}, {"messages", messages}, {"stream", true}};
            if (thinking) body["chat_template_kwargs"] = {{"enable_thinking", *thinking}};
            const auto fragment_limit = first_message ? args.sigint_fragments : std::nullopt;
            const auto delay = first_message ? args.sigint_seconds : std::nullopt;
            first_message = false;

            Turn turn;
            httplib::Client client("127.0.0.1", port);
            client.set_tcp_nodelay(true);
            client.set_connection_timeout(5);
            client.set_read_timeout(24 * 3600);
            {
                std::lock_guard<std::mutex> lock(mutex);
                generating = true;
                current = &client;
            }
            std::thread timer;
            std::mutex timer_mutex;
            std::condition_variable timer_wake;
            bool timer_done = false;
            if (delay) timer = std::thread([&] {
                std::unique_lock<std::mutex> lock(timer_mutex);
                if (!timer_wake.wait_for(lock, std::chrono::duration<double>(*delay), [&] { return timer_done; })) std::raise(SIGINT);
            });
            if (fragment_limit && *fragment_limit == 0) std::raise(SIGINT);
            bool thinking_open = false;
            std::size_t fragments = 0;
            std::string pending;
            httplib::Request request;
            request.method = "POST";
            request.path = "/v1/chat/completions";
            request.body = body.dump();
            request.set_header("Content-Type", "application/json");
            request.response_handler = [&](const httplib::Response& response) { turn.status = response.status; return !interrupted; };
            request.content_receiver = [&](const char* data, std::size_t size, std::size_t, std::size_t) {
                if (turn.status != 200) { turn.error.append(data, size); return !interrupted.load(); }
                pending.append(data, size);
                std::size_t start = 0;
                for (;;) {
                    const auto end = pending.find('\n', start);
                    if (end == std::string::npos) break;
                    std::string event = pending.substr(start, end - start);
                    start = end + 1;
                    if (!event.empty() && event.back() == '\r') event.pop_back();
                    if (event.compare(0, 5, "data:") != 0) continue;
                    auto payload = event.substr(5);
                    if (!payload.empty() && payload.front() == ' ') payload.erase(0, 1);
                    if (payload == "[DONE]") continue;
                    const auto value = Json::parse(payload, nullptr, false);
                    if (value.is_discarded()) continue;
                    account(turn.accounting, value);
                    if (!value.contains("choices") || !value["choices"].is_array() || value["choices"].empty()) continue;
                    const auto& delta = value["choices"][0].value("delta", Json::object());
                    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                        const auto text = delta["reasoning_content"].get<std::string>();
                        if (!text.empty()) {
                            if (!thinking_open) { std::cerr << "[thinking]\n"; thinking_open = true; }
                            std::cerr << text << std::flush;
                            turn.reasoning += text;
                            if (fragment_limit && ++fragments == *fragment_limit) std::raise(SIGINT);
                        }
                    }
                    if (delta.contains("content") && delta["content"].is_string()) {
                        const auto text = delta["content"].get<std::string>();
                        if (!text.empty()) {
                            if (thinking_open) { std::cerr << "\n[/thinking]\n" << std::flush; thinking_open = false; }
                            std::cout << text << std::flush;
                            turn.text += text;
                            if (fragment_limit && ++fragments == *fragment_limit) std::raise(SIGINT);
                        }
                    }
                }
                pending.erase(0, start);
                return !interrupted.load();
            };
            const auto result = client.send(request);
            if (timer.joinable()) {
                { std::lock_guard<std::mutex> lock(timer_mutex); timer_done = true; }
                timer_wake.notify_all();
                timer.join();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                generating = false;
                current = nullptr;
                turn.cancelled = interrupted.exchange(false);
            }
            if (thinking_open) std::cerr << "\n[/thinking]\n" << std::flush;
            turn.transport_failed = !result && !turn.cancelled;
            if (turn.transport_failed) {
                // A connection that fails because edge0-strata exited ends the chat, as it does before a request.
                for (int i = 0; i < 20 && engine.running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!engine.running()) {
                    engine_exited = true;
                    break;
                }
            }
            if (turn.transport_failed || (turn.status != 200 && !turn.cancelled)) {
                std::string message = turn.transport_failed ? "edge0-strata did not answer" : "edge0-strata returned HTTP " + std::to_string(turn.status);
                const auto error = Json::parse(turn.error, nullptr, false);
                if (!error.is_discarded() && error.contains("error") && error["error"].is_object() && error["error"].contains("message"))
                    message += ": " + error["error"]["message"].dump();
                std::cerr << message << "; use /new to start a new conversation\n";
                continue;
            }
            std::cout << '\n';
            edge0::GenerationResult result_view;
            result_view.text = turn.text;
            result_view.reasoning = turn.reasoning;
            result_view.state_valid = true;
            const auto stop = turn.cancelled ? std::string("cancelled") : stop_reason(turn.accounting.finish);
            result_view.stop_reason = stop == "end_of_generation" ? edge0::StopReason::end_of_generation
                                      : stop == "length"           ? edge0::StopReason::length
                                      : stop == "stop_string"      ? edge0::StopReason::stop_string
                                      : stop == "tool_calls"       ? edge0::StopReason::tool_calls
                                      : stop == "cancelled"        ? edge0::StopReason::cancelled
                                                                   : edge0::StopReason::failed;
            const auto& a = turn.accounting;
            std::cerr << "[edge0-strata] reused " << (a.reused_tokens ? std::to_string(*a.reused_tokens) : "n/a")
                      << ", computed " << (a.prompt_tokens_read ? std::to_string(*a.prompt_tokens_read) : "n/a") << ", first token ";
            if (a.prompt_seconds) std::cerr << *a.prompt_seconds << " s"; else std::cerr << "n/a";
            std::cerr << ", ";
            if (a.decode_tokens_per_second) std::cerr << *a.decode_tokens_per_second << " tok/s"; else std::cerr << "n/a tok/s";
            std::cerr << ", " << stop << '\n';
            history = cli::completed_history(candidate, history, result_view);
            if (result_view.stop_reason == edge0::StopReason::failed) std::cerr << stop << "; use /new to start a new conversation\n";
        }
        end_watcher();
        if (interrupted) {
            // Ctrl-C at the prompt that the watcher had not acted on yet (input ended at the same moment).
            engine.stop(60);
            return 130;
        }
        if (engine_exited) {
            // A failure, like a failed start: exit code 1, with the log's last lines and its path.
            engine.stop(60);
            std::cerr << "edge0-strata has exited (exit code " << engine.exit_code().value_or(-1) << "); restart chat\n";
            report_log();
            return 1;
        }
        if (std::cin.bad()) throw std::runtime_error("Cannot read chat input");
        engine.stop(60);
        return 0;
    } catch (const ConfigError& error) {
        std::cerr << "configuration_error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
}
