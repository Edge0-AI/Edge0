#include "serve_delegate.h"
#include "serve_delegate_status.h"
#include "serve_status.h"
#include "serve_protocol.h"
#include "edge0/diagnostics.h"
#include <httplib.h>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

// serve in delegate mode: edge0-strata runs on a loopback port and this process is
// its proxy. Generation endpoints are forwarded byte for byte (SSE chunk by chunk);
// /health, /v1/models and /status are answered here in Edge0's format (/health also
// carries edge0-strata's own fields); GET /v1/status, /metrics and /props pass through.
namespace delegate {
namespace {
using Json = nlohmann::ordered_json;
std::atomic<bool> interrupted{false};
void on_signal(int value) {
#ifdef _WIN32
    // The Windows CRT resets a signal handler before calling it. Keep repeated stop requests graceful.
    std::signal(value, on_signal);
#else
    (void)value;
#endif
    interrupted = true;
}
std::mutex log_mutex;
void log(const std::string& text) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cerr << text << '\n' << std::flush;
}
std::string timestamp() {
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
        << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000 << 'Z';
    return out.str();
}
Json error_body(const std::string& message, const std::string& code, int status=500, const std::string& path={}) {
    if(path=="/v1/messages" || path.rfind("/v1/messages/",0)==0)return portal::Failure(status,message,code).body(portal::Protocol::messages);
    return {{"error", {{"message", message}, {"type", "server_error"}, {"code", code}, {"param", nullptr}}}};
}

// One forwarded request: a worker thread performs the upstream call and hands the
// response over as it arrives; the HTTP handler (or its chunked provider) relays it.
struct Forward {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> chunks;
    bool headers = false, done = false, failed = false, complete = false;
    int status = 0;
    std::string content_type, body, failure_code, failure_message;
    std::atomic<bool> cancel{false};
    std::unique_ptr<httplib::Client> client;
    std::thread worker;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::string endpoint, source;
    bool stream = false, client_gone = false;
    Accounting accounting;
    std::string pending;
    void abort() {
        cancel = true;
        if (client) client->stop();
    }
    ~Forward() {
        abort();
        if (worker.joinable()) worker.join();
    }
};

struct ServeArgs {
    CommonArgs common;
    std::string host = "127.0.0.1", key_file;
    int port = 8000, write_timeout = 30;
    std::size_t max_body = 16 * 1024 * 1024;
};
ServeArgs parse(int argc, char** argv) {
    ServeArgs args;
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--verbose") continue;
        if (key == "--no-warmup") unsupported_option(key);
        if (i + 1 >= argc) throw ConfigError("invalid_argument: missing value for " + key);
        const std::string value = argv[++i];
        auto number = [&](std::int64_t limit) {
            if (value.empty() || value.size() > 18 || value.find_first_not_of("0123456789") != std::string::npos || std::stoll(value) > limit)
                throw ConfigError("invalid_argument: invalid value for " + key);
            return std::stoll(value);
        };
        if (common_argument(key, value, args.common)) {}
        else if (key == "--host") args.host = value;
        else if (key == "--port") args.port = static_cast<int>(number(65535));
        else if (key == "--api-key-file") args.key_file = value;
        else if (key == "--write-timeout-seconds") { args.write_timeout = static_cast<int>(number(INT32_MAX)); if (!args.write_timeout) throw ConfigError("invalid_argument: invalid write timeout"); }
        else if (key == "--max-body-bytes") args.max_body = static_cast<std::size_t>(number(INT64_MAX));
        else unsupported_option(key);
    }
    return args;
}
}

int serve(int argc, char** argv, const Match& match) {
    try {
        const auto args = parse(argc, argv);
        const char* cuda_path = std::getenv("CUDA_PATH");
        const auto options = resolve(args.common, match, executable_path(argv[0]), host_platform(), cuda_path ? cuda_path : "");
        for (const auto& warning : options.warnings) std::cerr << "warning: " << warning << '\n';
        std::string api_token;
        if (!args.key_file.empty()) {
            std::ifstream input(args.key_file, std::ios::binary);
            if (!input) throw ConfigError("Cannot read API key file");
            std::ostringstream content;
            content << input.rdbuf();
            api_token = content.str();
            const auto first = api_token.find_first_not_of(" \t\r\n\f\v");
            if (first == std::string::npos) throw ConfigError("API key file is empty");
            api_token = api_token.substr(first, api_token.find_last_not_of(" \t\r\n\f\v") - first + 1);
        }
        if (!portal::loopback(args.host) && args.host != "localhost" && api_token.empty())
            throw ConfigError("Non-loopback listening requires --api-key-file");

        interrupted = false;
        auto old_int = std::signal(SIGINT, on_signal), old_term = std::signal(SIGTERM, on_signal);
#ifdef _WIN32
        auto old_break = std::signal(SIGBREAK, on_signal);
        struct RestoreBreak { void (*handler)(int); ~RestoreBreak() { std::signal(SIGBREAK, handler); } } restore_break{old_break};
#endif
        struct Restore {
            void (*a)(int); void (*b)(int);
            ~Restore() { std::signal(SIGINT, a); std::signal(SIGTERM, b); }
        } restore{old_int, old_term};

        log("edge0-strata: starting " + match.registration->id + " from " + options.strata_dir.string() + " on cuda:" + std::to_string(options.gpu));
        auto launched = launch(*match.registration, options, {}, &interrupted);
        auto& engine = *launched->engine;
        const int upstream_port = engine.port();
        log("edge0-strata: ready (pid " + std::to_string(engine.pid()) + ", port " + std::to_string(upstream_port) + ")");
        auto fetch_status = [&]() -> Json {
            httplib::Client client("127.0.0.1", upstream_port);
            client.set_tcp_nodelay(true);
            client.set_connection_timeout(1);
            client.set_read_timeout(3);
            auto response = client.Get("/v1/status");
            if (!response || response->status != 200) return nullptr;
            auto value = Json::parse(response->body, nullptr, false);
            return value.is_discarded() ? Json(nullptr) : value;
        };
        std::optional<std::int64_t> context;
        if (const auto status = fetch_status(); status.is_object() && status.contains("context") && status["context"].is_object() &&
                                                status["context"].contains("native") && status["context"]["native"].is_number_integer())
            context = status["context"]["native"].get<std::int64_t>();

        const auto started = std::chrono::steady_clock::now();
        const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#ifdef _WIN32
        const auto pid = GetCurrentProcessId();
#else
        const auto pid = getpid();
#endif
        const auto started_at = timestamp();
        const std::string device = "cuda:" + std::to_string(options.gpu);
        std::atomic<bool> stopping{false};
        std::mutex state_mutex;
        ProxyHistory history;
        std::set<std::shared_ptr<Forward>> active;
        std::condition_variable idle;
        auto save = [&](const ProxyRecord& record) {
            std::lock_guard<std::mutex> lock(state_mutex);
            history.add(record);
            log(record_json(record).dump());
        };
        auto finish = [&](const std::shared_ptr<Forward>& f, int http_status) {
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                active.erase(f);
                idle.notify_all();
            }
            ProxyRecord record;
            record.time = timestamp();
            record.endpoint = f->endpoint;
            record.source = f->source;
            record.stream = f->stream;
            record.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - f->started).count();
            record.http_status = http_status;
            record.accounting = f->accounting;
            record.entered = http_status < 400 || http_status >= 500;
            if (http_status == 429) record.entered = false;
            if (f->client_gone) record.stop_reason = "cancelled";
            else if (http_status >= 400) record.stop_reason = "failed";
            else record.stop_reason = stop_reason(f->accounting.finish);
            if (http_status >= 400) {
                const auto body = Json::parse(f->body, nullptr, false);
                if (!body.is_discarded() && body.contains("error") && body["error"].is_object()) {
                    const auto& error = body["error"];
                    if (error.contains("code") && error["code"].is_string()) record.code = error["code"];
                    else if (error.contains("type") && error["type"].is_string()) record.code = error["type"];
                    if (error.contains("message") && error["message"].is_string()) record.message = error["message"];
                    if (error.contains("param") && error["param"].is_string()) record.param = error["param"];
                }
            }
            if (!f->failure_code.empty()) {record.code = f->failure_code;record.message = f->failure_message;}
            save(record);
        };

        httplib::Server server;
        server.set_tcp_nodelay(true);
        server.set_socket_options([](socket_t socket) {
#ifdef _WIN32
            const auto option = SO_EXCLUSIVEADDRUSE;
#else
            const auto option = SO_REUSEADDR;
#endif
            if (!httplib::set_socket_opt(socket, SOL_SOCKET, option, 1)) throw std::runtime_error("Cannot configure exclusive server binding");
        });
        server.new_task_queue = [] { return new httplib::ThreadPool(16); };
        server.set_payload_max_length(args.max_body);
        server.set_write_timeout(args.write_timeout);
        auto send_error = [](httplib::Response& res, int status, const std::string& message, const std::string& code, const std::string& path=std::string{}) {
            res.status = status;
            res.set_content(error_body(message, code, status, path).dump(), "application/json");
        };
        server.set_error_handler([&](const httplib::Request& req, httplib::Response& res) {
            if (res.status == 413) send_error(res, 413, "Request body too large", "request_too_large", req.path);
            else if (res.body.empty()) res.set_content(proxy_error(res.status, req.path).dump(), "application/json");
        });
        server.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
            if (portal::authorized(req.remote_addr, api_token, req.get_header_value("Authorization"), req.get_header_value("x-api-key")))
                return httplib::Server::HandlerResponse::Unhandled;
            send_error(res, 401, "Invalid API key", "invalid_api_key", req.path);
            ProxyRecord record;
            record.time = timestamp();
            record.endpoint = req.path;
            record.source = req.remote_addr;
            record.http_status = 401;
            record.code = "invalid_api_key";
            record.message = "Invalid API key";
            save(record);
            return httplib::Server::HandlerResponse::Handled;
        });
        int bound_port = 0;
        auto healthy = [&] { return !stopping && engine.running(); };
        server.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            if (stopping) send_error(res, 503, "Server shutting down", "shutting_down");
            else if (!engine.running()) send_error(res, 503, "edge0-strata has exited", "engine_exited");
            else {
                // Edge0's answer with edge0-strata's own /health fields merged in (status stays Edge0's), so that
                // tools written for edge0-strata, such as its benchmark, recognize the proxy.
                Json body{{"status", "ok"}};
                httplib::Client client("127.0.0.1", upstream_port);
                client.set_tcp_nodelay(true);
                client.set_connection_timeout(1);
                client.set_read_timeout(3);
                if (auto upstream = client.Get("/health"); upstream && upstream->status == 200) {
                    const auto value = Json::parse(upstream->body, nullptr, false);
                    if (value.is_object())
                        for (const auto& item : value.items())
                            if (item.key() != "status") body[item.key()] = item.value();
                }
                res.set_content(body.dump(), "application/json");
            }
        });
        // edge0-strata's own description endpoints, passed through unchanged for its tools (Edge0's is /status).
        for (const char* path : {"/v1/status", "/metrics", "/props"})
            server.Get(path, [&, path](const httplib::Request&, httplib::Response& res) {
                if (!healthy()) {
                    if (stopping) send_error(res, 503, "Server shutting down", "shutting_down");
                    else send_error(res, 503, "edge0-strata has exited", "engine_exited");
                    return;
                }
                httplib::Client client("127.0.0.1", upstream_port);
                client.set_tcp_nodelay(true);
                client.set_connection_timeout(5);
                client.set_read_timeout(60);
                auto upstream = client.Get(path);
                if (!upstream) {
                    send_error(res, 502, "edge0-strata did not answer " + std::string(path), "upstream_unavailable");
                    return;
                }
                res.status = upstream->status;
                res.set_content(upstream->body, upstream->has_header("Content-Type") ? upstream->get_header_value("Content-Type")
                                                                                      : std::string("application/octet-stream"));
            });
        server.Post("/v1/messages/count_tokens", [&](const httplib::Request& req, httplib::Response& res) {
            if (!healthy()) {
                send_error(res, 503, stopping ? "Server shutting down" : "edge0-strata has exited",
                           stopping ? "shutting_down" : "engine_exited", req.path);
                return;
            }
            httplib::Client client("127.0.0.1", upstream_port);
            client.set_tcp_nodelay(true);
            client.set_connection_timeout(5);
            client.set_read_timeout(60);
            client.set_path_encode(false);
            httplib::Headers headers;
            for (const char* name : {"Accept", "anthropic-version", "anthropic-beta"})
                if (req.has_header(name)) headers.emplace(name, req.get_header_value(name));
            auto upstream = client.Post(req.target.empty() ? req.path : req.target, headers, req.body,
                                        req.get_header_value("Content-Type"));
            if (!upstream) {send_error(res, 502, "edge0-strata did not answer", "upstream_unavailable", req.path);return;}
            res.status = upstream->status;
            res.set_content(upstream->body, upstream->has_header("Content-Type") ? upstream->get_header_value("Content-Type")
                                                                               : "application/octet-stream");
        });
        auto model_object = [&] {
            return Json{{"id", options.alias}, {"object", "model"}, {"created", 0}, {"owned_by", "local"},
                        {"context_length", context ? Json(*context) : Json(nullptr)},
                        {"capabilities", {{"thinking", match.registration->thinking}, {"thinking_default", match.registration->thinking_default}}}};
        };
        server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(Json{{"object", "list"}, {"data", Json::array({model_object()})}}.dump(), "application/json");
        });
        server.Get(R"(/v1/models/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
            if (req.matches[1].str() != options.alias) send_error(res, 404, "Unknown model", "model_not_found");
            else res.set_content(model_object().dump(), "application/json");
        });
        server.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
            DelegateStatusInput input;
            input.now = std::chrono::steady_clock::now();
            input.instance = {{"id", std::to_string(pid) + "-" + std::to_string(unix_ms)}, {"pid", pid}, {"started_at", started_at},
                              {"uptime_seconds", std::chrono::duration<double>(input.now - started).count()}};
            input.listen = {{"host", args.host}, {"port", bound_port}, {"auth", !api_token.empty()}};
            input.alias = options.alias;
            input.directory = args.common.model;
            input.device = device;
            input.context = context;
            input.system = portal::system_json(edge0::query_system_status());
            input.external = engine.running() ? fetch_status() : Json(nullptr);
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                input.history = history;
                for (const auto& f : active)
                    if (!input.active || f->started < input.active->started) {
                        // The accounting is written under the forward's own mutex; copying it there gives the live
                        // phase and fragment count without racing the forwarding thread.
                        std::lock_guard<std::mutex> stream_lock(f->mutex);
                        input.active = ActiveRequest{f->endpoint, f->source, f->started, f->accounting};
                    }
                // Taken after the copy, so the elapsed time is never negative and never below the time stamped on
                // the first fragment of the copied accounting.
                input.now = std::chrono::steady_clock::now();
            }
            res.set_content(status_json(input).dump(), "application/json");
        });
        auto forward = [&](const httplib::Request& req, httplib::Response& res) {
            auto f = std::make_shared<Forward>();
            f->endpoint = req.path;
            f->source = req.remote_addr;
            f->accounting.endpoint = req.path;
            {
                const auto body = Json::parse(req.body, nullptr, false);
                f->stream = !body.is_discarded() && body.is_object() && body.contains("stream") && body["stream"] == true;
                f->accounting.streaming = f->stream;
            }
            if (!healthy()) {
                f->body = stopping ? error_body("Server shutting down", "shutting_down",503,req.path).dump()
                                   : error_body("edge0-strata has exited", "engine_exited",503,req.path).dump();
                res.status = 503;
                res.set_content(f->body, "application/json");
                finish(f, 503);
                return;
            }
            f->client = std::make_unique<httplib::Client>("127.0.0.1", upstream_port);
            f->client->set_tcp_nodelay(true);
            ActiveGuard guard([&] {
                f->abort();
                if (f->worker.joinable()) f->worker.join();
                f->failure_code = "internal_error";
                f->failure_message = "Internal error";
                finish(f, 500);
            });
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                active.insert(f);
            }
            guard.arm();
            f->client->set_connection_timeout(5);
            f->client->set_read_timeout(24 * 3600);
            f->client->set_write_timeout(60);
            f->client->set_path_encode(false);  // the target goes out exactly as received
            httplib::Request upstream;
            upstream.method = "POST";
            upstream.path = req.target.empty() ? req.path : req.target;  // keeps the query string (?beta=true)
            upstream.body = req.body;
            for (const char* name : {"Content-Type", "Accept", "anthropic-version", "anthropic-beta"})
                if (req.has_header(name)) upstream.set_header(name, req.get_header_value(name));
            std::weak_ptr<Forward> weak = f;
            Forward* raw = f.get();
            upstream.response_handler = [raw](const httplib::Response& response) {
                std::lock_guard<std::mutex> lock(raw->mutex);
                raw->status = response.status;
                raw->content_type = response.get_header_value("Content-Type");
                raw->headers = true;
                raw->changed.notify_all();
                return !raw->cancel;
            };
            upstream.content_receiver = [raw](const char* data, std::size_t size, std::size_t, std::size_t) {
                std::lock_guard<std::mutex> lock(raw->mutex);
                raw->chunks.emplace_back(data, size);
                raw->changed.notify_all();
                return !raw->cancel;
            };
            f->worker = std::thread([raw, upstream] {
                auto result = raw->client->send(upstream);
                std::lock_guard<std::mutex> lock(raw->mutex);
                raw->failed = !result;
                raw->done = true;
                raw->changed.notify_all();
            });
            {
                std::unique_lock<std::mutex> lock(f->mutex);
                while (!f->headers && !f->done) {
                    f->changed.wait_for(lock, std::chrono::milliseconds(20));
                    if (!f->headers && req.is_connection_closed()) { f->client_gone = true; lock.unlock(); f->abort(); lock.lock(); }
                }
            }
            auto incomplete = [&, f] {
                for (int i = 0; i < 20 && !f->client_gone && !stopping && engine.running(); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                const int status = f->client_gone ? 499 : stopping || !engine.running() ? 503 : 502;
                f->failure_code = f->client_gone ? "cancelled" : stopping ? "shutting_down" : status == 503 ? "engine_exited" : "engine_unavailable";
                f->failure_message = f->client_gone ? "Client disconnected" : stopping ? "Server shutting down" : status == 503 ? "edge0-strata has exited" : "edge0-strata did not finish its response";
                f->status = status;
                f->body = error_body(f->failure_message, f->failure_code, status, f->endpoint).dump();
            };
            if (!f->headers) {
                if (f->worker.joinable()) f->worker.join();
                incomplete();
                res.status = f->status;
                res.set_content(f->body, "application/json");
                finish(f, f->status);
                guard.release();
                return;
            }
            res.status = f->status;
            if (f->content_type.compare(0, 17, "text/event-stream") == 0) {
                res.set_chunked_content_provider(
                    f->content_type,
                    [f, &req, incomplete, &options](std::size_t, httplib::DataSink& sink) {
                        std::unique_lock<std::mutex> lock(f->mutex);
                        for (;;) {
                            while (!f->chunks.empty()) {
                                auto chunk = std::move(f->chunks.front());
                                f->chunks.pop_front();
                                try {
                                    // The request's age when this chunk is forwarded; the first text fragment is
                                    // stamped with it (current.first_token_seconds).
                                    const double elapsed =
                                        std::chrono::duration<double>(std::chrono::steady_clock::now() - f->started).count();
                                    account_sse(f->accounting, f->pending, chunk.data(), chunk.size(), elapsed);
                                } catch (const std::exception&) {
                                    // Accounting never affects forwarding; the record keeps what was read.
                                }
                                lock.unlock();
                                const bool written = sink.write(chunk.data(), chunk.size());
                                lock.lock();
                                if (!written) { f->client_gone = true; lock.unlock(); f->abort(); return false; }
                            }
                            if (f->done) break;
                            f->changed.wait_for(lock, std::chrono::milliseconds(20));
                            if (f->chunks.empty() && !f->done && (!sink.is_writable() || req.is_connection_closed())) {
                                f->client_gone = true;
                                lock.unlock();
                                f->abort();
                                return false;
                            }
                        }
                        f->complete = !f->failed && f->accounting.complete;
                        lock.unlock();
                        if (!f->complete) {
                            incomplete();
                            if (f->client_gone) return false;
                            portal::Request request;
                            request.protocol = f->endpoint == "/v1/messages" ? portal::Protocol::messages
                                : f->endpoint == "/v1/responses" ? portal::Protocol::responses : portal::Protocol::chat;
                            portal::Encoder encoder(request, options.alias, "");
                            const portal::Failure error(f->status, f->failure_message, f->failure_code);
                            // A truncated last line must not consume the appended event as part of its data.
                            if (!sink.write("\n\n", 2)) {f->client_gone = true;return false;}
                            for (const auto& event : encoder.fail(error, f->accounting.response_id, f->accounting.next_sequence))
                                if (!sink.write(event.data(), event.size())) {f->client_gone = true;return false;}
                        }
                        sink.done();
                        return true;
                    },
                    [f, finish](bool) {
                        f->abort();
                        if (f->worker.joinable()) f->worker.join();
                        finish(f, f->status);
                    });
                guard.release();
                return;
            }
            {
                std::unique_lock<std::mutex> lock(f->mutex);
                while (!f->done) {
                    f->changed.wait_for(lock, std::chrono::milliseconds(20));
                    if (!f->done && req.is_connection_closed()) { f->client_gone = true; lock.unlock(); f->abort(); lock.lock(); }
                }
                for (auto& chunk : f->chunks) f->body += chunk;
                f->chunks.clear();
            }
            if (f->worker.joinable()) f->worker.join();
            const bool json_content = f->content_type.find("application/json") != std::string::npos || f->content_type.find("+json") != std::string::npos;
            f->complete = !f->failed && (!json_content || !Json::parse(f->body, nullptr, false).is_discarded());
            if (!f->complete) {
                incomplete();
                res.status = f->status;
                res.set_content(f->body, "application/json");
                finish(f, f->status);
                guard.release();
                return;
            }
            try {
                const auto parsed = Json::parse(f->body, nullptr, false);
                // Under the forward's mutex: /status copies the accounting of a request that is still active.
                std::lock_guard<std::mutex> lock(f->mutex);
                account(f->accounting, parsed);
            } catch (const std::exception&) {
                // Accounting never affects forwarding: the body goes out as received and the request is still
                // recorded (fields that could not be read stay null) and leaves the active set.
            }
            res.set_content(f->body, f->content_type.empty() ? "application/json" : f->content_type.c_str());
            finish(f, f->status);
            guard.release();
        };
        for (const char* path : {"/v1/chat/completions", "/v1/responses", "/v1/messages"}) server.Post(path, forward);

        const int port = args.port == 0 ? server.bind_to_any_port(args.host) : (server.bind_to_port(args.host, args.port) ? args.port : -1);
        if (port < 0) throw std::runtime_error("Cannot bind server");
        bound_port = port;
        std::atomic<bool> done{false};
        std::thread watcher([&] {
            bool reported = false;
            while (!done) {
                if (!reported && !engine.running()) {
                    reported = true;
                    log("edge0-strata exited (exit code " + std::to_string(engine.exit_code().value_or(-1)) + "); generation requests now return 503");
                }
                if (interrupted) {
                    log("stopping acceptance");
                    stopping = true;
                    server.stop();
                    std::unique_lock<std::mutex> lock(state_mutex);
                    if (!idle.wait_for(lock, std::chrono::seconds(60), [&] { return active.empty(); })) {
                        log("active requests did not finish within 60 s; cancelling them");
                        for (const auto& f : active) f->abort();
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        log("Listening: http://" + args.host + ":" + std::to_string(port) + " model=" + options.alias);
        const bool listened = server.listen_after_bind();
        done = true;
        watcher.join();
        log("active requests finished");
        engine.stop(60);
        log("edge0-strata stopped (exit code " + std::to_string(engine.exit_code().value_or(-1)) + ")");
        return listened ? 0 : 1;
    } catch (const ConfigError& error) {
        log(std::string("configuration_error: ") + error.what());
        return 2;
    } catch (const std::exception& error) {
        log(error.what());
        return 1;
    }
}
}
