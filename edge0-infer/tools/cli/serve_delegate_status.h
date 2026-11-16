#pragma once
// /status, request records and response accounting for the delegate-mode proxy.
// Pure functions: the proxy feeds them what it forwarded and what edge0-strata's
// /v1/status reported; nothing here performs I/O.
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <optional>
#include <string>
#include <vector>

namespace delegate {
using StatusJson = nlohmann::ordered_json;

// What the proxy can learn from one forwarded response (the external engine's own numbers).
struct Accounting {
    std::optional<std::int64_t> input_tokens, reused_tokens, generated_tokens, prompt_tokens_read, predicted_tokens;
    std::optional<double> prompt_seconds, decode_seconds, prefill_tokens_per_second, decode_tokens_per_second;
    // Live progress of a request still in flight. The external engine reports its counts only in the last chunk,
    // so the proxy counts what it forwards: `streamed_tokens` is the number of streamed fragments that carried
    // text (one token each), `first_token_seconds` the request's age when the first of them arrived.  `streaming`
    // says whether the request asked for a stream; a non-streaming request has no fragments to count.  None of
    // these is used once the request ends: its record keeps only the engine's own numbers.
    std::optional<std::int64_t> streamed_tokens;
    std::optional<double> first_token_seconds;
    bool saw_content = false, streaming = false, complete = false;
    std::string endpoint, response_id;
    std::size_t next_sequence = 0;
    std::string finish;  // the external engine's finish reason, as sent
};
// Folds one JSON body or SSE data object into the accounting (later values win).
void account(Accounting&, const StatusJson& body);
// The number of text-carrying fragments in one streamed event (0 or 1 per Chat choice, 0 or 1 for a Responses or
// Messages event): Chat `delta.content`, `delta.reasoning_content` or `delta.tool_calls[].function.arguments`;
// Responses `response.output_text.delta`, `response.reasoning_text.delta`, `response.function_call_arguments.delta`
// and `response.custom_tool_call_input.delta`; Messages `content_block_delta` with a `text_delta`, `thinking_delta`
// or `input_json_delta`.  Empty text, a role, a finish reason, usage or timings alone count 0.
std::int64_t content_fragments(const StatusJson& event);
// Splits SSE text into events and accounts each data object; keeps the unfinished tail in `pending`.  `elapsed` is
// the request's age when this chunk arrived: the first fragment that carries text is stamped with it.
void account_sse(Accounting&, std::string& pending, const char* data, std::size_t size, double elapsed = 0);
std::string stop_reason(const std::string& finish);

struct ProxyRecord {
    std::string time, endpoint, source;
    int http_status = 0;
    bool stream = false, entered = false;
    double total_seconds = 0;
    std::string code, message, param, stop_reason;
    Accounting accounting;
};
StatusJson record_json(const ProxyRecord&);

struct ProxyTotals {
    std::size_t requests = 0, rejected_busy = 0, rejected_invalid = 0, unauthorized = 0;
    std::int64_t input_tokens = 0, reused_tokens = 0, generated_tokens = 0, prompt_tokens_read = 0, predicted_tokens = 0;
    double decode_seconds = 0, prompt_seconds = 0, first_seconds = 0;
    std::size_t first_count = 0;
    double decode_seconds_rated = 0;
};
struct ProxyHistory {
    ProxyTotals totals;
    std::deque<ProxyRecord> recent;
    std::optional<ProxyRecord> last;
    void add(const ProxyRecord&);
};

struct ActiveRequest {
    std::string endpoint, source;
    std::chrono::steady_clock::time_point started;
    // A copy of the forwarded response's accounting, taken under the forward's mutex when the status is built.
    Accounting accounting;
};
struct DelegateStatusInput {
    StatusJson instance, listen, system;
    std::string alias, directory, device;
    std::optional<std::int64_t> context;
    StatusJson external = nullptr;  // edge0-strata's /v1/status, or null when it did not answer
    std::optional<ActiveRequest> active;
    std::chrono::steady_clock::time_point now;
    ProxyHistory history;
};
StatusJson status_json(const DelegateStatusInput&);
// Arm only after registration; release after finishing or handing ownership to a provider.
class ActiveGuard {
public:
    explicit ActiveGuard(std::function<void()> finish) : finish_(std::move(finish)) {}
    ActiveGuard(const ActiveGuard&) = delete;
    ActiveGuard& operator=(const ActiveGuard&) = delete;
    ~ActiveGuard() noexcept { if (armed_) { try { finish_(); } catch (...) {} } }
    void arm() noexcept { armed_ = true; }
    void release() noexcept { armed_ = false; }
private:
    std::function<void()> finish_;
    bool armed_ = false;
};
StatusJson proxy_error(int status, const std::string& path = {});
// Replaces system.memory with the external engine's machine.ram reading.
StatusJson system_memory(const StatusJson& portal_system, const StatusJson& external);
}
