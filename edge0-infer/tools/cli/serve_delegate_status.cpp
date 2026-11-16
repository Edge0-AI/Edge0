#include "serve_delegate_status.h"
#include "serve_protocol.h"
#include <limits>

#include <exception>

namespace delegate {
namespace {
using Json = StatusJson;
std::optional<std::int64_t> integer(const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_number()) return std::nullopt;
    return object[key].get<std::int64_t>();
}
std::optional<double> number(const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_number()) return std::nullopt;
    return object[key].get<double>();
}
template <class T> void take(std::optional<T>& target, const std::optional<T>& value) {
    if (value) target = value;
}
template <class T> Json nullable(const std::optional<T>& value) { return value ? Json(*value) : Json(nullptr); }
// A string field, or "" when it is missing or not a string (a field of the wrong type counts as missing).
std::string text(const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_string()) return "";
    return object[key].get<std::string>();
}
void finish_from(Accounting& a, const Json& value) {
    if (value.is_string() && !value.get<std::string>().empty()) a.finish = value.get<std::string>();
}
// A Responses object's `status` is a finish only when terminal; in_progress and queued leave it unknown.
void responses_status(Accounting& a, const Json& response) {
    const auto status = text(response, "status");
    if (status == "completed" || status == "incomplete" || status == "failed" || status == "cancelled") a.finish = status;
}
}

void account(Accounting& a, const Json& body) {
    if (!body.is_object()) return;
    const std::string type = text(body, "type");
    if (body.contains("usage") && body["usage"].is_object()) {
        const auto& usage = body["usage"];
        const auto prompt = integer(usage, "prompt_tokens");
        take(a.input_tokens, prompt);
        if (prompt) {
            // Chat usage: prompt_tokens stands even when the body also carries input_tokens.
        } else if (type == "message") {
            // Anthropic Messages: input_tokens counts only the uncached part; the request's input is all three
            // (a missing one counts as 0).
            const auto uncached = integer(usage, "input_tokens");
            const auto created = integer(usage, "cache_creation_input_tokens");
            const auto read = integer(usage, "cache_read_input_tokens");
            if (uncached || created || read) a.input_tokens = uncached.value_or(0) + created.value_or(0) + read.value_or(0);
        } else if (type != "message_delta") {
            // message_delta repeats input counts; the one in message_start stands, the delta only adds the output.
            take(a.input_tokens, integer(usage, "input_tokens"));
        }
        take(a.generated_tokens, integer(usage, "completion_tokens"));
        take(a.generated_tokens, integer(usage, "output_tokens"));
        if (usage.contains("prompt_tokens_details") && !a.reused_tokens)
            take(a.reused_tokens, integer(usage["prompt_tokens_details"], "cached_tokens"));
    }
    if (body.contains("timings") && body["timings"].is_object()) {
        const auto& t = body["timings"];
        take(a.reused_tokens, integer(t, "cache_n"));
        take(a.prompt_tokens_read, integer(t, "prompt_n"));
        take(a.predicted_tokens, integer(t, "predicted_n"));
        if (auto ms = number(t, "prompt_ms")) a.prompt_seconds = *ms / 1000;
        if (auto ms = number(t, "predicted_ms")) a.decode_seconds = *ms / 1000;
        take(a.prefill_tokens_per_second, number(t, "prompt_per_second"));
        take(a.decode_tokens_per_second, number(t, "predicted_per_second"));
    }
    if (body.contains("choices") && body["choices"].is_array())
        for (const auto& choice : body["choices"])
            if (choice.is_object() && choice.contains("finish_reason")) finish_from(a, choice["finish_reason"]);
    if (body.contains("stop_reason")) finish_from(a, body["stop_reason"]);
    // A Responses object (the non-streaming body, or nested in a streamed event) states its finish in `status`.
    if (text(body, "object") == "response") responses_status(a, body);
    if (body.contains("delta") && body["delta"].is_object() && body["delta"].contains("stop_reason"))
        finish_from(a, body["delta"]["stop_reason"]);
    // Responses and Anthropic Messages nest the final object.
    for (const char* key : {"response", "message"})
        if (body.contains(key) && body[key].is_object()) {
            account(a, body[key]);
            if (std::string(key) == "response") responses_status(a, body[key]);
        }
}

std::int64_t content_fragments(const Json& event) {
    if (!event.is_object()) return 0;
    std::int64_t fragments = 0;
    if (event.contains("choices") && event["choices"].is_array())
        for (const auto& choice : event["choices"]) {
            if (!choice.is_object() || !choice.contains("delta") || !choice["delta"].is_object()) continue;
            const auto& delta = choice["delta"];
            bool carries = !text(delta, "content").empty() || !text(delta, "reasoning_content").empty();
            if (!carries && delta.contains("tool_calls") && delta["tool_calls"].is_array())
                for (const auto& call : delta["tool_calls"])
                    if (call.is_object() && call.contains("function") && !text(call["function"], "arguments").empty()) carries = true;
            if (carries) ++fragments;
        }
    const std::string type = text(event, "type");
    if (type == "response.output_text.delta" || type == "response.reasoning_text.delta" ||
        type == "response.function_call_arguments.delta" || type == "response.custom_tool_call_input.delta")
        if (!text(event, "delta").empty()) ++fragments;
    if (type == "content_block_delta" && event.contains("delta") && event["delta"].is_object()) {
        const auto& delta = event["delta"];
        const std::string kind = text(delta, "type");
        if ((kind == "text_delta" && !text(delta, "text").empty()) || (kind == "thinking_delta" && !text(delta, "thinking").empty()) ||
            (kind == "input_json_delta" && !text(delta, "partial_json").empty()))
            ++fragments;
    }
    return fragments;
}

void account_sse(Accounting& a, std::string& pending, const char* data, std::size_t size, double elapsed) {
    pending.append(data, size);
    std::size_t start = 0;
    for (;;) {
        const auto end = pending.find('\n', start);
        if (end == std::string::npos) break;
        std::string line = pending.substr(start, end - start);
        start = end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.compare(0, 5, "data:") != 0) continue;
        auto payload = line.substr(5);
        if (!payload.empty() && payload.front() == ' ') payload.erase(0, 1);
        if (payload == "[DONE]") {
            if (a.endpoint == "/v1/chat/completions") a.complete = true;
            continue;
        }
        const auto value = Json::parse(payload, nullptr, false);
        if (value.is_discarded()) continue;
        try {
            const auto type = text(value, "type");
            if (a.endpoint == "/v1/responses") {
                if (type == "response.completed" || type == "response.incomplete" || type == "response.failed") a.complete = true;
                if (value.contains("response") && value["response"].is_object()) {
                    const auto id = text(value["response"], "id");
                    if (!id.empty()) a.response_id = id;
                }
                const auto id = text(value, "response_id");
                if (!id.empty()) a.response_id = id;
                if (value.contains("sequence_number") && value["sequence_number"].is_number_integer() && value["sequence_number"] >= 0) {
                    const auto n = value["sequence_number"].get<std::uint64_t>();
                    if (n < std::numeric_limits<std::size_t>::max()) a.next_sequence = std::max(a.next_sequence, static_cast<std::size_t>(n) + 1);
                }
            } else if (a.endpoint == "/v1/messages" && (type == "message_stop" || type == "error")) a.complete = true;
            else if (a.endpoint == "/v1/chat/completions" && value.contains("error")) a.complete = true;
            account(a, value);
            if (const auto fragments = content_fragments(value)) {
                a.streamed_tokens = a.streamed_tokens.value_or(0) + fragments;
                if (!a.saw_content) a.first_token_seconds = elapsed;
                a.saw_content = true;
            }
        } catch (const std::exception&) {
            // Accounting never affects forwarding: an event that cannot be read is left out of the record.
        }
    }
    pending.erase(0, start);
}

StatusJson proxy_error(int status, const std::string& path) {
    return portal::http_failure(status).body(
            path == "/v1/messages" || path.rfind("/v1/messages/", 0) == 0 ? portal::Protocol::messages : portal::Protocol::chat);
}

std::string stop_reason(const std::string& finish) {
    if (finish == "stop" || finish == "end_turn" || finish == "completed") return "end_of_generation";
    if (finish == "length" || finish == "max_tokens" || finish == "incomplete") return "length";
    if (finish == "tool_calls" || finish == "tool_use") return "tool_calls";
    if (finish == "stop_sequence") return "stop_string";
    if (finish == "cancel" || finish == "cancelled") return "cancelled";
    return finish.empty() ? "unknown" : finish;
}

Json record_json(const ProxyRecord& r) {
    Json result{{"time", r.time}, {"endpoint", r.endpoint}, {"source", r.source}, {"http_status", r.http_status},
                {"stream", r.stream}, {"total_seconds", r.total_seconds}};
    if (!r.code.empty()) result["code"] = r.code;
    if (!r.message.empty()) result["message"] = r.message;
    if (!r.param.empty()) result["param"] = r.param;
    if (r.entered) {
        const auto& a = r.accounting;
        result["stop_reason"] = r.stop_reason;
        result["input_tokens"] = nullable(a.input_tokens);
        result["reused_tokens"] = nullable(a.reused_tokens);
        result["generated_tokens"] = nullable(a.generated_tokens);
        result["first_token_seconds"] = nullable(a.prompt_seconds);
        result["decode_tokens_per_second"] = nullable(a.decode_tokens_per_second);
        result["prefill_tokens_per_second"] = nullable(a.prefill_tokens_per_second);
        result["decode_seconds"] = nullable(a.decode_seconds);
    }
    return result;
}

void ProxyHistory::add(const ProxyRecord& r) {
    auto& t = totals;
    if (!r.entered) {
        if (r.http_status == 401) ++t.unauthorized;
        else if (r.http_status == 429) ++t.rejected_busy;
        else ++t.rejected_invalid;
    } else {
        const auto& a = r.accounting;
        ++t.requests;
        t.input_tokens += a.input_tokens.value_or(0);
        t.reused_tokens += a.reused_tokens.value_or(0);
        t.generated_tokens += a.generated_tokens.value_or(0);
        if (a.prompt_seconds) {
            ++t.first_count;
            t.first_seconds += *a.prompt_seconds;
            if (a.prompt_tokens_read) { t.prompt_tokens_read += *a.prompt_tokens_read; t.prompt_seconds += *a.prompt_seconds; }
        }
        if (a.decode_seconds) {
            t.decode_seconds += *a.decode_seconds;
            if (a.predicted_tokens) { t.predicted_tokens += *a.predicted_tokens; t.decode_seconds_rated += *a.decode_seconds; }
        }
        last = r;
    }
    recent.push_front(r);
    if (recent.size() > 100) recent.pop_back();
}

Json system_memory(const Json& portal_system, const Json& external) {
    Json system = portal_system;
    const Json ram = external.is_object() && external.contains("machine") && external["machine"].is_object() &&
                             external["machine"].contains("ram") ? external["machine"]["ram"] : Json(nullptr);
    auto value = [&](std::optional<double> gib) {
        if (!gib) return Json{{"bytes", nullptr}, {"source", "edge0-strata"}, {"reason", "not reported by edge0-strata"}};
        return Json{{"bytes", static_cast<std::int64_t>(*gib * 1073741824.0)}, {"source", "edge0-strata"}, {"reason", ""}};
    };
    const auto total = number(ram, "total_gib"), used = number(ram, "used_gib");
    system["memory"] = {{"total", value(total)},
                        {"available", value(total && used ? std::optional<double>(*total - *used) : std::nullopt)}};
    return system;
}

Json status_json(const DelegateStatusInput& s) {
    const auto& external = s.external;
    Json gpu = nullptr;
    if (external.is_object() && external.contains("machine") && external["machine"].is_object() &&
        external["machine"].contains("gpu") && external["machine"]["gpu"].is_object())
        gpu = external["machine"]["gpu"];
    Json devices = nullptr;
    if (gpu.is_object()) {
        const auto total = number(gpu, "total_mib"), used = number(gpu, "used_mib");
        devices = Json::array({Json{{"name", s.device},
                                    {"backend", s.device == "metal" ? "metal" : "cuda"},
                                    {"description", gpu.contains("name") && gpu["name"].is_string() ? gpu["name"] : Json(nullptr)},
                                    {"memory_total_bytes", total ? Json(static_cast<std::int64_t>(*total * 1048576.0)) : Json(nullptr)},
                                    {"memory_free_bytes", total && used ? Json(static_cast<std::int64_t>((*total - *used) * 1048576.0)) : Json(nullptr)}}});
    }
    Json version = external.is_object() && external.contains("engine") && external["engine"].is_string() ? external["engine"] : Json(nullptr);
    Json engine{{"version", version},          {"source_commit", nullptr}, {"source_modified", nullptr},
                {"source_reason", nullptr},    {"ggml_origin", nullptr},   {"ggml_version", nullptr},
                {"backends", nullptr},         {"devices", devices}};
    Json model{{"alias", s.alias}, {"directory", s.directory}, {"context_length", nullable(s.context)},
               {"device", s.device}, {"prefill_device", s.device}};
    Json current = nullptr;
    if (s.active) {
        const auto started = std::chrono::duration<double>(s.active->started.time_since_epoch()).count();
        const auto& a = s.active->accounting;
        // Progress counted from what was forwarded so far.  A stream is in prefill until its first text fragment
        // and in decode after it; a non-streaming request forwards nothing until it ends, so its phase and count
        // are unknown (null) rather than shown as a prefill that never ends.  Input and reused counts appear once
        // the forwarded data carries them (Messages' message_start; Chat's last chunk).
        Json phase = nullptr, generated = nullptr;
        if (a.streaming) {
            phase = a.saw_content ? "decode" : "prefill";
            generated = nullable(a.streamed_tokens);
        }
        current = {{"endpoint", s.active->endpoint}, {"source", s.active->source}, {"phase", phase},
                   {"started_steady_seconds", started},
                   {"elapsed_seconds", std::chrono::duration<double>(s.now - s.active->started).count()},
                   {"input_tokens", nullable(a.input_tokens)}, {"reused_tokens", nullable(a.reused_tokens)},
                   {"computed_tokens", nullptr}, {"pending_tokens", nullptr}, {"generated_tokens", generated}};
        if (a.streaming && a.first_token_seconds) current["first_token_seconds"] = *a.first_token_seconds;
    }
    Json recent = Json::array(), last = nullptr;
    for (const auto& r : s.history.recent) recent.push_back(record_json(r));
    if (s.history.last) {
        // Required in `last`, n/a for a delegated model: null. Optional in recent_requests: absent.
        last = record_json(*s.history.last);
        for (const auto* field : {"error_stage", "reuse_method", "computed_tokens", "decode_forwards", "checkpoint_restored"})
            last[field] = nullptr;
        last["ended_at"] = s.history.last->time;
    }
    const auto& t = s.history.totals;
    Json totals{{"requests", t.requests}, {"rejected_busy", t.rejected_busy}, {"rejected_invalid", t.rejected_invalid},
                {"unauthorized", t.unauthorized}, {"input_tokens", t.input_tokens}, {"reused_tokens", t.reused_tokens},
                {"computed_tokens", nullptr}, {"generated_tokens", t.generated_tokens}, {"checkpoint_restores", nullptr},
                {"continued_turns", nullptr}, {"decode_seconds", t.decode_seconds}};
    if (t.first_count) totals["first_token_seconds_mean"] = t.first_seconds / double(t.first_count);
    if (t.prompt_seconds > 0) totals["prefill_tokens_per_second"] = double(t.prompt_tokens_read) / t.prompt_seconds;
    if (t.decode_seconds_rated > 0) totals["decode_tokens_per_second"] = double(t.predicted_tokens) / t.decode_seconds_rated;
    return {{"instance", s.instance}, {"listen", s.listen}, {"engine", engine}, {"model", model}, {"current", current},
            {"last", last}, {"totals", totals}, {"memory", {{"ledger", nullptr}, {"resident", nullptr}, {"physical", nullptr}}},
            {"system", system_memory(s.system, external)}, {"recent_requests", recent}};
}
}
