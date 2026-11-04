#pragma once
#include "edge0/diagnostics.h"
#include "edge0/generation.h"
#include <nlohmann/json.hpp>
#include <deque>
namespace portal {
using Json = nlohmann::ordered_json;
struct RequestRecord {
    std::string time, endpoint, source;
    int http_status = 0;
    bool stream = false, entered = false, result_available = true;
    double total_seconds = 0;
    std::string code, message, param, stop_reason;
    int error_stage = 0;
    edge0::ReuseMethod reuse = edge0::ReuseMethod::first;
    std::size_t input_tokens = 0, reused_tokens = 0, computed_tokens = 0, generated_tokens = 0;
    bool checkpoint_restored = false;
    edge0::RequestSummary summary;
    edge0::WeightActivity experts;
};
RequestRecord completed_record(RequestRecord, const edge0::GenerationResult&);
// Records a failure exactly as the client receives it: once a stream has opened, the HTTP status is already 200.
RequestRecord failed_record(RequestRecord, int status, std::string code, std::string message, std::string param, bool stream_opened);
Json record_json(const RequestRecord&);
struct Totals {
    std::size_t requests = 0, rejected_busy = 0, rejected_invalid = 0, unauthorized = 0;
    std::size_t input_tokens = 0, reused_tokens = 0, computed_tokens = 0, generated_tokens = 0;
    std::size_t checkpoint_restores = 0, continued_turns = 0, first_count = 0, prefill_tokens = 0, decode_forwards = 0;
    double first_seconds = 0, decode_seconds = 0;
    edge0::ExpertActivity experts;
    bool experts_applicable = false;
};
void accumulate(Totals&, const RequestRecord&);
Json totals_json(const Totals&);
struct History {
    Totals totals;
    std::deque<RequestRecord> recent;
    std::optional<RequestRecord> last;
    void add(const RequestRecord&);
};
bool loopback(const std::string&);
bool authorized(const std::string& source, const std::string& token, const std::string& authorization);
Json memory_json(const edge0::MemorySnapshot&);
Json system_json(const edge0::SystemStatus&);
Json engine_json(const edge0::Capabilities&);
Json model_json(const std::string& alias, const edge0::ModelDescription&);
struct StatusInput {
    Json instance, listen, engine, model;
    edge0::GenerationProgress progress;
    std::string endpoint, source;
    std::chrono::steady_clock::time_point now;
    edge0::MemorySnapshot memory;
    edge0::SystemStatus system;
    History history;
};
Json status_json(const StatusInput&);
}
