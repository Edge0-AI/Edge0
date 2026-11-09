#pragma once
#include "edge0/weight_activity.h"
#include "edge0/engine.h"
#include "edge0/text.h"
#include <map>
#include <memory>
namespace edge0 {
namespace app { struct ModelHandle; struct SessionHandle; }
using ModelReference = std::shared_ptr<app::ModelHandle>;
using SessionReference = std::shared_ptr<app::SessionHandle>;
struct CancelAccess;
// A runtime-bound, single-use handle. Copies share cancellation and use status.
// Generation consumes it after acquiring the activity gate; busy does not consume it.
class CancelToken {
public:
    CancelToken() = default;
private:
    struct State;
    std::shared_ptr<State> state_;
    friend struct CancelAccess;
    friend void request_cancel(const CancelToken&) noexcept;
};
Result<CancelToken> create_cancel_token(const RuntimeReference&);
// Signal-handler safe: one lock-free atomic store; no locks or allocation.
// Keep the token alive throughout the call. Copies share a single-use request.
void request_cancel(const CancelToken&) noexcept;
enum class ConfigurationSource { platform, model, application, request };
struct LoadOverrides {
    std::optional<std::string> device;
    std::optional<std::int64_t> context;
    std::map<std::string,std::string> model_options;
    std::optional<std::int64_t> prefill_chunk;
    std::optional<std::string> prefill_device = {}, stage_policy = {};
};
struct LoadConfiguration {
    std::string directory, device;
    std::int64_t context = 0;
    std::int64_t prefill_chunk = 4096;
    std::map<std::string,std::string> model_options;
    std::map<std::string,ConfigurationSource> sources;
    std::string prefill_device = {}, stage_policy = "fixed";
};
struct GenerationOverrides {
    std::optional<double> temperature, top_p, repetition_penalty;
    std::optional<std::int64_t> top_k, max_tokens;
    std::optional<std::uint64_t> seed;
    std::optional<bool> thinking;
    std::optional<std::vector<std::string>> stop;
    std::optional<double> presence_penalty, frequency_penalty;
    std::optional<std::map<std::int32_t,double>> logit_bias;
    std::optional<bool> logprobs;
    std::optional<std::int64_t> top_logprobs;
};
struct GenerationConfiguration {
    double temperature = 0, top_p = 1, repetition_penalty = 1;
    std::int64_t top_k = 0, max_tokens = 256;
    std::optional<std::uint64_t> seed;
    bool thinking = false;
    std::vector<std::string> stop;
    double presence_penalty = 0, frequency_penalty = 0;
    std::map<std::int32_t,double> logit_bias;
    bool logprobs = false;
    std::int64_t top_logprobs = 0;
    std::map<std::string,ConfigurationSource> sources;
};
enum class StopReason { end_of_generation, length, capacity, cancelled, failed, stop_string, tool_calls };
enum class OutputKind { answer, reasoning, tool_call_begin, tool_call_arguments, tool_call_end, tool_call_rejected, token_logprob };
struct LogprobCandidate {
    std::int32_t token = 0;
    std::string bytes;
    double logprob = 0;
};
struct TokenLogprob {
    std::int32_t token = 0;
    std::string bytes;
    double logprob = 0;
    std::vector<LogprobCandidate> top;
    OutputKind section = OutputKind::answer;
};
struct RejectedToolCall { std::size_t index = 0; std::string text, reason; };
struct OutputEvent {
    OutputKind kind = OutputKind::answer;
    std::string text;
    std::size_t index = 0;
    ToolCall tool_call;
    RejectedToolCall rejected;
    TokenLogprob logprob;
};
enum class GenerationStage { none, input, forward, sampling, delivery };
struct RequestSummary {
    std::optional<double> first_token_seconds, decode_seconds, decode_tokens_per_second;
    std::optional<std::size_t> decode_forwards;
    double total_seconds = 0;
    std::string first_token_reason, decode_reason;
    std::string prefill_device = {};
    std::size_t handoffs = 0, handoff_bytes = 0;
    double handoff_seconds = 0;
};
enum class ReuseMethod { first, keep, truncate, restore, cold };
struct GenerationResult {
    RequestSummary request_summary;
    WeightActivity weight_activity;
    StopReason stop_reason = StopReason::failed;
    GenerationStage error_stage = GenerationStage::none;
    std::optional<Error> error;
    std::string text;
    std::string reasoning;
    std::vector<ToolCall> tool_calls;
    std::vector<RejectedToolCall> rejected_tool_calls;
    std::size_t reasoning_tokens = 0;
    std::vector<TokenLogprob> logprobs;
    std::optional<std::string> matched_stop;
    std::vector<std::int32_t> input, generated;
    std::size_t generated_count = 0, reused_tokens = 0, computed_tokens = 0;
    std::int64_t e = 0;
    std::optional<std::int32_t> tail_token;
    bool tail_is_end = false, state_valid = true, input_complete = false;
    ReuseMethod reuse_method = ReuseMethod::first;
    std::size_t common_prefix = 0;
    bool cancel_requested = false;
    std::size_t rolled_back_tokens = 0;
    bool checkpoint_restored = false;
    std::optional<std::int64_t> checkpoint_boundary;
    double checkpoint_save_seconds = 0;
    bool checkpoint_saved = false;
    bool cold = true, reset_healthy_cache = false;
    std::size_t decode_plan_builds = 0, decode_plan_reuses = 0;
    std::size_t budget_peak = 0, budget_current = 0;
    GenerationConfiguration configuration;
};
// One registered asset file as listed in the asset directory's `assets.tsv`.
struct AssetIdentity {
    std::string role, filename;
    std::uint64_t bytes = 0;
    std::string sha256;
};
struct ModelDescription {
    std::string adapter, directory;
    std::int64_t context = 0;
    std::uint64_t context_limit = 0;
    bool thinking_supported = false, thinking_default = false, thinking_enabled_supported = false;
    bool tools_supported = false;
    GenerationConfiguration recommended;
    std::optional<GenerationConfiguration> recommended_thinking;
    LoadConfiguration loading;
    // Registered files sorted by role. The digests are the registered values read when the model
    // was loaded, not values verified by this load (`assets_verified` is false).
    std::vector<AssetIdentity> assets;
    bool assets_verified = false;
};
Result<ModelDescription> query_model(const ModelReference&);
// A receiver returns an error to stop delivery; callbacks may request release or unload.
using OutputReceiver = std::function<std::optional<Error>(const OutputEvent&)>;
Result<LoadConfiguration> resolve_load_configuration(const RuntimeReference&, const std::string&, const LoadOverrides& = {});
Result<ModelReference> load_model(const RuntimeReference&, const LoadConfiguration&);
Result<SessionReference> create_session(const ModelReference&);
Result<GenerationConfiguration> resolve_generation_configuration(const ModelReference&, const GenerationOverrides& = {});
Result<GenerationResult> generate(const SessionReference&, const std::vector<TextMessage>&,
    const GenerationConfiguration&, const GenerationOverrides& = {}, OutputReceiver = {}, const CancelToken* = nullptr);
struct TokenCount {
    std::size_t input_tokens = 0;
    std::int64_t capacity = 0;
    std::optional<std::size_t> reused_tokens, pending_tokens, available_tokens;
};
Result<GenerationResult> generate(const SessionReference&, const ChatInput&,
    const GenerationConfiguration&, const GenerationOverrides& = {}, OutputReceiver = {}, const CancelToken* = nullptr);
Result<TokenCount> count_tokens(const ModelReference&, const ChatInput&, const GenerationConfiguration&);
Result<TokenCount> count_tokens(const SessionReference&, const ChatInput&, const GenerationConfiguration&);
Result<TokenCount> count_tokens(const ModelReference&, const std::vector<TextMessage>&, const GenerationConfiguration&);
Result<TokenCount> count_tokens(const SessionReference&, const std::vector<TextMessage>&, const GenerationConfiguration&);
Result<std::string> reset_conversation(const SessionReference&);
Result<std::string> release_session(const SessionReference&);
Result<std::string> unload_model(const ModelReference&);
}
