#pragma once
#include "edge0/generation.h"
#include "edge0/timing.h"
namespace edge0 {
enum class BlockPhase { prefill, decode };
struct GenerationBlock {
    std::size_t start = 0, length = 0;
    BlockPhase phase = BlockPhase::prefill;
    bool sample = false;
    std::string device = {};
};
struct ReplayInput {
    std::vector<std::int32_t> tokens;
    std::vector<GenerationBlock> blocks;
};
enum class CancelPointKind { before_compute, after_reuse, before_block, in_block, before_step };
struct CancelPoint {
    CancelPointKind kind;
    std::size_t block = 0, segment = 0, token = 0;
};
struct GenerationCheckOptions {
    // Empty disables trace computation and readback.

    bool fresh_graphs = false, retain_logits = false;
    bool breakdown = false;
    bool skip_checkpoint_restore = false;
    bool ignore_end_of_generation = false;
    // One-based acceptance index; absent disables the extra clock read.
    std::optional<std::size_t> timing_marker = std::nullopt;
    std::string route_trace = {};
    std::function<bool(const CancelPoint&)> cancel_probe = {};
    bool skip_block_rollback = false;

};
struct GenerationCheckResult {
    GenerationResult result;
    std::optional<CancelPoint> cancel_point;
    std::vector<double> block_snapshot_seconds;
    ReplayInput replay;
    std::vector<DecodeTiming> decode_timing;
    std::vector<std::size_t> decode_readback_bytes;
    std::vector<double> prefill_seconds;
    std::optional<double> marker_seconds;
    std::size_t end_token_count = 0;
    std::optional<std::size_t> first_end_token_position; // One-based.

    std::size_t random_draws = 0, host_sampled_steps = 0, sampling_readback_bytes = 0;
    std::vector<double> sampling_seconds;
    std::vector<float> logits;
    std::size_t vocabulary = 0;
};
struct SessionCheckState {
    std::int64_t capacity = 0, e = 0;
    std::vector<std::int32_t> tokens;
    bool valid = false, released = false;
    std::optional<std::int64_t> checkpoint_boundary;
};
Result<GenerationCheckResult> generate_tokens_check(const SessionReference&, const std::vector<std::int32_t>&,
    const GenerationConfiguration&, const GenerationOverrides& = {}, const GenerationCheckOptions& = {}, OutputReceiver = {}, const CancelToken* = nullptr);
Result<GenerationCheckResult> generate_chat_check(const SessionReference&, const ChatInput&,
    const GenerationConfiguration&, const GenerationOverrides& = {}, const GenerationCheckOptions& = {}, OutputReceiver = {}, const CancelToken* = nullptr);
Result<GenerationCheckResult> replay_check(const SessionReference&, const ReplayInput&);
Result<TimingDistribution> measure_submission_overhead(const RuntimeReference&, const std::string& device, std::size_t samples = 200);
Result<SessionCheckState> query_session_check(const SessionReference&);
}
