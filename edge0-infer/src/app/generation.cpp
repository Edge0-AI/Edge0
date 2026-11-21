#include "app/runtime_hooks.h"
#include "app/generation_internal.h"
#include "app/adapters.h"
#include "app/sampling.h"
#include "app/text/decode.h"
#include "app/text/history.h"
#include "app/text/stop.h"
#include <set>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace edge0::app {
ReuseMethod reuse_method(session::ReuseKind kind) {
    switch (kind) {
    case session::ReuseKind::first: return ReuseMethod::first;
    case session::ReuseKind::keep: return ReuseMethod::keep;
    case session::ReuseKind::truncate: return ReuseMethod::truncate;
    case session::ReuseKind::restore: return ReuseMethod::restore;
    case session::ReuseKind::cold: return ReuseMethod::cold;
    }
    throw std::logic_error("Unknown enum value");
}
OutputKind output_kind(model::ParsedKind kind) {
    switch (kind) {
    case model::ParsedKind::answer: return OutputKind::answer;
    case model::ParsedKind::reasoning: return OutputKind::reasoning;
    case model::ParsedKind::tool_call_begin: return OutputKind::tool_call_begin;
    case model::ParsedKind::tool_call_arguments: return OutputKind::tool_call_arguments;
    case model::ParsedKind::tool_call_end: return OutputKind::tool_call_end;
    case model::ParsedKind::tool_call_rejected: return OutputKind::tool_call_rejected;
    }
    throw std::logic_error("Unknown enum value");
}
}
namespace edge0 {
struct CancelToken::State {
    std::shared_ptr<RuntimeRegistry> registry;
    std::atomic<bool> requested{false}, used{false};
};
static_assert(std::atomic<bool>::is_always_lock_free);
struct CancelAccess {
    static CancelToken create(std::shared_ptr<RuntimeRegistry> registry) {
        CancelToken token; token.state_ = std::make_shared<CancelToken::State>();
        token.state_->registry = std::move(registry); return token;
    }
    static bool consume(const CancelToken* token, const std::shared_ptr<RuntimeRegistry>& registry) {
        return !token || (token->state_ && token->state_->registry == registry && !token->state_->used.exchange(true));
    }
    static bool requested(const CancelToken* token) {
        return token && token->state_ && token->state_->requested.load(std::memory_order_relaxed);
    }
};
Result<CancelToken> create_cancel_token(const RuntimeReference& runtime) {
    auto resources = RuntimeAccess::acquire(runtime);
    if (!resources) return resources.error();
    return CancelAccess::create(resources.value().registry);
}
void request_cancel(const CancelToken& token) noexcept {
    if (token.state_) token.state_->requested.store(true, std::memory_order_relaxed);
}
}
namespace edge0 {
namespace {
Error invalid(const std::string& message) { return {ErrorCategory::invalid_argument,message}; }
session::Identity identity(const ModelReference& model) {
    const auto& d = model->rules->description;
    return {model->loaded->adapter ? model->loaded->adapter->id() : d.model, d.template_sha256, model->rules->tokenizer_identity.empty() ? d.model + ":" + d.pre : model->rules->tokenizer_identity};
}
Error busy() { return {ErrorCategory::busy,"Runtime has an active call or callback"}; }
void reclaim_model_locked(const ModelReference&);
struct Activity {
    std::shared_ptr<RuntimeRegistry> registry;
    ~Activity() { if(registry) { std::lock_guard<std::mutex> lock(registry->mutex); registry->active=false; for(const auto& model:registry->models) reclaim_model_locked(model); } }
    exec::Status enter(const std::shared_ptr<RuntimeRegistry>& r) {
        std::lock_guard<std::mutex> lock(r->mutex);
        if(r->ended) return Error{ErrorCategory::runtime_ended,"Runtime has ended"};
        if(r->active) return busy();
        r->active=true; registry=r; return {};
    }
};
exec::Status readable(const ModelReference& model) {
    if(!model) return invalid("Missing model reference");
    std::lock_guard<std::mutex> lock(model->registry->mutex);
    if(model->registry->ended) return Error{ErrorCategory::runtime_ended,"Runtime has ended"};
    if(model->registry->callback_thread==std::this_thread::get_id()) return busy();
    if(!model->loaded) return invalid("Model was released");
    if(model->unloading) return Error{ErrorCategory::unloading,"Model is unloading"};
    return {};
}
exec::Status validate(const GenerationConfiguration& c, bool) {
    if(!std::isfinite(c.temperature)||c.temperature<0||c.top_k<0||!std::isfinite(c.top_p)||c.top_p<=0||c.top_p>1||
       !std::isfinite(c.repetition_penalty)||c.repetition_penalty<=0||c.max_tokens<=0 ||
       !std::isfinite(c.presence_penalty)||std::abs(c.presence_penalty)>2 ||
       !std::isfinite(c.frequency_penalty)||std::abs(c.frequency_penalty)>2 ||
       c.top_logprobs<0||c.top_logprobs>20||(!c.logprobs && c.top_logprobs>0)) return invalid("Invalid generation configuration");
    if(c.stop.size()>16) return invalid("Too many stop strings");
    std::set<std::string> stops;
    for(const auto& stop:c.stop)
        if(stop.empty() || stop.size()>256 || !text::valid_utf8(stop) || !stops.insert(stop).second) return invalid("Invalid stop string");
    for(const auto& bias:c.logit_bias)
        if(bias.first<0 || !std::isfinite(bias.second) || std::abs(bias.second)>100) return invalid("Invalid logit bias");
    return {};
}
exec::Status validate_model(const GenerationConfiguration& c,const ModelReference& model) {
    if(auto error=validate(c,false)) return error;
    if(c.thinking && !model->rules->description.thinking_enabled) return Error{ErrorCategory::unsupported,"Thinking is not supported"};
    const auto vocabulary=model->loaded->assets.definition.vocabulary_size;
    for(const auto& bias:c.logit_bias) if(static_cast<std::uint64_t>(bias.first)>=vocabulary) return invalid("Bias token outside vocabulary");
    return {};
}
exec::Status overlay(GenerationConfiguration& c,const GenerationOverrides& o,ConfigurationSource source) {
    GenerationConfiguration values;
    values.logprobs=c.logprobs;
#define VALUE(field) if(o.field) values.field=*o.field;
    VALUE(temperature) VALUE(top_k) VALUE(top_p) VALUE(repetition_penalty) VALUE(max_tokens)
    VALUE(stop) VALUE(presence_penalty) VALUE(frequency_penalty) VALUE(logit_bias) VALUE(logprobs) VALUE(top_logprobs)
#undef VALUE
    if(auto e=validate(values,false)) return e;
#define APPLY(field) if(o.field) { c.field=*o.field; c.sources[#field]=source; }
    APPLY(temperature) APPLY(top_k) APPLY(top_p) APPLY(repetition_penalty) APPLY(max_tokens) APPLY(seed) APPLY(thinking)
    APPLY(stop) APPLY(presence_penalty) APPLY(frequency_penalty) APPLY(logit_bias) APPLY(logprobs) APPLY(top_logprobs)
#undef APPLY
    return {};
}
Result<GenerationConfiguration> recommended_configuration(const ModelReference& model,bool thinking) {
    GenerationConfiguration result;
    for(const auto* field:{"temperature","top_k","top_p","repetition_penalty","max_tokens","seed","thinking",
        "stop","presence_penalty","frequency_penalty","logit_bias","logprobs","top_logprobs"}) result.sources[field]=ConfigurationSource::platform;
    const auto& d=model->rules->description;
    const auto& recommendation=thinking?d.sampling_thinking_on:d.sampling_thinking_off;
    GenerationOverrides o;
    if(recommendation) {
        o.temperature=recommendation->temperature;o.top_k=recommendation->top_k;o.top_p=recommendation->top_p;
        o.repetition_penalty=recommendation->repetition_penalty;o.presence_penalty=recommendation->presence_penalty;
        o.frequency_penalty=recommendation->frequency_penalty;o.max_tokens=recommendation->max_tokens;
    } else {
        const auto& defaults=model->loaded->generation_defaults;
        o.temperature=defaults.temperature;o.top_k=defaults.top_k;o.top_p=defaults.top_p;o.repetition_penalty=defaults.repetition_penalty;
    }
    o.thinking=thinking;
    if(auto error=overlay(result,o,ConfigurationSource::model)) return *error;
    return result;
}
exec::Status refresh_recommendation(GenerationConfiguration& c,const ModelReference& model) {
    const auto recommended=recommended_configuration(model,c.thinking);
    if(!recommended) return recommended.error();
#define REFRESH(field) if(c.sources.count(#field) && c.sources.at(#field)==ConfigurationSource::model) c.field=recommended.value().field;
    REFRESH(temperature) REFRESH(top_k) REFRESH(top_p) REFRESH(repetition_penalty)
    REFRESH(presence_penalty) REFRESH(frequency_penalty) REFRESH(max_tokens)
#undef REFRESH
    return {};
}
void reclaim_model_locked(const ModelReference& model) {
    if(!model->unloading || model->registry->active) return;
    for(const auto& session:model->registry->sessions)
        if(session->model==model && session->state && !session->state->query().released) return;
    model->loaded.reset(); model->rules.reset();
}
void reclaim_session_locked(const SessionReference& session) {
    if(session->running||!session->state) return;
    session->state->request_release();
    reclaim_model_locked(session->model);
}
}
namespace app {
exec::Status resolve_context_capacity(LoadConfiguration& c,std::optional<std::int64_t> requested,
    std::optional<std::int64_t> recommended,std::uint64_t maximum) {
    if(recommended && *recommended<=0) return invalid("Invalid recommended context capacity");
    c.context=requested.value_or(static_cast<std::int64_t>(std::min<std::uint64_t>(recommended.value_or(4096),maximum)));
    if(c.context<=0 || static_cast<std::uint64_t>(c.context)>maximum) return invalid("Context capacity is outside model limits");
    c.sources["context"]=requested?ConfigurationSource::request:recommended?ConfigurationSource::model:ConfigurationSource::platform;
    return {};
}
exec::Status resolve_prefill_chunk(LoadConfiguration& configuration,std::optional<std::int64_t> requested) {
    const auto chunk=requested.value_or(4096);
    if(chunk<=0 || chunk>configuration.context) return invalid("Prefill chunk is outside context capacity");
    configuration.prefill_chunk=chunk;
    configuration.sources["prefill_chunk"]=requested?ConfigurationSource::request:ConfigurationSource::platform;
    return {};
}
void cleanup_registered(const std::shared_ptr<RuntimeRegistry>& registry) {
    std::lock_guard<std::mutex> lock(registry->mutex);
    for(const auto& model:registry->models) model->unloading=true;
    for(const auto& session:registry->sessions) { session->releasing=true; reclaim_session_locked(session); }
    for(const auto& model:registry->models) { model->loaded.reset(); model->rules.reset(); }
    registry->sessions.clear(); registry->models.clear();
}
Result<ModelReference> register_model(const RuntimeReference& runtime,std::shared_ptr<model::LoadedModel> loaded,
    std::shared_ptr<text::Rules> rules,const models::ModelCompute& compute,const LoadConfiguration& configuration,
    std::vector<AssetIdentity> assets) {
    auto resources=RuntimeAccess::acquire(runtime); if(!resources) return resources.error();
    auto result=std::make_shared<ModelHandle>(runtime,resources.value().registry);
    result->assets=std::move(assets);
    result->loaded=std::move(loaded); result->rules=std::move(rules); result->compute=&compute; result->configuration=configuration;
    std::lock_guard<std::mutex> lock(result->registry->mutex);
    result->registry->models.push_back(result);
    return result;
}
}
Result<LoadConfiguration> resolve_load_configuration(const RuntimeReference& runtime,const std::string& directory,const LoadOverrides& overrides) {
    auto resources=RuntimeAccess::acquire(runtime); if(!resources) return resources.error();
    { std::lock_guard<std::mutex> lock(resources.value().registry->mutex);
      if(resources.value().registry->callback_thread==std::this_thread::get_id()) return busy(); }
    auto catalog=model::open_catalog(directory,app::compiled_adapters()); if(!catalog) return catalog.error();
    auto adapter=model::select_adapter(catalog.value(),app::compiled_adapters()); if(!adapter) return adapter.error();
    auto assets=model::check_compatibility(catalog.value(),*adapter.value()); if(!assets) return assets.error();
    auto declaration=adapter.value()->describe_weight_storage(assets.value().indexes,overrides.model_options);
    if(!declaration) return declaration.error();
    LoadConfiguration result; result.directory=directory;
    if(overrides.device) { result.device=*overrides.device; result.sources["device"]=ConfigurationSource::request; }
    else {
        if(runtime.available_devices().size()!=1) return invalid("Specify a device for a multi-device runtime");
        result.device=runtime.available_devices().front(); result.sources["device"]=ConfigurationSource::platform;
    }
    if(!resources.value().backends.count(result.device)) return invalid("Device is outside runtime scope");
    result.prefill_device = overrides.prefill_device.value_or(result.device);
    result.stage_policy = overrides.stage_policy.value_or("fixed");
    result.sources["prefill_device"] = overrides.prefill_device ? ConfigurationSource::request : ConfigurationSource::platform;
    result.sources["stage_policy"] = overrides.stage_policy ? ConfigurationSource::request : ConfigurationSource::platform;
    if (!resources.value().backends.count(result.prefill_device))
        return Error{ErrorCategory::unsupported, "Prefill device is outside runtime scope"};
    const auto hooks = resources.value().registry->hooks;
    if (result.stage_policy != "fixed" && (!hooks || !hooks->policies.count(result.stage_policy)))
        return invalid("Unknown stage policy: " + result.stage_policy);
    const auto maximum=assets.value().definition.context_length;
    if(auto error=app::resolve_context_capacity(result,overrides.context,adapter.value()->describe_text().recommended_context,maximum)) return *error;
    if(auto error=app::resolve_prefill_chunk(result,overrides.prefill_chunk)) return *error;
    if(auto error=model::validate_context_options(declaration.value(),overrides.model_options,result.context)) return *error;
    for(const auto& option:declaration.value().configuration) {
        const auto found=overrides.model_options.find(option.name);
        result.model_options[option.name]=found==overrides.model_options.end()?option.default_value:found->second;
        result.sources[option.name]=found==overrides.model_options.end()?ConfigurationSource::model:ConfigurationSource::request;
    }
    for(const auto& option:overrides.model_options) if(!result.model_options.count(option.first)) return invalid("Unknown model option: "+option.first);
    return result;
}
Result<ModelReference> load_model(const RuntimeReference& runtime,const LoadConfiguration& configuration) {
    auto resources=RuntimeAccess::acquire(runtime); if(!resources) return resources.error();
    Activity activity; if(auto e=activity.enter(resources.value().registry)) return *e;
    auto resolved=resolve_load_configuration(runtime,configuration.directory,{configuration.device,configuration.context,configuration.model_options,configuration.prefill_chunk, configuration.prefill_device.empty() ? std::nullopt : std::optional<std::string>(configuration.prefill_device), configuration.stage_policy});
    if(!resolved) return resolved.error();
    auto budget=resources.value().budget;
    model::LoadRequest request; request.directory=configuration.directory; request.device=configuration.device;
    request.context=configuration.context;
    request.prefill_device=resolved.value().prefill_device; request.stage_policy=resolved.value().stage_policy;
    if (auto hooks = resources.value().registry->hooks) {
        request.consumption=hooks->consumption; request.transfer=hooks->transfer; request.direct=hooks->direct;
        const auto policy=hooks->policies.find(request.stage_policy);
        if(policy!=hooks->policies.end()) request.policy_override=policy->second;
    }
    auto catalog=model::open_catalog(configuration.directory,app::compiled_adapters()); if(!catalog) return catalog.error();
    auto selected=app::select_compiled(catalog.value()); if(!selected) return selected.error();
    const auto* compute=selected.value().compute;
    request.supports_handoff=compute->supports_handoff();
    request.options=configuration.model_options;
    request.on_close=[compute](model::LoadedModel& model) { compute->close_model(model); };
    auto loaded=model::load(request,{selected.value().adapter},budget,resources.value().backends); if(!loaded) return loaded.error();
    auto rules=text::load_rules(loaded.value()->assets.indexes,*loaded.value()->adapter); if(!rules) return rules.error();
    // Registered identities come from the catalog already read above; no payload is read or hashed.
    std::vector<AssetIdentity> identities;
    for(const auto& item:catalog.value().files) identities.push_back({item.second.role,item.second.filename,item.second.bytes,item.second.sha256});
    return app::register_model(runtime,loaded.value(),std::make_shared<text::Rules>(rules.value()),*compute,configuration,std::move(identities));
}
Result<SessionReference> create_session(const ModelReference& model) {
    if(!model) return invalid("Missing model reference");
    auto resources=RuntimeAccess::acquire(model->runtime); if(!resources) return resources.error();
    Activity activity; if(auto e=activity.enter(model->registry)) return *e;
    if(auto e=readable(model)) return *e;
    auto description=model->compute->describe_states(model->loaded->assets.definition); if(!description) return description.error();
    auto budget=resources.value().budget;
    auto state=session::Session::create(budget,model->loaded->stages,model->configuration.context,description.value().kv,description.value().updated_entries);
    if(!state) return state.error();
    const auto loaded=model->loaded; const auto* compute=model->compute;
    auto error=state.value()->initialize([&](session::KVCache& kv,session::StateStore& states) {
        return compute->initialize_session(loaded,kv,states,budget,loaded->stages);
    },[loaded,compute](session::KVCache& kv,session::StateStore& states) { compute->finish_pending_weights(loaded,states,models::FinishReason::failure); compute->release_session(loaded,kv,states); });
    if(error) return *error;
    if (auto e = state.value()->configure_reuse(identity(model), description.value().evolves_as_whole, description.value().supports_save_restore,
        [compute](session::KVCache&, session::StateStore& s) { return compute->save_state(s); },
        [compute](session::KVCache&, session::StateStore& s) { return compute->restore_state(s); },
        [compute, supported=description.value().supports_save_restore](session::KVCache&, session::StateStore& s) {
            return supported ? compute->discard_saved_state(s) : exec::Status{};
        })) return *e;
    auto result=std::make_shared<app::SessionHandle>(); result->model=model; result->state=state.value();
    std::lock_guard<std::mutex> lock(model->registry->mutex); model->registry->sessions.push_back(result);
    return result;
}
Result<GenerationConfiguration> resolve_generation_configuration(const ModelReference& model,const GenerationOverrides& application) {
    if(auto e=readable(model)) return *e;
    std::lock_guard<std::mutex> lock(model->registry->mutex);
    if(!model->loaded||model->unloading) return Error{ErrorCategory::unloading,"Model is unloading"};
    auto recommended=recommended_configuration(model,application.thinking.value_or(model->rules->description.default_thinking));
    if(!recommended) return recommended.error();
    auto result=recommended.value();
    if(auto e=overlay(result,application,ConfigurationSource::application)) return *e;
    if(auto e=validate_model(result,model)) return *e;
    return result;
}
Result<ModelDescription> query_model(const ModelReference& model) {
    if(!model) return invalid("Missing model reference");
    std::lock_guard<std::mutex> lock(model->registry->mutex);
    if(model->registry->ended) return Error{ErrorCategory::runtime_ended,"Runtime has ended"};
    if(model->registry->callback_thread==std::this_thread::get_id()) return busy();
    if(!model->loaded) return invalid("Model was reclaimed");
    if(model->unloading) return Error{ErrorCategory::unloading,"Model is unloading"};
    ModelDescription result;
    result.adapter=model->loaded->adapter?model->loaded->adapter->id():model->rules->description.model;
    result.directory=model->configuration.directory;
    result.context=model->configuration.context;
    result.context_limit=model->loaded->assets.definition.context_length;
    result.thinking_supported=model->rules->description.thinking;
    result.thinking_enabled_supported=model->rules->description.thinking_enabled;
    result.thinking_default=model->rules->description.default_thinking;
    result.tools_supported=model->rules->description.tools;
    result.loading=model->configuration;
    result.assets=model->assets; result.assets_verified=false;
    auto recommended=recommended_configuration(model,result.thinking_default);
    if(!recommended) return recommended.error();
    result.recommended=recommended.value();
    if(result.thinking_enabled_supported) {
        auto thinking=recommended_configuration(model,true);if(!thinking) return thinking.error();
        result.recommended_thinking=thinking.value();
    }
    return result;
}
Result<std::string> release_session(const SessionReference& session) {
    if(!session) return invalid("Missing session reference");
    auto registry=session->model->registry;
    std::lock_guard<std::mutex> lock(registry->mutex);
    if(registry->ended) return Error{ErrorCategory::runtime_ended,"Runtime has ended"};
    session->releasing=true;
    reclaim_session_locked(session);
    return std::string("Release accepted");
}
Result<std::string> unload_model(const ModelReference& model) {
    if(!model) return invalid("Missing model reference");
    std::lock_guard<std::mutex> lock(model->registry->mutex);
    if(model->registry->ended) return Error{ErrorCategory::runtime_ended,"Runtime has ended"};
    model->unloading=true;
    // Keep the underlying model usable until the active block reaches its boundary.
    reclaim_model_locked(model);
    return std::string("Unload accepted");
}
Result<SessionCheckState> query_session_check(const SessionReference& session) {
    if(!session) return invalid("Missing session reference");
    auto registry=session->model->registry;
    std::lock_guard<std::mutex> lock(registry->mutex);
    if(registry->ended) return Error{ErrorCategory::runtime_ended,"Runtime has ended"};
    if(registry->callback_thread==std::this_thread::get_id()) return busy();
    const auto state=session->state->query();
    return SessionCheckState{state.capacity,state.e,state.tokens,state.valid&&!session->weights_failed,state.released,state.checkpoint};
}
}

namespace edge0 {
namespace {
struct GenerationRun {
    const SessionReference& session;
    const std::vector<std::int32_t>* given;
    const ChatInput* chat;
    const GenerationConfiguration& configuration;
    const GenerationOverrides& overrides;
    const GenerationCheckOptions& options;
    OutputReceiver& receiver;
    const CancelToken* cancel_token;
    const RuntimeResources& resources;
    std::shared_ptr<model::LoadedModel> loaded;
    std::shared_ptr<text::Rules> rules;
    ModelReference model=session->model;
    GenerationCheckResult output = {};
    GenerationResult& result=output.result;
    exec::Budget budget=resources.budget;
    using TimePoint=std::chrono::steady_clock::time_point;
    TimePoint started = {};
    std::optional<TimePoint> first_token = {},last_token = {};
    session::Snapshot before = {};
    text::Decoder decoder = {};
    bool receiver_stopped=false,output_stopped=false,host_sampling=false;
    std::size_t block_index=0;
    std::unique_ptr<model::OutputParser> parser = {};
    std::optional<text::StopMatcher> stop_matcher = {};
    std::map<std::size_t,std::string> call_ids = {};
    std::set<std::string> used_ids = {};
    app::GeneratedCounts generated_counts = {};
    std::optional<TokenLogprob> token_logprob = {};
    app::SamplingHistory history = {};
    std::optional<app::SamplingRandom> random = {};
    std::optional<models::GreedyOutput> greedy = {};
    std::shared_ptr<exec::Backend> prefill_backend = {};
    exec::TransferStats initial_transfers = {};
    exec::TransferStats transfer_stats() const {
        auto stats=model->compute->transfer_stats(*loaded,session->state->states());
        const auto kv=session->state->kv().transfer_stats();
        stats.count+=kv.count; stats.bytes+=kv.bytes; stats.seconds+=kv.seconds; return stats;
    }

    static double seconds(TimePoint from,TimePoint to) {
        return std::chrono::duration<double>(to-from).count();
    }
    bool requested() {
        return CancelAccess::requested(cancel_token) || session->releasing.load() || model->unloading.load();
    }
    bool check_cancel(CancelPoint point,bool lifecycle=true) {
        const bool probe = options.cancel_probe && options.cancel_probe(point);
        const bool stop = CancelAccess::requested(cancel_token) || probe ||
            (lifecycle && (session->releasing.load() || model->unloading.load()));
        result.cancel_requested = result.cancel_requested || stop;
        if (stop && !output.cancel_point) output.cancel_point = point;
        return stop;
    }
    void fail(Error error,GenerationStage stage) {
        result.stop_reason=StopReason::failed; result.error=std::move(error); result.error_stage=stage;
    }
    Result<bool> prepare() {
        result.weight_activity.applicable=!loaded->pools.empty();
        result.configuration=configuration;
        if(auto e=validate_model(result.configuration,model)) return *e;
        if(auto e=overlay(result.configuration,overrides,ConfigurationSource::request)) return *e;
        if(auto error=refresh_recommendation(result.configuration,model)) return *error;
        if(auto e=validate_model(result.configuration,model)) return *e;
        if(options.timing_marker && *options.timing_marker==0) return invalid("Timing marker must be positive");
        started=model->clock_now();
        if(given) result.input=*given;
        else {
            auto rendered=text::render_input(*rules,*chat,{result.configuration.thinking,{}});
            if(!rendered) return rendered.error();
            result.input=rendered.value().tokens;
        }
        output.vocabulary=loaded->assets.definition.vocabulary_size;
        if(result.input.empty()) return invalid("Empty input tokens");
        for(auto token:result.input) if(token<0||std::size_t(token)>=output.vocabulary) return invalid("Token outside vocabulary");
        before=session->state->query();
        if(loaded->stages.mixed()) initial_transfers=transfer_stats();
        if(result.input.size()>std::uint64_t(before.capacity)) return Error{ErrorCategory::capacity_exceeded,"Input exceeds session capacity"};
        const auto prefix = text::match_history(result.input, before, identity(model));
        auto reuse = session->state->check_reuse(static_cast<std::int64_t>(prefix), static_cast<std::int64_t>(result.input.size()), identity(model));
        if (!reuse) return reuse.error();
        if (check_cancel({CancelPointKind::before_compute})) {
            result.stop_reason = StopReason::cancelled;
            result.reuse_method = app::reuse_method(reuse.value().kind);
            result.common_prefix = prefix; result.cold = false;
            result.e = before.e; result.checkpoint_boundary = before.checkpoint;
            result.state_valid = before.valid;
            result.budget_current = resources.budget.current();
            result.budget_peak = resources.budget.peak();
            output.replay = {before.tokens, session->blocks};
            result.request_summary.total_seconds = seconds(started, model->clock_now());
            result.request_summary.first_token_reason = "no_accepted_token";
            result.request_summary.decode_reason = "cancelled";
            return false;
        }
        auto finished_prior = model->compute->finish_pending_weights(loaded,session->state->states(),models::FinishReason::normal);
        if (!finished_prior) return finished_prior.error();
        result.weight_activity += finished_prior.value();
        const auto& plan = reuse.value();
        if (auto error = session->state->apply_reuse(plan, [&](session::KVCache& kv,session::StateStore& states) {
            return model->compute->reset_session(loaded,kv,states);
        }, options.skip_checkpoint_restore)) return *error;
        result.common_prefix = prefix;
        result.reused_tokens = static_cast<std::size_t>(plan.k);
        {
            std::lock_guard<std::mutex> lock(resources.registry->progress_mutex);
            resources.registry->progress = {GenerationPhase::prefill, result.input.size(), result.reused_tokens,
                0, result.input.size() - result.reused_tokens, 0, started, {}};
        }
        result.reuse_method = app::reuse_method(plan.kind);
        result.checkpoint_restored = plan.kind == session::ReuseKind::restore && !options.skip_checkpoint_restore;
        result.cold = plan.kind == session::ReuseKind::cold || plan.kind == session::ReuseKind::first;
        result.reset_healthy_cache = plan.kind == session::ReuseKind::cold && before.e > 0;
        while (!session->blocks.empty() && session->blocks.back().start >= result.reused_tokens) session->blocks.pop_back();
        if (!session->blocks.empty() && session->blocks.back().start + session->blocks.back().length > result.reused_tokens)
            session->blocks.back().length = result.reused_tokens - session->blocks.back().start;
        if(auto e=session->state->begin_use()) return *e;
        return true;

    }
    bool emit(const OutputEvent& event) {
        if(receiver) {
            { std::lock_guard<std::mutex> lock(model->registry->mutex); model->registry->callback_thread=std::this_thread::get_id(); }
            exec::Status error;
            try { error=receiver(event); }
            catch(const std::exception& e) { error=invalid(std::string("Output receiver threw: ")+e.what()); }
            catch(...) { error=invalid("Output receiver threw"); }
            { std::lock_guard<std::mutex> lock(model->registry->mutex); model->registry->callback_thread={}; }
            if(error) { receiver_stopped=true; fail(*error,GenerationStage::delivery); return false; }
        }
        switch(event.kind) {
        case OutputKind::answer:result.text+=event.text;break;
        case OutputKind::reasoning:result.reasoning+=event.text;break;
        case OutputKind::tool_call_end:result.tool_calls.push_back(event.tool_call);break;
        case OutputKind::tool_call_rejected:result.rejected_tool_calls.push_back(event.rejected);break;
        case OutputKind::token_logprob:result.logprobs.push_back(event.logprob);break;
        case OutputKind::tool_call_begin:case OutputKind::tool_call_arguments:break;
        }
        return true;

    }
    bool classified(const model::ParsedOutput& part) {
        OutputEvent event;event.kind=app::output_kind(part.kind);event.text=part.text;event.index=part.index;event.tool_call=part.call;
        if(part.kind==model::ParsedKind::tool_call_begin) {
            std::random_device random_id;
            std::string id;
            do {
                std::ostringstream out;out<<"call_"<<std::hex<<std::setfill('0');
                for(int i=0;i<3;++i) out<<std::setw(8)<<static_cast<std::uint32_t>(random_id());
                id=out.str();
            } while(!used_ids.insert(id).second);
            call_ids[part.index]=id;
        }
        if(call_ids.count(part.index)) event.tool_call.id=call_ids.at(part.index);
        if(part.kind==model::ParsedKind::tool_call_rejected) event.rejected={part.index,part.text,part.reason};
        if(part.kind==model::ParsedKind::answer) event.text=stop_matcher->feed(part.text);
        if((event.kind!=OutputKind::answer && event.kind!=OutputKind::reasoning) || !event.text.empty())
            if(!emit(event)) return false;
        if(stop_matcher->matched()) {
            result.matched_stop=stop_matcher->matched();result.stop_reason=StopReason::stop_string;output_stopped=true;
        } else if(chat && !chat->parallel_tool_calls &&
                  (part.kind==model::ParsedKind::tool_call_end || part.kind==model::ParsedKind::tool_call_rejected)) {
            result.stop_reason=StopReason::tool_calls;output_stopped=true;
        }
        return !output_stopped;

    }
    bool deliver(const std::string& fragment) {
        return parser->feed(fragment,[this](const model::ParsedOutput& part) { return classified(part); });
    }
    void abandon_observation(const std::unique_ptr<models::ForwardResult>& owner) {
        model->compute->finish_pending_weights(loaded,session->state->states(),models::FinishReason::failure);
        owner->end_observation();
    }
    bool sample_step(const std::unique_ptr<models::ForwardResult>& owner) {
            const auto readback_before = owner->readback_bytes();
            const auto sampling_started = options.breakdown && host_sampling ? model->clock_now() : std::chrono::steady_clock::time_point{};
            if (host_sampling) {
                ++output.host_sampled_steps;
                auto row = owner->logits();
                output.sampling_readback_bytes += owner->readback_bytes() - readback_before;
                if (!row || row.value().size() != output.vocabulary) {
                    if (options.breakdown) output.sampling_seconds.push_back(seconds(sampling_started, model->clock_now()));
                    abandon_observation(owner);
                    fail(row ? invalid("Invalid logits row length") : row.error(), GenerationStage::sampling);
                    return false;
                }
                if (result.configuration.temperature > 0 && !random) {
                    if (!result.configuration.seed) {
                        std::random_device device;
                        result.configuration.seed = (static_cast<std::uint64_t>(device()) << 32) | static_cast<std::uint64_t>(device());
                        result.configuration.sources["seed"] = ConfigurationSource::platform;
                    }
                    random.emplace(*result.configuration.seed);
                }
                auto uniform = [&] {
                    const auto value = random->uniform();
                    output.random_draws = random->draws();
                    return value;
                };
                auto choice = app::sample_logits(row.value(), result.configuration, history, uniform, generated_counts);
                if (options.breakdown) output.sampling_seconds.push_back(seconds(sampling_started, model->clock_now()));
                if (!choice) { abandon_observation(owner); fail(choice.error(), GenerationStage::sampling); return false; }
                greedy = models::GreedyOutput{choice.value(), 0};
                if(result.configuration.logprobs) {
                    token_logprob=app::raw_logprob(row.value(),choice.value(),static_cast<std::size_t>(result.configuration.top_logprobs));
                    token_logprob->bytes=rules->tokenizer.token_to_piece(choice.value(),true);
                    for(auto& candidate:token_logprob->top) candidate.bytes=rules->tokenizer.token_to_piece(candidate.token,true);
                }
                if (options.retain_logits) output.logits.insert(output.logits.end(), row.value().begin(), row.value().end());
            } else {
                auto choice=owner->greedy_output();
                if(!choice) { abandon_observation(owner); fail(choice.error(),GenerationStage::sampling); return false; }
                greedy=choice.value();
                if(options.retain_logits) {
                    auto row=owner->logits();
                    if(!row) { abandon_observation(owner); fail(row.error(),GenerationStage::sampling); return false; }
                    output.logits.insert(output.logits.end(),row.value().begin(),row.value().end());
                }
                output.sampling_readback_bytes += owner->readback_bytes() - readback_before;
            }

        return true;
    }
    bool forward(std::vector<std::int32_t> tokens,exec::Phase phase,bool sample) {
        auto backend=loaded->stages.of(phase);
        if(phase==exec::Phase::prefill) {
            if(!prefill_backend) {
                if(loaded->stages.mixed()) {
                    auto facts=model->compute->stage_facts(*loaded,session->state->states(),loaded->stages.prefill);
                    facts.tokens=result.input.size()-result.reused_tokens;
                    facts.handoff_bytes+=session->state->kv().handoff_bytes(loaded->stages.prefill,session->state->query().e);
                    auto chosen=exec::choose_prefill_backend(loaded->stages,loaded->mixed->policy,loaded->mixed->candidates,facts);
                    if(!chosen) { fail(chosen.error(),GenerationStage::forward); return false; }
                    prefill_backend=chosen.value();
                } else prefill_backend=backend;
            }
            backend=prefill_backend;
            for(const auto& entry:resources.backends) if(entry.second==backend) { result.request_summary.prefill_device=entry.first; break; }
        }
        auto update=session->state->begin_update(tokens);
        if(!update) { fail(update.error(),GenerationStage::forward); return false; }
        models::ForwardRequest request{tokens,update.value().step,phase,{tokens.size()-1},{}};
        request.backend=backend;
        request.plan_mode=options.fresh_graphs?models::PlanMode::fresh:models::PlanMode::automatic;
        request.greedy=sample; request.retain_logits=sample&&(options.retain_logits || host_sampling);
        models::ForwardProgress progress; request.progress=&progress;
        request.route_trace=options.route_trace;
        request.breakdown=options.breakdown; request.clock_now=model->clock_now;
        request.skip_block_rollback = options.skip_block_rollback;
        if (cancel_token || options.cancel_probe) request.cancel_block = [&](std::size_t segment) {
            return check_cancel({CancelPointKind::in_block, block_index, segment, result.generated.size()}, false);
        };
        const auto block_started=options.breakdown && phase==exec::Phase::prefill?model->clock_now():std::chrono::steady_clock::time_point{};
        select_cpu_threads(resources, phase);
        if(loaded->stages.mixed()) if(auto error=app::prepare_stage_forward(*model->compute,loaded,session->state->kv(),session->state->states(),request,resources.registry.get())) {
            session->state->report_interruption(update.value(),false); fail(*error,GenerationStage::forward); return false;
        }
        auto computed=model->compute->forward(loaded,request,session->state->kv(),session->state->states(),budget);
        if (progress.block_snapshot_seconds > 0) output.block_snapshot_seconds.push_back(progress.block_snapshot_seconds);
        if(!computed) {
            result.cancel_requested = result.cancel_requested || requested();
            result.weight_activity+=progress.activity;
            if (progress.cancelled) {
                auto error = session->state->report_interruption(update.value(), false);
                if (error) { fail(*error, GenerationStage::forward); return false; }
                result.stop_reason = StopReason::cancelled; result.cancel_requested = true;
                result.rolled_back_tokens += progress.rolled_back_tokens;
                return false;
            }
            session->state->report_interruption(update.value(),progress.submitted);
            fail(computed.error(),GenerationStage::forward); return false;
        }
        const auto& owner=computed.value();
        if(options.breakdown) {
            if(phase==exec::Phase::prefill) output.prefill_seconds.push_back(seconds(block_started,model->clock_now()));
            else if(auto timing=owner->decode_timing()) output.decode_timing.push_back(*timing);
        }
        result.weight_activity+=owner->weight_activity();
        auto commit=session->state->commit_update(update.value(),owner->completion(),owner->updated_entries());
        if(commit) {
            session->state->report_interruption(update.value(),true);
            abandon_observation(owner); fail(*commit,GenerationStage::forward); return false;
        }
        result.computed_tokens+=tokens.size();
        if (phase == exec::Phase::prefill) {
            std::lock_guard<std::mutex> lock(resources.registry->progress_mutex);
            auto& observed = resources.registry->progress;
            observed.computed_tokens += tokens.size();
            observed.pending_tokens = observed.input_tokens - observed.reused_tokens - observed.computed_tokens;
        }
        session->blocks.push_back({static_cast<std::size_t>(update.value().step.e), tokens.size(),
            phase == exec::Phase::prefill ? BlockPhase::prefill : BlockPhase::decode, sample});
        for(const auto& entry:resources.backends) if(entry.second==backend) { session->blocks.back().device=entry.first; break; }
        if(phase==exec::Phase::decode) {
            if(owner->reused_plan()) ++result.decode_plan_reuses; else ++result.decode_plan_builds;
        }
        if(sample && !sample_step(owner)) return false;
        if(options.breakdown && phase==exec::Phase::decode) output.decode_readback_bytes.push_back(owner->readback_bytes());
        if(auto error=owner->end_observation()) {
            model->compute->finish_pending_weights(loaded,session->state->states(),models::FinishReason::failure);
            session->weights_failed=true; session->state->invalidate(); fail(*error,GenerationStage::forward); return false;
        }
        return true;

    }
    void save_checkpoint(const models::StateDescription& description,std::size_t n) {
        if (!description.supports_save_restore || n < 2) return;
        const auto state = session->state->query();
        if (state.e != static_cast<std::int64_t>(n - 1) || state.checkpoint == state.e) return;
        const auto begin = model->clock_now();
        result.checkpoint_saved = !session->state->save_checkpoint();
        result.checkpoint_save_seconds += std::chrono::duration<double>(model->clock_now() - begin).count();

    }
    bool prefill(const models::StateDescription& description,std::size_t n,std::size_t first) {
        bool okay=true;
        if (check_cancel({CancelPointKind::after_reuse})) { result.stop_reason=StopReason::cancelled; okay=false; }
        if (okay) save_checkpoint(description,n);
        for(std::size_t offset=result.reused_tokens;okay && offset<n;) {
            if (session->releasing.load() || model->unloading.load()) {
                result.cancel_requested=true; result.stop_reason=StopReason::cancelled; okay=false; break;
            }
            const auto count=std::min<std::size_t>(model->configuration.prefill_chunk,offset<first?first-offset:n-offset);
            okay=forward({result.input.begin()+offset,result.input.begin()+offset+count},exec::Phase::prefill,offset+count==n);
            if(!okay) break;
            offset+=count;
            ++block_index;
            if (check_cancel({CancelPointKind::before_block, block_index})) {
                result.stop_reason=StopReason::cancelled; okay=false; break;
            }
            save_checkpoint(description,n);
        }
        result.input_complete=okay;
        return okay;

    }
    void decode(bool okay) {
        const auto maximum_source=result.configuration.sources.find("max_tokens");
        const bool cap_recommended=maximum_source!=result.configuration.sources.end() &&
            (maximum_source->second==ConfigurationSource::model || maximum_source->second==ConfigurationSource::platform);
        const auto remaining=static_cast<std::uint64_t>(before.capacity)-result.input.size();
        const auto maximum=cap_recommended && remaining>0?std::min<std::uint64_t>(result.configuration.max_tokens,remaining):
            static_cast<std::uint64_t>(result.configuration.max_tokens);
        while(okay) {
            if (session->releasing.load() || model->unloading.load()) {
                result.cancel_requested=true; result.stop_reason=StopReason::cancelled; break;
            }
            if(!greedy||!std::isfinite(greedy->sum)||greedy->token<0||std::size_t(greedy->token)>=output.vocabulary) {
                fail(invalid("Non-finite logits sum or invalid greedy token"),GenerationStage::sampling); break;
            }
            const auto token=greedy->token;
            result.generated.push_back(token); result.tail_token=token;
            if (host_sampling) {history.insert(token);++generated_counts[token];}
            const bool eog=rules->tokenizer.eog().count(token)!=0;
            if(eog) {
                ++output.end_token_count;
                if(!output.first_end_token_position) output.first_end_token_position=result.generated.size();
            } else ++result.generated_count;
            const bool ends=eog&&!options.ignore_end_of_generation;
            const bool length=(options.ignore_end_of_generation?result.generated.size():result.generated_count)>=maximum;
            const bool capacity=session->state->query().e>=before.capacity;
            if(result.generated.size()==1) first_token=model->clock_now();
            {
                std::lock_guard<std::mutex> lock(resources.registry->progress_mutex);
                auto& observed = resources.registry->progress;
                observed.phase = GenerationPhase::decode;
                observed.first_token = first_token;
                observed.generated_tokens = result.generated.size();
            }
            if(options.timing_marker && result.generated.size()==*options.timing_marker)
                output.marker_seconds=seconds(started,model->clock_now());
            if(ends||length||capacity) last_token=model->clock_now();
            if(ends) { result.stop_reason=StopReason::end_of_generation; result.tail_is_end=true; break; }
            try {
                const auto section=parser->section();
                if(!eog && section==model::ParsedKind::reasoning) ++result.reasoning_tokens;
                const auto raw_piece=rules->tokenizer.token_to_piece(token,true);
                const auto piece=parser->preserve_control_piece(raw_piece)?raw_piece:rules->tokenizer.token_to_piece(token,false);
                const bool continued=deliver(decoder.decode_increment(piece));
                if(receiver_stopped) break;
                if(!eog && token_logprob) {
                    token_logprob->section=app::output_kind(section);
                    OutputEvent event;event.kind=OutputKind::token_logprob;event.logprob=*token_logprob;
                    if(!emit(event)) break;
                }
                if(!continued || output_stopped) {last_token=model->clock_now();break;}
            } catch(const std::exception& error) { fail(invalid(error.what()),GenerationStage::delivery); break; }
            if(length) { result.stop_reason=StopReason::length; break; }
            if(capacity) { result.stop_reason=StopReason::capacity; break; }
            if(check_cancel({CancelPointKind::before_step, block_index, 0, result.generated.size()})) { result.stop_reason=StopReason::cancelled; break; }
            if(!forward({token},exec::Phase::decode,true)) break;
            result.tail_token.reset();
        }


    }
    void finish() {
        const auto finished_text = text::finish_turn(decoder, result.input.size(), result.generated,
            session->state->query().e, result.stop_reason == StopReason::end_of_generation, result.error_stage != GenerationStage::delivery);
        result.tail_token = finished_text.tail; result.tail_is_end = finished_text.tail_is_end;
        if(result.error_stage!=GenerationStage::delivery && !output_stopped) {
            if(deliver(finished_text.text)) parser->finish([this](const model::ParsedOutput& part) { return classified(part); });
        }
        if(result.error_stage!=GenerationStage::delivery) {
            const auto pending=stop_matcher->finish();
            if(!pending.empty()) {OutputEvent event;event.kind=OutputKind::answer;event.text=pending;emit(event);}
        }
        const auto reason=((result.stop_reason==StopReason::failed && result.error_stage!=GenerationStage::delivery) || (result.stop_reason==StopReason::cancelled && (session->releasing.load() || model->unloading.load())))?models::FinishReason::failure:models::FinishReason::normal;
        auto finished=model->compute->finish_pending_weights(loaded,session->state->states(),reason);
        if(!finished) {
            fail(finished.error(),GenerationStage::forward);
            session->weights_failed=true; session->state->invalidate();
        } else result.weight_activity+=finished.value();
        result.cancel_requested = result.cancel_requested || requested();
        const auto after=session->state->query();
        result.checkpoint_boundary = after.checkpoint;
        output.replay = {after.tokens, session->blocks};
        result.e=after.e; result.state_valid=after.valid&&!session->weights_failed;
        result.budget_current=budget.current(); result.budget_peak=budget.peak();
        auto& summary=result.request_summary;
        summary.total_seconds=seconds(started,model->clock_now());
        if(loaded->stages.mixed()) {
            const auto stats=transfer_stats();
            summary.handoffs=stats.count-initial_transfers.count;
            summary.handoff_bytes=stats.bytes-initial_transfers.bytes;
            summary.handoff_seconds=stats.seconds-initial_transfers.seconds;
        }
        if(first_token) summary.first_token_seconds=seconds(started,*first_token);
        else summary.first_token_reason="no_accepted_token";
        if(result.stop_reason==StopReason::failed)
            summary.decode_reason=receiver_stopped?"ended_by_receiver":"ended_by_failure";
        else if(result.stop_reason==StopReason::cancelled) summary.decode_reason="cancelled";
        else if(result.generated.size()<2) summary.decode_reason="fewer_than_two_tokens";
        else if(first_token && last_token) {
            summary.decode_seconds=seconds(*first_token,*last_token);
            summary.decode_forwards=result.generated.size()-1;
            if(*summary.decode_seconds>0) summary.decode_tokens_per_second=*summary.decode_forwards / *summary.decode_seconds;
            else summary.decode_reason="clock_resolution";
        } else summary.decode_reason="missing_last_token_boundary";


    }
    Result<GenerationCheckResult> execute() {
        auto prepared=prepare();
        if(!prepared) return prepared.error();
        if(!prepared.value()) return std::move(output);
        model::TextOptions parsing_options{result.configuration.thinking,{},
            chat && chat->tool_choice.mode!=ToolChoiceMode::none?chat->tools:std::vector<ToolDefinition>{}};
        parser=rules->adapter->output_parser(parsing_options);
        stop_matcher.emplace(result.configuration.stop);
        host_sampling = result.configuration.temperature > 0 || result.configuration.repetition_penalty != 1 ||
            result.configuration.presence_penalty!=0 || result.configuration.frequency_penalty!=0 ||
            !result.configuration.logit_bias.empty() || result.configuration.logprobs;
        if (host_sampling) history.insert(result.input.begin(), result.input.end());
        auto description=model->compute->describe_states(loaded->assets.definition);
        if(!description) return description.error();
        const auto n=result.input.size();
        const auto first=description.value().evolves_as_whole&&n>=2?n-1:n;
        auto suffix = text::prepare_suffix(result.input, result.reused_tokens);
        if (!suffix) return suffix.error();
        const bool okay=prefill(description.value(),n,first);
        decode(okay);
        finish();
        return std::move(output);
    }
};
Result<GenerationCheckResult> generate_impl(const SessionReference& session,
    const std::vector<std::int32_t>* given,const ChatInput* chat,
    const GenerationConfiguration& configuration,const GenerationOverrides& overrides,
    const GenerationCheckOptions& options,OutputReceiver receiver,const CancelToken* cancel_token) {
    if(!session) return invalid("Missing session reference");
    const auto model=session->model;
    auto resources=RuntimeAccess::acquire(model->runtime); if(!resources) return resources.error();
    Activity activity; if(auto e=activity.enter(model->registry)) return *e;
    if (!CancelAccess::consume(cancel_token, model->registry)) return invalid("Invalid or used cancellation token");
    std::shared_ptr<model::LoadedModel> loaded;
    std::shared_ptr<text::Rules> rules;
    {
        std::lock_guard<std::mutex> lock(model->registry->mutex);
        if(session->releasing||session->state->query().released) return invalid("Session was released");
        if(model->unloading) return Error{ErrorCategory::unloading,"Model is unloading"};
        if(session->weights_failed || !session->state->query().valid) return Error{ErrorCategory::session_invalid,"Session state is invalid"};
        loaded=model->loaded; rules=model->rules; session->running=true;
    }
    struct Finish {
        SessionReference session;
        ~Finish() {
            {
                std::lock_guard<std::mutex> lock(session->model->registry->progress_mutex);
                session->model->registry->progress = {};
            }
            session->state->end_use();
            std::lock_guard<std::mutex> lock(session->model->registry->mutex);
            session->running=false;
            if(session->releasing) reclaim_session_locked(session);
        }
    } finish{session};
    GenerationRun run{session,given,chat,configuration,overrides,options,receiver,
        cancel_token,resources.value(),loaded,rules};
    return run.execute();
}
}
Result<GenerationResult> generate(const SessionReference& session,const std::vector<TextMessage>& messages,
    const GenerationConfiguration& configuration,const GenerationOverrides& overrides,OutputReceiver receiver,const CancelToken* cancel_token) {
    return generate(session, ChatInput{messages, {}}, configuration, overrides, std::move(receiver), cancel_token);
}
Result<GenerationResult> generate(const SessionReference& session,const ChatInput& chat,
    const GenerationConfiguration& configuration,const GenerationOverrides& overrides,OutputReceiver receiver,const CancelToken* cancel_token) {
    auto result=generate_impl(session,nullptr,&chat,configuration,overrides,{},std::move(receiver),cancel_token);
    if(!result) return result.error();
    return result.value().result;
}
Result<GenerationCheckResult> generate_tokens_check(const SessionReference& session,const std::vector<std::int32_t>& tokens,
    const GenerationConfiguration& configuration,const GenerationOverrides& overrides,const GenerationCheckOptions& options,OutputReceiver receiver,const CancelToken* cancel_token) {
    return generate_impl(session,&tokens,nullptr,configuration,overrides,options,std::move(receiver),cancel_token);
}
}

namespace edge0 {
Result<GenerationCheckResult> generate_chat_check(const SessionReference& session, const ChatInput& chat,
    const GenerationConfiguration& configuration, const GenerationOverrides& overrides,
    const GenerationCheckOptions& options, OutputReceiver receiver, const CancelToken* cancel_token) {
    return generate_impl(session, nullptr, &chat, configuration, overrides, options, std::move(receiver), cancel_token);
}
Result<TokenCount> count_tokens(const ModelReference& model, const ChatInput& chat, const GenerationConfiguration& configuration) {
    if (auto error = readable(model)) return *error;
    std::lock_guard<std::mutex> lock(model->registry->mutex);
    if (!model->loaded || model->unloading) return invalid("Model was released");
    if (auto error = validate_model(configuration, model)) return *error;
    auto rendered = text::render_input(*model->rules, chat, {configuration.thinking, {}});
    if (!rendered) return rendered.error();
    TokenCount result; result.input_tokens = rendered.value().tokens.size(); result.capacity = model->configuration.context;
    return result;
}
Result<TokenCount> count_tokens(const SessionReference& session, const ChatInput& chat, const GenerationConfiguration& configuration) {
    if (!session) return invalid("Missing session reference");
    const auto model = session->model;
    if (auto error = readable(model)) return *error;
    std::lock_guard<std::mutex> lock(model->registry->mutex);
    if (session->running) return busy();
    const auto state = session->state->query();
    if (session->releasing || state.released) return invalid("Session was released");
    if (!state.valid || session->weights_failed) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (!model->loaded || model->unloading) return invalid("Model was released");
    if (auto error = validate_model(configuration, model)) return *error;
    auto rendered = text::render_input(*model->rules, chat, {configuration.thinking, {}});
    if (!rendered) return rendered.error();
    const auto& input = rendered.value().tokens;
    auto reuse = session->state->check_reuse(text::match_history(input, state, identity(model)), input.size(), identity(model));
    if (!reuse) return reuse.error();
    const auto k = static_cast<std::size_t>(reuse.value().k);
    return TokenCount{input.size(), state.capacity, k, input.size() - k, static_cast<std::size_t>(state.capacity) - k};
}
Result<TokenCount> count_tokens(const ModelReference& model, const std::vector<TextMessage>& messages, const GenerationConfiguration& configuration) {
    return count_tokens(model, ChatInput{messages, {}}, configuration);
}
Result<TokenCount> count_tokens(const SessionReference& session, const std::vector<TextMessage>& messages, const GenerationConfiguration& configuration) {
    return count_tokens(session, ChatInput{messages, {}}, configuration);
}
Result<std::string> reset_conversation(const SessionReference& session) {
    if (!session) return invalid("Missing session reference");
    const auto model = session->model;
    Activity activity; if (auto error = activity.enter(model->registry)) return *error;
    const auto state = session->state->query();
    if (session->releasing || state.released) return invalid("Session was released");
    if (!state.valid || session->weights_failed) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (auto error = readable(model)) return *error;
    auto finished = model->compute->finish_pending_weights(model->loaded, session->state->states(), models::FinishReason::normal);
    if (!finished) return finished.error();
    if (auto error = session->state->reset([&](session::KVCache& kv, session::StateStore& states) {
        return model->compute->reset_session(model->loaded, kv, states);
    })) return *error;
    session->blocks.clear();
    return std::string("Conversation reset");
}
Result<GenerationCheckResult> replay_check(const SessionReference& session, const ReplayInput& input) {
    if (!session) return invalid("Missing session reference");
    const auto model = session->model;
    auto resources = RuntimeAccess::acquire(model->runtime); if (!resources) return resources.error();
    Activity activity; if (auto error = activity.enter(model->registry)) return *error;
    if (auto error = readable(model)) return *error;
    const auto before = session->state->query();
    if (before.e || before.released || session->releasing) return invalid("Replay requires an empty session");
    if (!before.valid || session->weights_failed) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    std::size_t end = 0;
    const auto vocabulary = model->loaded->assets.definition.vocabulary_size;
    for (const auto& block : input.blocks) {
        if (!block.length || block.start != end || block.length > input.tokens.size() - end ||
            (block.phase == BlockPhase::decode && block.length != 1)) return invalid("Invalid replay blocks");
        const auto backend = block.device.empty() ? model->loaded->stages.of(block.phase==BlockPhase::prefill?exec::Phase::prefill:exec::Phase::decode) :
            (resources.value().backends.count(block.device)?resources.value().backends.at(block.device):nullptr);
        if(!backend) return invalid("Replay device is outside runtime scope");
        if(block.phase==BlockPhase::decode) {
            if(backend!=model->loaded->stages.decode) return invalid("Replay decode device differs from configured device");
        } else if(model->loaded->mixed) {
            const auto& candidates=model->loaded->mixed->candidates;
            if(std::find(candidates.begin(),candidates.end(),backend)==candidates.end()) return invalid("Replay prefill device is outside policy candidates");
        } else if(backend!=model->loaded->stages.prefill) return invalid("Replay prefill device differs from configured device");
        end += block.length;
    }
    if (end != input.tokens.size() || end > static_cast<std::size_t>(before.capacity)) return invalid("Invalid replay capacity");
    for (auto token : input.tokens) if (token < 0 || static_cast<std::size_t>(token) >= vocabulary) return invalid("Invalid replay token");
    if (auto error = session->state->begin_use()) return *error;
    {
        std::lock_guard<std::mutex> lock(model->registry->mutex); session->running = true;
    }
    struct FinishReplay {
        SessionReference session;
        bool drained=false;
        exec::Status drain(models::FinishReason reason) {
            if (drained) return {};
            drained=true;
            const auto result=session->model->compute->finish_pending_weights(session->model->loaded,session->state->states(),reason);
            if (!result) {
                session->weights_failed=true;
                session->state->invalidate();
                return result.error();
            }
            return {};
        }
        ~FinishReplay() {
            drain(models::FinishReason::failure);
            {
                std::lock_guard<std::mutex> lock(session->model->registry->progress_mutex);
                session->model->registry->progress = {};
            }
            session->state->end_use();
            std::lock_guard<std::mutex> lock(session->model->registry->mutex); session->running = false;
            if (session->releasing) reclaim_session_locked(session);
        }
    } finish{session};
    GenerationCheckResult output; output.vocabulary = vocabulary;
    auto budget = resources.value().budget;
    for (const auto& block : input.blocks) {
        std::vector<std::int32_t> tokens(input.tokens.begin() + block.start, input.tokens.begin() + block.start + block.length);
        auto update = session->state->begin_update(tokens); if (!update) return update.error();
        models::ForwardRequest request{tokens, update.value().step, block.phase == BlockPhase::prefill ? exec::Phase::prefill : exec::Phase::decode, {tokens.size()-1}, {}};
        request.backend=block.device.empty()?model->loaded->stages.of(request.phase):resources.value().backends.at(block.device);
        request.greedy = true; request.retain_logits = true;
        models::ForwardProgress progress; request.progress = &progress;
        select_cpu_threads(resources.value(), request.phase);
        if(model->loaded->stages.mixed()) if(auto error=app::prepare_stage_forward(*model->compute,model->loaded,session->state->kv(),session->state->states(),request,resources.value().registry.get())) {
            session->state->report_interruption(update.value(),false); return *error;
        }
        auto result = model->compute->forward(model->loaded, request, session->state->kv(), session->state->states(), budget);
        if (!result) { session->state->report_interruption(update.value(), progress.submitted); return result.error(); }
        auto& owner = result.value();
        if (auto error = session->state->commit_update(update.value(), owner->completion(), owner->updated_entries())) {
            session->state->report_interruption(update.value(), true); owner->end_observation(); return *error;
        }
        auto logits = owner->logits(); auto greedy = owner->greedy_output();
        auto ended = owner->end_observation();
        if (!logits) return logits.error();
        if (!greedy) return greedy.error();
        if (ended) return *ended;
        output.logits.insert(output.logits.end(), logits.value().begin(), logits.value().end());
        output.result.generated.push_back(greedy.value().token);
        output.result.computed_tokens += block.length;
    }
    if (auto error=finish.drain(models::FinishReason::normal)) return *error;
    session->blocks = input.blocks; output.replay = input;
    output.result.e = session->state->query().e; output.result.input_complete = true;
    output.result.stop_reason = StopReason::length;
    return output;
}
}
