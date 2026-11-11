#include "app/generation_internal.h"
#include "app/adapters.h"
#include "tests/support/runtime.h"
#include "edge0/diagnostics.h"
#include "models/edge0_35b/output.h"
#include "app/text/unicode.h"
#include <regex>
#include <limits>
#include <algorithm>
#include <cstdio>
extern "C" void file_read_counter_start();
extern "C" unsigned file_read_counter_stop();
#include <thread>
#include <condition_variable>
#include <mutex>
using namespace edge0;
using test::take;
struct FakeAdapter : model::Adapter {
    bool classified=false;
    mutable model::TextOptions seen;
    FakeAdapter():Adapter({}) {}
    std::string id() const override {return "fake";}
    model::TextDescription describe_text() const override {return {};}
    model::TextResult text_format(const std::vector<model::Message>& messages,const model::TextOptions& options) const override {seen=options;model::TextResult r;r.thinking=options.thinking.value_or(false);for(const auto& m:messages)r.fragments.push_back({m.content,false});return r;}
    std::unique_ptr<model::OutputParser> output_parser(const model::TextOptions& options) const override {
        // Borrow the real model parser for its streaming format; the framework does not depend on it.
        return classified?models::edge0_35b::output_parser(options):std::make_unique<model::OutputParser>();
    }
    Result<model::ModelDefinition> describe_assets(const model::AssetView&) const override {return model::ModelDefinition{};}
};
struct FakeResult : models::ForwardResult {
    bool sample_failure=false, error_seen=false;
    std::vector<float> row{1,0,0};
    int* logits_calls=nullptr;
    bool allow_logits=false;
    std::size_t bytes=128; // Route readback precedes sampling and must be excluded.
    const bool* pending=nullptr;
    exec::Completion completed{true,false,{}};
    std::optional<DecodeTiming> timing;
    std::optional<DecodeTiming> decode_timing() const override { return timing; }
    models::GreedyOutput choice{0,1};
    std::vector<std::string> updated;
    const exec::Completion& completion() const override{return completed;}
    Result<std::vector<float>> logits() override{CHECK(allow_logits);++*logits_calls;bytes+=row.size()*sizeof(float);return row;}
    Result<std::vector<models::ObservedTensor>> observations() override{return std::vector<models::ObservedTensor>{};}
    exec::Status end_observation() override{if(error_seen) CHECK(pending && !*pending);return {};}
    std::size_t weight_registrations() const override{return 0;}
    const std::vector<std::string>& updated_entries() const override{return updated;}
    Result<models::GreedyOutput> greedy_output() override{
        if(sample_failure) {error_seen=true;return Error{ErrorCategory::backend_error,"sample read failed"};}
        bytes+=8;return choice;
    }
    std::size_t readback_bytes() const override{return bytes;}
    bool reused_plan() const override{return false;}
    WeightActivity weight_activity() const override { return {{2,1,1,64},{3,0,3,192}}; }
};
struct FakeCompute : models::ModelCompute {
    mutable std::vector<std::int64_t> blocks;
    mutable int calls=0,resets=0,releases=0,logits_calls=0;
    std::vector<float> row{1,0,0};
    bool whole=true, checkpoints=false, save_failure=false;
    mutable int saves=0,restores=0,discards=0;
    bool tail_delay=false,tail_failure=false,sample_failure=false,finish_failure=false;
    mutable bool pending=false;
    mutable std::vector<models::FinishReason> finishes;
    std::function<void(int)> on_forward;
    std::function<void()> on_save;
    int cancelled_at=-1;
    int fail_at=-1,end_at=-1;
    bool submitted=false, fail_after_segment=false;
    models::GreedyOutput choice{0,1};
    Result<models::StateDescription> describe_states(const model::ModelDefinition&) const override {
        models::StateDescription d;d.evolves_as_whole=whole;d.supports_save_restore=checkpoints;
        d.kv={{"k",GGML_TYPE_F32,{4},1,1,session::Visibility::causal}};return d;
    }
    exec::Status initialize_session(const std::shared_ptr<model::LoadedModel>&,session::KVCache&,session::StateStore&,exec::Budget&,const exec::Stages&) const override{return {};}
    Result<std::unique_ptr<models::ForwardResult>> forward(const std::shared_ptr<model::LoadedModel>&,const models::ForwardRequest& r,
        session::KVCache& kv,session::StateStore&,exec::Budget&) const override {
        pending=tail_delay||tail_failure||sample_failure;
        const int index=calls++;blocks.push_back(r.step.n);
        if (on_forward) on_forward(index);
        if (index == cancelled_at) {
            CHECK(r.cancel_block); r.cancel_block(1);
            r.progress->submitted = true; r.progress->cancelled = true;
            r.progress->rolled_back_tokens = r.tokens.size();
            return Error{ErrorCategory::invalid_argument,"cancelled block"};
        }
        if(index==fail_at && fail_after_segment) { if(r.progress) { r.progress->submitted=submitted; r.progress->activity.decode={2,0,2,128}; } return Error{ErrorCategory::backend_error,"segment failed"}; }
        if(index==fail_at&&!submitted)return Error{ErrorCategory::backend_error,"before submit"};
        auto result=std::make_unique<FakeResult>();result->choice=choice;result->row=row;result->logits_calls=&logits_calls;result->allow_logits=r.retain_logits;
        result->sample_failure=sample_failure;result->pending=&pending;
        if(r.breakdown && r.phase==exec::Phase::decode) {
            const auto start=r.clock_now();
            DecodeTiming timing;timing.prepare_seconds=.1;timing.wait_seconds=.2;timing.submit_seconds=.3;timing.route_seconds=.1;timing.segments=3;
            timing.total_seconds=std::chrono::duration<double>(r.clock_now()-start).count();result->timing=timing;
        }
        if(index==end_at)result->choice.token=2;
        result->completed.written={take(kv.storage("k"))->allocation};
        if(index==fail_at)result->completed={false,true,{}};
        return std::unique_ptr<models::ForwardResult>(std::move(result));
    }
    Result<WeightActivity> finish_pending_weights(const std::shared_ptr<model::LoadedModel>&,
        session::StateStore&,models::FinishReason reason) const override {
        finishes.push_back(reason);
        const bool active=pending;pending=false;
        if(active && tail_delay) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if(finish_failure || (active && tail_failure && reason==models::FinishReason::normal))
            return Error{ErrorCategory::backend_error,"tail read failed"};
        return WeightActivity{};
    }
    exec::Status save_state(session::StateStore&) const override {++saves;if(on_save)on_save();if(save_failure)return Error{ErrorCategory::backend_error,"save failed"};return {};}
    exec::Status restore_state(session::StateStore&) const override {++restores;return {};}
    exec::Status discard_saved_state(session::StateStore&) const override {++discards;return {};}
    exec::Status reset_session(const std::shared_ptr<model::LoadedModel>&,session::KVCache&,session::StateStore&) const override{CHECK(!pending);++resets;return {};}
    void release_session(const std::shared_ptr<model::LoadedModel>&,session::KVCache&,session::StateStore&) const override{CHECK(!pending);++releases;}
    void close_model(model::LoadedModel&) const override{}
};
struct Fixture {
    FakeAdapter adapter;
    FakeCompute compute;
    RuntimeReference runtime=take(open_runtime({{"CPU"},{}}));
    ModelReference model;
    SessionReference session;
    GenerationConfiguration config;
    std::size_t model_bytes=0;
    explicit Fixture(std::int64_t capacity=1024,bool checkpoints=false) {
        compute.checkpoints=checkpoints;
        auto resources=take(RuntimeAccess::acquire(runtime));
        auto loaded=std::make_shared<model::LoadedModel>(resources.budget);loaded->stages={resources.backends.at("CPU"),resources.backends.at("CPU")};
        loaded->adapter=&adapter;
        loaded->storages.push_back(take(loaded->weights.create_storage(loaded->stages.prefill,{{{GGML_TYPE_F32,{4}},1}}, edge0::exec::Purpose::resident_weights)));
        model_bytes=resources.budget.current();
        loaded->assets.definition.vocabulary_size=3;
        loaded->assets.definition.context_length=capacity;
        loaded->generation_defaults.temperature=1;loaded->generation_defaults.top_k=20;loaded->generation_defaults.top_p=.95;
        text::TokenData data;data.tokens={"a","b","<|im_end|>"};data.types={1,1,3};
        auto rules=std::make_shared<text::Rules>(text::Rules{text::Bpe(data),{},&adapter});
        LoadConfiguration load;load.context=capacity;load.prefill_chunk=std::min<std::int64_t>(512,capacity);load.device="CPU";
        model=take(app::register_model(runtime,loaded,rules,compute,load));
        session=take(create_session(model));
        GenerationOverrides application;application.temperature=0;config=take(resolve_generation_configuration(model,application));
    }
    ~Fixture(){take(finish_runtime_use(runtime));}
    GenerationResult run(std::size_t n=3,std::int64_t maximum=3,OutputReceiver receiver={}) {
        GenerationOverrides o;o.max_tokens=maximum;
        auto result=take(generate_tokens_check(session,std::vector<std::int32_t>(n,0),config,o,{},receiver)).result;
        CHECK(!result.checkpoint_saved);
        return result;
    }
};
int memory_calls=0;
edge0::platform::ProcessMemory fake_memory() {
    ++memory_calls;
    return {{{edge0::platform::MemoryValues{11,22}},{}},{{edge0::platform::MemoryValues{33,44}},{}}};
}
edge0::platform::ProcessMemory failed_memory() {
    ++memory_calls;
    return {{std::nullopt,"injected RSS failure"},{std::nullopt,"injected physical failure"}};
}
int clock_reads=0;
std::chrono::steady_clock::time_point counted_clock() {
    return std::chrono::steady_clock::time_point{std::chrono::seconds(++clock_reads)};
}
void exact_window(const GenerationResult& r) {
    const auto& s=r.request_summary;
    CHECK(s.first_token_seconds && s.decode_seconds && s.decode_forwards && s.decode_tokens_per_second);
    CHECK(*s.first_token_seconds<=s.total_seconds && *s.decode_seconds<=s.total_seconds);
    CHECK(*s.decode_forwards==r.generated.size()-1);
    CHECK(*s.decode_tokens_per_second==*s.decode_forwards / *s.decode_seconds);
    CHECK(s.decode_reason.empty());
}
void missing_window(const GenerationResult& r,const std::string& reason) {
    const auto& s=r.request_summary;
    CHECK(!s.decode_seconds && !s.decode_forwards && !s.decode_tokens_per_second);
    CHECK(s.decode_reason==reason && s.total_seconds>0);
}
void multi_turn_tests() {
    Fixture f(1024,true);GenerationOverrides limits;limits.max_tokens=3;
    auto run=[&](std::vector<std::int32_t> input,ReuseMethod method,std::size_t k,std::size_t computed,bool saved) {
        auto r=take(generate_tokens_check(f.session,input,f.config,limits)).result;
        CHECK(r.reuse_method==method && r.reused_tokens==k && r.computed_tokens==computed && r.checkpoint_saved==saved);
        return r;
    };
    run({0,0,0},ReuseMethod::first,0,5,true);
    run({0,0,0},ReuseMethod::restore,2,3,false);
    run({0,0,0},ReuseMethod::restore,2,3,false);CHECK(f.compute.restores==2 && f.compute.saves==1);
    run({0,0,0,0,0,0,1},ReuseMethod::keep,5,4,true);
    run({0,0,0,0,0,0,1,0,0,0,1},ReuseMethod::keep,9,4,true);
    run({0,0,0,0,0,0,1,0,0,0,1,1},ReuseMethod::restore,10,4,true);
    run({1,1,1},ReuseMethod::cold,0,5,true);
    const auto before=take(query_session_check(f.session));
    for(auto chat:std::vector<ChatInput>{{{{"tool","a"}},{}},{{{"user","a",{{"id","tool","{}"}}}},{}},{{{"user","a"}},{{"{}"}}},{{{"user","a",{},std::string("reason")}},{}},{{{"user","a",{},std::nullopt,"id"}},{}}}) {
        auto rejected=generate(f.session,chat,f.config);
        CHECK(!rejected && rejected.error().category==(chat.tools.empty()?ErrorCategory::invalid_argument:ErrorCategory::unsupported));
        const auto after=take(query_session_check(f.session));CHECK(after.e==before.e && after.tokens==before.tokens && after.checkpoint_boundary==before.checkpoint_boundary);
    }
    ChatInput too_long{{{"user",std::string(1025,'a')}}, {}};
    const auto oversized=take(count_tokens(f.session,too_long,f.config));CHECK(oversized.input_tokens==1025 && oversized.capacity==1024);
    auto refused=generate(f.session,too_long,f.config);CHECK(!refused && refused.error().category==ErrorCategory::capacity_exceeded);
    CHECK(take(query_session_check(f.session)).tokens==before.tokens && take(query_session_check(f.session)).checkpoint_boundary==before.checkpoint_boundary);
    ChatInput chat{{{"user","bbbaaa"}}, {}};
    const auto count=take(count_tokens(f.session,chat,f.config));
    auto result=take(generate(f.session,chat,f.config,limits));CHECK(count.reused_tokens==result.reused_tokens);
    CHECK(take(count_tokens(f.model,chat,f.config)).input_tokens==6);
    take(reset_conversation(f.session));CHECK(take(query_session_check(f.session)).e==0);
    auto callback=[&](const OutputEvent&)->exec::Status {
        auto count_busy=count_tokens(f.session,chat,f.config);CHECK(!count_busy && count_busy.error().category==ErrorCategory::busy);
        auto reset_busy=reset_conversation(f.session);CHECK(!reset_busy && reset_busy.error().category==ErrorCategory::busy);return {};
    };
    take(generate(f.session,chat,f.config,limits,callback));
    take(release_session(f.session));auto released=reset_conversation(f.session);CHECK(!released && released.error().category==ErrorCategory::invalid_argument);
}
static_assert(noexcept(request_cancel(std::declval<const CancelToken&>())));
static_assert(std::atomic<bool>::is_always_lock_free);
void cancellation_tests() {
    auto unchanged = [](const SessionCheckState& a, const SessionCheckState& b) {
        CHECK(a.e==b.e && a.tokens==b.tokens && a.checkpoint_boundary==b.checkpoint_boundary && a.valid==b.valid);
    };
    std::optional<CancelToken> old;
    {
        Fixture f(1024,true); GenerationOverrides limits; limits.max_tokens=4;
        take(generate_tokens_check(f.session,{0,0,0},f.config,limits));
        auto token=take(create_cancel_token(f.runtime)); auto copy=token; request_cancel(copy);
        const auto before=take(query_session_check(f.session)); const auto calls=f.compute.calls;
        auto result=take(generate_tokens_check(f.session,{0,0,0},f.config,limits,{}, {}, &token)).result;
        CHECK(result.stop_reason==StopReason::cancelled && result.cancel_requested && !result.input_complete);
        CHECK(!result.cold && !result.reused_tokens && !result.computed_tokens && f.compute.calls==calls);
        unchanged(before,take(query_session_check(f.session)));
        auto used=generate_tokens_check(f.session,{0,0,0},f.config,limits,{}, {}, &copy);
        CHECK(!used && used.error().category==ErrorCategory::invalid_argument);
        unchanged(before,take(query_session_check(f.session)));
        old=take(create_cancel_token(f.runtime));
    }
    {
        Fixture f; const auto before=take(query_session_check(f.session));
        auto result=generate_tokens_check(f.session,{0},f.config,{}, {}, {}, &*old);
        CHECK(!result && result.error().category==ErrorCategory::invalid_argument);
        unchanged(before,take(query_session_check(f.session)));
        auto token=take(create_cancel_token(f.runtime));
        auto result2=take(generate(f.session,std::vector<TextMessage>{{"user","a"}},f.config,{}, {}, &token));
        CHECK(result2.stop_reason!=StopReason::cancelled);
        request_cancel(token); auto next=take(create_cancel_token(f.runtime));
        CHECK(take(generate(f.session,ChatInput{{{"user","a"}},{}},f.config,{}, {}, &next)).stop_reason!=StopReason::cancelled);
    }
    for(bool failure:{false,true}) {
        Fixture f(1024,true); f.model->configuration.prefill_chunk=2;
        auto token=take(create_cancel_token(f.runtime)); auto busy_token=take(create_cancel_token(f.runtime));
        std::mutex mutex; std::condition_variable condition; bool entered=false, resume=false;
        f.compute.on_forward=[&](int index) {
            if(index!=1)return;
            std::unique_lock<std::mutex> lock(mutex); entered=true; condition.notify_one();
            condition.wait(lock,[&]{return resume;});
        };
        if(failure) { f.compute.fail_at=1; f.compute.fail_after_segment=true; f.compute.submitted=true; }
        std::thread cancelling([&] {
            { std::unique_lock<std::mutex> lock(mutex); condition.wait(lock,[&]{return entered;}); }
            auto rejected=generate_tokens_check(f.session,{0},f.config,{}, {}, {}, &busy_token);
            CHECK(!rejected && rejected.error().category==ErrorCategory::busy);
            request_cancel(token);
            { std::lock_guard<std::mutex> lock(mutex);resume=true; } condition.notify_one();
        });
        auto result=take(generate_tokens_check(f.session,std::vector<std::int32_t>(11,0),f.config,{}, {}, {}, &token)).result;
        cancelling.join();
        CHECK(result.cancel_requested && result.state_valid==!failure);
        CHECK(result.stop_reason==(failure?StopReason::failed:StopReason::cancelled));
        CHECK(result.e==(failure?2:4));
        if(!failure) CHECK(generate_tokens_check(f.session,{0},f.config,{}, {}, {}, &busy_token));
    }
    for(bool last:{false,true}) {
        Fixture f; auto token=take(create_cancel_token(f.runtime));int fragments=0;
        GenerationOverrides limits; limits.max_tokens=last?2:4;
        auto receiver=[&](const OutputEvent&)->std::optional<Error>{if(++fragments==2)request_cancel(token);return {};};
        auto result=take(generate_tokens_check(f.session,{0},f.config,limits,{},receiver,&token)).result;
        CHECK(fragments==2 && result.cancel_requested && result.generated.size()==2);
        CHECK(f.compute.finishes.back()==models::FinishReason::normal);
        CHECK(result.stop_reason==(last?StopReason::length:StopReason::cancelled));
    }
    {
        Fixture f(1024,true); f.model->configuration.prefill_chunk=2;
        GenerationOverrides limits;limits.max_tokens=4;
        take(generate_tokens_check(f.session,{0,0,0},f.config,limits));
        const auto before=take(query_session_check(f.session));
        f.compute.cancelled_at=f.compute.calls;
        auto token=take(create_cancel_token(f.runtime));
        auto result=take(generate_tokens_check(f.session,std::vector<std::int32_t>(11,0),f.config,limits,{}, {}, &token)).result;
        CHECK(result.stop_reason==StopReason::cancelled && result.state_valid && result.rolled_back_tokens==2);
        CHECK(!result.computed_tokens && result.e==before.e && result.checkpoint_boundary==before.checkpoint_boundary);
        unchanged(before,take(query_session_check(f.session)));
    }
    for (bool unload : {false,true}) {
        Fixture f(1024,true);
        f.compute.on_save=[&] { if(unload)take(unload_model(f.model));else take(release_session(f.session)); };
        auto result=take(generate_tokens_check(f.session,{0,0,0},f.config)).result;
        CHECK(result.stop_reason==StopReason::cancelled && result.cancel_requested);
        CHECK(f.compute.calls==1 && result.e==2 && !result.input_complete);
        CHECK(f.compute.finishes.back()==models::FinishReason::failure);
    }
    for (bool delivery_failure : {false,true}) {
        Fixture f;auto token=take(create_cancel_token(f.runtime));
        auto receiver=[&](const OutputEvent&)->std::optional<Error> {
            request_cancel(token);
            if(delivery_failure) return Error{ErrorCategory::backend_error,"Delivery failed after cancellation"};
            f.compute.finish_failure=true;return {};
        };
        auto result=take(generate_tokens_check(f.session,{0},f.config,{}, {},receiver,&token)).result;
        CHECK(result.stop_reason==StopReason::failed && result.cancel_requested);
        CHECK(result.error_stage==(delivery_failure?GenerationStage::delivery:GenerationStage::forward));
        CHECK(result.state_valid==delivery_failure);
        f.compute.finish_failure=false;
    }
    for(auto point:{CancelPointKind::before_compute,CancelPointKind::after_reuse,CancelPointKind::before_block}) {
        Fixture f(1024,true);f.model->configuration.prefill_chunk=2;
        GenerationCheckOptions options; options.cancel_probe=[&](const CancelPoint& seen) {return seen.kind==point && (point!=CancelPointKind::before_block || seen.block==5);};
        auto r=take(generate_tokens_check(f.session,std::vector<std::int32_t>(11,0),f.config,{},options));
        CHECK(r.result.stop_reason==StopReason::cancelled && r.result.state_valid && !r.result.checkpoint_saved);
        CHECK(r.result.e==(point==CancelPointKind::before_block?10:0));
        CHECK(r.cancel_point && r.cancel_point->kind==point);
    }
}
void thinking_tools_tests() {
    {
        LoadConfiguration c;
        CHECK(!app::resolve_context_capacity(c,{},16384,8192));
        CHECK(c.context==8192 && c.sources.at("context")==ConfigurationSource::model);
        CHECK(!app::resolve_context_capacity(c,{},16384,32768));
        CHECK(c.context==16384 && c.sources.at("context")==ConfigurationSource::model);
        CHECK(!app::resolve_context_capacity(c,4096,16384,32768));
        CHECK(c.context==4096 && c.sources.at("context")==ConfigurationSource::request);
        CHECK(!app::resolve_context_capacity(c,{},{},64));
        CHECK(c.context==64 && c.sources.at("context")==ConfigurationSource::platform);
        CHECK(app::resolve_context_capacity(c,32769,16384,32768));
    }
    auto prepare=[](Fixture& f,const std::string& output) {
        f.adapter.classified=true;
        f.model->rules->description.thinking=true;f.model->rules->description.thinking_enabled=true;
        f.model->rules->description.tools=true;
        text::TokenData data;
        std::string encoded;for(unsigned char c:output) encoded+=text::unicode_byte_to_utf8(c);
        data.tokens={encoded,"b","<|im_end|>"};data.types={1,1,3};f.model->rules->tokenizer=text::Bpe(data);
    };
    const std::string call="<tool_call>\n<function=f>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call>";
    const ToolDefinition tool{R"({"function":{"name":"f","parameters":{"properties":{"s":{"type":"string"}}}}})"};
    {
        Fixture f;prepare(f,"unused");
        text::TokenData data;data.tokens={"reason","</think>","answer","<|im_end|>"};data.types={1,3,1,3};
        f.model->rules->tokenizer=text::Bpe(data);f.model->loaded->assets.definition.vocabulary_size=4;
        f.compute.on_forward=[&](int index) {f.compute.choice.token=index;};
        GenerationOverrides o;o.thinking=true;o.max_tokens=4;
        const auto r=take(generate_tokens_check(f.session,{0},f.config,o)).result;
        CHECK(r.reasoning=="reason" && r.text=="answer" && r.reasoning_tokens==2 && r.stop_reason==StopReason::end_of_generation);
    }
    {
        Fixture f;prepare(f," reason </think>\n\nanswer\n"+call+"\n"+call);
        ChatInput chat{{{"user","b"}},{tool}};
        GenerationOverrides o;o.thinking=true;o.max_tokens=1;
        std::vector<OutputEvent> events;
        const auto count=take(count_tokens(f.session,chat,[&] {auto c=f.config;c.thinking=true;return c;}()));
        auto r=take(generate(f.session,chat,f.config,o,[&](const OutputEvent& e)->exec::Status {events.push_back(e);return {};}));
        CHECK(r.reasoning=="reason" && r.text=="answer" && r.tool_calls.size()==2 && r.rejected_tool_calls.empty());
        CHECK(r.input.size()==count.input_tokens && r.reused_tokens==*count.reused_tokens && r.reasoning_tokens==1);
        CHECK(std::regex_match(r.tool_calls[0].id,std::regex("call_[0-9a-f]{24}")) && r.tool_calls[0].id!=r.tool_calls[1].id);
        std::string answer,reasoning,args;std::size_t ends=0;
        for(const auto& e:events) {
            if(e.kind==OutputKind::answer) answer+=e.text;
            if(e.kind==OutputKind::reasoning) reasoning+=e.text;
            if(e.kind==OutputKind::tool_call_begin) args.clear();
            if(e.kind==OutputKind::tool_call_arguments) args+=e.text;
            if(e.kind==OutputKind::tool_call_end) {CHECK(args==e.tool_call.arguments_json && e.tool_call.id==r.tool_calls[ends++].id);}
        }
        CHECK(answer==r.text && reasoning==r.reasoning && ends==2);
        chat.parallel_tool_calls=false;
        r=take(generate(f.session,chat,f.config,o));CHECK(r.stop_reason==StopReason::tool_calls && r.tool_calls.size()==1 && r.generated.size()==1);
        chat.tool_choice.mode=ToolChoiceMode::none;o.thinking=false;
        r=take(generate(f.session,chat,f.config,o));CHECK(r.tool_calls.empty() && r.text.find("<tool_call>")!=std::string::npos && f.adapter.seen.tools.empty());
        o.thinking=true;
        r=take(generate(f.session,chat,f.config,o,[](const OutputEvent& e)->exec::Status {
            if(e.kind==OutputKind::reasoning) return Error{ErrorCategory::invalid_argument,"reject reasoning"};return {};
        }));CHECK(r.error_stage==GenerationStage::delivery && r.reasoning.empty());
        for(auto mode:{ToolChoiceMode::required,ToolChoiceMode::function}) {
            chat.tool_choice.mode=mode;const auto before=take(query_session_check(f.session));
            auto rejected=generate(f.session,chat,f.config,o);CHECK(!rejected && rejected.error().category==ErrorCategory::unsupported);
            const auto after=take(query_session_check(f.session));CHECK(after.e==before.e && after.tokens==before.tokens);
        }
    }
    {
        Fixture f;prepare(f,"secretSTOP</think>\n\nabcSTOPtail");GenerationOverrides o;o.thinking=true;o.stop=std::vector<std::string>{"STOP"};o.logprobs=true;o.top_logprobs=2;
        std::vector<OutputKind> kinds;
        auto result=take(generate_tokens_check(f.session,{1},f.config,o,{},[&](const OutputEvent& e)->exec::Status {kinds.push_back(e.kind);return {};}));
        CHECK(result.result.stop_reason==StopReason::stop_string && result.result.matched_stop=="STOP" && result.result.text=="abc");
        CHECK(result.result.reasoning=="secretSTOP" && result.result.generated==std::vector<std::int32_t>{0});
        CHECK(result.result.e==1 && f.compute.calls==1 && result.host_sampled_steps==1 && result.sampling_readback_bytes==12);
        CHECK(result.result.logprobs.size()==1 && result.result.logprobs[0].section==OutputKind::reasoning && kinds.back()==OutputKind::token_logprob);
    }
    {
        Fixture f;auto& d=f.model->rules->description;d.thinking=true;d.thinking_enabled=true;
        d.sampling_thinking_off=model::SamplingRecommendation{.7,.8,1,20,32768,1.5,0};
        d.sampling_thinking_on=model::SamplingRecommendation{1,.95,1,20,32768,1.5,0};
        auto base=take(resolve_generation_configuration(f.model));CHECK(base.temperature==.7 && base.presence_penalty==1.5 && base.max_tokens==32768);
        GenerationOverrides application;application.thinking=true;application.top_p=.4;
        base=take(resolve_generation_configuration(f.model,application));CHECK(base.temperature==1 && base.top_p==.4);
        GenerationOverrides request;request.thinking=false;request.temperature=0;request.max_tokens=1;
        auto result=take(generate_tokens_check(f.session,{0},base,request)).result;
        CHECK(result.configuration.top_k==20 && result.configuration.top_p==.4 && result.configuration.sources.at("top_p")==ConfigurationSource::application);
        request.temperature.reset();result=take(generate_tokens_check(f.session,{0},base,request)).result;
        CHECK(result.configuration.temperature==.7 && result.configuration.sources.at("temperature")==ConfigurationSource::model);
        const auto description=take(query_model(f.model));CHECK(description.thinking_supported && description.recommended_thinking && description.recommended_thinking->temperature==1);
    }
    for(bool explicit_limit:{false,true}) {
        Fixture f(6);f.config.max_tokens=9;f.config.sources["max_tokens"]=explicit_limit?ConfigurationSource::application:ConfigurationSource::model;
        // A declaration makes the recommendation stable when a request is merged.
        if(!explicit_limit) f.model->rules->description.sampling_thinking_off=model::SamplingRecommendation{0,1,1,0,9,0,0};
        const auto r=take(generate_tokens_check(f.session,{0,0,0},f.config)).result;
        CHECK(r.stop_reason==(explicit_limit?StopReason::capacity:StopReason::length));
        CHECK(r.generated.size()==(explicit_limit?4:3));
    }
    {Fixture f(3);auto r=take(generate_tokens_check(f.session,{0,0,0},f.config)).result;CHECK(r.stop_reason==StopReason::capacity);}
    for(int kind=0;kind<9;++kind) {
        Fixture f;GenerationOverrides o;
        if(kind==0)o.presence_penalty=3;if(kind==1)o.frequency_penalty=std::numeric_limits<double>::quiet_NaN();
        if(kind==2)o.logit_bias=std::map<std::int32_t,double>{{3,1}};
        if(kind==3)o.logit_bias=std::map<std::int32_t,double>{{0,101}};
        if(kind==4)o.top_logprobs=1;
        if(kind==5)o.stop=std::vector<std::string>{""};if(kind==6)o.stop=std::vector<std::string>(17,"x");
        if(kind==7)o.stop=std::vector<std::string>{"\xff"};if(kind==8)o.stop=std::vector<std::string>{"x","x"};
        const auto before=take(query_session_check(f.session));const auto r=generate_tokens_check(f.session,{0},f.config,o);
        CHECK(!r && r.error().category==ErrorCategory::invalid_argument);
        const auto after=take(query_session_check(f.session));CHECK(before.tokens==after.tokens && before.e==after.e);
    }
}
int main() {
    {
        Fixture f;f.compute.tail_failure=true;
        const ReplayInput input{{0},{{0,1,BlockPhase::prefill,true}}};
        const auto result=replay_check(f.session,input);
        CHECK(!result && result.error().message=="tail read failed");
        CHECK(!f.compute.pending && f.compute.finishes.size()==1);
        CHECK(f.compute.finishes.back()==models::FinishReason::normal);
        CHECK(!take(query_session_check(f.session)).valid);
        const auto again=generate_tokens_check(f.session,{0},f.config);
        CHECK(!again && again.error().category==ErrorCategory::session_invalid);
    }
    {
        Fixture f;f.compute.fail_at=0;f.compute.finish_failure=true;
        const ReplayInput input{{0},{{0,1,BlockPhase::prefill,true}}};
        const auto result=replay_check(f.session,input);
        CHECK(!result && result.error().message=="before submit");
        CHECK(f.compute.finishes.size()==1 && f.compute.finishes.back()==models::FinishReason::failure);
        CHECK(!take(query_session_check(f.session)).valid);
        f.compute.finish_failure=false;
    }
    CHECK(app::reuse_method(session::ReuseKind::first)==ReuseMethod::first);
    CHECK(app::reuse_method(session::ReuseKind::keep)==ReuseMethod::keep);
    CHECK(app::reuse_method(session::ReuseKind::truncate)==ReuseMethod::truncate);
    CHECK(app::reuse_method(session::ReuseKind::restore)==ReuseMethod::restore);
    CHECK(app::reuse_method(session::ReuseKind::cold)==ReuseMethod::cold);
    CHECK(app::output_kind(model::ParsedKind::answer)==OutputKind::answer);
    CHECK(app::output_kind(model::ParsedKind::reasoning)==OutputKind::reasoning);
    CHECK(app::output_kind(model::ParsedKind::tool_call_begin)==OutputKind::tool_call_begin);
    CHECK(app::output_kind(model::ParsedKind::tool_call_arguments)==OutputKind::tool_call_arguments);
    CHECK(app::output_kind(model::ParsedKind::tool_call_end)==OutputKind::tool_call_end);
    CHECK(app::output_kind(model::ParsedKind::tool_call_rejected)==OutputKind::tool_call_rejected);
    {
        Fixture f;f.compute.fail_at=1;
        ReplayInput input{{0,1},{{0,1,BlockPhase::prefill,true},{1,1,BlockPhase::decode,true}}};
        auto result=replay_check(f.session,input);
        CHECK(!result && f.compute.finishes.back()==models::FinishReason::failure);
    }
    {
        Fixture f(1024,true);f.model->clock_now=counted_clock;clock_reads=0;
        const auto result=take(generate_tokens_check(f.session,{0,0,0},f.config,{})).result;
        CHECK(result.checkpoint_saved && result.checkpoint_save_seconds==1);
    }

    {
        std::set<std::string> ids;
        for (const auto& entry : app::compiled_models()) {
            CHECK(entry.adapter && entry.compute);
            CHECK(ids.insert(entry.adapter->id()).second);
        }
        model::Catalog catalog; catalog.architecture="unknown";
        auto selected=app::select_compiled(catalog);
        CHECK(!selected && selected.error().category==ErrorCategory::unsupported);
        for (const auto& id : ids) CHECK(selected.error().message.find(id)!=std::string::npos);
        std::vector<model::ConfigurationItem> items{
            {"mode", {"a","b"}, "a", false, "b"}, {"plain", {}, "default"}};
        auto reference=app::apply_reference_values(items,{});
        CHECK(reference.at("mode")=="b" && !reference.count("plain"));
        CHECK(app::apply_reference_values(items,{{"mode","a"}}).at("mode")=="a");
    }
    thinking_tools_tests();
    {
        Fixture f; GenerationOverrides o;o.max_tokens=1;
        auto g=take(generate_tokens_check(f.session,{0,0},f.config,o));
        CHECK(g.random_draws==0 && g.host_sampled_steps==0 && g.sampling_readback_bytes==8 && f.compute.logits_calls==0);
        // ID 0 occurs only in the reused prefix. Full history: .8 < 2; suffix-only: 8 > 2.
        f.compute.row={8,-100,2};o.repetition_penalty=10;
        auto r=take(generate_tokens_check(f.session,{0,0,1},f.config,o));
        CHECK(r.result.reused_tokens==2 && r.result.generated==std::vector<std::int32_t>{2});
        CHECK(r.random_draws==0 && r.host_sampled_steps==1 && r.sampling_readback_bytes==12);
    }
    {
        Fixture f; f.compute.row={8,3,-100};
        GenerationOverrides o;o.repetition_penalty=2;o.max_tokens=1;
        auto r=take(generate_tokens_check(f.session,{0,0,0},f.config,o));
        CHECK(r.result.generated==std::vector<std::int32_t>{0}); // Once: 4 > 3; repeated: 1 < 3.
        take(reset_conversation(f.session));f.compute.row={8,5,-100};o.max_tokens=2;
        r=take(generate_tokens_check(f.session,{1},f.config,o));
        CHECK((r.result.generated==std::vector<std::int32_t>{0,0})); // Accepted ID 0 becomes penalized: 4 > 2.5.
        take(reset_conversation(f.session));f.compute.row={8,-100,5};
        r=take(generate_tokens_check(f.session,{1},f.config,o));
        CHECK((r.result.generated==std::vector<std::int32_t>{0,2})); // Missing generated history would select 0 twice.
    }
    {
        Fixture f;GenerationOverrides o;o.temperature=.8;o.max_tokens=4;
        auto a=take(generate_tokens_check(f.session,{0},f.config,o));
        CHECK(a.result.configuration.seed && a.random_draws==a.result.generated.size());
        CHECK(a.result.configuration.sources.at("seed")==ConfigurationSource::platform);
        o.seed=a.result.configuration.seed;take(reset_conversation(f.session));
        auto b=take(generate_tokens_check(f.session,{0},f.config,o));
        CHECK(a.result.generated==b.result.generated);
        f.compute.row={1,0};take(reset_conversation(f.session));
        auto bad=take(generate_tokens_check(f.session,{0},f.config,o));
        CHECK(bad.result.stop_reason==StopReason::failed && bad.result.error_stage==GenerationStage::sampling && bad.result.state_valid);
        CHECK(bad.random_draws==0);
        f.compute.row={1,std::numeric_limits<float>::quiet_NaN(),0};take(reset_conversation(f.session));
        bad=take(generate_tokens_check(f.session,{0},f.config,o));
        CHECK(bad.result.error_stage==GenerationStage::sampling && bad.result.state_valid && bad.random_draws==0);
    }


    cancellation_tests();
    multi_turn_tests();
    {
        Fixture f(1024,true);f.compute.save_failure=true;
        GenerationOverrides limits;limits.max_tokens=3;
        auto first=take(generate_tokens_check(f.session,{0,0,0},f.config,limits));
        CHECK(first.result.stop_reason==StopReason::length && !first.result.checkpoint_saved && !first.result.checkpoint_boundary);
        auto again=take(generate_tokens_check(f.session,{0,0,0},f.config));
        CHECK(again.result.reuse_method==ReuseMethod::cold && !again.result.checkpoint_saved && again.result.state_valid);
    }

    {
        Fixture f;f.compute.tail_delay=true;
        auto result=f.run();CHECK(result.request_summary.total_seconds>=.05);
        CHECK(!f.compute.pending && f.compute.finishes.back()==models::FinishReason::normal);
        f.run();CHECK(f.compute.resets==1 && !f.compute.pending);
        take(release_session(f.session));CHECK(f.compute.finishes.back()==models::FinishReason::failure);
    }
    {
        Fixture f;f.compute.tail_failure=true;auto result=f.run();
        CHECK(result.stop_reason==StopReason::failed && !result.state_valid && !f.compute.pending);
        CHECK(!take(query_session_check(f.session)).valid);
        auto again=generate_tokens_check(f.session,{0},f.config);
        CHECK(!again && again.error().category==ErrorCategory::session_invalid);
    }
    {
        Fixture f;f.compute.sample_failure=true;auto result=f.run();
        CHECK(result.stop_reason==StopReason::failed && result.error_stage==GenerationStage::sampling);
        CHECK(!f.compute.pending && f.compute.finishes.back()==models::FinishReason::failure);
    }
    {
        Fixture f;f.compute.tail_delay=true;f.compute.fail_at=2;f.compute.submitted=true;
        auto result=f.run();CHECK(result.stop_reason==StopReason::failed);
        CHECK(!f.compute.pending && f.compute.finishes.back()==models::FinishReason::failure);
    }

    {
        LoadConfiguration configuration;configuration.context=8192;
        CHECK(!app::resolve_prefill_chunk(configuration,{}));
        CHECK(configuration.prefill_chunk==4096 && configuration.sources.at("prefill_chunk")==ConfigurationSource::platform);
        CHECK(!app::resolve_prefill_chunk(configuration,3));
        CHECK(configuration.prefill_chunk==3 && configuration.sources.at("prefill_chunk")==ConfigurationSource::request);
        for(auto invalid:{-1,0,8193}) {
            auto error=app::resolve_prefill_chunk(configuration,invalid);
            CHECK(error && error->category==ErrorCategory::invalid_argument);
        }
    }
    {
        Fixture f(8192);f.model->configuration.prefill_chunk=4096;
        auto result=f.run(5000,1);
        CHECK(f.compute.blocks==std::vector<std::int64_t>({4096,903,1}));
        CHECK(result.e==5000 && result.generated_count==1);
    }
    for(bool enabled:{false,true}) {
        Fixture f;f.model->clock_now=counted_clock;clock_reads=0;
        GenerationCheckOptions options;options.breakdown=enabled;
        GenerationOverrides overrides;overrides.max_tokens=64;
        auto result=take(generate_tokens_check(f.session,{0},f.config,overrides,options));
        CHECK(result.decode_timing.size()==(enabled?63:0));
        CHECK(result.prefill_seconds.size()==(enabled?1:0));
        CHECK(clock_reads==(enabled?132:4));
        CHECK(result.result.generated==std::vector<std::int32_t>(64,0));
        for(const auto& timing:result.decode_timing) {
            CHECK(timing.prepare_seconds+timing.wait_seconds+timing.submit_seconds+timing.route_seconds<=timing.total_seconds);
            CHECK(timing.segments==3);
        }
    }
    {
        Fixture f;
        f.model->configuration.prefill_chunk = 4;
        f.model->clock_now = counted_clock; clock_reads = 0;
        CHECK(take(query_generation_progress(f.runtime)).phase == GenerationPhase::idle);
        std::vector<std::size_t> computed, generated;
        std::optional<GenerationProgress> last;
        GenerationCheckOptions options;
        options.cancel_probe = [&](const CancelPoint& point) {
            if (point.kind == CancelPointKind::before_block) {
                const auto progress = take(query_generation_progress(f.runtime));
                CHECK(progress.phase == GenerationPhase::prefill && progress.input_tokens == 13);
                CHECK(progress.pending_tokens == 13 - progress.computed_tokens && !progress.first_token);
                computed.push_back(progress.computed_tokens);
            }
            return false;
        };
        GenerationOverrides overrides; overrides.max_tokens = 4;
        auto value = take(generate_tokens_check(f.session, std::vector<std::int32_t>(13,0), f.config, overrides, options,
            [&](const OutputEvent&) -> exec::Status {
                last = take(query_generation_progress(f.runtime));
                CHECK(last->phase == GenerationPhase::decode && last->first_token);
                if (generated.empty() || generated.back() != last->generated_tokens) generated.push_back(last->generated_tokens);
                return {};
            }));
        CHECK(computed == std::vector<std::size_t>({4,8,12,13}));
        CHECK(generated == std::vector<std::size_t>({1,2,3,4}));
        CHECK(last && std::chrono::duration<double>(*last->first_token - last->started).count() == *value.result.request_summary.first_token_seconds);
        CHECK(clock_reads == 4 && take(query_generation_progress(f.runtime)).phase == GenerationPhase::idle);
    }
    for (int n : {4,64}) {
        Fixture f;
        f.model->clock_now = counted_clock; clock_reads = 0;
        std::atomic<bool> done{false};
        std::atomic<std::size_t> polls{0};
        std::thread observer([&] {
            while (!done) { take(query_generation_progress(f.runtime)); ++polls; }
        });
        while (!polls) std::this_thread::yield();
        auto value = f.run(1,n);
        done = true; observer.join();
        CHECK(clock_reads == 4 && value.generated == std::vector<std::int32_t>(n,0));
        CHECK(take(query_generation_progress(f.runtime)).phase == GenerationPhase::idle);
    }
    for(int n:{4,64}) {
        Fixture f;f.model->clock_now=counted_clock;clock_reads=0;
        auto r=f.run(1,n);CHECK(clock_reads==4);exact_window(r);
        CHECK(*r.request_summary.first_token_seconds==1 && *r.request_summary.decode_seconds==1);
        CHECK(r.request_summary.total_seconds==3 && clock_reads==4);
    }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;
      auto r=f.run(1,1);missing_window(r,"fewer_than_two_tokens");CHECK(clock_reads==4); }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;f.compute.fail_at=0;
      auto r=f.run(1,4);CHECK(!r.request_summary.first_token_seconds && !r.request_summary.first_token_reason.empty());
      missing_window(r,"ended_by_failure");CHECK(clock_reads==2); }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;f.compute.end_at=2;
      auto r=f.run(1,9);exact_window(r);CHECK(r.generated.size()==3 && r.generated_count==2);CHECK(clock_reads==4); }
    { Fixture f(4);f.model->clock_now=counted_clock;clock_reads=0;
      auto r=f.run(1,9);exact_window(r);CHECK(r.stop_reason==StopReason::capacity && clock_reads==4); }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;f.compute.fail_at=2;
      auto r=f.run(1,9);CHECK(r.generated.size()==2 && r.request_summary.first_token_seconds);
      missing_window(r,"ended_by_failure");CHECK(clock_reads==3); }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;int received=0;
      auto r=f.run(1,9,[&](const OutputEvent&)->std::optional<Error> {
          if(++received==2) return Error{ErrorCategory::invalid_argument,"receiver stopped"};return {};
      });CHECK(r.generated.size()==2 && r.request_summary.first_token_seconds);
      missing_window(r,"ended_by_receiver");CHECK(clock_reads==3); }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;int received=0;
      auto r=f.run(1,9,[&](const OutputEvent&)->std::optional<Error> {
          if(++received==2) take(release_session(f.session));return {};
      });CHECK(r.generated.size()==2);missing_window(r,"cancelled");CHECK(clock_reads==3); }
    { Fixture f;f.model->clock_now=counted_clock;clock_reads=0;int received=0;
      auto r=f.run(1,2,[&](const OutputEvent&)->std::optional<Error> {
          if(++received==2) return Error{ErrorCategory::invalid_argument,"last delivery failed"};return {};
      });missing_window(r,"ended_by_receiver");CHECK(clock_reads==4); }
    for(std::size_t marker:{1,4,5}) {
        Fixture f;f.model->clock_now=counted_clock;clock_reads=0;
        GenerationCheckOptions options;options.timing_marker=marker;
        GenerationOverrides overrides;overrides.max_tokens=4;
        auto r=take(generate_tokens_check(f.session,{0},f.config,overrides,options));
        exact_window(r.result);CHECK(clock_reads==(marker<=4?5:4));
        CHECK(bool(r.marker_seconds)==(marker<=4));
        if(r.marker_seconds) {
            CHECK(*r.marker_seconds>=*r.result.request_summary.first_token_seconds);
            CHECK(*r.marker_seconds<=*r.result.request_summary.first_token_seconds+*r.result.request_summary.decode_seconds);
        }
    }
    for(bool ignore:{false,true}) {
        Fixture f;f.compute.end_at=2;f.model->clock_now=counted_clock;clock_reads=0;
        GenerationCheckOptions options;options.ignore_end_of_generation=ignore;options.timing_marker=2;
        GenerationOverrides overrides;overrides.max_tokens=8;
        auto r=take(generate_tokens_check(f.session,{0},f.config,overrides,options));
        CHECK(r.end_token_count==1 && r.first_end_token_position==3);
        CHECK(r.result.generated.size()==(ignore?8:3));CHECK(r.result.generated_count==(ignore?7:2));
        CHECK(r.result.stop_reason==(ignore?StopReason::length:StopReason::end_of_generation));
        exact_window(r.result);CHECK(clock_reads==5 && r.marker_seconds);
        CHECK(*r.marker_seconds>*r.result.request_summary.first_token_seconds);
        CHECK(*r.marker_seconds<*r.result.request_summary.first_token_seconds+*r.result.request_summary.decode_seconds);
    }

#ifndef _WIN32
    // The read counter interposes fread via dlsym(RTLD_NEXT), which is POSIX-only
    // (Windows provides a no-op stub). Skip the mechanism check there; the
    // query_model "reads nothing" check below still holds (stub returns 0).
    {
        FILE* file=std::tmpfile();CHECK(file);
        CHECK(std::fputs("probe",file)>=0);std::rewind(file);
        char bytes[5]{};file_read_counter_start();
        CHECK(std::fread(bytes,1,5,file)==5);
        CHECK(file_read_counter_stop()>=1);std::fclose(file);
    }
#endif
    {
        Fixture f;
        auto resources=take(RuntimeAccess::acquire(f.runtime));
        resources.registry->memory_query=fake_memory;
        memory_calls=0;
        auto before=take(memory_snapshot(f.runtime));
        CHECK(memory_calls==1);
        CHECK(before.ledger.limit_bytes==resources.budget.limit());
        CHECK(before.ledger.current_bytes==resources.budget.current());
        CHECK(before.ledger.peak_bytes==resources.budget.peak());
        CHECK(before.ledger.scope==MemoryScope::runtime_lifetime);
        CHECK(before.rss.scope==MemoryScope::process_lifetime);
        CHECK(before.physical.scope==MemoryScope::process_lifetime);
        CHECK(before.rss.values->current_bytes==11 && before.rss.values->peak_bytes==22);
        CHECK(before.physical.values->current_bytes==33 && before.physical.values->peak_bytes==44);
        const auto result=f.run(); CHECK(!result.weight_activity.applicable);
        CHECK(memory_calls==1);
        resources.registry->memory_query=failed_memory;
        auto failed=take(memory_snapshot(f.runtime));
        CHECK(memory_calls==2);
        CHECK(!failed.rss.values && failed.rss.reason=="injected RSS failure");
        CHECK(!failed.physical.values && failed.physical.reason=="injected physical failure");
        CHECK(failed.ledger.current_bytes==resources.budget.current());
        CHECK(failed.ledger.peak_bytes==resources.budget.peak());
    }
    {
        Fixture f;
        f.model->loaded->assets.definition.context_length=8192;
        f.model->configuration.directory="nonexistent-query-fixture";
        f.model->configuration.model_options={{"example","on"}};
        f.model->configuration.sources={{"device",ConfigurationSource::platform},{"context",ConfigurationSource::request},{"example",ConfigurationSource::model}};
        file_read_counter_start();
        auto result=query_model(f.model);
        CHECK(file_read_counter_stop()==0);
        auto description=take(result);
        CHECK(description.context==1024 && description.context_limit==8192);
        CHECK(description.directory=="nonexistent-query-fixture");
        CHECK(description.loading.sources==f.model->configuration.sources);
        CHECK(description.loading.model_options==f.model->configuration.model_options);
        CHECK(description.recommended.temperature==1 && description.recommended.top_k==20 && description.recommended.top_p==.95);
        CHECK(description.adapter=="fake" && !description.thinking_supported);
        CHECK(!description.thinking_default && !description.thinking_enabled_supported && !description.tools_supported);
        take(release_session(f.session)); take(unload_model(f.model));
        auto reclaimed=query_model(f.model);
        CHECK(!reclaimed && reclaimed.error().category==ErrorCategory::invalid_argument);
        CHECK(!query_model({}));
    }

    {
        Fixture f;
        weights::PoolDescription pool;
        pool.slots=1;pool.experts.resize(1);
        for(auto& layout:pool.projections) layout={GGML_TYPE_F32,{4}};
        auto& loaded=*f.model->loaded;
        loaded.pools.push_back(take(weights::ExpertPool::create_pools(loaded.weights,loaded.stages.prefill,pool)));
        auto r=f.run();
        CHECK(r.weight_activity.applicable);
        CHECK(r.weight_activity.decode.demand==std::size_t(f.compute.calls)*2);
    }
    { Fixture f; auto r=f.run(); CHECK(r.weight_activity.decode.demand==std::size_t(f.compute.calls)*2); CHECK(r.weight_activity.prefill.bytes==std::size_t(f.compute.calls)*192); }
    for(bool submitted:{false,true}) {
        Fixture f; f.compute.fail_at=0; f.compute.fail_after_segment=true; f.compute.submitted=submitted;
        auto r=f.run(); CHECK(r.state_valid==!submitted); CHECK(r.weight_activity.decode.bytes==128);
        auto next=generate_tokens_check(f.session,{0},f.config);
        if(submitted) {
            CHECK(!next && next.error().category==ErrorCategory::session_invalid);
            auto reset=reset_conversation(f.session);CHECK(!reset && reset.error().category==ErrorCategory::session_invalid);
        } else CHECK(next);
    }
    for(const auto& item:std::vector<std::pair<std::size_t,std::vector<std::int64_t>>>{{17,{16,1}},{600,{512,87,1}},{1,{1}}}) {
        Fixture f;auto r=f.run(item.first,1);CHECK(f.compute.blocks==item.second);CHECK(r.e==std::int64_t(item.first)&&r.generated_count==1);
    }
    {Fixture f;f.compute.whole=false;f.run(17,1);CHECK(f.compute.blocks==std::vector<std::int64_t>{17});}
    {Fixture f(6);auto r=f.run(3,16);CHECK(r.stop_reason==StopReason::capacity&&r.generated_count==4&&r.e==6);}
    for(int end:{1,3}) {
        Fixture f;f.compute.end_at=end;auto r=f.run(3,3);
        CHECK(r.stop_reason==StopReason::end_of_generation&&r.generated_count==std::size_t(end-1));
        CHECK(r.tail_is_end&&r.generated.back()==2&&r.text==std::string(std::size_t(end-1),'a'));
        CHECK(f.compute.calls==end+1);
    }
    for(float sum:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        Fixture f;f.compute.choice.sum=sum;auto r=f.run();CHECK(r.stop_reason==StopReason::failed&&r.error_stage==GenerationStage::sampling&&r.state_valid);
    }
    {Fixture f;f.compute.choice.token=3;auto r=f.run();CHECK(r.error_stage==GenerationStage::sampling&&r.state_valid);}
    {Fixture f;auto r=f.run(3,3,[](const OutputEvent&)->exec::Status{return Error{ErrorCategory::invalid_argument,"delivery"};});CHECK(r.error_stage==GenerationStage::delivery&&r.state_valid&&r.text.empty());}
    {
        Fixture f;f.compute.fail_at=1;
        auto r=f.run(17,3);CHECK(r.state_valid&&r.e==16&&!r.input_complete);
        f.compute.fail_at=-1;auto next=f.run(17,1);
        CHECK(next.stop_reason==StopReason::length&&!next.reset_healthy_cache&&next.reused_tokens==16&&f.compute.resets==0);
    }
    for(bool submitted:{false,true}) {
        Fixture f;f.compute.fail_at=0;f.compute.submitted=submitted;auto r=f.run();
        CHECK(r.stop_reason==StopReason::failed&&r.state_valid==!submitted&&r.e==0);
        f.compute.fail_at=-1;
        auto next=generate_tokens_check(f.session,{0},f.config);
        if(submitted)CHECK(!next&&next.error().category==ErrorCategory::session_invalid);else CHECK(next);
    }
    {
        Fixture f(4);auto before=take(query_session_check(f.session));
        auto rejected=generate_tokens_check(f.session,{0,0,0,0,0},f.config);
        CHECK(!rejected&&rejected.error().category==ErrorCategory::capacity_exceeded);
        CHECK(take(query_session_check(f.session)).e==before.e);
        auto recommended=take(resolve_generation_configuration(f.model));
        CHECK(recommended.temperature==1&&recommended.top_k==20&&recommended.top_p==.95);
        CHECK(recommended.sources.at("temperature")==ConfigurationSource::model&&recommended.sources.at("max_tokens")==ConfigurationSource::platform);
        GenerationOverrides application;application.top_p=.7;auto c=take(resolve_generation_configuration(f.model,application));
        CHECK(c.top_p==.7&&c.sources.at("top_p")==ConfigurationSource::application);
        std::vector<GenerationOverrides> bad(9);
        bad[0].temperature=-1;bad[1].temperature=std::numeric_limits<double>::quiet_NaN();bad[2].temperature=std::numeric_limits<double>::infinity();
        bad[3].top_p=0;bad[4].top_p=1.5;bad[5].top_k=-1;bad[6].repetition_penalty=0;bad[7].max_tokens=0;bad[8].max_tokens=-1;
        auto resources=take(RuntimeAccess::acquire(f.runtime));const auto occupied=resources.budget.current();
        for(const auto& invalid:bad) {
            auto r=generate_tokens_check(f.session,{0},f.config,invalid);
            CHECK(!r&&r.error().category==ErrorCategory::invalid_argument);
            CHECK(f.session->state->query().e==0&&f.session->state->query().tokens.empty()&&resources.budget.current()==occupied);
        }
        for(int kind:{0,1,2}) {
            GenerationOverrides o;if(kind==0)o.temperature=.7;if(kind==1)o.repetition_penalty=1.1;if(kind==2)o.thinking=true;
            o.max_tokens=1;
            auto r=generate_tokens_check(f.session,{0},f.config,o);
            if(kind==2) CHECK(!r&&r.error().category==ErrorCategory::unsupported);
            else {CHECK(r && r.value().result.generated.size()==1);take(reset_conversation(f.session));}
        }
    }
    {
        Fixture f;int callbacks=0;
        auto r=f.run(3,3,[&](const OutputEvent&)->exec::Status {
            ++callbacks;
            auto formal=generate(f.session,std::vector<TextMessage>{},f.config);CHECK(!formal&&formal.error().category==ErrorCategory::busy);
            auto nested=generate_tokens_check(f.session,{0},f.config);CHECK(!nested&&nested.error().category==ErrorCategory::busy);
            auto load=load_model(f.runtime,{});CHECK(!load&&load.error().category==ErrorCategory::busy);
            auto session=create_session(f.model);CHECK(!session&&session.error().category==ErrorCategory::busy);
            auto end=finish_runtime_use(f.runtime);CHECK(!end&&end.error().category==ErrorCategory::busy);
            auto config=resolve_generation_configuration(f.model);CHECK(!config&&config.error().category==ErrorCategory::busy);
            return {};
        });CHECK(r.stop_reason==StopReason::length&&callbacks==3);
    }
    for(bool unload:{false,true}) {
        Fixture f;auto resources=take(RuntimeAccess::acquire(f.runtime));const auto occupied=resources.budget.current();
        auto r=f.run(3,16,[&](const OutputEvent&)->exec::Status {
            if(unload)take(unload_model(f.model));else take(release_session(f.session));
            CHECK(resources.budget.current()==occupied);return {};
        });CHECK(r.stop_reason==StopReason::cancelled);
        if(unload) {
            auto next=generate_tokens_check(f.session,{0},f.config);CHECK(!next&&next.error().category==ErrorCategory::unloading);
            CHECK(resources.budget.current()==occupied);take(release_session(f.session));
        } else {auto next=generate_tokens_check(f.session,{0},f.config);CHECK(!next&&next.error().category==ErrorCategory::invalid_argument);}
        CHECK(resources.budget.current()==(unload?0:f.model_bytes));
        if(!unload)take(unload_model(f.model));
        CHECK(resources.budget.current()==0);
        auto released=generate_tokens_check(f.session,{0},f.config);
        CHECK(!released&&released.error().category==ErrorCategory::invalid_argument);
        auto closed=create_session(f.model);CHECK(!closed&&closed.error().category==ErrorCategory::invalid_argument);
    }
    {Fixture f;auto r=f.run(3,16,[&](const OutputEvent&)->exec::Status {std::thread t([&]{take(release_session(f.session));});t.join();return {};});CHECK(r.stop_reason==StopReason::cancelled&&f.session->state->query().released);}
    {
        Fixture f;f.run();auto before=f.session->state->query();
        auto resources=take(RuntimeAccess::acquire(f.runtime));const auto occupied=resources.budget.current();
        GenerationOverrides bad;bad.top_p=0;
        CHECK(!generate_tokens_check(f.session,{0},f.config,bad));
        auto after=f.session->state->query();
        CHECK(after.e==before.e&&after.tokens==before.tokens&&resources.budget.current()==occupied&&f.compute.resets==0);
    }
    std::cout<<"generation boundaries, configuration, failures, callback activity and threaded release passed\n";
}
