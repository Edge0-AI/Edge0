#include "edge0/diagnostics.h"

#include "app/generation_internal.h"
#include "app/runtime_hooks.h"
#include "models/edge0_35b/compute.h"
#include "tests/support/runtime.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
using namespace edge0;
using test::take;
std::vector<GenerationCheckResult> run(const std::string& directory, int mode) {
    auto runtime=take(open_runtime({{"CPU"},{}}));
    auto hooks=take(app::runtime_hooks(runtime));
    if(mode) CHECK(!app::add_backend_alias(runtime,"CPU-prefill","CPU"));
    bool fail_transfer=false;
    std::map<ggml_tensor*,std::size_t> range_bytes,full_bytes;
    if(mode==2) {
        hooks->consumption=[](auto,auto) { return exec::Consumption::copy; };
        hooks->transfer=[&](ggml_tensor* source,ggml_tensor* target,const std::vector<exec::TransferRange>& ranges)->exec::Status {
            if(fail_transfer) return Error{ErrorCategory::backend_error,"Injected handoff failure"};
            if(ranges.empty()) {full_bytes[target]+=ggml_nbytes(source);ggml_backend_tensor_copy(source,target);}
            else for(const auto& range:ranges) if(range.bytes) {
                range_bytes[target]+=range.bytes;
                std::vector<unsigned char> bytes(range.bytes);
                ggml_backend_tensor_get(source,bytes.data(),range.offset,bytes.size());
                ggml_backend_tensor_set(target,bytes.data(),range.offset,bytes.size());
            }
            return {};
        };
    }
    LoadOverrides loading; loading.device="CPU";
    if(mode) loading.prefill_device="CPU-prefill";
    loading.context=128; loading.prefill_chunk=128;
    loading.model_options={{"routing","actual"},{"prefill_ahead_min_tokens","1"}};
    auto model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory,loading))));
    auto session=take(create_session(model));
    GenerationOverrides application; application.temperature=0;application.presence_penalty=0;application.max_tokens=4;
    auto config=take(resolve_generation_configuration(model,application));
    GenerationCheckOptions checking;checking.retain_logits=true;checking.ignore_end_of_generation=true;
    std::vector<GenerationCheckResult> results;
    std::vector<std::int32_t> input{1,2,3,4,5};
    for(int i=0;i<3;++i) {
        range_bytes.clear();full_bytes.clear();
        auto result=take(generate_tokens_check(session,input,config,{},checking));
        if(result.result.error) std::cerr<<result.result.error->message<<'\n';
        CHECK(result.result.stop_reason==StopReason::length);
        CHECK(result.result.generated.size()==4);
        if(mode==2 && i==1) {
            for(auto layer:model->loaded->assets.definition.layer_groups.at("deltanet")) {
                const auto delta=take(session->state->states().get<models::edge0_35b::DeltaState>("deltanet."+std::to_string(layer)));
                for(const auto& state:{delta->convolution,delta->recurrent}) for(const auto& backend:{model->loaded->stages.prefill,model->loaded->stages.decode}) {
                    auto* target=take(state->on(backend))->tensor();
                    CHECK(full_bytes[target]==ggml_nbytes(target));
                }
            }
            const auto states=take(model->compute->describe_states(model->loaded->assets.definition));
            for(const auto& entry:states.kv) {
                const auto storage=take(session->state->kv().storage(entry.name));
                auto* target=take(storage->on(model->loaded->stages.prefill))->tensor();
                const auto row_bytes=ggml_nbytes(target)/static_cast<std::size_t>(target->ne[entry.token_axis]);
                CHECK(range_bytes[target]==3*row_bytes);
            }
        }
        const auto ledger = take(memory_snapshot(runtime)).ledger;
        std::size_t by_purpose = 0, by_device = 0;
        for (const auto& entry : ledger.by_purpose) by_purpose += entry.second;
        for (const auto& entry : ledger.by_device) by_device += entry.second.current_bytes;
        CHECK(by_purpose == ledger.current_bytes && by_device == ledger.current_bytes);
        for (auto purpose : {MemoryPurpose::resident_weights, MemoryPurpose::experts, MemoryPurpose::session_state})
            CHECK(ledger.by_purpose.at(purpose) > 0);
        CHECK(ledger.by_device.count("CPU") == 1);
        CHECK(ledger.by_device.at("CPU").current_bytes > 0);
        CHECK(ledger.by_device.at("CPU").memory == MemoryLocation::system);
        if (mode == 2) {
            const auto description = take(model->compute->describe_states(model->loaded->assets.definition));
            for (const auto& entry : description.kv) {
                const auto state = take(session->state->kv().storage(entry.name));
                for (const auto& buffer : state->replicas->buffers) {
                    CHECK(buffer->charges.front()->purpose() == exec::Purpose::session_state);
                    CHECK(exec::buffer_device(ggml_backend_buffer_get_type(buffer->buffer)) == "CPU");
                    CHECK(ledger.by_device.at("CPU").current_bytes >= buffer->charges.front()->size());
                }
            }
        }
        results.push_back(result);
        input=take(query_session_check(session)).tokens; input.push_back(6+i);input.push_back(9+i);
    }
    auto append=[&](const std::vector<std::int32_t>& tokens,const GenerationCheckOptions& options) {
        auto result=take(generate_tokens_check(session,tokens,config,{},options));
        CHECK(!result.result.error);results.push_back(result);return result;
    };
    if(mode==2) {
        const auto before=take(query_session_check(session));
        fail_transfer=true;
        auto failed=take(generate_tokens_check(session,input,config,{},checking));
        CHECK(failed.result.error && failed.result.error->category==ErrorCategory::backend_error);
        const auto after=take(query_session_check(session));
        CHECK(after.valid && after.e==before.e && after.tokens==before.tokens);
        fail_transfer=false;
        hooks->skip_handoff=true;
        failed=take(generate_tokens_check(session,input,config,{},checking));
        CHECK(failed.result.error && failed.result.error->category==ErrorCategory::invalid_argument);
        CHECK(take(query_session_check(session)).valid);
        hooks->skip_handoff=false;
    }
    append(input,checking);
    append(input,checking);
    auto cancelled=checking;
    cancelled.cancel_probe=[](const CancelPoint& point){return point.kind==CancelPointKind::before_step && point.token==2;};
    input=take(query_session_check(session)).tokens;input.push_back(11);input.push_back(12);
    const auto interrupted=append(input,cancelled);
    CHECK(interrupted.result.stop_reason==StopReason::cancelled);
    append(input,checking);
    input=take(query_session_check(session)).tokens;
    input.insert(input.end(),{15,16,17,18,19});
    auto block_cancel=checking;
    block_cancel.cancel_probe=[](const CancelPoint& point){return point.kind==CancelPointKind::in_block && point.segment==2;};
    const auto rolled_back=append(input,block_cancel);
    CHECK(rolled_back.result.stop_reason==StopReason::cancelled && rolled_back.result.rolled_back_tokens>0);
    append(input,checking);
    take(reset_conversation(session));
    const auto reset_result=append({1,2,3,4,5},checking);
    CHECK(!reset_result.result.text.empty());
    GenerationOverrides stopping;stopping.stop=std::vector<std::string>{reset_result.result.text.substr(0,1)};
    take(reset_conversation(session));
    auto stopped=take(generate_tokens_check(session,{1,2,3,4,5},config,stopping,checking));
    CHECK(stopped.result.stop_reason==StopReason::stop_string);
    CHECK(stopped.result.generated.size()==1 && stopped.result.request_summary.handoffs==0);
    results.push_back(stopped);
    input=take(query_session_check(session)).tokens;input.push_back(13);input.push_back(14);
    append(input,checking);
    if(!mode) {
        CHECK(!model->loaded->stages.runs && model->loaded->stages.prefill==model->loaded->stages.decode);
        CHECK(model->loaded->stages.prefill == model->loaded->stages.decode);
        CHECK(hooks->hand_off_calls==0);
        for(const auto& result:results) CHECK(result.result.request_summary.handoffs==0);
    }
    if(mode==1) for(const auto& result:results) CHECK(result.result.request_summary.handoff_bytes==0);
    if(mode==2) for(const auto& result:results) {
        const auto& summary=result.result.request_summary;
        if(result.result.stop_reason==StopReason::stop_string) {
            CHECK(summary.handoffs==0 && summary.handoff_bytes==0 && summary.handoff_seconds==0);
        } else {
            CHECK(summary.handoffs>0 && summary.handoff_bytes>0 && summary.handoff_seconds>0 && summary.handoff_seconds<=summary.total_seconds);
        }
    }
    take(release_session(session));take(unload_model(model));
    session.reset();model.reset();
    CHECK(take(memory_snapshot(runtime)).ledger.current_bytes==0);
    take(finish_runtime_use(runtime));
    return results;
}
void policy_replay(const std::string& directory, bool copy) {
    auto runtime=take(open_runtime({{"CPU"},{}}));
    auto hooks=take(app::runtime_hooks(runtime));
    CHECK(!app::add_backend_alias(runtime,"CPU-prefill","CPU"));
    if(copy)hooks->consumption=[](auto,auto){return exec::Consumption::copy;};
    std::size_t choices=0,candidate_calls=0;
    hooks->policies["threshold"]={
        [&candidate_calls](const exec::Stages& s){++candidate_calls;return std::vector<std::shared_ptr<exec::Backend>>{s.prefill,s.decode};},
        [&choices](const exec::Stages& s,const exec::StageFacts& f){++choices;return f.tokens<4?s.decode:s.prefill;}};
    LoadOverrides load;load.device="CPU";load.prefill_device="CPU-prefill";load.stage_policy="threshold";
    load.context=128;load.prefill_chunk=128;load.model_options={{"routing","actual"},{"prefill_ahead_min_tokens","1"}};
    const auto defaults=take(resolve_load_configuration(runtime,directory));
    CHECK(defaults.prefill_device==defaults.device && defaults.stage_policy=="fixed");
    CHECK(defaults.sources.at("prefill_device")==ConfigurationSource::platform && defaults.sources.at("stage_policy")==ConfigurationSource::platform);
    const auto configured=take(resolve_load_configuration(runtime,directory,load));
    CHECK(configured.sources.at("prefill_device")==ConfigurationSource::request && configured.sources.at("stage_policy")==ConfigurationSource::request);
    auto invalid_load=load;invalid_load.prefill_device="outside";
    auto invalid_config=resolve_load_configuration(runtime,directory,invalid_load);
    CHECK(!invalid_config && invalid_config.error().category==ErrorCategory::unsupported);
    invalid_load=load;invalid_load.stage_policy="missing";
    invalid_config=resolve_load_configuration(runtime,directory,invalid_load);
    CHECK(!invalid_config && invalid_config.error().category==ErrorCategory::invalid_argument);
    auto model=take(load_model(runtime,configured));
    CHECK(candidate_calls==1);
    CHECK(take(RuntimeAccess::acquire(runtime)).budget.current()==model->loaded->estimate.total_bytes);
    for(bool next:{false,true})
        CHECK((model->loaded->temporary_on(model->loaded->stages.prefill,next)==model->loaded->temporary_on(model->loaded->stages.decode,next))==!copy);
    auto session=take(create_session(model));
    GenerationOverrides generation;generation.temperature=0;generation.presence_penalty=0;generation.max_tokens=4;
    auto config=take(resolve_generation_configuration(model,generation));
    GenerationCheckOptions check;check.retain_logits=true;check.ignore_end_of_generation=true;
    const auto check_zero_replicas=[&]() {
        for(auto layer:model->loaded->assets.definition.layer_groups.at("deltanet")) {
            const auto delta=take(session->state->states().get<models::edge0_35b::DeltaState>("deltanet."+std::to_string(layer)));
            for(const auto& storage:{delta->convolution,delta->recurrent})
                for(const auto& backend:{model->loaded->stages.prefill,model->loaded->stages.decode}) {
                    const auto view=take(storage->on(backend));CHECK(view->current());
                    std::vector<unsigned char> bytes(ggml_nbytes(view->tensor()));
                    ggml_backend_tensor_get(view->tensor(),bytes.data(),0,bytes.size());
                    CHECK(std::all_of(bytes.begin(),bytes.end(),[](auto byte){return byte==0;}));
                }
        }
    };
    check_zero_replicas();
    bool fresh=true;
    for(const auto& input:std::vector<std::vector<std::int32_t>>{{1,2,3},{1,2,3,4,5},{1,2,3}}) {
        if(!fresh)take(reset_conversation(session));
        fresh=false;
        check_zero_replicas();
        const auto original=take(generate_tokens_check(session,input,config,{},check));
        CHECK(original.result.stop_reason==StopReason::length);
        const std::string selected=input.size()<4?"CPU":"CPU-prefill";
        CHECK(original.result.request_summary.prefill_device==selected);
        if(input.size()<4) {
            CHECK(original.result.request_summary.handoffs==0);
            CHECK(original.result.request_summary.handoff_bytes==0);
            CHECK(original.result.request_summary.handoff_seconds==0);
        }
        for(const auto& block:original.replay.blocks) CHECK(block.device==(block.phase==BlockPhase::prefill?selected:"CPU"));
        const auto calls=choices;
        take(reset_conversation(session));
        auto invalid=original.replay;invalid.blocks.front().device="outside";
        auto rejected=replay_check(session,invalid);
        CHECK(!rejected && rejected.error().category==ErrorCategory::invalid_argument);
        const auto replayed=take(replay_check(session,original.replay));
        CHECK(choices==calls && candidate_calls==1);
        // Replay exposes every block; generation exposes sampled blocks only.
        std::vector<float> sampled;
        for(std::size_t i=0;i<original.replay.blocks.size();++i) if(original.replay.blocks[i].sample) {
            const auto start=replayed.logits.begin()+i*original.vocabulary;
            sampled.insert(sampled.end(),start,start+original.vocabulary);
        }
        CHECK(sampled.size()==original.logits.size());
        CHECK(std::memcmp(sampled.data(),original.logits.data(),original.logits.size()*sizeof(float))==0);
    }
    take(release_session(session));take(unload_model(model));session.reset();model.reset();take(finish_runtime_use(runtime));
}
int main() {
    const auto* tiny=std::getenv("MOE_TINY");if(!tiny)return 77;
    const std::string directory=std::string(tiny)+"/tiny-a";
    for(bool copy:{false,true})policy_replay(directory,copy);
    const auto baseline=run(directory,0);
    for(int mode:{1,2}) {
        const auto mixed=run(directory,mode);
        CHECK(mixed.size()==baseline.size());
        for(std::size_t i=0;i<mixed.size();++i) {
            CHECK(mixed[i].result.generated==baseline[i].result.generated);
            CHECK(mixed[i].sampling_readback_bytes==baseline[i].sampling_readback_bytes);
            CHECK(mixed[i].logits.size()==baseline[i].logits.size());
            CHECK(std::memcmp(mixed[i].logits.data(),baseline[i].logits.data(),mixed[i].logits.size()*sizeof(float))==0);
        }
    }
    std::cout<<"mixed stages: single and dual-instance shared/copy continuation matched bitwise\n";
}
