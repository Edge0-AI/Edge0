#include "app/generation_internal.h"
#include "tests/support/runtime.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
using namespace edge0;
using test::take;
int main() {
    const auto* reference=std::getenv("MOE_TINY");if(!reference)return 77;
    auto runtime=take(open_runtime({{"CPU"},{}}));
    const std::string directory=std::string(reference)+"/tiny-a";
    for(std::int64_t capacity:{std::int64_t(0),std::numeric_limits<std::int64_t>::max()}) {
        LoadOverrides invalid;invalid.context=capacity;
        auto result=resolve_load_configuration(runtime,directory,invalid);
        CHECK(!result&&result.error().category==ErrorCategory::invalid_argument);
    }
    auto defaults=take(resolve_load_configuration(runtime,directory));
    CHECK(defaults.context==16384 && defaults.sources.at("context")==ConfigurationSource::model);
    CHECK(defaults.prefill_chunk==4096 && defaults.sources.at("prefill_chunk")==ConfigurationSource::platform);
    CHECK(defaults.model_options.at("prefill_read")=="auto" && defaults.sources.at("prefill_read")==ConfigurationSource::model);
    CHECK(defaults.model_options.at("prefill_ahead_min_tokens")=="3000" && defaults.sources.at("prefill_ahead_min_tokens")==ConfigurationSource::model);
    for(std::int64_t chunk:{0,16385}) {
        LoadOverrides invalid;invalid.prefill_chunk=chunk;
        auto result=resolve_load_configuration(runtime,directory,invalid);
        CHECK(!result && result.error().category==ErrorCategory::invalid_argument);
    }
    LoadOverrides short_context;short_context.context=64;
    CHECK(!resolve_load_configuration(runtime,directory,short_context));
    short_context.prefill_chunk=3;
    CHECK(!resolve_load_configuration(runtime,directory,short_context));
    short_context.model_options["prefill_ahead_min_tokens"]="64";
    auto explicit_chunk=take(resolve_load_configuration(runtime,directory,short_context));
    CHECK(explicit_chunk.model_options.at("prefill_ahead_min_tokens")=="64" && explicit_chunk.sources.at("prefill_ahead_min_tokens")==ConfigurationSource::request);
    CHECK(explicit_chunk.prefill_chunk==3 && explicit_chunk.sources.at("prefill_chunk")==ConfigurationSource::request);
    LoadOverrides unknown;unknown.model_options={{"unknown","on"}};
    CHECK(!resolve_load_configuration(runtime,directory,unknown));
    for(bool lora:{false,true}) {
        LoadOverrides load;load.context=64;load.prefill_chunk=64;load.model_options={{"routing","actual"},{"lora",lora?"on":"off"},{"prefill_ahead_min_tokens","1"}};
        auto configuration=take(resolve_load_configuration(runtime,directory,load));
        CHECK(configuration.sources.at("device")==ConfigurationSource::platform);
        auto model=take(load_model(runtime,configuration));auto session=take(create_session(model));
        GenerationOverrides application;application.temperature=0;application.presence_penalty=0;
        auto config=take(resolve_generation_configuration(model,application));
        auto x=take(render_prompt(directory,std::vector<edge0::TextMessage>{{"user","hello"}})).tokens;
        auto y=take(render_prompt(directory,std::vector<edge0::TextMessage>{{"user","hello world"}})).tokens;
        CHECK(x!=y);
        GenerationOverrides request;request.max_tokens=16;
        std::string delivered;
        auto a=take(generate_tokens_check(session,x,config,request,{false,true},[&](const OutputEvent& s)->exec::Status {if(s.kind==OutputKind::answer)delivered+=s.text;return {};}));
        CHECK(a.result.stop_reason==StopReason::length&&a.result.generated_count==16);
        CHECK(a.result.e==std::int64_t(x.size()+15)&&!a.result.reset_healthy_cache&&a.result.cold);
        auto state=take(query_session_check(session));
        auto expected=x;expected.insert(expected.end(),a.result.generated.begin(),a.result.generated.end()-1);
        CHECK(state.tokens==expected&&state.e==std::int64_t(expected.size()));
        CHECK(delivered==take(detokenize(directory,a.result.generated)).text&&delivered==a.result.text);
        const auto states=take(model->compute->describe_states(model->loaded->assets.definition));
        auto storage=take(session->state->kv().storage(states.kv.front().name));
        auto* tensor=storage->tensor();std::vector<unsigned char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
        for(std::int64_t row=0;row<tensor->ne[1];++row) {
            bool nonzero=false;for(std::size_t i=0;i<tensor->nb[1];++i)nonzero|=bytes[std::size_t(row)*tensor->nb[1]+i]!=0;
            CHECK(nonzero==(row<state.e));
        }
        storage.reset();
        auto b=take(generate_tokens_check(session,y,config,request,{false,true}));
        auto fresh=take(create_session(model));
        auto c=take(generate_tokens_check(fresh,y,config,request,{false,true}));
        CHECK(b.result.generated==c.result.generated&&b.logits==c.logits);
        CHECK(b.result.reset_healthy_cache&&b.result.cold&&!c.result.reset_healthy_cache);
        request.max_tokens=1;
        auto single=take(generate_tokens_check(session,x,config,request));
        CHECK(single.result.e==std::int64_t(x.size())&&single.result.decode_plan_builds==0&&single.result.decode_plan_reuses==0);
        take(release_session(session));take(release_session(fresh));take(unload_model(model));
        load.context=std::int64_t(x.size()+3);load.prefill_chunk=load.context;
        model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory,load))));session=take(create_session(model));
        config=take(resolve_generation_configuration(model,application));request.max_tokens=16;
        auto capacity=take(generate_tokens_check(session,x,config,request));
        CHECK(capacity.result.stop_reason==StopReason::capacity&&capacity.result.generated_count==4&&capacity.result.e==*load.context);
        CHECK(std::equal(capacity.result.generated.begin(),capacity.result.generated.end(),a.result.generated.begin()));
        take(release_session(session));take(unload_model(model));
    }
    {
        auto resources=take(RuntimeAccess::acquire(runtime));CHECK(resources.budget.current()==0);
    }
    auto model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory))));auto session=take(create_session(model));
    auto budget=take(RuntimeAccess::acquire(runtime)).budget;
    take(finish_runtime_use(runtime));CHECK(budget.current()==0);
    CHECK(!create_session(model)&&create_session(model).error().category==ErrorCategory::runtime_ended);
    CHECK(!release_session(session)&&release_session(session).error().category==ErrorCategory::runtime_ended);
    std::cout<<"T6 boundary/KV, T7 cold reset, T8 delivery/capacity, runtime cleanup passed\n";
}
