#include "app/generation_internal.h"
#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/compute.h"
#include "tests/support/runtime.h"
#include <cstdlib>
#include <atomic>
using namespace edge0;
using test::take;
namespace edge=models::edge0_35b;
int main() {
    const auto* root=std::getenv("MOE_TINY");if(!root)return 77;
    const auto directory=std::string(root)+"/tiny-a";
    auto runtime=take(open_runtime({{"CPU"},{}}));auto resources=take(RuntimeAccess::acquire(runtime));
    auto load=[&] {
        model::LoadRequest request;request.directory=directory;request.device="CPU";
        return take(model::load(request,{&edge::adapter()},resources.budget,resources.backends));
    };
    auto loaded=load(); auto description=take(edge::describe_states(loaded->assets.definition));
    std::atomic<std::size_t> started{0},finished{0};
    loaded->weights.before_read=[&](std::size_t) {
        auto index=started.fetch_add(1);++finished;
        if(index>=24 && index<36)throw std::runtime_error("third layer read failure");
    };
    {
        auto kv=take(session::KVCache::create(resources.budget,loaded->stages.prefill,1024,description.kv));session::StateStore states;
        CHECK(!edge::initialize_session(loaded,*kv,states,resources.budget,loaded->stages));
        models::ForwardRequest request{{-1},{0,1},exec::Phase::prefill,{0},{}};models::ForwardProgress progress;request.progress=&progress;
        CHECK(!edge::forward(loaded,request,*kv,states,resources.budget)); CHECK(!progress.submitted && started==0);
        request.tokens.assign(1,1); auto result=edge::forward(loaded,request,*kv,states,resources.budget);
        CHECK(!result && progress.submitted); CHECK(started==36 && finished==36);
        for(const auto& pool:loaded->pools)for(std::size_t i=0;i<12;++i)CHECK(pool->storage()->state(i)!=weights::RegionState::writing);
        CHECK(loaded->pools[2]->storage()->state(0)==weights::RegionState::isolated);
        edge::release_session(loaded,*kv,states);
    }
    // A new session cannot make an isolated model healthy again.
    {
        auto kv=take(session::KVCache::create(resources.budget,loaded->stages.prefill,1024,description.kv));session::StateStore states;
        CHECK(!edge::initialize_session(loaded,*kv,states,resources.budget,loaded->stages));
        models::ForwardRequest request{{1},{0,1},exec::Phase::prefill,{0},{}};
        auto result=edge::forward(loaded,request,*kv,states,resources.budget);CHECK(!result);
        CHECK(result.error().message.find("isolated")!=std::string::npos); CHECK(started==36);
        edge::release_session(loaded,*kv,states);
    }
    loaded.reset(); loaded=load(); started=0;finished=0;
    auto rules=std::make_shared<text::Rules>(take(text::load_rules(directory)));
    LoadConfiguration config;config.context=1024;config.device="CPU";
    auto model=take(app::register_model(runtime,loaded,rules,edge::compute(),config));auto session=take(create_session(model));
    GenerationOverrides overrides;overrides.temperature=0;overrides.max_tokens=1;
    auto generation=take(resolve_generation_configuration(model,overrides));
    loaded->weights.before_read=[&](std::size_t) {auto index=started.fetch_add(1);++finished;if(index>=24)throw std::runtime_error("third layer");};
    auto bad=generate_tokens_check(session,{-1},generation);CHECK(!bad);CHECK(take(query_session_check(session)).valid);
    auto failed=take(generate_tokens_check(session,{1},generation)).result;
    CHECK(failed.stop_reason==StopReason::failed && !failed.state_valid && started==36 && finished==36);
    auto rejected=generate_tokens_check(session,{1},generation);CHECK(!rejected && rejected.error().category==ErrorCategory::session_invalid);
    take(release_session(session));take(unload_model(model));loaded.reset();
    {
        model::LoadRequest request;request.directory=directory;request.device="CPU";request.options={{"prefill_read","ahead"}};
        auto ahead=take(model::load(request,{&edge::adapter()},resources.budget,resources.backends));
        auto handle=take(app::register_model(runtime,ahead,rules,edge::compute(),config));auto active=take(create_session(handle));
        std::atomic<std::size_t> reads_started{0},reads_finished{0};std::atomic<bool> overlap{false},violation{false};
        ahead->weights.before_read=[&](std::size_t) {
            ++reads_started;
            bool writing=false,using_bank=false,second_layer=false;
            for(const auto& bank:{ahead->temporary,ahead->temporary_next}) {
                auto state=take(ahead->weights.query_state(bank,0));
                writing|=state.state==weights::RegionState::writing;using_bank|=state.uses>0;
                if(state.state==weights::RegionState::writing && state.uses) violation=true;
                if(bank==ahead->temporary_next && state.state==weights::RegionState::writing) second_layer=true;
            }
            if(writing && using_bank) overlap=true;
            ++reads_finished;
            if(second_layer) throw std::runtime_error("ahead second-layer read failure");
        };
        auto failure=take(generate_tokens_check(active,{1,2,3},generation)).result;
        CHECK(failure.stop_reason==StopReason::failed && !failure.state_valid);
        CHECK(reads_started>0 && reads_started==reads_finished && overlap && !violation);
        for(const auto& bank:{ahead->temporary,ahead->temporary_next})
            for(std::size_t i=0;i<ahead->assets.definition.parameters.at("n_expert")*3;++i)
                CHECK(bank->state(i)!=weights::RegionState::writing);
        take(release_session(active));take(unload_model(handle));
    }
    auto healthy=load();auto fresh=take(app::register_model(runtime,healthy,rules,edge::compute(),config));auto fresh_session=take(create_session(fresh));
    auto recovered=take(generate_tokens_check(fresh_session,{1},generation)).result;
    CHECK(recovered.stop_reason==StopReason::length && recovered.state_valid && recovered.generated.size()==1);
    healthy.reset(); resources.backends.clear(); take(finish_runtime_use(runtime)); CHECK(resources.budget.current()==0);
    std::cout<<"T9: pre-submit valid; third-layer failure drained 36 writers; no later layer; isolated new session; formal session_invalid; reload healthy\n";
}
