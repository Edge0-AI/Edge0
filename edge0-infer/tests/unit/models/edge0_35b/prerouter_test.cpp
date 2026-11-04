#include "models/edge0_35b/compute.h"
#include "models/edge0_35b/assets.h"
#include "model/loading.h"
#include "tests/support/runtime.h"
#include <cstdlib>
#include <thread>
#include <set>
using namespace edge0;
using test::take;
namespace edge=edge0::models::edge0_35b;
using Ledger = std::array<std::size_t,3>;
const char* purposes[]{"buffer","metadata","workspace"};
Ledger snapshot(const exec::Budget& budget) {
    const Ledger entries{(budget.current(exec::Purpose::resident_weights) + budget.current(exec::Purpose::experts) + budget.current(exec::Purpose::session_state) + budget.current(exec::Purpose::transfer)),budget.current(edge0::exec::Purpose::metadata),budget.current(edge0::exec::Purpose::workspace)};
    CHECK(entries[0]+entries[1]+entries[2]==budget.current());
    return entries;
}
Ledger released(const Ledger& before,const Ledger& after) {
    Ledger result{};
    for(std::size_t i=0;i<result.size();++i) { CHECK(before[i]>=after[i]);result[i]=before[i]-after[i]; }
    return result;
}
std::size_t total(const Ledger& entries) { return entries[0]+entries[1]+entries[2]; }
struct Run {
    std::vector<std::vector<float>> logits;
    std::vector<std::size_t> ledger;
    std::vector<Ledger> ledger_parts;
    Ledger head_resident{},first_plan{},second_plan{},finished{};
    std::vector<std::int32_t> tokens;
    WeightActivity activity;
};
Run run(const std::shared_ptr<exec::Backend>& backend,const std::string& root,const std::string& mode,
        bool resident=false,bool discontinuous=false) {
    exec::Budget budget(1024ULL*1024*1024);Run result;
    model::LoadRequest load;load.directory=root;load.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle()));
    load.options={{"routing",mode},{"experts",resident?"resident":"slots"}};
    auto model=take(model::load(load,{&edge::adapter()},budget,{{load.device,backend}}));
    // Read actual reservations from the role's storage, once per allocation.
    std::set<const exec::Buffer*> allocations;
    for(const auto& entry:model->resident) if(entry.first.find("prerouter_")!=std::string::npos) {
        auto allocation=entry.second.storage->allocation();
        if(!allocations.insert(allocation.get()).second) continue;
        for(const auto& charge:allocation->charges) {
            bool recognized=false;
            for(std::size_t i=0;i<3;++i) if(charge->purpose()==std::array<exec::Purpose,3>{exec::Purpose::resident_weights,exec::Purpose::metadata,exec::Purpose::workspace}[i]) {
                result.head_resident[i]+=charge->size();recognized=true;
            }
            CHECK(recognized);
        }
    }
    CHECK((mode=="actual")==allocations.empty());
    auto description=take(edge::describe_states(model->assets.definition));
    auto kv=take(session::KVCache::create(budget,backend,256,description.kv));session::StateStore states;
    CHECK(!edge::initialize_session(model,*kv,states,budget, exec::Stages{backend, backend}));
    for(int step=-1;step<32;++step) {
        const std::int64_t position=step<0?0:step+3+(discontinuous && step>=2?1:0);
        models::ForwardRequest request{step<0?std::vector<std::int32_t>{1,2,3}:std::vector<std::int32_t>{std::int32_t(10+step)},
            {position,step<0?3:1},step<0?exec::Phase::prefill:exec::Phase::decode,{std::size_t(step<0?2:0)},{}};
        request.greedy=true;request.breakdown=true;
        model->weights.before_read={};
        const auto step_before=snapshot(budget);
        auto forwarded=edge::forward(model,request,*kv,states,budget);
        if(!forwarded) std::cerr<<forwarded.error().message<<"\n";
        CHECK(forwarded);const auto& output=forwarded.value();
        auto choice=take(output->greedy_output());
        if(step>=0) {
            const bool first=step==0 || (discontinuous && step==2);
            const std::size_t expected=mode=="actual"?136:first?328:232;
            CHECK(output->readback_bytes()==expected);
            CHECK(output->decode_timing()->segments==(mode=="predicted" && !first?3:9));
            result.tokens.push_back(choice.token);
            result.logits.push_back(take(output->logits()));
        }
        result.activity+=output->weight_activity();
        CHECK(!output->end_observation());
        result.ledger.push_back(budget.current());
        result.ledger_parts.push_back(snapshot(budget));
        if(step>=2) CHECK(snapshot(budget)==step_before);
        auto prediction=take(states.get<edge::PrerouterState>("aux.prerouter"));
        if(step<0 && prediction) {
            CHECK(prediction->position==-1);
            for(const auto& forecast:prediction->predictions) CHECK(forecast.ids.empty() && forecast.scores.empty());
        }

    }
    const auto before=budget.current();result.activity+=take(edge::finish_pending_weights(model,states,models::FinishReason::normal));
    CHECK(budget.current()==before);
    result.finished=snapshot(budget);
    // Attribute plan reservations by releasing each cached plan separately.
    // This measures their ledger entries independently of cross-mode totals.
    auto plan_before=snapshot(budget);
    states.remove("aux.predicted_plans").reset();
    result.second_plan=released(plan_before,snapshot(budget));
    plan_before=snapshot(budget);
    states.remove("aux.expert_plans").reset();
    result.first_plan=released(plan_before,snapshot(budget));
    auto prediction=take(states.get<edge::PrerouterState>("aux.prerouter"));
    if(prediction) for(const auto& pending:prediction->pending) CHECK(!pending);
    for(std::size_t i=3;i<result.ledger.size();++i) CHECK(result.ledger[i]==result.ledger[2]);
    if(prediction) {
        models::ForwardRequest prefill{{42},{discontinuous?36:35,1},exec::Phase::prefill,{0},{"prerouter_in-1"}};
        auto forwarded=edge::forward(model,prefill,*kv,states,budget); CHECK(forwarded);
        const auto& output=forwarded.value();
        CHECK(take(output->observations()).empty());
        CHECK(!output->end_observation());
        CHECK(prediction->position==-1);
        for(const auto& forecast:prediction->predictions) CHECK(forecast.ids.empty() && forecast.scores.empty());
        for(const auto& previous:prediction->previous) CHECK(previous.empty());
        for(const auto& pending:prediction->pending) CHECK(!pending);
    }
    CHECK(!edge::reset_session(model,*kv,states));CHECK(!take(states.get<edge::PrerouterState>("aux.prerouter")));
    edge::release_session(model,*kv,states);kv.reset();model.reset();CHECK(budget.current()==0);
    return result;
}
void failing(const std::shared_ptr<exec::Backend>& backend,const std::string& root,const std::string& mode,bool tail) {
    exec::Budget budget(1024ULL*1024*1024);
    model::LoadRequest load;load.directory=root;load.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle()));load.options={{"routing",mode}};
    auto model=take(model::load(load,{&edge::adapter()},budget,{{load.device,backend}}));
    auto kv=take(session::KVCache::create(budget,backend,256,take(edge::describe_states(model->assets.definition)).kv));
    session::StateStore states;CHECK(!edge::initialize_session(model,*kv,states,budget, exec::Stages{backend, backend}));
    for(int step=-1;step<=0;++step) {
        models::ForwardRequest request{step<0?std::vector<std::int32_t>{1,2,3}:std::vector<std::int32_t>{10},
            {step<0?0:3,step<0?3:1},step<0?exec::Phase::prefill:exec::Phase::decode,{std::size_t(step<0?2:0)},{}};
        auto output=edge::forward(model,request,*kv,states,budget);CHECK(output);
        if(step==0) model->weights.before_read=[](std::size_t) {throw std::runtime_error("Injected prefetch read failure");};
        CHECK(!output.value()->end_observation());
    }
    model->weights.before_read={};
    if(tail) {
        auto finished=edge::finish_pending_weights(model,states,models::FinishReason::normal);
        CHECK(!finished && finished.error().category==ErrorCategory::backend_error);
    } else {
        models::ForwardRequest request{{11},{4,1},exec::Phase::decode,{0},{}};
        auto output=edge::forward(model,request,*kv,states,budget);
        CHECK(!output && output.error().category==ErrorCategory::backend_error);
    }
    auto prediction=take(states.get<edge::PrerouterState>("aux.prerouter"));
    for(const auto& pending:prediction->pending) CHECK(!pending);
    CHECK(model->pools.back()->storage()->state(0)==weights::RegionState::isolated ||
          model->pools.back()->storage()->state(1)==weights::RegionState::isolated);
    const auto reads=model->weights.read_count();
    CHECK(edge::finish_pending_weights(model,states,models::FinishReason::failure));
    CHECK(reads==model->weights.read_count());
    edge::release_session(model,*kv,states);kv.reset();model.reset();CHECK(budget.current()==0);
}
void failure_before_end(const std::shared_ptr<exec::Backend>& backend,const std::string& root,const std::string& mode,bool unwind) {
    exec::Budget budget(1024ULL*1024*1024);
    model::LoadRequest load;load.directory=root;load.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle()));load.options={{"routing",mode}};
    auto model=take(model::load(load,{&edge::adapter()},budget,{{load.device,backend}}));
    auto kv=take(session::KVCache::create(budget,backend,256,take(edge::describe_states(model->assets.definition)).kv));
    session::StateStore states;CHECK(!edge::initialize_session(model,*kv,states,budget, exec::Stages{backend, backend}));
    for(int step=-1;step<=1;++step) {
        models::ForwardRequest request{step<0?std::vector<std::int32_t>{1,2,3}:std::vector<std::int32_t>{std::int32_t(10+step)},
            {step<0?0:step+3,step<0?3:1},step<0?exec::Phase::prefill:exec::Phase::decode,{std::size_t(step<0?2:0)},{}};
        try {
            auto output=edge::forward(model,request,*kv,states,budget);CHECK(output);
            if(step==1 && unwind) throw std::runtime_error("Injected observer failure");
            if(step==1) {
                CHECK(edge::finish_pending_weights(model,states,models::FinishReason::failure));
                const auto reads=model->weights.read_count();
                CHECK(!output.value()->end_observation());
                CHECK(model->weights.read_count()==reads);
            } else CHECK(!output.value()->end_observation());
        } catch(const std::runtime_error&) {
            CHECK(unwind && step==1);
            auto prediction=take(states.get<edge::PrerouterState>("aux.prerouter"));
            CHECK(!prediction->pending.back());
            CHECK(edge::finish_pending_weights(model,states,models::FinishReason::failure));
        }
    }
    auto prediction=take(states.get<edge::PrerouterState>("aux.prerouter"));
    CHECK(prediction->position==-1);
    for(const auto& pending:prediction->pending) CHECK(!pending);
    edge::release_session(model,*kv,states);kv.reset();model.reset();CHECK(budget.current()==0);
}
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    const auto* reference=std::getenv("MOE_TINY");if(!reference)return 77;
    const auto root=std::string(reference)+"/tiny-a";
    for(const auto* mode:{"predicted"}) for(bool tail:{false,true}) failing(backend,root,mode,tail);
    for(const auto* mode:{"predicted"}) for(bool unwind:{false,true}) failure_before_end(backend,root,mode,unwind);
    auto actual=run(backend,root,"actual");
    auto predicted=run(backend,root,"predicted"),resident=run(backend,root,"predicted",true);
    CHECK(actual.activity.prefetch.initiated==0 && actual.activity.prefetch.bytes==0 && actual.activity.prefetch.tail_bytes==0);
    CHECK(predicted.activity.prefetch.initiated==32*6*4 && predicted.activity.prefetch.tail_bytes>0);
    CHECK(predicted.tokens==resident.tokens);
    const bool metal=ggml_backend_dev_type(ggml_backend_get_device(backend->handle()))==GGML_BACKEND_DEVICE_TYPE_GPU;
    if(!metal) {CHECK(actual.logits.front()==predicted.logits.front());CHECK(predicted.logits==resident.logits);}
    else for(std::size_t step=0;step<predicted.logits.size();++step) for(std::size_t i=0;i<predicted.logits[step].size();++i)
        CHECK(std::abs(predicted.logits[step][i]-resident.logits[step][i])<=1e-3);
    auto gaps=run(backend,root,"predicted",false,true);
    CHECK(gaps.activity.prefetch.discarded==1);
    std::cout<<"32-step routing: byte counts, segments, slots/resident, steady ledger, finish/reset/release, position gaps passed\n";
    std::cout<<"ledger actual="<<actual.ledger[2]<<" predicted="<<predicted.ledger[2]<<'\n';
    if(argc>1 && std::string(argv[1])=="--ledger") {
        CHECK(actual.head_resident==Ledger{} && actual.second_plan==Ledger{});
        const auto head_plan=released(predicted.first_plan,actual.first_plan);
        Ledger difference{};
        for(std::size_t i=0;i<3;++i) {
            difference[i]=predicted.head_resident[i]+head_plan[i];
            std::cout<<purposes[i]<<": head_resident="<<predicted.head_resident[i]
                     <<" head_plan="<<head_plan[i]<<" D="<<difference[i]
                     <<" second_plan="<<predicted.second_plan[i]<<'\n';
        }
        CHECK(total(predicted.head_resident)>0 && total(head_plan)>0 && total(predicted.second_plan)>0);
        const auto verify=[&](const Ledger& a,const Ledger& predicted_entries,bool second) {
            for(std::size_t i=0;i<3;++i)
                CHECK(predicted_entries[i]==a[i]+difference[i]+(second?predicted.second_plan[i]:0));
        };
        // Index 0 is prefill; index 1 is the first decode step.
        for(std::size_t step=1;step<actual.ledger_parts.size();++step)
            verify(actual.ledger_parts[step],predicted.ledger_parts[step],step>=2);
        verify(actual.finished,predicted.finished,true);
        std::cout<<"head resident="<<total(predicted.head_resident)<<"; head plan="<<total(head_plan)
                 <<"; D="<<total(difference)<<"; second plan="<<total(predicted.second_plan)
                 <<"; predicted minus actual="<<total(predicted.finished)-total(actual.finished)
                 <<"; all decode steps and finish verified\n";
    }
}
