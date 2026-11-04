#include "models/edge0_35b/compute.h"
#include "models/edge0_35b/assets.h"
#include "model/loading.h"
#include "tests/support/runtime.h"
#include <cstdlib>
using namespace edge0;
using test::take;
namespace edge=edge0::models::edge0_35b;
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    const auto* reference=std::getenv("MOE_TINY");if(!reference)return 77;
    exec::Budget budget(1024ULL*1024*1024);
    for(const auto* route:{"actual","predicted"}) {
        model::LoadRequest load;load.directory=std::string(reference)+"/tiny-a";
        load.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle()));load.options={{"routing",route},{"prefill_read","ahead"},{"prefill_ahead_min_tokens","1"}};
        auto model=take(model::load(load,{&edge::adapter()},budget,{{load.device,backend}}));
        auto healthy = [&] {
            auto check = [&](const std::shared_ptr<weights::Storage>& storage, std::size_t count) {
                if (!storage) return;
                for (std::size_t i=0; i<count; ++i) {
                    auto info=take(model->weights.query_state(storage,i));
                    CHECK(info.state!=weights::RegionState::isolated && info.state!=weights::RegionState::writing);
                    CHECK(!info.pending);
                }
            };
            for (std::size_t i=0; i<model->pools.size(); ++i)
                check(model->pools[i]->storage(), model->pool_descriptions[i].slots*3);
            const auto declaration=take(model->adapter->describe_weight_storage(model->assets.indexes,model->options));
            std::size_t count=0; for (const auto& array:declaration.temporary) count+=array.regions;
            check(model->temporary,count); check(model->temporary_next,count);
        };
        auto description=take(edge::describe_states(model->assets.definition));
        CHECK(description.block_snapshot_bytes == description.checkpoint_bytes);
        auto kv=take(session::KVCache::create(budget,backend,1024,description.kv));session::StateStore states;
        CHECK(!edge::initialize_session(model,*kv,states,budget, exec::Stages{backend, backend}));
        const auto snapshot = take(states.get<edge::CheckpointState>("aux.block_snapshot")); CHECK(snapshot);
        CHECK(!edge::reset_session(model,*kv,states)); CHECK(take(states.get<edge::CheckpointState>("aux.block_snapshot")) == snapshot);
        std::int64_t e=0;
        auto step=[&](exec::Phase phase) {
            models::ForwardRequest request{{1},{e,1},phase,{0},{}};request.greedy=true;
            request.cancel_block=[](std::size_t) { CHECK(false); return true; };
            auto computed=edge::forward(model,request,*kv,states,budget);CHECK(computed);const auto& output=computed.value();take(output->greedy_output());
            const bool reused=output->reused_plan();CHECK(!output->end_observation());++e;return reused;
        };
        step(exec::Phase::prefill);step(exec::Phase::decode);step(exec::Phase::decode);
        const auto checkpoint_e=e;
        CHECK(edge::finish_pending_weights(model,states,models::FinishReason::normal));
        healthy();
        CHECK(!edge::save_state(states));
        std::vector<std::pair<std::shared_ptr<exec::StateStorage>,std::vector<unsigned char>>> saved;
        for(auto layer:model->assets.definition.layer_groups.at("deltanet")) {
            auto state=take(states.get<edge::DeltaState>("deltanet."+std::to_string(layer)));
            for(auto storage:{state->convolution,state->recurrent}) {
                std::vector<unsigned char> bytes(ggml_nbytes(storage->tensor()));ggml_backend_tensor_get(storage->tensor(),bytes.data(),0,bytes.size());
                saved.emplace_back(storage,std::move(bytes));
            }
        }
        for(int i=0;i<4;++i)step(exec::Phase::decode);
        CHECK(edge::finish_pending_weights(model,states,models::FinishReason::normal));
        CHECK(!edge::restore_state(states));CHECK(!take(states.get<edge::PrerouterState>("aux.prerouter")));e=checkpoint_e;
        std::size_t index=0;
        for(auto layer:model->assets.definition.layer_groups.at("deltanet")) {
            auto state=take(states.get<edge::DeltaState>("deltanet."+std::to_string(layer)));
            for(auto storage:{state->convolution,state->recurrent}) {
                CHECK(storage==saved[index].first);
                std::vector<unsigned char> bytes(ggml_nbytes(storage->tensor()));ggml_backend_tensor_get(storage->tensor(),bytes.data(),0,bytes.size());
                CHECK(bytes==saved[index++].second);
            }
        }
        CHECK(step(exec::Phase::decode));
        CHECK(edge::finish_pending_weights(model,states,models::FinishReason::normal));
        healthy();
        CHECK(!edge::restore_state(states));
        const auto used=budget.current();CHECK(!edge::discard_saved_state(states));CHECK(budget.current()==used);
        CHECK(edge::restore_state(states));
        models::ForwardProgress progress;
        models::ForwardRequest interrupted{{1,2},{e,2},exec::Phase::prefill,{1},{}};
        interrupted.progress=&progress;
        interrupted.cancel_block=[](std::size_t segment) { return segment==3; };
        CHECK(!edge::forward(model,interrupted,*kv,states,budget));
        CHECK(progress.cancelled && progress.rolled_back_tokens==2);
        healthy();
        progress={};
        interrupted.cancel_block=[&](std::size_t) { states.remove("aux.block_snapshot"); return true; };
        CHECK(!edge::forward(model,interrupted,*kv,states,budget));
        CHECK(progress.submitted && !progress.cancelled && progress.rolled_back_tokens==0);
        edge::release_session(model,*kv,states); CHECK(!take(states.get<edge::CheckpointState>("aux.block_snapshot")));
    }
    std::cout<<"checkpoint bytes, stable storage, repeated restore and decode plan reuse passed\n";
}
