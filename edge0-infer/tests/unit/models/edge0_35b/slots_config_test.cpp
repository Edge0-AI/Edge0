#include "model/loading.h"
#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/compute.h"
#include "tests/support/model_fixture.h"
#include "tests/support/runtime.h"
using namespace edge0;
using test::take;
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv); const auto& adapter=models::edge0_35b::adapter();
    for (const auto threshold:{1,2,3000,4096}) {
        model::ModelOptions options={{"prefill_read","auto"},{"prefill_ahead_min_tokens",std::to_string(threshold)}};
        CHECK(!models::edge0_35b::use_ahead_prefill(options,threshold-1));
        CHECK(models::edge0_35b::use_ahead_prefill(options,threshold)==(threshold>1));
        CHECK(models::edge0_35b::use_ahead_prefill(options,threshold+1));
        CHECK(!models::edge0_35b::use_ahead_prefill(options,1));
    }
    fixture::Temp temp; auto assets=fixture::metadata(2,16,2);
    assets["base"].metadata["qwen3next.expert_used_count"]=std::uint32_t(4);
    assets["lora"].metadata["adapter.lora.alpha"]=4.0f;
    assets["prerouter"].metadata["prerouter.top_k"]=std::uint32_t(4);
    auto definition=take(adapter.describe_assets(assets));
    for(const auto* role:{"base","lora","prerouter"}) {
        std::vector<model::TensorRequirement> tensors;
        for(const auto& tensor:definition.tensors)if(tensor.role==role)tensors.push_back(tensor);
        fixture::gguf(temp.path/(std::string(role)+".gguf"),assets.at(role).metadata,tensors);
    }
    fixture::write(temp.path/"model.json","{\"format_version\":1,\"architecture\":\"edge0_35b\"}");
    fixture::write(temp.path/"generation.json","{}");fixture::write(temp.path/"tokenizer.json","{}");
    fixture::write(temp.path/"template",fixture::read(std::filesystem::path(MOE_SOURCE_ROOT)/"tests/fixtures/models/edge0_35b/chat_template.jinja"));
    fixture::catalog(temp.path,{{"base","base.gguf"},{"lora","lora.gguf"},{"prerouter","prerouter.gguf"},{"model","model.json"},{"generation_config","generation.json"},{"tokenizer","tokenizer.json"},{"chat_template","template"}});
    exec::Budget budget(64*1024*1024);std::atomic<std::size_t> reads{0};
    model::LoadRequest request;request.directory=temp.path.string();request.device="CPU";request.before_read=[&](std::size_t){++reads;};
    for(const model::ModelOptions& options:std::vector<model::ModelOptions>{
        {{"prefill_ahead_min_tokens","0"}},{{"prefill_ahead_min_tokens","4097"}},
        {{"prefill_ahead_min_tokens","-1"}},{{"prefill_ahead_min_tokens","1.5"}},{{"prefill_ahead_min_tokens","184467440737095516160"}},
        {{"expert_slots","3"}},{{"expert_slots","17"}},{{"expert_slots","abc"}},{{"expert_slots","4.5"}},
        {{"experts","resident"},{"expert_slots","4"}},{{"experts","bad"}},{{"prefill_read","bad"}},{{"expert_read","bad"}}}) {
        request.options=options;auto rejected=model::load(request,{&adapter},budget,{{"CPU",backend}});
        CHECK(!rejected && rejected.error().category==ErrorCategory::invalid_argument);CHECK(reads==0 && budget.current()==0);
    }
    request.context=64;request.options={{"prefill_ahead_min_tokens","65"}};
    auto outside_context=model::load(request,{&adapter},budget,{{"CPU",backend}});
    CHECK(!outside_context && outside_context.error().category==ErrorCategory::invalid_argument && reads==0);
    request.context=4096;
    platform::file_testing::disable_direct_read=true;
    request.options={{"expert_read","direct"}};
    { auto unsupported=model::load(request,{&adapter},budget,{{"CPU",backend}});
      CHECK(!unsupported && unsupported.error().category==ErrorCategory::unsupported && reads==0 && budget.current()==0); }
    platform::file_testing::disable_direct_read=false;
    request.options={{"prefill_read","used"}};std::size_t required=0;
    {auto loaded=take(model::load(request,{&adapter},budget,{{"CPU",backend}}));required=loaded->estimate.total_bytes;CHECK(required==budget.current());}
    std::size_t single_temporary=0;
    {
        auto loaded=take(model::load(request,{&adapter},budget,{{"CPU",backend}}));
        CHECK(loaded->temporary && !loaded->temporary_next);
        single_temporary=loaded->estimate.temporary_bytes;
    }
    for (const auto* read:{"auto","ahead"}) {
        request.options={{"prefill_read",read},{"prefill_ahead_min_tokens","2"}};
        {
            auto loaded=take(model::load(request,{&adapter},budget,{{"CPU",backend}}));
            CHECK(loaded->temporary && loaded->temporary_next && loaded->temporary!=loaded->temporary_next);
            CHECK(loaded->estimate.temporary_bytes==2*single_temporary);
            CHECK(loaded->estimate.total_bytes==required+single_temporary);
            const auto source=loaded->pool_descriptions[0].experts[0][0];
            auto write=take(loaded->weights.begin_write(loaded->temporary,{{0,"bank0",source}}));
            CHECK(!loaded->weights.fill_regions(write));
            auto bindings=take(loaded->weights.complete(write));
            auto view=take(loaded->weights.acquire_views(loaded->temporary,bindings[0]));
            auto blocked=loaded->weights.begin_write(loaded->temporary,{{1,"protected",source}});
            CHECK(!blocked);
            auto other=take(loaded->weights.begin_write(loaded->temporary_next,{{0,"bank1",source}}));
            CHECK(!loaded->weights.fill_regions(other));CHECK(loaded->weights.complete(other));
            CHECK(!loaded->weights.release_use(view.use));
            std::atomic<std::size_t> maximum_writing{0}, observed_reads{0};
            std::atomic<bool> overlap{false}, violation{false};
            loaded->weights.before_read=[&](std::size_t) {
                std::size_t writing=0;
                bool computing=false;
                for(const auto& bank:{loaded->temporary,loaded->temporary_next}) {
                    const auto state=take(loaded->weights.query_state(bank,0));
                    writing+=state.state==weights::RegionState::writing?1:0;
                    computing|=state.uses>0;
                    if(state.state==weights::RegionState::writing && state.uses>0) violation=true;
                }
                auto previous=maximum_writing.load();
                while(previous<writing && !maximum_writing.compare_exchange_weak(previous,writing)) {}
                if(writing && computing) overlap=true;
                ++observed_reads;
            };
            auto states_description=take(models::edge0_35b::describe_states(loaded->assets.definition));
            auto kv=take(session::KVCache::create(budget,backend,64,states_description.kv));
            session::StateStore states;
            CHECK(!models::edge0_35b::initialize_session(loaded,*kv,states,budget, exec::Stages{backend, backend}));
            models::ForwardRequest forward{{1,2},{0,2},exec::Phase::prefill,{1},{}};
            auto result=models::edge0_35b::forward(loaded,forward,*kv,states,budget);
            if(!result) std::cerr<<result.error().message<<"\n";
            CHECK(result); auto& output=result.value();
            CHECK(output->completion().success);
            CHECK(!output->end_observation());
            CHECK(observed_reads>0 && maximum_writing==1 && overlap && !violation);
            loaded->weights.before_read={};
            models::edge0_35b::release_session(loaded,*kv,states);
            std::cout<<read<<": maximum writing banks=1; active compute bank never written\n";
        }
        reads=0;exec::Budget ahead_short(required+single_temporary-1);
        auto ahead_rejected=model::load(request,{&adapter},ahead_short,{{"CPU",backend}});
        CHECK(!ahead_rejected && ahead_rejected.error().category==ErrorCategory::budget_exceeded && reads==0);
    }
    request.options={{"prefill_read","used"}};
    reads=0;exec::Budget insufficient(required-1);
    auto rejected=model::load(request,{&adapter},insufficient,{{"CPU",backend}});
    CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded && reads==0 && insufficient.current()==0);
    std::cout<<"T8: all invalid options rejected before payload reads; one-byte admission rejection\n";
}
