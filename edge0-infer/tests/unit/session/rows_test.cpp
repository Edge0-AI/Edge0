#include "session/storage.h"
#include "tests/support/runtime.h"
#include <numeric>
using namespace edge0;
using test::take;
int main(int argc, char** argv) {
    auto backend=test::backend(argc,argv);
    exec::Budget budget(64*1024*1024);
    for (std::int64_t heads : {1,2}) for (session::Step step : {session::Step{13,1},session::Step{255,3}}) {
        std::vector<std::vector<unsigned char>> results;
        for (bool rows : {false,true}) {
            auto cache=take(session::KVCache::create(budget,backend,512,
                {{"k",GGML_TYPE_F16,{256,heads},1,256,session::Visibility::causal}}));
            auto storage=take(cache->storage("k"));
            auto graph=take(exec::create_graph(budget,backend,exec::Phase::prefill));
            auto* c=graph->context();
            auto* input=take(graph->input("values",{GGML_TYPE_F32,{256,step.n,heads}}));
            take(graph->bind_states(storage));
            ggml_tensor* output;
            if (rows) {
                auto* indices=take(graph->input("rows",{GGML_TYPE_I64,{step.n}}));
                output=ggml_set_rows(c,storage->tensor(),input,indices);
            } else output=ggml_cpy(c,input,take(cache->write_range(c,"k",step)).view);
            take(graph->bind_state_write(storage,output));
            CHECK(!graph->output(output));
            auto plan=take(exec::build_plan(graph));
            std::vector<float> values(static_cast<std::size_t>(256*step.n*heads));
            for (std::size_t i=0;i<values.size();++i) values[i]=float(i%511)/32;
            CHECK(!plan->set_inputs("values",values.data(),values.size()*sizeof(float)));
            if (rows) {
                std::vector<std::int64_t> indices(static_cast<std::size_t>(step.n));
                std::iota(indices.begin(),indices.end(),step.e);
                CHECK(!plan->set_inputs("rows",indices.data(),indices.size()*sizeof(std::int64_t)));
            }
            CHECK(!plan->submit()); CHECK(take(plan->collect_completion()).success);
            CHECK(!plan->end_observation());
            results.emplace_back(ggml_nbytes(storage->tensor()));
            ggml_backend_tensor_get(storage->tensor(),results.back().data(),0,results.back().size());
        }
        CHECK(results[0]==results[1]);
    }
    CHECK(budget.current()==0);
    std::cout << "KV row writes equal slice copies for one/two heads and alignment crossing\n";
}
