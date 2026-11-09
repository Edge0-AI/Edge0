#include "weights/expert_slots.h"
#include "tests/support/runtime.h"
#include <chrono>
#include <filesystem>
#include <fstream>
using namespace edge0;
using namespace edge0::weights;
using test::take;
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    exec::Budget budget(64*1024*1024);
    auto path=std::filesystem::current_path()/"build"/("slots-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::array<float,6*3*16> data{};
    for(std::size_t i=0;i<data.size();++i) data[i]=float(int(i%17)-8)*0.125f;
    {std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<char*>(data.data()),sizeof(data));}
    {
        Weights w(budget);
        PoolDescription description; description.slots=4;
        for(auto& layout:description.projections) layout={GGML_TYPE_F32,{4,4}};
        description.experts.resize(6);
        for(std::size_t e=0;e<6;++e) for(std::size_t p=0;p<3;++p)
            description.experts[e][p]={path.string(),(e*3+p)*64,64,{GGML_TYPE_F32,{4,4}}};
        auto pool=take(ExpertPool::create_pools(w,backend,description));
        auto run=[&](std::vector<std::int32_t> ids) {return take(pool->await_ready(take(pool->prepare(ids))));};
        auto first=run({0,1,2,3}); CHECK(first.activity.loads==4 && first.activity.bytes==4*3*64);
        auto pending=take(pool->prepare({1,2,3,4})); CHECK(!pool->prepare({1}));
        CHECK(!w.acquire_views(pool->storage(),0,first.bindings[0]));
        auto second=take(pool->await_ready(pending)); CHECK(second.activity.hits==3 && second.activity.loads==1);
        CHECK(second.mapping.at(4)==first.mapping.at(0));
        // Missing first, later hit protected even when it is least recently used.
        auto hit=run({1,1}); CHECK(hit.activity.demand==1 && hit.activity.loads==0);
        CHECK(hit.bindings[0][0].generation==second.bindings[0][0].generation);
        auto third=run({0,1,3,4}); CHECK(third.mapping.at(0)==first.mapping.at(2));
        auto reads=w.read_count(); CHECK(!pool->prepare({0,1,2,3,4})); CHECK(w.read_count()==reads);
        auto view=take(w.acquire_views(pool->storage(),0,third.bindings[0])); CHECK(!pool->prepare({2})); CHECK(view.use->valid());
        CHECK(!w.release_use(view.use));
        pending=take(pool->prepare({0,1,3,4})); CHECK(!pool->abandon(pending));
        auto current=take(pool->prepare({4,0,1,3}));
        CHECK(!pool->abandon(pending)); CHECK(!pool->prepare({2}));
        auto ready=take(pool->await_ready(current)); CHECK(ready.activity.loads==0);
        // Compare whole-array mul_mat_id with a resident tensor containing the
        // same experts in model order. IDs in the slot graph come from bindings.
        auto reference=take(w.create_storage(backend,{{{GGML_TYPE_F32,{4,4,6}},1}}, edge0::exec::Purpose::resident_weights));
        std::array<float,6*16> matrix{};
        for(std::size_t e=0;e<6;++e) std::copy_n(data.begin()+std::ptrdiff_t(e*3*16),16,matrix.begin()+std::ptrdiff_t(e*16));
        // Source size is validated before custom conversion, so use a separate file.
        auto refpath=path.string()+"-reference";
        {std::ofstream f(refpath,std::ios::binary);f.write(reinterpret_cast<char*>(matrix.data()),sizeof(matrix));}
        auto tx=take(w.begin_write(reference,{{0,"reference",{refpath,0,sizeof(matrix),{GGML_TYPE_F32,{4,4,6}}}}}));
        CHECK(!w.fill_regions(tx)); auto rb=take(w.complete(tx));
        auto compute=[&](View weight,const std::array<std::int32_t,4>& ids) {
            auto graph=take(exec::create_graph(budget,backend,exec::Phase::decode));
            CHECK(!graph->bind_weights(weight.tensor,backend,weight.use));
            auto* x=take(graph->input("x",{GGML_TYPE_F32,{4,1,1}}));
            auto* selected=take(graph->input("ids",{GGML_TYPE_I32,{4,1}}));
            auto* output=ggml_mul_mat_id(graph->context(),weight.tensor,x,selected); CHECK(!graph->output(output));
            auto plan=take(exec::build_plan(graph)); float input[]{1,2,3,4};
            CHECK(!plan->set_inputs("x",input,sizeof(input))); CHECK(!plan->set_inputs("ids",ids.data(),sizeof(ids)));
            CHECK(!plan->submit()); CHECK(take(plan->collect_completion()).success);
            std::array<float,16> result{}; CHECK(!plan->read_output(output,result.data(),0,sizeof(result)));
            CHECK(!plan->end_observation()); CHECK(!plan->release_plan()); return result;
        };
        std::array<std::int32_t,4> ids{4,0,1,3}, mapped{};
        for(std::size_t i=0;i<4;++i)mapped[i]=std::int32_t(ready.mapping.at(ids[i]));
        auto actual=compute(take(w.acquire_views(pool->storage(),0,ready.bindings[0])),mapped);
        auto expected=compute(take(w.acquire_views(reference,rb[0])),ids);
        CHECK(std::memcmp(actual.data(),expected.data(),sizeof(actual))==0);
        CHECK(!w.close(reference)); std::filesystem::remove(refpath);
        {
            auto separate=description; separate.layer=1;
            auto closing_pool=take(ExpertPool::create_pools(w,backend,separate));
            auto loaded=take(closing_pool->await_ready(take(closing_pool->prepare({0}))));
            auto held=take(w.acquire_views(closing_pool->storage(),0,loaded.bindings[0]));
            auto closed=std::async(std::launch::async,[&] {return closing_pool->close();});
            CHECK(closed.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout);
            CHECK(!w.release_use(held.use)); CHECK(!closed.get());
        }
        // Failure after a writer starts isolates the pool, even with no reader.
        w.before_read=[](std::size_t) {throw std::runtime_error("injected");};
        pending=take(pool->prepare({2})); CHECK(!pool->await_ready(pending));
        CHECK(!pool->prepare({0})); CHECK(!pool->close());
    }
    CHECK(budget.current()==0); std::filesystem::remove(path);
    std::cout<<"slots: hit protection, empty/LRU, duplicate demand, group protection, isolation, mul_mat_id bitwise match; ledger=0\n";
}
