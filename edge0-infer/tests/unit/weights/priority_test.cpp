#include "weights/expert_slots.h"
#include "tests/support/model_fixture.h"
#include "tests/support/runtime.h"
#include <condition_variable>
#include <fstream>
#include <thread>
using namespace edge0;
using test::take;
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    fixture::Temp temp;
    std::vector<float> data(64,1);
    { std::ofstream f(temp.path/"weights",std::ios::binary); f.write(reinterpret_cast<const char*>(data.data()),256); }
    for(bool promote:{false,true}) {
        exec::Budget budget(1024*1024); weights::Weights weights(budget,1);
        auto storage=take(weights.create_storage(backend,{{{GGML_TYPE_F32,{4}},4}}, edge0::exec::Purpose::resident_weights));
        std::mutex mutex;std::condition_variable changed;bool blocked=false,released=false;
        std::vector<std::size_t> order;
        weights.before_read=[&](std::size_t region) {
            std::unique_lock<std::mutex> lock(mutex);order.push_back(region);
            if(region==0) { blocked=true;changed.notify_all();changed.wait(lock,[&]{return released;}); }
        };
        auto start=[&](std::size_t region,weights::Priority priority) {
            auto w=take(weights.begin_write(storage,{{region,std::to_string(region),{(temp.path/"weights").string(),region*16,16,{GGML_TYPE_F32,{4}}}}}));
            CHECK(!weights.fill_regions(w,priority));return w;
        };
        auto a=start(0,weights::Priority::background);
        {std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return blocked;});}
        auto b=start(1,weights::Priority::background),c=start(2,weights::Priority::background),d=start(3,weights::Priority::urgent);
        CHECK(!weights.ready(a));CHECK(!weights.ready(b));
        if(promote) CHECK(!weights.promote(c));
        // An already executing job is not moved or duplicated.
        CHECK(!weights.promote(a));
        {std::lock_guard<std::mutex> lock(mutex);released=true;}changed.notify_all();
        // Wait for all reads without promoting additional jobs.
        while(!weights.ready(b)) std::this_thread::yield();
        for(auto w:{a,b,c,d}) CHECK(weights.complete(w));
        CHECK((order==(promote?std::vector<std::size_t>{0,3,2,1}:std::vector<std::size_t>{0,3,1,2})));
        CHECK(!weights.close(storage));
    }
    // await_ready promotes the entire background pool preparation while preserving publication.
    {
        exec::Budget budget(1024*1024);weights::Weights weights(budget,1);
        auto blocker=take(weights.create_storage(backend,{{{GGML_TYPE_F32,{4}},2}}, edge0::exec::Purpose::resident_weights));
        std::mutex mutex;std::condition_variable changed;bool blocked=false,released=false;std::vector<std::size_t> order;
        weights.before_read=[&](std::size_t region) {
            std::unique_lock<std::mutex> lock(mutex);order.push_back(region);
            if(order.size()==1) {blocked=true;changed.notify_all();changed.wait(lock,[&]{return released;});}
        };
        weights::Source source{(temp.path/"weights").string(),0,16,{GGML_TYPE_F32,{4}}};
        auto first=take(weights.begin_write(blocker,{{1,"block",source}}));CHECK(!weights.fill_regions(first));
        {std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return blocked;});}
        auto other=take(weights.begin_write(blocker,{{0,"other",source}}));CHECK(!weights.fill_regions(other,weights::Priority::background));
        weights::PoolDescription desc;desc.slots=1;desc.projections.fill(source.layout);desc.experts.push_back({source,source,source});
        auto pool=take(weights::ExpertPool::create_pools(weights,backend,desc));
        auto pending=take(pool->prepare({0},weights::Priority::background));CHECK(!pool->prepare({0}));
        // Release the worker only after await_ready has promoted its requests.
        weights.after_promote=[&] { {std::lock_guard<std::mutex> lock(mutex);released=true;} changed.notify_all(); };
        auto prepared=take(pool->await_ready(pending));
        weights.after_promote={};
        CHECK(prepared.mapping.at(0)==0 && prepared.bindings[0].size()==1);
        CHECK(weights.complete(first));CHECK(weights.complete(other));
        CHECK(order.size()==5 && order.back()==0 && order[1]==0 && order[2]==1 && order[3]==2);
        CHECK(!pool->close());CHECK(!weights.close(blocker));
    }
    std::cout<<"urgent/background FIFO, promotion, executing jobs and pool await publication passed\n";
}
