#include "weights/storage.h"
#include "tests/support/runtime.h"
#include <chrono>
#include <fstream>
#include <filesystem>
using namespace edge0;
using namespace edge0::weights;
using test::take;
using namespace std::chrono_literals;
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    exec::Budget budget(64*1024*1024);
    auto path=std::filesystem::current_path()/"build"/("lifecycle-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    float data[4]{1,2,3,4};
    { std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<char*>(data),sizeof(data)); }
    {
        Weights w(budget);
        auto s=take(w.create_storage(backend,{{{GGML_TYPE_F32,{4}},3}}, edge0::exec::Purpose::resident_weights,Protection::group));
        auto metadata=std::make_shared<int>(42);
        auto item=[&](std::size_t r,std::string id) { return LoadItem{r,std::move(id),{path.string(),0,sizeof(data),{GGML_TYPE_F32,{4}}},metadata}; };
        auto publish=[&](std::vector<LoadItem> items) {
            auto tx=take(w.begin_write(s,items)); CHECK(!w.fill_regions(tx)); return take(w.complete(tx));
        };
        auto b=publish({item(0,"a"),item(1,"b")});
        CHECK(take(w.query_state("a")).metadata==metadata);
        auto tx=take(w.begin_write(s,{item(0,"c")}));
        CHECK(take(w.query_state("c")).pending==tx);
        CHECK(take(w.query_state("c")).state==RegionState::writing);
        CHECK(w.evict(s,{0})); CHECK(!w.acquire_views(s,0,b));
        CHECK(!w.abandon(tx)); CHECK(take(w.query_state("a")).binding.generation==b[0].generation);
        auto view=take(w.acquire_views(s,0,b));
        CHECK(view.tensor->ne[1]==3); CHECK(!w.begin_write(s,{item(2,"d")})); CHECK(w.evict(s,{1}));
        CHECK(take(w.query_state(s,0)).uses==1); CHECK(!w.release_use(view.use));
        auto stale=b; b[1]=publish({item(1,"b")})[0];
        CHECK(!w.acquire_views(s,0,stale)); CHECK(take(w.query_state(s,0)).uses==0);
        tx=take(w.begin_write(s,{item(2,"d")}));
        auto waiting=std::async(std::launch::async,[&] { return w.await_write(take(w.query_state("d")).pending); });
        CHECK(waiting.wait_for(20ms)==std::future_status::timeout);
        CHECK(!w.fill_regions(tx)); auto ready=take(w.complete(tx));
        CHECK(take(waiting.get())[0].id==ready[0].id);
        CHECK(!w.evict(s,{2})); CHECK(!w.query_state("d"));
        // Restore only if the old ID is still unclaimed.
        auto other=take(w.create_storage(backend,{{{GGML_TYPE_F32,{4}},1}}, edge0::exec::Purpose::resident_weights));
        tx=take(w.begin_write(s,{item(0,"temporary")}));
        auto second=take(w.begin_write(other,{item(0,"a")})); CHECK(!w.fill_regions(second)); take(w.complete(second));
        CHECK(!w.abandon(tx)); CHECK(take(w.query_state(s,0)).state==RegionState::empty);
        CHECK(!w.close(other));
        // Two independent ranges: one fails while the other still owns its range.
        tx=take(w.begin_write(s,{item(0,"bad")}));
        auto first=take(w.writable_regions(tx,0,0,8));
        CHECK(!w.writable_regions(tx,0,4,8)); CHECK(!w.writable_regions(tx,0,16,1));
        auto last=take(w.writable_regions(tx,0,8,8));
        CHECK(first->write(7,data,2)); CHECK(!first->write(0,data,8));
        CHECK(!first->finish(Error{ErrorCategory::backend_error,"injected"}));
        auto completion=std::async(std::launch::async,[&] { return w.complete(tx); });
        CHECK(completion.wait_for(20ms)==std::future_status::timeout);
        CHECK(!last->write(0,data+2,8)); CHECK(!last->finish()); CHECK(!completion.get());
        CHECK(take(w.query_state(s,0)).state==RegionState::isolated);
        CHECK(w.evict(s,{0})); CHECK(!w.begin_write(s,{item(0,"again")})); CHECK(!w.acquire_views(s,0,{b[1]}));
        CHECK(!w.close(s)); tx.reset(); first.reset(); last.reset(); second.reset();
        // Abandon isolates only regions whose writers actually acquired access.
        auto partial=take(w.create_storage(backend,{{{GGML_TYPE_F32,{4}},2}}, edge0::exec::Purpose::resident_weights));
        tx=take(w.begin_write(partial,{item(0,"partial-a"),item(1,"partial-b")}));
        CHECK(!w.fill_regions(tx)); auto originals=take(w.complete(tx));
        tx=take(w.begin_write(partial,{item(0,"partial-c"),item(1,"partial-d")}));
        auto partial_writer=take(w.writable_regions(tx,0,0,16));
        CHECK(!partial_writer->write(0,data,16)); CHECK(!partial_writer->finish()); CHECK(!w.abandon(tx));
        CHECK(partial->state(0)==RegionState::isolated && partial->state(1)==RegionState::published);
        CHECK(take(w.query_state("partial-b")).binding.generation==originals[1].generation);
        CHECK(!w.close(partial));
        // Successful custom writers publish exactly one generation.
        auto target=take(w.create_storage(backend,{{{GGML_TYPE_F32,{4}},1}}, edge0::exec::Purpose::resident_weights));
        tx=take(w.begin_write(target,{item(0,"custom")}));
        auto writer=take(w.writable_regions(tx,0,0,16)); CHECK(!writer->write(0,data,16)); CHECK(!writer->finish());
        auto binding=take(w.complete(tx)); CHECK(binding[0].generation==1);
        view=take(w.acquire_views(target,binding[0]));
        auto closing=std::async(std::launch::async,[&] { return w.close(target); });
        CHECK(closing.wait_for(20ms)==std::future_status::timeout);
        CHECK(!w.release_use(view.use)); CHECK(!closing.get());
    }
    CHECK(budget.current()==0); std::filesystem::remove(path);
    std::cout<<"weights lifecycle: queries, rollback, bounded writers, publication, group views and close passed\n";
}
