#include "weights/storage.h"
#include "tests/support/runtime.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <numeric>
using namespace edge0;
using test::take;
struct Files {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/("moe-batch-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<unsigned char> bytes;
    explicit Files(std::size_t size):bytes(size) {
        std::filesystem::create_directory(directory);
        for(std::size_t i=0;i<size;++i) bytes[i]=static_cast<unsigned char>(i%251);
        for(const auto* name:{"a","b"}) { std::ofstream f(directory/name,std::ios::binary);f.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())); }
    }
    ~Files(){std::filesystem::remove_all(directory);}
};
struct HookScope {
    platform::file_testing::Hooks hooks;
    HookScope(){platform::file_testing::hooks=&hooks;}
    ~HookScope(){platform::file_testing::hooks=nullptr;}
};
std::pair<std::size_t,std::size_t> group(const std::shared_ptr<exec::Backend>& backend,std::size_t count,std::size_t width,
    std::size_t gap,std::size_t jobs,bool different_file=false,bool different_mode=false,bool fail=false) {
    Files files(count*(width+gap));
    exec::Budget budget(32*1024*1024);weights::Weights weights(budget,1);
    auto storage=take(weights.create_storage(backend,{{{GGML_TYPE_I8,{static_cast<std::int64_t>(width)}},count}}, edge0::exec::Purpose::resident_weights,weights::Protection::group));
    std::vector<weights::LoadItem> items;
    for(std::size_t i=0;i<count;++i) {
        weights::Source source{(files.directory/(different_file && i==1?"b":"a")).string(),i*(width+gap),width,{GGML_TYPE_I8,{static_cast<std::int64_t>(width)}}};
        if(different_mode && i==1) source.read_mode=weights::ReadMode::direct;
        items.push_back({i,"item-"+std::to_string(i),source});
    }
    HookScope scope;
    if(fail) { scope.hooks.read_limit=width;scope.hooks.fail_read_call=2; }
    auto write=take(weights.begin_write(storage,items));CHECK(!weights.fill_regions(write));
    auto result=weights.complete(write);
    if(fail) {
        CHECK(!result && result.error().category==ErrorCategory::backend_error);
        for(std::size_t i=0;i<count;++i) CHECK(storage->state(i)==weights::RegionState::isolated);
    } else {
        CHECK(result);CHECK(scope.hooks.vector_calls==jobs);
        for(std::size_t i=0;i<count;++i) {
            CHECK(storage->state(i)==weights::RegionState::published);
            CHECK(std::memcmp(storage->tensor(i)->data,files.bytes.data()+i*(width+gap),width)==0);
        }
    }
    CHECK(scope.hooks.max_segments<=64 && scope.hooks.max_bytes<=4*1024*1024);
    const auto calls=std::make_pair(scope.hooks.opens.load(),scope.hooks.identities.load());
    if(!fail) {
        std::vector<std::size_t> regions(count);std::iota(regions.begin(),regions.end(),0);
        CHECK(!weights.evict(storage,regions));
        const auto before=scope.hooks.opens.load();
        auto again=take(weights.begin_write(storage,items));CHECK(!weights.fill_regions(again));take(weights.complete(again));
        CHECK(scope.hooks.opens-before==((different_file||different_mode)?2:1));
    }
    CHECK(!weights.close(storage));
    return calls;
}
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    const auto one=group(backend,1,16,0,1);
    CHECK(one==group(backend,8,16,0,1));CHECK(one==group(backend,64,16,0,1));
    group(backend,65,16,0,2);
    group(backend,5,1024*1024,0,2);
    group(backend,1,5*1024*1024,0,2);
    group(backend,4,16,16,4);
    group(backend,3,16,0,3,true);
    if(weights::supports_direct_read()) group(backend,3,16,0,3,false,true);
    group(backend,3,16,0,1,false,false,true);
    {
        Files files(29);auto file=take(platform::ReadFile::open((files.directory/"a").string()));
        auto identity=take(file->identity());std::vector<unsigned char> a(5),b(13),c(11);
        HookScope scope;scope.hooks.read_limit=3;
        CHECK(!file->read_vector(0,{{a.data(),a.size()},{b.data(),b.size()},{c.data(),c.size()}},identity));
        std::vector<unsigned char> all=a;all.insert(all.end(),b.begin(),b.end());all.insert(all.end(),c.begin(),c.end());
        CHECK(all==files.bytes && scope.hooks.vector_calls==10);
        CHECK(file->read_vector(1,{{all.data(),all.size()}},identity));
    }
    std::cout<<"file identity/open counts independent of k; cache reuse; vector adjacency, caps, short reads and group isolation passed\n";
}
