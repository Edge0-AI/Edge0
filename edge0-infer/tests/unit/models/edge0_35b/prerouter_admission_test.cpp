#include "models/edge0_35b/assets.h"
#include "model/loading.h"
#include "tests/support/model_fixture.h"
#include "tests/support/runtime.h"
using namespace edge0;
using test::take;
int main(int argc,char** argv) {
    CHECK(argc==2);auto backend=test::backend(0,nullptr);const auto& adapter=models::edge0_35b::adapter();
    fixture::Temp temp;fixture::assets(temp.path,adapter,argv[1]);
    auto catalog=take(model::open_catalog(temp.path.string()));
    std::vector<std::pair<std::string,std::string>> files;
    for(const auto& file:catalog.files) files.emplace_back(file.first,file.second.filename);
    auto metadata=fixture::metadata();auto definition=take(adapter.describe_assets(metadata));
    std::vector<model::TensorRequirement> heads;
    for(const auto& tensor:definition.tensors) if(tensor.role=="prerouter") heads.push_back(tensor);
    exec::Budget budget(128*1024*1024);std::atomic<std::size_t> reads{0};
    model::LoadRequest request;request.directory=temp.path.string();request.device="CPU";request.context=4096;
    request.before_read=[&](std::size_t){++reads;};
    for(const auto* fault:{"role","missing","shape","extra","owners","top_k"}) {
        auto tensors=heads;auto meta=metadata.at("prerouter").metadata;auto listed=files;
        std::string diagnostic="prerouter";
        if(std::string(fault)=="role") listed.erase(std::remove_if(listed.begin(),listed.end(),[](const auto& f){return f.first=="prerouter";}),listed.end());
        if(std::string(fault)=="missing") {diagnostic=tensors.back().name;tensors.pop_back();}
        if(std::string(fault)=="shape") {diagnostic=tensors[0].name;++tensors[0].shape[1];}
        if(std::string(fault)=="extra") {auto extra=tensors[0];extra.name="unexpected_prerouter";diagnostic=extra.name;tensors.push_back(extra);}
        if(std::string(fault)=="owners") meta["prerouter.owners"]=std::vector<std::uint32_t>{1};
        if(std::string(fault)=="top_k") meta["prerouter.top_k"]=std::uint32_t(1);
        fixture::gguf(temp.path/"prerouter.gguf",meta,tensors);fixture::catalog(temp.path,listed);
        for(const auto* mode:{"actual","predicted"}) {
            request.options={{"routing",mode}};
            auto rejected=model::load(request,{&adapter},budget,{{"CPU",backend}});
            CHECK(!rejected && rejected.error().category==ErrorCategory::incompatible);
            CHECK(rejected.error().message.find(diagnostic)!=std::string::npos);
            CHECK(reads==0 && budget.current()==0);
        }
    }
    fixture::gguf(temp.path/"prerouter.gguf",metadata.at("prerouter").metadata,heads);fixture::catalog(temp.path,files);
    for(const model::ModelOptions& options:std::vector<model::ModelOptions>{
            {{"routing","predicted"},{"lora","off"}},{{"routing","prefetch"}}}) {
        request.options=options;auto rejected=model::load(request,{&adapter},budget,{{"CPU",backend}});
        CHECK(!rejected && rejected.error().category==ErrorCategory::invalid_argument && reads==0 && budget.current()==0);
    }
    std::size_t actual=0,predicted=0;
    for(const auto* mode:{"actual","predicted"}) {
        request.options={{"routing",mode}};
        auto model=take(model::load(request,{&adapter},budget,{{"CPU",backend}}));
        CHECK(std::find(model->assets.unused_roles.begin(),model->assets.unused_roles.end(),"prerouter")==model->assets.unused_roles.end());
        std::size_t count=0;for(const auto& item:model->resident) if(item.first.rfind("prerouter:",0)==0) ++count;
        CHECK(count==(std::string(mode)=="actual"?0:3));
        if(std::string(mode)=="actual") actual=model->estimate.total_bytes;
        else predicted=model->estimate.total_bytes;
    }
    CHECK(predicted>actual);exec::Budget tight(predicted-1);reads=0;
    request.options={{"routing","predicted"}};
    auto rejected=model::load(request,{&adapter},tight,{{"CPU",backend}});
    CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded && reads==0 && tight.current()==0);
    request.options={{"routing","actual"}};CHECK(model::load(request,{&adapter},tight,{{"CPU",backend}}));
    std::cout<<"mandatory role and tensor/metadata rejection in all modes; zero payload reads; lists and exact-budget admission passed\n";
}
