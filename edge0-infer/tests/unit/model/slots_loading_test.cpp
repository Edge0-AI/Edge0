#include "model/loading.h"
#include "tests/support/model_fixture.h"
#include "tests/support/runtime.h"
using namespace edge0;
using test::take;
struct Fake final:model::Adapter {
    bool slots=true;
    Fake():Adapter({"base"}){}
    std::string id()const override{return "fake";}
    model::TextDescription describe_text()const override { model::Sha256 hash;hash.update("template");return {hash.finish(),"gpt2","qwen35",false,false}; }
    model::TextResult text_format(const std::vector<model::Message>&,const model::TextOptions&)const override{return {};}
    Result<model::ModelDefinition> describe_assets(const model::AssetView&)const override {
        model::ModelDefinition d;
        d.vocabulary_size=4;d.context_length=4096;d.layer_count=1;d.expert_count=1;d.experts_per_token=1;d.required_roles={"base","chat_template"};d.tensors={{"base","a","f32",{{4,1,1,1}}}};return d;
    }
    Result<model::WeightDescription> describe_weight_storage(const model::AssetView& assets,const model::ModelOptions&)const override {
        const auto& tensor=assets.at("base").tensors.at("a");
        weights::Source source{tensor.path,tensor.offset,16,{GGML_TYPE_F32,{4}}};
        model::WeightDescription d;d.resident.push_back({"a","base","a",source});
        if(slots) {
            weights::PoolDescription pool;pool.slots=2;
            for(auto& layout:pool.projections)layout=source.layout;
            pool.experts.resize(4);for(auto& expert:pool.experts)expert.fill(source);
            d.pools.push_back(pool);for(const auto& layout:pool.projections)d.temporary.push_back({layout,4});
        }
        return d;
    }
};
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);Fake fake;fixture::Temp temp;
    model::Metadata metadata;metadata["tokenizer.ggml.model"]=std::string("gpt2");metadata["tokenizer.ggml.pre"]=std::string("qwen35");
    fixture::gguf(temp.path/"base.gguf",metadata,{{"base","a","f32",{{4,1,1,1}}}});
    fixture::write(temp.path/"template","template");fixture::write(temp.path/"generation.json","{}");
    fixture::write(temp.path/"model.json","{\"format_version\":1,\"architecture\":\"fake\"}");
    fixture::catalog(temp.path,{{"base","base.gguf"},{"chat_template","template"},{"generation_config","generation.json"},{"model","model.json"}});
    model::LoadRequest request;request.directory=temp.path.string();request.device="CPU";
    std::size_t reads=0;request.before_read=[&](std::size_t){++reads;};
    {
        exec::Budget budget(1024*1024);
        auto unknown=request;unknown.options["unknown-option"]="value";
        auto rejected=model::load(unknown,{&fake},budget,{{"CPU",backend}});
        CHECK(!rejected && rejected.error().category==ErrorCategory::invalid_argument);
        CHECK(rejected.error().message.find("unknown-option")!=std::string::npos);
        CHECK(reads==0 && budget.current()==0);
    }
    for(bool slots:{false,true}) {
        fake.slots=slots;exec::Budget budget(1024*1024);std::size_t required=0;
        {
            auto loaded=take(model::load(request,{&fake},budget,{{"CPU",backend}})); required=loaded->estimate.total_bytes;
            CHECK(required==budget.current());CHECK(loaded->estimate.resident_bytes>0);
            CHECK((loaded->estimate.pool_bytes>0)==slots && (loaded->estimate.temporary_bytes>0)==slots);
            CHECK(loaded->pools.size()==std::size_t(slots));CHECK(bool(loaded->temporary)==slots);
            if(slots)CHECK(loaded->pools[0]->storage()->state(0)==weights::RegionState::empty);
        }
        CHECK(budget.current()==0);reads=0;exec::Budget insufficient(required-1);
        auto rejected=model::load(request,{&fake},insufficient,{{"CPU",backend}});
        CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded);
        CHECK(reads==0 && insufficient.current()==0);
    }
    std::cout<<"three-part admission, actual allocation and one-byte rejection passed\n";
}
