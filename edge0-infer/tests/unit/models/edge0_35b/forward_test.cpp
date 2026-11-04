#include "model/loading.h"
#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/compute.h"
#include "tests/support/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <regex>
using namespace edge0;
namespace edge = models::edge0_35b;
using test::take;
bool numerical_failure=false;
struct Session {
    std::shared_ptr<model::LoadedModel> model;
    std::shared_ptr<session::KVCache> kv;
    session::StateStore states;
    Session(std::shared_ptr<model::LoadedModel> m,exec::Budget& budget):model(std::move(m)) {
        kv=take(session::KVCache::create(budget,model->stages.prefill,1024,take(edge::describe_states(model->assets.definition)).kv));
        CHECK(!edge::initialize_session(model,*kv,states,budget,model->stages));
    }
    ~Session() { edge::release_session(model,*kv,states); }
};
struct Output { std::vector<float> logits; std::vector<models::ObservedTensor> observed; };
Output run(Session& s,exec::Budget& budget,const std::vector<std::int32_t>& tokens,int e,bool observe) {
    models::ForwardRequest request{tokens,{e,std::int64_t(tokens.size())},e?exec::Phase::decode:exec::Phase::prefill,{}, {}};
    request.output_positions.resize(tokens.size()); std::iota(request.output_positions.begin(),request.output_positions.end(),0);
    if(observe) request.observe=edge::observation_names(s.model->assets.definition);
    auto result=edge::forward(s.model,request,*s.kv,s.states,budget);
    if(!result) { std::cerr<<result.error().message<<'\n'; std::exit(1); }
    const auto& owner=result.value();
    CHECK(owner->completion().success && owner->completion().written.size()==16);
    CHECK(owner->weight_registrations()==s.model->resident.size());
    std::cout<<"forward registrations="<<owner->weight_registrations()<<" written="<<owner->completion().written.size()<<'\n';
    Output out{take(owner->logits()),take(owner->observations())};
    CHECK(out.logits.size()==tokens.size()*s.model->assets.definition.parameters.at("n_vocab"));
    for(float value:out.logits) CHECK(std::isfinite(value));
    if(observe) {
        for(const auto& name:request.observe) {
            if(name.find("linear_attn_out-")==0) {
                const auto layer=std::stoul(name.substr(std::string("linear_attn_out-").size()));
                const auto& delta=s.model->assets.definition.layer_groups.at("deltanet");
                if(std::find(delta.begin(),delta.end(),layer)==delta.end())continue;
            }
            auto found=std::find_if(out.observed.begin(),out.observed.end(),[&](const auto& item){return item.name==name;});
            CHECK(found!=out.observed.end()&&!found->data.empty());
        }
        for(const auto& item:out.observed) {
            CHECK(item.layout.type==GGML_TYPE_F32||item.layout.type==GGML_TYPE_I32);
            CHECK(!item.layout.dimensions.empty());
            std::size_t expected=ggml_type_size(item.layout.type);
            for(auto dimension:item.layout.dimensions) {
                CHECK(dimension>0&&std::size_t(dimension)<=item.data.size()/expected);
                expected*=std::size_t(dimension);
            }
            CHECK(item.data.size()==expected);
        }
        for(const auto& item:out.observed) if(item.layout.type==GGML_TYPE_F32) {
            CHECK(item.data.size()%sizeof(float)==0);
            for(std::size_t i=0;i<item.data.size();i+=sizeof(float)) {
                float value; std::memcpy(&value,item.data.data()+i,sizeof(value)); CHECK(std::isfinite(value));
            }
        }
    }
    CHECK(!owner->end_observation());
    // A published region accepts an identical reload only after every use ends.
    for(const auto& item:s.model->resident) {
        auto split=item.first.find(':');
        const auto& tensor=s.model->assets.indexes.at(item.first.substr(0,split)).tensors.at(item.first.substr(split+1));
        auto* resident=item.second.storage->tensor(item.second.binding.region);
        std::vector<std::int64_t> dims(resident->ne,resident->ne+3);
        while(dims.size()>1&&dims.back()==1)dims.pop_back();
        auto write=take(s.model->weights.begin_write(item.second.storage,{{item.second.binding.region,item.first,
            {tensor.path,tensor.offset,std::size_t(tensor.bytes),{resident->type,dims}}}}));
        // Destruction without fill restores the original published binding.
    }
    return out;
}
std::vector<float> read(ggml_tensor* t) {
    std::vector<float> out(ggml_nelements(t));
    if(t->type==GGML_TYPE_F32) ggml_backend_tensor_get(t,out.data(),0,out.size()*4);
    else { std::vector<ggml_fp16_t> h(out.size()); ggml_backend_tensor_get(t,h.data(),0,h.size()*2); for(std::size_t i=0;i<h.size();++i)out[i]=ggml_fp16_to_fp32(h[i]); }
    return out;
}
double compare(const std::vector<float>& a,const std::vector<float>& b,double atol,double rtol,bool enforce=true) {
    CHECK(!a.empty()&&a.size()==b.size()); double maximum=0; bool reported=false;
    for(std::size_t i=0;i<a.size();++i) { const auto delta=std::abs(double(a[i])-b[i]); maximum=std::max(maximum,delta); if(!std::isfinite(delta)||(enforce&&delta>atol+rtol*std::abs(b[i]))) { numerical_failure=true; if(!reported) std::cerr<<"mismatch i="<<i<<" a="<<a[i]<<" b="<<b[i]<<" delta="<<delta<<'\n'; reported=true; } }
    return maximum;
}
std::vector<std::int32_t> input(const std::string& file) {
    std::ifstream stream(file); CHECK(stream.good());
    const std::string text((std::istreambuf_iterator<char>(stream)),{});
    auto key=text.find("\"input\""); CHECK(key!=std::string::npos);
    auto begin=text.find('[',key),end=text.find(']',begin); CHECK(end!=std::string::npos);
    std::string values=text.substr(begin+1,end-begin-1); std::vector<std::int32_t> result;
    std::regex number("[0-9]+");
    for(auto it=std::sregex_iterator(values.begin(),values.end(),number);it!=std::sregex_iterator();++it)result.push_back(std::stoi(it->str()));
    return result;
}
void flash_probe(const std::shared_ptr<exec::Backend>& backend) {
    exec::Budget budget(64ULL*1024*1024);
    auto execute=[&](int queries) {
        auto graph=take(exec::create_graph(budget,backend,exec::Phase::prefill));
        auto* q=take(graph->input("q",{GGML_TYPE_F32,{256,queries,1}}));
        auto* k=take(graph->input("k",{GGML_TYPE_F16,{256,256,1}}));
        auto* v=take(graph->input("v",{GGML_TYPE_F16,{256,256,1}}));
        auto* output=ggml_flash_attn_ext(graph->context(),q,k,v,nullptr,1.0f/16,0,0);
        CHECK(ggml_prec_set_acc(output,GGML_PREC_F32)); CHECK(!graph->output(output));
        auto plan=take(exec::build_plan(graph));
        std::vector<float> queries_data(256*queries);
        std::vector<ggml_fp16_t> keys(256*256),values(keys.size());
        for(std::size_t i=0;i<queries_data.size();++i) queries_data[i]=std::sin(float(i%256)*.37f);
        for(std::size_t i=0;i<keys.size();++i) { keys[i]=ggml_fp32_to_fp16(std::sin(float(i)*.13f)); values[i]=ggml_fp32_to_fp16(std::cos(float(i)*.17f)*.1f); }
        CHECK(!plan->set_inputs("q",queries_data.data(),queries_data.size()*4));
        CHECK(!plan->set_inputs("k",keys.data(),keys.size()*2));
        CHECK(!plan->set_inputs("v",values.data(),values.size()*2));
        CHECK(!plan->submit()); CHECK(take(plan->collect_completion()).success);
        std::vector<float> result(256*queries);
        CHECK(!plan->read_output(output,result.data(),0,result.size()*4));
        CHECK(!plan->end_observation()); CHECK(!plan->release_plan());
        result.resize(256); return result;
    };
    auto multi=execute(64),single=execute(1); double maximum=0;
    for(std::size_t i=0;i<multi.size();++i) maximum=std::max(maximum,std::abs(double(multi[i])-single[i]));
    std::cout<<"CPU flash identical Q/K/V, requested GGML_PREC_F32, query counts 64/1: max_abs="<<maximum<<" budget_current="<<budget.current()<<'\n';
    CHECK(budget.current()==0);
}
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv); const bool metal=argc==3;
    if(argc==2 && std::string(argv[1])=="--flash-probe") { flash_probe(backend); return 0; }
    const char* reference=std::getenv("MOE_TINY"); if(!reference) { std::cout<<"SKIP: missing MOE_TINY\n"; return 77; }
    exec::Budget budget(1024ULL*1024*1024);
    for(bool lora:{false,true}) {
        model::LoadRequest req; req.directory=std::string(reference)+"/tiny-a"; req.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle())); req.options={{"routing","actual"},{"lora",lora?"on":"off"},{"experts","resident"}}; req.on_close=edge::close_model;
        auto model=take(model::load(req,{&edge::adapter()},budget,{{req.device,backend}}));
        const auto baseline=budget.current();
        for(const char* name:{"plain-mixed","chat-intro","long-mixed"}) {
            const auto tokens=input(std::string(MOE_SOURCE_ROOT)+"/tests/fixtures/answers/tiny-base-cpu-flash/"+name+"/tokens.json");
            CHECK(tokens.size()>1); const auto vocabulary=model->assets.definition.parameters.at("n_vocab");
            Session whole(model,budget),split(model,budget),silent(model,budget),other(model,budget);
            models::ForwardRequest invalid{{},{0,0},exec::Phase::prefill,{0},{}};
            CHECK(!edge::forward(model,invalid,*whole.kv,whole.states,budget));
            invalid.tokens={1}; invalid.step={1024,1};
            CHECK(!edge::forward(model,invalid,*whole.kv,whole.states,budget));
            std::cout<<"case="<<name<<" lora="<<lora<<'\n';
            const bool preceding_failure=numerical_failure; numerical_failure=false;
            auto full=run(whole,budget,tokens,0,true);
            auto quiet=run(silent,budget,tokens,0,false);
            auto t7=compare(full.logits,quiet.logits,0,0,!metal);
            if(!metal) CHECK(std::memcmp(full.logits.data(),quiet.logits.data(),full.logits.size()*sizeof(float))==0);
            auto first=run(split,budget,{tokens.begin(),tokens.end()-1},0,true);
            auto isolated=run(other,budget,{tokens.begin(),tokens.end()-1},0,true);
            CHECK(first.logits==isolated.logits);
            auto last=run(split,budget,{tokens.back()},int(tokens.size()-1),true);
            auto otherlast=run(other,budget,{tokens.back()},int(tokens.size()-1),true);
            CHECK(last.logits==otherlast.logits);
            std::vector<float> tail(full.logits.end()-vocabulary,full.logits.end());
            const double tolerance=metal?1e-3:1e-4;
            auto combined_data=[&](const std::string& name) {
                auto prefix=std::find_if(first.observed.begin(),first.observed.end(),[&](const auto& x){return x.name==name;});
                auto suffix=std::find_if(last.observed.begin(),last.observed.end(),[&](const auto& x){return x.name==name;});
                CHECK(prefix!=first.observed.end()&&suffix!=last.observed.end());
                auto data=prefix->data; data.insert(data.end(),suffix->data.begin(),suffix->data.end());
                return data;
            };
            int first_route_layer=-1;
            double first_route_gap=0;
            for(const auto& item:full.observed) if(item.name.find("ffn_moe_topk-")==0) {
                const auto combined=combined_data(item.name);
                CHECK(item.data.size()==combined.size());
                if(item.data==combined)continue;
                const auto layer=std::stoi(item.name.substr(std::string("ffn_moe_topk-").size()));
                const auto probabilities=combined_data("ffn_moe_probs-"+std::to_string(layer));
                double maxgap=0; std::size_t changes=0,first_position=0;
                const auto experts=model->assets.definition.parameters.at("n_expert");
                const auto top=model->assets.definition.parameters.at("top_k");
                for(std::size_t j=0;j<combined.size()/4;++j) {
                    std::int32_t whole_id,split_id;
                    std::memcpy(&whole_id,item.data.data()+4*j,4); std::memcpy(&split_id,combined.data()+4*j,4);
                    CHECK(whole_id>=0&&std::uint64_t(whole_id)<experts&&split_id>=0&&std::uint64_t(split_id)<experts);
                    if(whole_id==split_id)continue;
                    if(!changes)first_position=j/top;
                    ++changes; float a,b;
                    const auto offset=(j/top)*experts;
                    std::memcpy(&a,probabilities.data()+4*(offset+whole_id),4);
                    std::memcpy(&b,probabilities.data()+4*(offset+split_id),4);
                    CHECK(std::isfinite(a)&&std::isfinite(b));
                    const auto gap=std::abs(double(a)-b);
                    maxgap=std::max(maxgap,gap);
                    std::cout<<"T6 route position layer="<<layer<<" token="<<j/top<<" rank="<<j%top
                             <<" whole_expert="<<whole_id<<" split_expert="<<split_id
                             <<" whole_expert_split_probability="<<a<<" split_expert_probability="<<b
                             <<" gap="<<gap<<'\n';
                }
                if(first_route_layer<0||layer<first_route_layer) { first_route_layer=layer; first_route_gap=maxgap; }
                std::cout<<"T6 route mismatch "<<item.name<<" positions="<<changes<<" first_token="<<first_position<<" max_probability_gap="<<maxgap<<'\n';
            }
            const bool no_flip=first_route_layer<0;
            std::cout<<"T6 first_route_layer="<<first_route_layer<<" first_route_gap="<<first_route_gap<<" tied="<<(!no_flip&&first_route_gap<=tolerance)<<'\n';
            double layer_max=0;
            for(const auto& obs:full.observed) if(obs.layout.type==GGML_TYPE_F32) {
                auto combined=combined_data(obs.name); CHECK(combined.size()==obs.data.size());
                std::vector<float> a(combined.size()/4),b(a.size());
                std::memcpy(a.data(),combined.data(),combined.size()); std::memcpy(b.data(),obs.data.data(),obs.data.size());
                const bool enforced=false;
                const auto maximum=compare(a,b,0,0,false);
                layer_max=std::max(layer_max,maximum);
                std::cout<<"split layer "<<obs.name<<" max="<<maximum<<" enforced="<<enforced<<'\n';
            }
            auto t6=compare(last.logits,tail,0,0,false);
            std::cout<<"T6 logits_max="<<t6<<" enforced=0"<<'\n';
            for(const auto& decl:take(edge::describe_states(model->assets.definition)).kv) {
                auto a=read(take(whole.kv->storage(decl.name))->tensor()),b=read(take(split.kv->storage(decl.name))->tensor());
                const auto maximum=compare(b,a,0,0,false);
                t6=std::max(t6,maximum);
                std::cout<<"T6 KV "<<decl.name<<" max="<<maximum<<'\n';
            }
            for(auto layer:model->assets.definition.layer_groups.at("deltanet")) {
                auto a=take(whole.states.get<edge::DeltaState>("deltanet."+std::to_string(layer)));
                auto b=take(split.states.get<edge::DeltaState>("deltanet."+std::to_string(layer)));
                CHECK(a&&b&&a->convolution&&b->convolution&&a->recurrent&&b->recurrent);
                const auto convolution=compare(read(b->convolution->tensor()),read(a->convolution->tensor()),0,0,false);
                const auto recurrent=compare(read(b->recurrent->tensor()),read(a->recurrent->tensor()),0,0,false);
                t6=std::max({t6,convolution,recurrent});
                std::cout<<"T6 state deltanet."<<layer<<" convolution_max="<<convolution<<" recurrent_max="<<recurrent<<'\n';
            }
            std::cout<<"T6 layer_max="<<layer_max<<'\n';
            std::cout<<name<<" lora="<<lora<<" T6_max="<<t6<<" T7_max="<<t7<<" T10=identical T6_status="<<(numerical_failure?"failed":"passed")<<'\n';
            numerical_failure|=preceding_failure;
        }
        CHECK(budget.current()==baseline);
        {
            Session held(model,budget); auto retained=model;
            CHECK(!model::request_unload(model));
            models::ForwardRequest r{{1},{0,1},exec::Phase::prefill,{0},{}};
            auto rejected=edge::forward(retained,r,*held.kv,held.states,budget);
            CHECK(!rejected&&rejected.error().category==ErrorCategory::conflict);
        }
        CHECK(budget.current()==0);
    }
    return numerical_failure?1:0;
}
