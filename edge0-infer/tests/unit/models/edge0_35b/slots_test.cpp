#include "models/edge0_35b/compute.h"
#include "models/edge0_35b/assets.h"
#include "model/loading.h"
#include "tests/support/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>
using namespace edge0;
using test::take;
namespace edge=models::edge0_35b;
std::vector<std::int32_t> input_tokens(const std::string& path) {
    std::ifstream f(path); CHECK(f.good()); std::string text((std::istreambuf_iterator<char>(f)),{});
    auto start=text.find('[',text.find("\"input\"")),end=text.find(']',start); CHECK(end!=std::string::npos);
    text=text.substr(start+1,end-start-1); std::regex number("[0-9]+"); std::vector<std::int32_t> tokens;
    for(auto it=std::sregex_iterator(text.begin(),text.end(),number);it!=std::sregex_iterator();++it)tokens.push_back(std::stoi(it->str()));
    return tokens;
}
struct Output { std::vector<std::int32_t> tokens; std::vector<float> logits; WeightActivity activity; };
Output run(const std::shared_ptr<exec::Backend>& backend,const std::string& directory,const std::vector<std::int32_t>& input,
    bool lora,int mode) {
    exec::Budget budget(1024ULL*1024*1024); Output out;
    {
        model::LoadRequest load; load.directory=directory;
        load.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle()));
        load.options={{"routing","actual"},{"lora",lora?"on":"off"},{"prefill_read","used"}};
        if(mode==1) load.options["expert_slots"]="16";
        if(mode==2) load.options["prefill_read"]="layer";
        if(mode==3) load.options["experts"]="resident";
        if(mode>=5) {
            load.options["prefill_read"]="auto";
            load.options["prefill_ahead_min_tokens"]=std::to_string(mode==5?input.size()-1:mode==6?input.size():1);
        }
        auto model=take(model::load(load,{&edge::adapter()},budget,{{load.device,backend}}));
        auto description=take(edge::describe_states(model->assets.definition));
        auto kv=take(session::KVCache::create(budget,backend,1024,description.kv)); session::StateStore states;
        CHECK(!edge::initialize_session(model,*kv,states,budget, exec::Stages{backend, backend}));
        std::int64_t e=0; std::size_t builds=0,reuses=0;
        auto forward=[&](const std::vector<std::int32_t>& tokens,bool sample,exec::Phase phase) {
            models::ForwardRequest r{tokens,{e,std::int64_t(tokens.size())},phase,{tokens.size()-1},{}};
            r.greedy=sample; if(mode==4) r.plan_mode=models::PlanMode::fresh;
            if(tokens.size()>1 && mode!=3) for(int layer=0;layer<8;++layer)r.observe.push_back("ffn_moe_topk-"+std::to_string(layer));
            const auto reads_before=model->weights.read_count();
            auto result=edge::forward(model,r,*kv,states,budget); if(!result) {std::cerr<<"mode="<<mode<<" e="<<e<<" "<<result.error().message<<'\n';std::exit(1);}
            auto& owner=result.value(); CHECK(owner->completion().success); CHECK(owner->weight_registrations()==(lora?255:143));
            auto activity=owner->weight_activity(); out.activity+=activity;
            CHECK(model->weights.read_count()-reads_before==3*(activity.decode.loads+activity.prefill.loads));
            if(tokens.size()>1 && mode!=3) {
                std::size_t demanded=0;
                for(const auto& observed:take(owner->observations())) {
                    std::set<std::int32_t> unique;
                    for(std::size_t i=0;i<observed.data.size();i+=4) {std::int32_t id;std::memcpy(&id,observed.data.data()+i,4);unique.insert(id);}
                    demanded+=unique.size();
                }
                CHECK(activity.prefill.demand==demanded);
                CHECK(activity.prefill.loads==((mode==2 || mode==5 || mode==7)?8*16:demanded)); CHECK(activity.prefill.bytes==activity.prefill.loads*7680);
            }
            if(tokens.size()==1 && mode!=3) {
                CHECK(activity.decode.demand==32 && activity.decode.hits+activity.decode.loads==32);
                CHECK(activity.decode.bytes==activity.decode.loads*7680);
                if(out.tokens.empty()) CHECK(activity.decode.loads==32);
            }
            if(sample) {
                auto choice=take(owner->greedy_output()); out.tokens.push_back(choice.token);
                CHECK(owner->readback_bytes()==(mode==3?8:136));
                auto logits=take(owner->logits()); out.logits.insert(out.logits.end(),logits.begin(),logits.end());
            }
            if(phase==exec::Phase::decode) {if(owner->reused_plan())++reuses;else ++builds;}
            CHECK(!owner->end_observation()); e+=std::int64_t(tokens.size());
        };
        CHECK(input.size()>1);
        forward({input.begin(),input.end()-1},false,exec::Phase::prefill);
        forward({input.back()},true,exec::Phase::prefill);
        for(int i=0;i<15;++i) forward({out.tokens.back()},true,exec::Phase::decode);
        if(mode!=4) CHECK(builds==1 && reuses==14); else CHECK(builds==15 && reuses==0);
        if(mode==1) for(const auto& pool:model->pools) CHECK(pool->activity().loads<=16);
        edge::release_session(model,*kv,states);
    }
    CHECK(budget.current()==0); return out;
}
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv); const auto* root=std::getenv("MOE_TINY"); if(!root)return 77;
    const bool metal=argc>1;
    for(bool lora:{false,true}) for(const auto* name:{"chat-intro","long-mixed","plain-mixed"}) {
        const auto role=lora?"lora":"base";
        auto input=input_tokens(std::string(MOE_SOURCE_ROOT)+"/tests/fixtures/answers/tiny-"+role+"-cpu-flash-split/"+name+"/tokens.json");
        auto baseline=run(backend,std::string(root)+"/tiny-a",input,lora,0);
        for(int mode=1;mode<=7;++mode) {
            auto output=run(backend,std::string(root)+"/tiny-a",input,lora,mode);
            CHECK(output.tokens==baseline.tokens); CHECK(output.logits.size()==baseline.logits.size());
            double maximum=0; bool same=std::memcmp(output.logits.data(),baseline.logits.data(),output.logits.size()*sizeof(float))==0;
            for(std::size_t i=0;i<output.logits.size();++i) {CHECK(std::isfinite(output.logits[i])&&std::isfinite(baseline.logits[i]));maximum=std::max(maximum,std::abs(double(output.logits[i])-baseline.logits[i]));}
            std::cout<<role<<' '<<name<<" mode="<<mode<<" max_abs="<<maximum<<" bitwise="<<same<<std::endl;
            if(mode==3 && metal) CHECK(maximum<=1e-3); else CHECK(same);
        }
        const auto& a=baseline.activity.decode;
        std::cout<<role<<' '<<name<<" demand="<<a.demand<<" hits="<<a.hits<<" loads="<<a.loads<<" bytes="<<a.bytes<<" hit_rate="<<double(a.hits)/double(a.demand)<<std::endl;
    }
}
