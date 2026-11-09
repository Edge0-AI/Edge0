#include "models/edge0_35b/compute.h"
#include "models/edge0_35b/assets.h"
#include "model/loading.h"
#include "tests/support/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
using namespace edge0;
using test::take;
namespace edge=edge0::models::edge0_35b;
int main(int argc,char** argv) {
    auto backend=test::backend(argc,argv);
    const auto* reference=std::getenv("MOE_TINY"); if(!reference)return 77;
    exec::Budget budget(1024ULL*1024*1024);
    for(bool lora:{false,true}) {
        model::LoadRequest load;load.directory=std::string(reference)+"/tiny-a";
        load.device=ggml_backend_dev_name(ggml_backend_get_device(backend->handle()));load.options={{"routing","actual"},{"lora",lora?"on":"off"},{"experts","resident"}};
        auto model=take(model::load(load,{&edge::adapter()},budget,{{load.device,backend}}));
        const auto baseline=budget.current();
        auto description=take(edge::describe_states(model->assets.definition));
        auto make=[&] {return take(session::KVCache::create(budget,backend,1024,description.kv));};
        {
            auto a=make(),b=make();session::StateStore sa,sb;
            CHECK(!edge::initialize_session(model,*a,sa,budget, exec::Stages{backend, backend}));CHECK(!edge::initialize_session(model,*b,sb,budget, exec::Stages{backend, backend}));
            std::map<std::string,std::pair<std::shared_ptr<exec::StateStorage>,std::shared_ptr<exec::StateStorage>>> identities;
            for(const auto& name:description.updated_entries) {
                auto state=take(sa.get<edge::DeltaState>(name));identities[name]={state->convolution,state->recurrent};
            }
            std::int64_t e=0;std::size_t builds=0,reuses=0;
            for(int step=-2;step<15;++step) {
                std::vector<std::int32_t> tokens=step==-2?std::vector<std::int32_t>{1,2,3,4}:std::vector<std::int32_t>{5+std::max(step,0)};
                std::vector<std::vector<float>> rows;
                for(int mode=0;mode<2;++mode) {
                    models::ForwardRequest request{tokens,{e,std::int64_t(tokens.size())},step<0?exec::Phase::prefill:exec::Phase::decode,{tokens.size()-1},{}};
                    request.greedy=true;request.plan_mode=mode?models::PlanMode::fresh:models::PlanMode::automatic;
                    auto output=edge::forward(model,request,mode?*b:*a,mode?sb:sa,budget);CHECK(output);
                    const auto& result=output.value();
                    CHECK(result->updated_entries()==description.updated_entries);
                    CHECK(result->weight_registrations()==(lora?255:143));
                    auto choice=take(result->greedy_output());CHECK(result->readback_bytes()==8);
                    rows.push_back(take(result->logits()));
                    CHECK(rows.back()[std::size_t(choice.token)]==*std::max_element(rows.back().begin(),rows.back().end()));
                    double sum=std::accumulate(rows.back().begin(),rows.back().end(),0.0);
                    CHECK(std::abs(sum-choice.sum)<=1e-3*std::max(std::abs(sum),1e-12));
                    if(step>=0&&!mode) {if(result->reused_plan())++reuses;else ++builds;}
                    CHECK(!result->end_observation());
                }
                if(rows[0]!=rows[1]) {
                    double maximum=0;for(std::size_t i=0;i<rows[0].size();++i)maximum=std::max(maximum,std::abs(double(rows[0][i])-rows[1][i]));
                    std::cerr<<"first decode difference step="<<step<<" max_abs="<<maximum<<'\n';return 1;
                }
                e+=std::int64_t(tokens.size());
            }
            CHECK(builds==1&&reuses==14);
            CHECK(!sa.put("aux.future",std::make_shared<int>(1)));
            auto auxiliary=sa.get<int>("aux.decode_plan");CHECK(!auxiliary && auxiliary.error().category==ErrorCategory::conflict);
            CHECK(!edge::reset_session(model,*a,sa));CHECK(!sa.get<int>("aux.decode_plan") && take(sa.get<int>("aux.future")));
            for(const auto& entry:identities) {
                auto state=take(sa.get<edge::DeltaState>(entry.first));
                CHECK(state->convolution==entry.second.first&&state->recurrent==entry.second.second);
                for(const auto& storage:{state->convolution,state->recurrent}) {
                    std::vector<unsigned char> data(ggml_nbytes(storage->tensor()));
                    ggml_backend_tensor_get(storage->tensor(),data.data(),0,data.size());
                    CHECK(std::all_of(data.begin(),data.end(),[](auto byte){return byte==0;}));
                }
            }
            identities.clear();
            models::ForwardRequest request{{1},{0,1},exec::Phase::prefill,{0},{}};
            request.greedy=true;request.retain_logits=false;
            {
                auto result=edge::forward(model,request,*a,sa,budget);CHECK(result);
                take(result.value()->greedy_output());CHECK(result.value()->readback_bytes()==8);
                CHECK(!result.value()->logits());CHECK(!result.value()->end_observation());
            }
            {
                request.step={1,1};request.phase=exec::Phase::decode;
                auto result=edge::forward(model,request,*a,sa,budget);CHECK(result);
                CHECK(result.value()->reused_plan());CHECK(!result.value()->end_observation());
            }
            edge::release_session(model,*a,sa);edge::release_session(model,*b,sb);
            for(const auto& name:description.updated_entries)CHECK(!take(sa.get<models::edge0_35b::DeltaState>(name))&&!take(sb.get<models::edge0_35b::DeltaState>(name)));
            CHECK(!take(sa.get<int>("aux.future")));
            CHECK(!take(sa.get<int>("aux.decode_plan"))&&!take(sb.get<int>("aux.decode_plan")));
        }
        CHECK(budget.current()==baseline);
        model.reset();CHECK(budget.current()==0);
    }
    std::cout<<"device greedy, state reports/reset, 1 build + 14 reuse, bitwise decode and release passed\n";
}
