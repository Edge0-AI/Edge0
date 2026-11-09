#include "edge0/diagnostics.h"

#include "app/generation_internal.h"
#include "platform/system.h"
#include "app/runtime_hooks.h"
#include "models/edge0_35b/compute.h"
#include "tests/support/runtime.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
using namespace edge0;
// Throwing assertions unwind backend-owned resources before process teardown.
#undef CHECK
#define CHECK(expression) do { if (!(expression)) throw std::runtime_error(std::string("check failed: ") + #expression); } while(false)
template<class T> T take(const Result<T>& value) {
    if(!value) throw std::runtime_error(value.error().message);
    return value.value();
}
struct RuntimeCleanup {
    RuntimeReference runtime;
    ModelReference model;
    SessionReference session;
    ~RuntimeCleanup() {
        if(session) {release_session(session);session.reset();}
        if(model) {unload_model(model);model.reset();}
        const auto memory=memory_snapshot(runtime);
        if(memory) std::cerr<<"cleanup current_bytes="<<memory.value().ledger.current_bytes<<'\n';
        finish_runtime_use(runtime);
    }
};
std::vector<GenerationCheckResult> run(const std::string& directory, const std::string& family, const std::string& metal, int mode) {
    auto runtime=take(open_runtime({mode?std::vector<std::string>{"CPU",metal}:std::vector<std::string>{metal},{}}));
    RuntimeCleanup cleanup{runtime,{}, {}};
    if(mode==2) {
        auto hooks=take(app::runtime_hooks(runtime));
        hooks->consumption=[](auto,auto) {return exec::Consumption::copy;};
    }
    LoadOverrides load;load.device=mode?"CPU":metal;load.prefill_device=metal;
    load.context=128;load.prefill_chunk=128;load.model_options={{"routing","actual"},{"prefill_ahead_min_tokens","1"}};
    auto& model=cleanup.model;model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory,load))));
    auto& session=cleanup.session;session=take(create_session(model));
    GenerationOverrides generation;generation.temperature=0;generation.presence_penalty=0;generation.max_tokens=4;
    auto config=take(resolve_generation_configuration(model,generation));
    GenerationCheckOptions check;check.retain_logits=true;check.ignore_end_of_generation=true;
    std::vector<GenerationCheckResult> results;
    std::vector<std::int32_t> input{1,2,3,4,5};
    for(int i=0;i<3;++i) {
        auto value=take(generate_tokens_check(session,input,config,{},check));
        if(value.result.error)std::cerr<<value.result.error->message<<'\n';
        CHECK(value.result.stop_reason==StopReason::length);
        results.push_back(value);
        input=take(query_session_check(session)).tokens;input.push_back(6+i);input.push_back(9+i);
    }
    auto append=[&](const std::vector<std::int32_t>& tokens,const GenerationCheckOptions& options) {
        auto result=take(generate_tokens_check(session,tokens,config,{},options));
        CHECK(!result.result.error);results.push_back(result);return result;
    };
    append(input,check);append(input,check);
    auto cancelled=check;cancelled.cancel_probe=[](const CancelPoint& point){return point.kind==CancelPointKind::before_step && point.token==2;};
    input=take(query_session_check(session)).tokens;input.push_back(11);input.push_back(12);
    CHECK(append(input,cancelled).result.stop_reason==StopReason::cancelled);
    append(input,check);
    input=take(query_session_check(session)).tokens;
    input.insert(input.end(),{15,16,17,18,19});
    auto block_cancel=check;
    block_cancel.cancel_probe=[](const CancelPoint& point){return point.kind==CancelPointKind::in_block && point.segment==2;};
    const auto rolled_back=append(input,block_cancel);
    CHECK(rolled_back.result.stop_reason==StopReason::cancelled && rolled_back.result.rolled_back_tokens>0);
    append(input,check);
    take(reset_conversation(session));
    const auto reset=append({1,2,3,4,5},check);
    CHECK(!reset.result.text.empty());
    GenerationOverrides stopping;stopping.stop=std::vector<std::string>{reset.result.text.substr(0,1)};
    take(reset_conversation(session));
    auto stopped=take(generate_tokens_check(session,{1,2,3,4,5},config,stopping,check));
    CHECK(stopped.result.stop_reason==StopReason::stop_string);results.push_back(stopped);
    input=take(query_session_check(session)).tokens;input.push_back(13);input.push_back(14);append(input,check);
    if(mode==2) {
        const auto ledger = take(memory_snapshot(runtime)).ledger;
        CHECK(ledger.by_device.at(metal).current_bytes > 0);
        // Metal unified memory is always system. Other backends follow the
        // device: a discrete Vulkan GPU is dedicated, a unified one is system.
        const auto expected = family=="metal" ? MemoryLocation::system : platform::device_memory(metal);
        CHECK(ledger.by_device.at(metal).memory == expected);
        auto& kv=session->state->kv();
        const auto description=take(model->compute->describe_states(model->loaded->assets.definition));
        CHECK(!kv.make_current(model->loaded->stages.prefill,session->state->query().e));
        CHECK(!model->compute->hand_off(model->loaded,session->state->states(),model->loaded->stages.prefill));
        for(auto layer:model->loaded->assets.definition.layer_groups.at("deltanet")) {
            const auto state=take(session->state->states().get<models::edge0_35b::DeltaState>("deltanet."+std::to_string(layer)));
            for(const auto& storage:{state->convolution,state->recurrent}) {
                const auto a=take(storage->on(model->loaded->stages.prefill)),b=take(storage->on(model->loaded->stages.decode));
                std::vector<unsigned char> left(ggml_nbytes(a->tensor())),right(left.size());
                ggml_backend_tensor_get(a->tensor(),left.data(),0,left.size());
                ggml_backend_tensor_get(b->tensor(),right.data(),0,right.size());
                CHECK(left==right);
            }
        }
        bool all_kv_equal=true;
        for(const auto& entry:description.kv) {
            const auto state=take(kv.storage(entry.name));
            const auto a=take(state->on(model->loaded->stages.prefill)),b=take(state->on(model->loaded->stages.decode));
            CHECK(a->allocation != b->allocation);
            CHECK(a->allocation->charges.front()->purpose() == exec::Purpose::session_state);
            CHECK(exec::buffer_device(ggml_backend_buffer_get_type(a->allocation->buffer)) == metal);
            CHECK(ledger.by_device.at(metal).current_bytes >= a->allocation->charges.front()->size());
            std::vector<unsigned char> left(ggml_nbytes(a->tensor())),right(left.size());
            ggml_backend_tensor_get(a->tensor(),left.data(),0,left.size());
            ggml_backend_tensor_get(b->tensor(),right.data(),0,right.size());
            const auto e=session->state->query().e;
            std::size_t first=left.size(),valid_differences=0,total_differences=0;
            for(std::size_t offset=0;offset<left.size();++offset) if(left[offset]!=right[offset]) {
                if(first==left.size()) first=offset;
                ++total_differences;
                const auto row=(offset/a->tensor()->nb[entry.token_axis])%static_cast<std::size_t>(a->tensor()->ne[entry.token_axis]);
                if(row<static_cast<std::size_t>(e)) ++valid_differences;
            }
            std::cerr<<"KV "<<entry.name<<" first_offset="<<first
                <<" row="<<(first==left.size()?-1:static_cast<std::int64_t>((first/a->tensor()->nb[entry.token_axis])%a->tensor()->ne[entry.token_axis]))
                <<" e="<<e<<" synced_prefill="<<take(kv.synced(entry.name,model->loaded->stages.prefill))
                <<" synced_decode="<<take(kv.synced(entry.name,model->loaded->stages.decode))
                <<" different_bytes="<<total_differences<<" valid_different_bytes="<<valid_differences<<'\n';
            const auto row_bytes=a->tensor()->nb[entry.token_axis];
            const auto plane_bytes=row_bytes*static_cast<std::size_t>(a->tensor()->ne[entry.token_axis]);
            const auto planes=ggml_nbytes(a->tensor())/plane_bytes;
            std::vector<unsigned char> a_row(row_bytes),b_row(row_bytes);
            for(std::size_t plane=0;plane<planes;++plane) for(std::int64_t row=0;row<e;++row) {
                const auto offset=plane*plane_bytes+static_cast<std::size_t>(row)*row_bytes;
                ggml_backend_tensor_get(a->tensor(),a_row.data(),offset,row_bytes);
                ggml_backend_tensor_get(b->tensor(),b_row.data(),offset,row_bytes);
                all_kv_equal=all_kv_equal && a_row==b_row;
            }
        }
        CHECK(all_kv_equal);
    }
    return results;
}
int main(int argc, char** argv) try {
    const auto* tiny=std::getenv("MOE_TINY");if(!tiny)return 77;
    const std::string family=argc>1?argv[1]:"metal";
    std::string device;
    for(const auto& candidate:query_capabilities().devices)if(candidate.backend==family){device=candidate.name;break;}
    if(device.empty())return 77;
    const auto directory=std::string(tiny)+"/tiny-a";
    const auto metal=run(directory,family,device,0),shared=run(directory,family,device,1),copied=run(directory,family,device,2);
    CHECK(shared.size()==copied.size());
    for(std::size_t i=0;i<shared.size();++i) {
        CHECK(shared[i].result.generated==copied[i].result.generated);
        CHECK(shared[i].logits.size()==copied[i].logits.size());
        CHECK(std::memcmp(shared[i].logits.data(),copied[i].logits.data(),shared[i].logits.size()*sizeof(float))==0);
    }
    CHECK(shared[0].vocabulary==metal[0].vocabulary);
    CHECK(std::memcmp(shared[0].logits.data(),metal[0].logits.data(),metal[0].vocabulary*sizeof(float))==0);
    std::cout<<"mixed "<<family<<": natural/copy logits and tokens, initial prefill and transferred KV matched bitwise\n";
}

catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
