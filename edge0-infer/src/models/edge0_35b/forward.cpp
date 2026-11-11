#include "models/edge0_35b/compute.h"
#include "models/edge0_35b/assets.h"
#include "model/loading.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <fstream>
#include <iomanip>
#include <exception>

namespace edge0::models::edge0_35b {
namespace {
template<class T> T take(Result<T> result) {
    if (!result) throw result.error();
    return std::move(result.value());
}
void check(exec::Status status) { if (status) throw *status; }
struct Timing {
    const ForwardRequest& request;
    bool enabled;
    DecodeTiming values;
    std::chrono::steady_clock::time_point started;
    explicit Timing(const ForwardRequest& r) : request(r), enabled(r.breakdown && r.phase==exec::Phase::decode), started(stamp()) {}
    std::chrono::steady_clock::time_point stamp() const { return enabled?request.clock_now():std::chrono::steady_clock::time_point{}; }
    void add(double& field, std::chrono::steady_clock::time_point from) const { if(enabled) field+=std::chrono::duration<double>(stamp()-from).count(); }
    std::optional<DecodeTiming> finish() { if(!enabled) return {}; add(values.total_seconds,started); return values; }
};
struct DecodePlan {
    std::shared_ptr<exec::Plan> plan;
    ggml_tensor* output = nullptr;
    GreedyTensors greedy{};
    std::vector<std::pair<int,ggml_tensor*>> resident_routes;
    std::vector<ggml_tensor*> ranges;
    std::map<std::string, ggml_tensor*> weights;
};
const std::shared_ptr<exec::Backend>& execution_backend(const std::shared_ptr<model::LoadedModel>& model,
    const ForwardRequest& request) {
    if (request.backend) {
        if (request.backend != model->stages.prefill && request.backend != model->stages.decode)
            throw Error{ErrorCategory::invalid_argument, "Forward backend is outside model stages"};
        return request.backend;
    }
    if (model->stages.mixed()) throw Error{ErrorCategory::invalid_argument, "Mixed forward requires a backend"};
    return model->stages.prefill;
}
struct ResultOwner final : ForwardResult {
    std::shared_ptr<model::LoadedModel> model;
    std::shared_ptr<exec::Plan> plan;
    exec::Completion completed;
    ggml_tensor* output = nullptr;
    std::vector<std::pair<std::string, ggml_tensor*>> observed;
    std::size_t registrations = 0, bytes_read = 0;
    std::vector<std::string> updated;
    GreedyTensors greedy{};
    bool persistent = false, reused = false, allow_logits = true;
    WeightActivity activity;
    std::optional<DecodeTiming> timing;
    std::optional<DecodeTiming> decode_timing() const override { return timing; }
    std::vector<ObservedTensor> captured;
    bool clear_temporary = false;
    std::shared_ptr<exec::Backend> backend;
    std::function<exec::Status()> after_observation;
    WeightActivity weight_activity() const override { return activity; }
    const std::vector<std::string>& updated_entries() const override { return updated; }
    std::size_t readback_bytes() const override { return bytes_read; }
    bool reused_plan() const override { return reused; }
    Result<GreedyOutput> greedy_output() override {
        if (!plan || !greedy.token) return Error{ErrorCategory::conflict, "Greedy output is unavailable"};
        GreedyOutput value{};
        if (auto e = plan->read_output(greedy.token, &value.token, 0, sizeof(value.token))) return *e;
        if (auto e = plan->read_output(greedy.sum, &value.sum, 0, sizeof(value.sum))) return *e;
        bytes_read += sizeof(value.token) + sizeof(value.sum);
        return value;
    }
    ~ResultOwner() override {
        if(std::uncaught_exceptions()) after_observation={};
        end_observation();
    }
    const exec::Completion& completion() const override { return completed; }
    std::size_t weight_registrations() const override { return registrations; }
    Result<std::vector<float>> logits() override {
        if (!plan || !allow_logits) return Error{ErrorCategory::conflict, "Logits output is unavailable"};
        std::vector<float> data(ggml_nelements(output));
        if (auto e = plan->read_output(output, data.data(), 0, data.size()*sizeof(float))) return *e;
        bytes_read += data.size()*sizeof(float);
        return data;
    }
    Result<std::vector<ObservedTensor>> observations() override {
        if (!plan) return Error{ErrorCategory::conflict, "Observation has ended"};
        std::vector<ObservedTensor> result=captured;
        for (auto& entry : observed) {
            auto* t = entry.second;
            ObservedTensor item{entry.first, {t->type, {t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}, {}};
            item.data.resize(ggml_nbytes(t));
            if (auto e = plan->read_output(t, item.data.data(), 0, item.data.size())) return *e;
            result.push_back(std::move(item));
        }
        return result;
    }
    exec::Status end_observation() override {
        if (!plan) return {};
        if (auto e = plan->end_observation()) return e;
        if (!persistent) if (auto e = plan->release_plan()) return e;
        plan.reset();
        completed.written.clear();
        if (clear_temporary) {
            const auto count=model->assets.definition.parameters.at("n_expert")*3;
            for(const auto& bank:{model->temporary_on(backend,false),model->temporary_on(backend,true)}) if(bank) {
                std::vector<std::size_t> regions;
                for(std::size_t i=0;i<count;++i) if(bank->state(i)==weights::RegionState::published) regions.push_back(i);
                if(auto e=model->weights.evict(bank,regions)) return e;
            }
            clear_temporary=false;
        }
        if(after_observation) { auto callback=std::move(after_observation); return callback(); }
        return {};
    }
};
struct HeadOutput { int layer; ggml_tensor *ids, *scores, *rank; };
struct Builder {
    std::shared_ptr<model::LoadedModel> model;
    std::shared_ptr<exec::Graph> graph;
    ggml_context* c;
    const ForwardRequest& request;
    session::KVCache& kv;
    session::StateStore& states;
    std::map<std::string, ggml_tensor*> bound;
    std::vector<std::pair<std::string,ggml_tensor*>> observed;
    std::int64_t n;
    float epsilon;
    ggml_tensor *positions, *mask;
    ggml_tensor* rows = nullptr;
    std::vector<ggml_tensor*> ranges;
    std::vector<std::string> updated;
    std::vector<HeadOutput> heads = {};
    std::vector<std::pair<int,ggml_tensor*>> trace_routes = {};
    ggml_tensor* identity = nullptr;
    std::int64_t p(const char* key) { return model->assets.definition.parameters.at(key); }
    ggml_tensor* weight(const std::string& name, const std::string& role="base") {
        const auto id=role+":"+name;
        auto found=bound.find(id); if(found!=bound.end()) return found->second;
        if(!model->resident_on(execution_backend(model,request)).count(id)) throw Error{ErrorCategory::incompatible,"Missing forward weight: "+id};
        const auto& entry=model->resident_on(execution_backend(model,request)).at(id);
        auto view=take(model->weights.acquire_views(entry.storage,entry.binding,execution_backend(model,request)));
        check(graph->bind_weights(view.tensor,execution_backend(model,request),view.use));
        bound.emplace(id,view.tensor);
        return view.tensor;
    }
    ggml_tensor* observe(ggml_tensor* t,const std::string& name,int layer=-1) {
        const auto label=layer<0?name:name+"-"+std::to_string(layer);
        if(std::find(request.observe.begin(),request.observe.end(),label)!=request.observe.end()) {
            auto* out=ggml_is_contiguous(t)?t:ggml_cont(c,t);
            check(graph->output(out)); observed.emplace_back(label,out);
        }
        return t;
    }
    // Mathematical references: llama.cpp 7ab4ee7 src/llama-graph.cpp,
    // llm_graph_context::build_lora_mm / build_lora_mm_id / build_norm.
    ggml_tensor* mm(const std::string& name,ggml_tensor* x,ggml_tensor* ids=nullptr) {
        auto* w=weight(name);
        auto multiply=[&](ggml_tensor* a,ggml_tensor* b) {
            return ids?ggml_mul_mat_id(c,a,b,ids):ggml_mul_mat(c,a,b);
        };
        auto* result=multiply(w,x);
        if(model->resident_on(execution_backend(model,request)).count("lora:"+name+".lora_a")) {
            auto* a=weight(name+".lora_a","lora"); auto* b=weight(name+".lora_b","lora");
            const auto alpha=take(model::read_metadata<float>(model->assets.indexes.at("lora").metadata,"adapter.lora.alpha"));
            result=ggml_add(c,result,ggml_scale(c,multiply(b,multiply(a,x)),*alpha/float(b->ne[0])));
        }
        return result;
    }
    ggml_tensor* norm(ggml_tensor* x,const std::string& w) { return ggml_mul(c,ggml_rms_norm(c,x,epsilon),weight(w)); }
    // llama.cpp 7ab4ee7 src/models/qwen3next.cpp: graph::build_layer_attn.
    ggml_tensor* attention(ggml_tensor* x,int layer) {
        const auto prefix="blk."+std::to_string(layer)+".";
        const auto d=p("head_dim"),h=p("n_head"),hk=p("n_head_kv");
        auto* full=ggml_reshape_3d(c,mm(prefix+"attn_q.weight",x),2*d,h,n);
        auto* q=ggml_view_3d(c,full,d,h,n,full->nb[1],full->nb[2],0);
        auto* gate=ggml_view_3d(c,full,d,h,n,full->nb[1],full->nb[2],d*sizeof(float));
        auto* k=ggml_reshape_3d(c,mm(prefix+"attn_k.weight",x),d,hk,n);
        auto* v=ggml_reshape_3d(c,mm(prefix+"attn_v.weight",x),d,hk,n);
        q=norm(q,prefix+"attn_q_norm.weight"); k=norm(k,prefix+"attn_k_norm.weight");
        const auto freq=float(model->assets.definition.numeric_parameters.at("qwen3next.rope.freq_base"));
        q=ggml_rope_ext(c,q,positions,nullptr,int(p("rope_dim")),GGML_ROPE_TYPE_NEOX,int(p("context_length")),freq,1,0,1,32,1);
        k=ggml_rope_ext(c,k,positions,nullptr,int(p("rope_dim")),GGML_ROPE_TYPE_NEOX,int(p("context_length")),freq,1,0,1,32,1);
        auto save=[&](ggml_tensor* t,const char* suffix) {
            const auto name="attn."+std::to_string(layer)+"."+suffix;
            auto storage=take(kv.storage(name)); take(graph->bind_states(storage));
            ggml_tensor* copy;
            if (rows) copy=ggml_set_rows(c,storage->tensor(),ggml_cont(c,ggml_permute(c,t,0,2,1,3)),rows);
            else {
                auto range=take(kv.write_range(c,name,request.step));
                copy=ggml_cpy(c,ggml_permute(c,t,0,2,1,3),range.view);
            }
            take(graph->bind_state_write(storage,copy));
            return take(kv.history_view(c,name,rows ? session::Step{0,kv.capacity()} : request.step,copy));
        };
        auto* history_k=save(k,"k"); auto* history_v=save(v,"v");
        auto* attn=ggml_flash_attn_ext(c,ggml_permute(c,q,0,2,1,3),history_k,history_v,mask,1/std::sqrt(float(d)),0,0);
        if (rows) {
            check(graph->bounded_view(history_k,1,exec::Strides::preserve,attn));
            check(graph->bounded_view(history_v,1,exec::Strides::preserve,attn));
            ranges.push_back(history_k); ranges.push_back(history_v);
            if (std::find(ranges.begin(),ranges.end(),mask)==ranges.end()) {
                check(graph->bounded_view(mask,0,exec::Strides::packed,attn)); ranges.push_back(mask);
            }
        }
        ggml_prec_set_acc(attn,GGML_PREC_F32);
        attn=ggml_reshape_2d(c,attn,d*h,n);
        gate=ggml_sigmoid(c,ggml_cont_2d(c,gate,d*h,n));
        return observe(mm(prefix+"attn_output.weight",ggml_mul(c,attn,gate)),"attn_output",layer);
    }
    // llama.cpp 7ab4ee7 src/models/qwen3next.cpp: build_qkvz,
    // build_layer_attn_linear; src/models/delta-net-base.cpp: build_conv_state,
    // build_delta_net_fused. Fused K=1 carries one final recurrent state.
    ggml_tensor* delta(ggml_tensor* x,int layer) {
        const auto prefix="blk."+std::to_string(layer)+".";
        const auto d=p("state_dim"),hk=p("n_key_head"),hv=p("n_value_head"),ratio=hv/hk;
        const auto channels=d*(2*hk+hv),kernel=p("conv_kernel");
        auto state=take(states.get<DeltaState>("deltanet."+std::to_string(layer)));
        if(!state) throw Error{ErrorCategory::conflict,"DeltaNet state is not initialized"};
        // Current GGUF uses the split QKV and gate projections, as in the
        // optimized build_qkvz branch; legacy ssm_in is not in this manifest.
        auto* qkv=mm(prefix+"attn_qkv.weight",x);
        auto* z=ggml_reshape_3d(c,mm(prefix+"attn_gate.weight",x),d,hv,n);
        auto* ba=ggml_reshape_3d(c,mm(prefix+"ssm_ba.weight",x),2*ratio,hk,n);
        auto* b=ggml_cont_2d(c,ggml_view_3d(c,ba,ratio,hk,n,ba->nb[1],ba->nb[2],0),hv,n);
        auto* a=ggml_cont_2d(c,ggml_view_3d(c,ba,ratio,hk,n,ba->nb[1],ba->nb[2],ratio*sizeof(float)),hv,n);
        auto* beta=ggml_reshape_3d(c,ggml_sigmoid(c,b),1,hv,n);
        auto* gate=ggml_reshape_3d(c,ggml_mul(c,ggml_softplus(c,ggml_add(c,a,weight(prefix+"ssm_dt.bias"))),weight(prefix+"ssm_a")),1,hv,n);
        // Each state storage has exactly one write per graph. Old values are read only via
        // concat and ggml_gated_delta_net, which are upstream of that write; no other node
        // may read the storage directly, or it could observe the updated state.
        auto* oldconv=take(graph->bind_states(state->convolution));
        auto* convinput=ggml_concat(c,oldconv,ggml_transpose(c,qkv),0);
        auto* last=ggml_view_2d(c,convinput,kernel-1,channels,convinput->nb[1],n*sizeof(float));
        take(graph->bind_state_write(state->convolution,ggml_cpy(c,last,oldconv)));
        auto* conv=ggml_silu(c,ggml_ssm_conv(c,convinput,weight(prefix+"ssm_conv1d.weight")));
        auto part=[&](std::int64_t heads,std::int64_t offset) {
            return ggml_view_3d(c,conv,d,heads,n,d*sizeof(float),channels*sizeof(float),offset*sizeof(float));
        };
        auto l2=[&](ggml_tensor* t) { return ggml_scale(c,ggml_rms_norm(c,t,epsilon/float(d)),1/std::sqrt(float(d))); };
        auto repeat=[&](ggml_tensor* t) {
            return ggml_reshape_3d(c,ggml_repeat_4d(c,ggml_reshape_4d(c,t,d,1,hk,n),d,ratio,hk,n),d,hv,n);
        };
        auto* q=repeat(l2(part(hk,0))); auto* k=repeat(l2(part(hk,d*hk)));
        auto* v=part(hv,2*d*hk);
        auto* oldstate=take(graph->bind_states(state->recurrent));
        auto* combined=ggml_gated_delta_net(c,q,k,v,gate,beta,oldstate,1);
        auto* output=ggml_view_3d(c,combined,d,hv,n,d*sizeof(float),d*hv*sizeof(float),0);
        observe(output,"attn_output",layer);
        auto* next=ggml_view_3d(c,combined,d,d,hv,d*sizeof(float),d*d*sizeof(float),d*hv*n*sizeof(float));
        take(graph->bind_state_write(state->recurrent,ggml_cpy(c,next,oldstate)));
        updated.push_back("deltanet."+std::to_string(layer));
        auto* gated=ggml_mul(c,norm(output,prefix+"ssm_norm.weight"),ggml_silu(c,z));
        return observe(mm(prefix+"ssm_out.weight",ggml_reshape_2d(c,gated,d*hv,n)),"linear_attn_out",layer);
    }
    // llama.cpp 7ab4ee7 src/llama-graph.cpp: build_moe_ffn;
    // src/models/qwen3next.cpp: graph::build_layer_ffn.
    std::pair<ggml_tensor*,ggml_tensor*> route(ggml_tensor* x,int layer) {
        const auto prefix="blk."+std::to_string(layer)+".";
        const auto experts=p("n_expert"),top=p("top_k");
        auto* probs=observe(ggml_soft_max(c,mm(prefix+"ffn_gate_inp.weight",x)),"ffn_moe_probs",layer);
        auto* ids=observe(ggml_argsort_top_k(c,probs,int(top)),"ffn_moe_topk",layer);
        if(!request.route_trace.empty()) trace_routes.emplace_back(layer,ggml_cont(c,ids));
        auto* weights=ggml_reshape_2d(c,ggml_get_rows(c,ggml_reshape_3d(c,probs,1,experts,n),ids),top,n);
        weights=observe(ggml_div(c,weights,ggml_clamp(c,ggml_sum_rows(c,weights),6.103515625e-5f,INFINITY)),"ffn_moe_weights_norm",layer);
        return {ids,weights};
    }
    void head(ggml_tensor* input,ggml_tensor* executed,int layer) {
        if(request.phase!=exec::Phase::decode || n!=1 || model->options.at("routing")=="actual" ||
           layer<p("prerouter_first_owner") || layer+1>=p("n_layer")) return;
        const auto experts=p("n_expert"),top=p("top_k");
        if(!identity) identity=take(graph->input("prerouter_identity",{GGML_TYPE_F32,{experts,experts}}));
        auto* selected=ggml_get_rows(c,identity,executed);
        auto* onehot=ggml_reshape_2d(c,ggml_sum_rows(c,ggml_cont(c,ggml_transpose(c,selected))),experts,1);
        auto* previous=take(graph->input("prerouter_previous_"+std::to_string(layer),{GGML_TYPE_F32,{experts,1}}));
        auto* x=observe(ggml_concat(c,ggml_concat(c,input,onehot,0),previous,0),"prerouter_in",layer);
        const auto prefix="blk."+std::to_string(layer)+".prerouter_";
        auto* h=ggml_mul_mat(c,weight(prefix+"fc1.weight","prerouter"),x);
        h=ggml_add(c,ggml_mul_mat(c,weight(prefix+"fc2.weight","prerouter"),ggml_gelu_erf(c,h)),
            ggml_mul_mat(c,weight(prefix+"linear_init.weight","prerouter"),x));
        observe(h,"prerouter_out",layer);
        auto* probs=ggml_soft_max(c,h);
        auto* ids=ggml_cont(c,ggml_argsort_top_k(c,probs,int(top)));
        check(graph->output(ids)); observe(ids,"prerouter_ids",layer);
        ggml_tensor *scores=nullptr,*rank=nullptr;
        {
            scores=ggml_reshape_2d(c,ggml_get_rows(c,ggml_reshape_3d(c,probs,1,experts,1),ids),top,1);
            scores=ggml_div(c,scores,ggml_sum_rows(c,scores)); check(graph->output(scores));
            observe(scores,"prerouter_scores",layer);
        }
        if(!request.route_trace.empty()) {
            rank=ggml_cont(c,ggml_argsort_top_k(c,probs,int(std::min(2*top,experts))));
        }
        heads.push_back({layer,ids,scores,rank});
    }
    ggml_tensor* ffn(ggml_tensor* x,int layer,ggml_tensor* ids=nullptr,ggml_tensor* weights=nullptr) {
        const auto prefix="blk."+std::to_string(layer)+".";
        const auto top=p("top_k"),embedding=p("n_embd");
        if (!ids) { auto routing=route(x,layer); ids=routing.first; weights=routing.second; }
        auto* input=ggml_reshape_3d(c,x,embedding,1,n);
        auto* gate=mm(prefix+"ffn_gate_exps.weight",input,ids);
        auto* up=mm(prefix+"ffn_up_exps.weight",input,ids);
        auto* down=mm(prefix+"ffn_down_exps.weight",ggml_mul(c,ggml_silu(c,gate),up),ids);
        down=ggml_mul(c,down,ggml_reshape_3d(c,weights,1,top,n));
        auto* sum=ggml_view_2d(c,down,embedding,n,down->nb[2],0);
        for(std::int64_t i=1;i<top;++i) sum=ggml_add(c,sum,ggml_view_2d(c,down,embedding,n,down->nb[2],i*down->nb[1]));
        observe(sum,"ffn_moe_out",layer);
        auto* shared=mm(prefix+"ffn_down_shexp.weight",ggml_mul(c,ggml_silu(c,mm(prefix+"ffn_gate_shexp.weight",x)),mm(prefix+"ffn_up_shexp.weight",x)));
        shared=observe(ggml_mul(c,shared,ggml_sigmoid(c,mm(prefix+"ffn_gate_inp_shexp.weight",x))),"ffn_shexp_gated",layer);
        return observe(ggml_add(c,sum,shared),"ffn_out",layer);
    }
};
}
namespace {
struct Segment : DecodePlan {
    ggml_tensor* route = nullptr;
    std::vector<std::pair<std::string,ggml_tensor*>> observed;
    std::vector<std::string> updated;
    std::vector<HeadOutput> heads;
    std::vector<std::pair<int,ggml_tensor*>> trace_routes;
};
struct SegmentedPlan {
    std::vector<std::shared_ptr<Segment>> segments;
    std::array<std::array<std::shared_ptr<exec::StateStorage>,3>,2> transfer;
};
Result<std::unique_ptr<ForwardResult>> forward_slots(const std::shared_ptr<model::LoadedModel>& model,
    const ForwardRequest& request,session::KVCache& kv,session::StateStore& states,exec::Budget& budget) {
    Timing timing(request);
    bool submitted=false;
    WeightActivity activity;
    std::shared_ptr<SegmentedPlan> group;
    std::shared_ptr<weights::Write> pending_prefill;
    std::shared_ptr<PrerouterState> prediction;
    std::string plan_key="aux.expert_plans";
    try {
        check(model->require_active());
        const auto& parameters=model->assets.definition.parameters;
        const auto n=request.step.n;
        if(n<=0 || n!=std::int64_t(request.tokens.size()) || request.output_positions.empty())
            throw Error{ErrorCategory::invalid_argument,"Invalid forward block"};
        for(auto token:request.tokens) if(token<0 || std::uint64_t(token)>=parameters.at("n_vocab"))
            throw Error{ErrorCategory::invalid_argument,"Token outside vocabulary"};
        for(auto pos:request.output_positions) if(pos>=request.tokens.size()) throw Error{ErrorCategory::invalid_argument,"Output position outside block"};
        const auto declarations=take(describe_states(model->assets.definition));
        for(const auto& entry:declarations.kv) take(kv.history_length(entry.name,request.step));
        const auto mask_values=take(kv.causal_mask(declarations.kv.front().name,request.step));
        const auto length=take(kv.history_length(declarations.kv.front().name,request.step));
        const auto layers=std::size_t(parameters.at("n_layer"));
        const auto top=std::int64_t(parameters.at("top_k")), embedding=std::int64_t(parameters.at("n_embd"));
        const bool slots=!model->pools.empty();
        const bool use_pool=n==1 && (!slots || execution_backend(model,request)==model->stages.decode);
        const bool predicted=model->options.at("routing")=="predicted";
        const bool decode=request.phase==exec::Phase::decode && n==1;
        const auto first=parameters.at("prerouter_first_owner");
        bool consume=false;
        if(predicted) {
            prediction=take(states.get<PrerouterState>("aux.prerouter"));
            if(!prediction) {
                prediction=std::make_shared<PrerouterState>();
                prediction->predictions.resize(layers); prediction->previous.resize(layers);
                prediction->pending.resize(layers);
                check(states.put("aux.prerouter",prediction));
            }
            consume=decode && prediction->position>=0 && request.step.e==prediction->position+1;
            if(!consume) {
                activity+=take(finish_pending_weights(model,states,FinishReason::normal));
                if(decode && prediction->position>=0) ++activity.prefetch.discarded;
                prediction->position=-1;
                for(auto& forecast:prediction->predictions) forecast={};
                for(auto& ids:prediction->previous) ids.clear();
            }
        }
        const bool merged=predicted && consume;
        if(merged) plan_key="aux.predicted_plans";
        if(!request.route_trace.empty()) plan_key+=".trace";
        if(model->stages.mixed()) plan_key+="."+std::to_string(reinterpret_cast<std::uintptr_t>(execution_backend(model,request).get()));
        const auto last_segment=merged?std::size_t(first+1):layers;
        auto next=std::make_shared<std::vector<Prediction>>(layers);
        std::vector<std::vector<std::int32_t>> executed(layers), trace_actual(layers), trace_heads(layers);
        const auto old_position=prediction?prediction->position:-1;
        const bool ahead=slots && use_ahead_prefill(model->options,n);
        const auto bank=[&](std::size_t layer) { return model->temporary_on(execution_backend(model,request),ahead && layer%2); };
        const auto load_items=[&](std::size_t layer,const std::set<std::int32_t>& selected) {
            std::vector<weights::LoadItem> items;
            const auto count=parameters.at("n_expert");
            for(auto expert:selected) {
                if(expert<0 || std::size_t(expert)>=count) throw Error{ErrorCategory::incompatible,"Route outside expert range"};
                for(std::size_t p=0;p<3;++p) {
                    const auto& source=model->pool_descriptions.at(layer).experts.at(std::size_t(expert))[p];
                    items.push_back({p*count+std::size_t(expert),"prefill/"+std::to_string(layer)+"/"+std::to_string(expert)+"/"+std::to_string(p),source});
                }
            }
            return items;
        };
        const bool persistent=decode && request.observe.empty() && request.plan_mode==PlanMode::automatic;
        group=persistent?take(states.get<SegmentedPlan>(plan_key)):nullptr;
        bool reused=bool(group);
        if(group) for(const auto& segment:group->segments) {
            std::map<ggml_tensor*,std::int64_t> ranges;
            for(auto* view:segment->ranges) ranges[view]=length;
            auto reuse=exec::check_plan_reuse(*segment->plan,n,ranges);
            if(reuse.decision==exec::ReuseDecision::unsupported) throw Error{ErrorCategory::unsupported,reuse.reason};
            if(reuse.decision==exec::ReuseDecision::rebuild) { states.remove(plan_key); group.reset(); reused=false; break; }
        }
        if(!group) {
            group=std::make_shared<SegmentedPlan>();
            for(auto& bank:group->transfer) {
                bank[0]=take(exec::allocate_state_storage(budget,execution_backend(model,request),{GGML_TYPE_F32,{embedding,n}}, edge0::exec::Purpose::transfer));
                bank[1]=take(exec::allocate_state_storage(budget,execution_backend(model,request),{GGML_TYPE_F32,{embedding,n}}, edge0::exec::Purpose::transfer));
                bank[2]=take(exec::allocate_state_storage(budget,execution_backend(model,request),{GGML_TYPE_F32,{top,n}}, edge0::exec::Purpose::transfer));
            }
        }
        auto result=std::make_unique<ResultOwner>(); result->model=model; result->reused=reused;
        result->persistent=persistent; result->allow_logits=request.retain_logits;
        result->completed.success=true;
        weights::Prepared prepared;
        std::vector<std::int32_t> mapped;
        const auto prepare_layer=[&](std::size_t layer,const std::vector<std::int32_t>& ids) {
            weights::Prepared ready;
            if(!slots) return ready;
            auto pool=model->pools.at(layer);
            if(prediction && prediction->pending[layer]) {
                if(!pool->ready(prediction->pending[layer])) ++activity.prefetch.late;
                const auto waited=timing.stamp();
                ready=take(pool->await_ready(prediction->pending[layer]));
                prediction->pending[layer].reset();
                timing.add(timing.values.wait_seconds,waited);
                activity.decode+=ready.activity; activity.prefetch.bytes+=ready.activity.bytes;
                return ready;
            }
            const auto started=timing.stamp();
            auto pending=take(pool->prepare(ids));
            timing.add(timing.values.prepare_seconds,started);
            const auto waited=timing.stamp();
            ready=take(pool->await_ready(pending));
            timing.add(timing.values.wait_seconds,waited); activity.decode+=ready.activity;
            return ready;
        };
        const auto start_prefetch=[model,prediction,next,first](std::size_t layer) -> exec::Status {
            if(!prediction || model->pools.empty() || layer<=first || next->at(layer).ids.empty()) return {};
            auto pending=model->pools.at(layer)->prepare(next->at(layer).ids,weights::Priority::background);
            if(!pending) return pending.error();
            prediction->pending[layer]=pending.value();
            return {};
        };
        std::vector<std::int32_t> positions(std::size_t(n),0), outputs;
        std::iota(positions.begin(),positions.end(),std::int32_t(request.step.e));
        for(auto pos:request.output_positions) outputs.push_back(std::int32_t(pos));
        const bool cancellable = request.phase == exec::Phase::prefill && n > 1 && bool(request.cancel_block);
        if (cancellable) {
            const auto begin = std::chrono::steady_clock::now();
            check(copy_block_snapshot(states, false));
            if (request.progress) request.progress->block_snapshot_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        }
        for(std::size_t index=0;index<=last_segment;++index) {
            const auto& attention_layers=edge0_35b::attention_layers(model->assets.definition);
            const bool merge_tail=merged && index==last_segment;
            const bool has_attention=merge_tail || (index<layers && std::find(attention_layers.begin(),attention_layers.end(),index)!=attention_layers.end());
            std::vector<weights::Prepared> tail_prepared(layers);
            if(merge_tail) for(std::size_t layer=first+1;layer<layers;++layer) {
                executed[layer]=prediction->predictions[layer].ids;
                tail_prepared[layer]=prepare_layer(layer,executed[layer]);
            }
            const auto binding_started=timing.stamp();
            std::array<weights::View,3> expert_views{};
            if(index && slots) for(std::size_t projection=0;projection<3;++projection) {
                const auto storage=use_pool?model->pools.at(index-1)->storage():bank(index-1);
                expert_views[projection]=take(model->weights.acquire_views(storage,execution_backend(model,request),projection,prepared.bindings[projection]));
            }
            std::shared_ptr<Segment> segment;
            if(reused) {
                segment=group->segments.at(index);
                for(const auto& entry:segment->weights) {
                    auto found=model->resident_on(execution_backend(model,request)).find(entry.first);
                    if(found!=model->resident_on(execution_backend(model,request)).end()) {
                        auto view=take(model->weights.acquire_views(found->second.storage,found->second.binding,execution_backend(model,request)));
                        check(segment->plan->bind_weights(view.tensor,view.use));
                    }
                }
                if(index && slots) for(auto& view:expert_views) check(segment->plan->bind_weights(view.tensor,view.use));
            } else {
                segment=std::make_shared<Segment>();
                auto graph=take(exec::create_graph(budget,execution_backend(model,request),request.phase,merge_tail?1024*(layers-first):1024));
                if(model->stages.mixed()) check(graph->coordinate(model->stages));
                check(graph->set_token_count(n)); auto* c=graph->context();
                ggml_tensor *pos=nullptr,*mask=nullptr,*rows=nullptr;
                if(has_attention) {
                    pos=take(graph->input("positions",{GGML_TYPE_I32,{n}}));
                    const auto maximum=decode?take(kv.history_length(declarations.kv.front().name,{0,kv.capacity()})):std::int64_t(mask_values.size())/n;
                    auto* input=take(graph->input("mask",{GGML_TYPE_F16,{maximum,n}}));
                    mask=decode?ggml_view_2d(c,input,maximum,n,input->nb[1],0):input;
                    if(decode) rows=take(graph->input("rows",{GGML_TYPE_I64,{n}}));
                }
                Builder b{model,graph,c,request,kv,states,{}, {},n,float(model->assets.definition.numeric_parameters.at("qwen3next.attention.layer_norm_rms_epsilon")),pos,mask,rows,{}, {}};
                ggml_tensor* x;
                if(!index) {
                    auto* tokens=take(graph->input("tokens",{GGML_TYPE_I32,{n}}));
                    x=ggml_get_rows(c,b.weight("token_embd.weight"),tokens);
                } else {
                    const auto prefix="blk."+std::to_string(index-1)+".";
                    static const char* names[]{"gate","up","down"};
                    for(std::size_t projection=0;slots && projection<3;++projection) {
                        const auto& view=expert_views[projection];
                        check(graph->bind_weights(view.tensor,execution_backend(model,request),view.use));
                        b.bound["base:"+prefix+"ffn_"+names[projection]+"_exps.weight"]=view.tensor;
                    }
                    auto& bank=group->transfer[(index-1)%2];
                    auto* residual=take(graph->bind_states(bank[0]));
                    auto* input=take(graph->bind_states(bank[1]));
                    auto* route_weights=take(graph->bind_states(bank[2]));
                    auto* ids=take(graph->input("experts",{GGML_TYPE_I32,{top,n}}));
                    x=b.observe(ggml_add(c,residual,b.ffn(input,int(index-1),ids,route_weights)),"l_out",int(index-1));
                }
                if(merge_tail) {
                    for(std::size_t layer=index;layer<layers;++layer) {
                        const auto prefix="blk."+std::to_string(layer)+".";
                        if(slots) for(std::size_t projection=0;projection<3;++projection) {
                            auto view=take(model->weights.acquire_views(model->pools[layer]->storage(),projection,tail_prepared[layer].bindings[projection]));
                            check(graph->bind_weights(view.tensor,execution_backend(model,request),view.use));
                            static const char* names[]{"gate","up","down"};
                            b.bound["base:"+prefix+"ffn_"+names[projection]+"_exps.weight"]=view.tensor;
                        }
                        auto* normalized=b.observe(b.norm(x,prefix+"attn_norm.weight"),"attn_norm",int(layer));
                        auto* out=std::find(attention_layers.begin(),attention_layers.end(),layer)!=attention_layers.end()?b.attention(normalized,int(layer)):b.delta(normalized,int(layer));
                        x=b.observe(ggml_add(c,x,out),"attn_residual",int(layer));
                        auto* input=b.norm(x,prefix+"post_attention_norm.weight");
                        if(!request.route_trace.empty()) b.route(input,int(layer));
                        auto* ids=take(graph->input("experts_"+std::to_string(layer),{GGML_TYPE_I32,{top,1}}));
                        auto* real_ids=take(graph->input("executed_"+std::to_string(layer),{GGML_TYPE_I32,{top,1}}));
                        auto* scores=take(graph->input("scores_"+std::to_string(layer),{GGML_TYPE_F32,{top,1}}));
                        check(graph->output(real_ids));
                        b.observe(real_ids,"executed",int(layer)); b.observe(scores,"executed_scores",int(layer));
                        b.head(input,real_ids,int(layer));
                        x=b.observe(ggml_add(c,x,b.ffn(input,int(layer),ids,scores)),"l_out",int(layer));
                    }
                }
                if(index<layers && !merge_tail) {
                    const auto prefix="blk."+std::to_string(index)+".";
                    auto* normalized=b.observe(b.norm(x,prefix+"attn_norm.weight"),"attn_norm",int(index));
                    const auto& attention=edge0_35b::attention_layers(model->assets.definition);
                    auto* out=std::find(attention.begin(),attention.end(),index)!=attention.end()?b.attention(normalized,int(index)):b.delta(normalized,int(index));
                    x=b.observe(ggml_add(c,x,out),"attn_residual",int(index));
                    auto* ffn_input=b.norm(x,prefix+"post_attention_norm.weight");
                    auto routing=b.route(ffn_input,int(index));
                    auto& bank=group->transfer[index%2];
                    check(graph->retain_output(x,bank[0]));
                    auto* retained=take(graph->bind_state_write(bank[1],ggml_cpy(c,ffn_input,take(graph->bind_states(bank[1])))));
                    check(graph->output(retained)); check(graph->retain_output(routing.second,bank[2]));
                    segment->route=ggml_cont(c,routing.first); check(graph->output(segment->route));
                    b.head(retained,segment->route,int(index));
                } else {
                    auto* out=take(graph->input("outputs",{GGML_TYPE_I32,{std::int64_t(outputs.size())}}));
                    x=b.observe(ggml_get_rows(c,b.norm(x,"output_norm.weight"),out),"result_norm");
                    segment->output=b.observe(b.mm("output.weight",x),"result_output"); check(graph->output(segment->output));
                    if(request.greedy || decode) segment->greedy=take(append_greedy(*graph,segment->output));
                }
                // Append diagnostics after the normal outputs so tracing cannot
                // reorder the computation or change its fusion boundaries.
                for(const auto& route:b.trace_routes) check(graph->output(route.second));
                for(const auto& head:b.heads) {
                    if(head.rank) check(graph->output(head.rank));
                }
                segment->plan=take(exec::build_plan(graph)); segment->ranges=std::move(b.ranges); segment->weights=std::move(b.bound);
                segment->heads=std::move(b.heads); segment->trace_routes=std::move(b.trace_routes);
                segment->observed=std::move(b.observed); segment->updated=std::move(b.updated);
                group->segments.push_back(segment);
            }
            auto plan=segment->plan;
            if(reused && merge_tail && slots) for(std::size_t layer=index;layer<layers;++layer)
                for(std::size_t projection=0;projection<3;++projection) {
                    auto view=take(model->weights.acquire_views(model->pools[layer]->storage(),projection,tail_prepared[layer].bindings[projection]));
                    check(plan->bind_weights(view.tensor,view.use));
                }
            timing.add(timing.values.binding_seconds,binding_started);
            if(!segment->heads.empty()) {
                const auto experts=parameters.at("n_expert");
                std::vector<float> identity(experts*experts,0);
                for(std::size_t i=0;i<experts;++i) identity[i*experts+i]=1;
                check(plan->set_inputs("prerouter_identity",identity.data(),identity.size()*sizeof(float)));
                for(const auto& head:segment->heads) {
                    std::vector<float> previous(experts,0);
                    for(auto id:prediction->previous[std::size_t(head.layer)]) previous.at(std::size_t(id))=1;
                    check(plan->set_inputs("prerouter_previous_"+std::to_string(head.layer),previous.data(),previous.size()*sizeof(float)));
                }
            }
            if(merge_tail) for(std::size_t layer=index;layer<layers;++layer) {
                auto ids=executed[layer];
                if(slots) for(auto& id:ids) id=std::int32_t(tail_prepared[layer].mapping.at(id));
                check(plan->set_inputs("experts_"+std::to_string(layer),ids.data(),ids.size()*sizeof(std::int32_t)));
                check(plan->set_inputs("executed_"+std::to_string(layer),executed[layer].data(),executed[layer].size()*sizeof(std::int32_t)));
                const auto& scores=prediction->predictions[layer].scores;
                check(plan->set_inputs("scores_"+std::to_string(layer),scores.data(),scores.size()*sizeof(float)));
            }
            for(auto* range:segment->ranges) check(plan->set_valid_ranges(range,length));
            if(has_attention) {
                check(plan->set_inputs("positions",positions.data(),positions.size()*sizeof(std::int32_t)));
                check(plan->set_inputs("mask",mask_values.data(),mask_values.size()*sizeof(ggml_fp16_t)));
                if(decode) { auto row=request.step.e; check(plan->set_inputs("rows",&row,sizeof(row))); }
            }
            if(!index) check(plan->set_inputs("tokens",request.tokens.data(),request.tokens.size()*sizeof(std::int32_t)));
            if(index) check(plan->set_inputs("experts",mapped.data(),mapped.size()*sizeof(std::int32_t)));
            if(index==last_segment) check(plan->set_inputs("outputs",outputs.data(),outputs.size()*sizeof(std::int32_t)));
            if(ahead && index<layers) {
                std::set<std::int32_t> all;
                for(std::size_t e=0;e<parameters.at("n_expert");++e) all.insert(std::int32_t(e));
                const auto items=load_items(index,all);
                pending_prefill=take(model->weights.begin_write(bank(index),items));
                check(model->weights.fill_regions(pending_prefill));
                activity.prefill.loads+=all.size();
                for(const auto& item:items) activity.prefill.bytes+=item.source.bytes;
            }
            const auto submit_started=timing.stamp();
            check(plan->submit()); submitted=true; if(request.progress) request.progress->submitted=true;
            auto completion=take(plan->collect_completion());
            timing.add(timing.values.submit_seconds,submit_started);
            if(timing.enabled) ++timing.values.segments;
            if(!completion.success) throw Error{ErrorCategory::backend_error,"Segment execution failed"};
            for(const auto& written:completion.written) {
                bool transfer=false;
                for(const auto& bank:group->transfer) for(const auto& state:bank) if(state->allocation==written) transfer=true;
                if(!transfer) result->completed.written.push_back(written);
            }
            result->updated.insert(result->updated.end(),segment->updated.begin(),segment->updated.end());
            result->registrations+=segment->weights.size();
            for(const auto& entry:segment->observed) {
                auto* tensor=entry.second;
                ObservedTensor observed{entry.first,{tensor->type,{tensor->ne[0],tensor->ne[1],tensor->ne[2],tensor->ne[3]}},{}};
                observed.data.resize(ggml_nbytes(tensor)); check(plan->read_output(tensor,observed.data.data(),0,observed.data.size()));
                result->captured.push_back(std::move(observed));
            }
            for(const auto& head:segment->heads) {
                auto& forecast=next->at(std::size_t(head.layer+1)); forecast.ids.resize(std::size_t(top));
                check(plan->read_output(head.ids,forecast.ids.data(),0,forecast.ids.size()*sizeof(std::int32_t)));
                result->bytes_read+=forecast.ids.size()*sizeof(std::int32_t);
                if(head.scores) { forecast.scores.resize(std::size_t(top)); check(plan->read_output(head.scores,forecast.scores.data(),0,forecast.scores.size()*sizeof(float))); result->bytes_read+=forecast.scores.size()*sizeof(float); }
                if(head.rank) {
                    auto& ranked=trace_heads[std::size_t(head.layer)];
                    ranked.resize(std::size_t(ggml_nelements(head.rank)));
                    check(plan->read_output(head.rank,ranked.data(),0,ranked.size()*sizeof(std::int32_t)));
                    result->bytes_read+=ranked.size()*sizeof(std::int32_t);
                }
            }
            for(const auto& route:segment->trace_routes) {
                auto& ids=trace_actual[std::size_t(route.first)]; ids.resize(std::size_t(ggml_nelements(route.second)));
                check(plan->read_output(route.second,ids.data(),0,ids.size()*sizeof(std::int32_t))); result->bytes_read+=ids.size()*sizeof(std::int32_t);
            }
            if(index==last_segment) { result->plan=plan; result->output=segment->output; result->greedy=segment->greedy; break; }
            const auto route_started=timing.stamp();
            mapped.resize(std::size_t(top*n)); check(plan->read_output(segment->route,mapped.data(),0,mapped.size()*sizeof(std::int32_t)));
            timing.add(timing.values.route_seconds,route_started);
            result->bytes_read+=mapped.size()*sizeof(std::int32_t);
            check(plan->end_observation());
            if(!persistent) check(plan->release_plan());
            if (cancellable && request.cancel_block(index + 1)) {
                if (pending_prefill) { take(model->weights.complete(pending_prefill)); pending_prefill.reset(); }
                activity += take(finish_pending_weights(model, states, FinishReason::normal));
                if (!request.skip_block_rollback) check(copy_block_snapshot(states, true));
                states.remove("aux.prerouter");
                states.remove(plan_key); group.reset();
                const auto count = parameters.at("n_expert") * 3;
                for (const auto& temporary : {model->temporary_on(execution_backend(model,request),false), model->temporary_on(execution_backend(model,request),true)}) if (temporary) {
                    std::vector<std::size_t> published;
                    for (std::size_t region = 0; region < count; ++region)
                        if (temporary->state(region) == weights::RegionState::published) published.push_back(region);
                    check(model->weights.evict(temporary, published));
                }
                if (request.progress) {
                    request.progress->cancelled = true;
                    request.progress->rolled_back_tokens = static_cast<std::size_t>(n);
                    request.progress->activity = activity;
                }
                return Error{ErrorCategory::invalid_argument, "Prefill block cancelled"};
            }
            if(decode && predicted && !merged && index>first+1) {
                check(start_prefetch(index-1)); activity.prefetch.initiated+=next->at(index-1).ids.size();
            }
            if(use_pool) {
                executed[index]=mapped;
                prepared=prepare_layer(index,mapped);
                if(slots) for(auto& expert:mapped) expert=std::int32_t(prepared.mapping.at(expert));
            } else if(slots) {
                std::set<std::int32_t> selected(mapped.begin(),mapped.end());
                const auto count=parameters.at("n_expert");
                const auto demand=selected.size();
                for(auto expert:selected) if(expert<0 || std::size_t(expert)>=count)
                    throw Error{ErrorCategory::incompatible,"Route outside expert range"};
                if(!ahead) {
                    if(model->options.at("prefill_read")=="layer") for(std::size_t e=0;e<count;++e) selected.insert(std::int32_t(e));
                    const auto items=load_items(index,selected);
                    pending_prefill=take(model->weights.begin_write(bank(index),items));
                    check(model->weights.fill_regions(pending_prefill));
                    activity.prefill.loads+=selected.size();
                    for(const auto& item:items) activity.prefill.bytes+=item.source.bytes;
                }
                auto bindings=take(model->weights.complete(pending_prefill));pending_prefill.reset();prepared={};
                for(const auto& binding:bindings) prepared.bindings[binding.region/count].push_back(binding);
                activity.prefill.demand+=demand;
            }
            if(request.progress) request.progress->activity=activity;
        }
        if(decode && predicted) {
            prediction->previous=executed; prediction->predictions=*next; prediction->position=request.step.e;
            result->after_observation=[start_prefetch,prediction,model,first,layers,merged]() -> exec::Status {
                if(prediction->position<0) return {};
                const auto begin=merged?first+1:layers-1;
                for(std::size_t layer=begin;layer<layers;++layer) if(auto error=start_prefetch(layer)) {
                    for(std::size_t i=0;i<layers;++i) if(prediction->pending[i]) { model->pools[i]->abandon(prediction->pending[i]); prediction->pending[i].reset(); }
                    return error;
                }
                return {};
            };
            if(slots) for(std::size_t layer=merged?first+1:layers-1;layer<layers;++layer) activity.prefetch.initiated+=next->at(layer).ids.size();
        }
        if(decode && !request.route_trace.empty()) {
            std::ofstream trace(request.route_trace,std::ios::app);
            if(!trace) throw Error{ErrorCategory::invalid_argument,"Cannot open route trace"};
            const auto array=[&](const std::vector<std::int32_t>& ids) { trace<<'['; for(std::size_t i=0;i<ids.size();++i) { if(i) trace<<','; trace<<ids[i]; } trace<<']'; };
            trace<<"{\"position\":"<<request.step.e<<",\"source_position\":"<<old_position<<",\"top_k\":"<<top<<",\"layers\":[";
            for(std::size_t layer=0;layer<layers;++layer) {
                if(layer) trace<<',';
                trace<<"{\"layer\":"<<layer<<",\"actual\":";array(trace_actual[layer]);
                trace<<",\"executed\":";array(executed[layer]); trace<<",\"head\":";array(trace_heads[layer]);
                trace<<",\"prepared\":";array(slots?next->at(layer).ids:std::vector<std::int32_t>{}); trace<<'}';
            }
            trace<<"]}\n";
            if(!trace) throw Error{ErrorCategory::backend_error,"Cannot write route trace"};
        }
        result->activity=activity; result->clear_temporary=slots && !use_pool; result->backend=execution_backend(model,request);
        if(persistent && !reused) { states.remove(plan_key); check(states.put(plan_key,group)); }
        result->timing=timing.finish();
        return std::unique_ptr<ForwardResult>(std::move(result));
    } catch(const Error& error) {
        if(pending_prefill) model->weights.abandon(pending_prefill);
        finish_pending_weights(model,states,FinishReason::failure);
        states.remove(plan_key); group.reset();
        if(request.progress) { request.progress->submitted=submitted; request.progress->activity=activity; }
        return error;
    } catch(const std::out_of_range&) {
        if(pending_prefill) model->weights.abandon(pending_prefill);
        finish_pending_weights(model,states,FinishReason::failure);
        states.remove(plan_key); group.reset();
        if(request.progress) { request.progress->submitted=submitted; request.progress->activity=activity; }
        return Error{ErrorCategory::incompatible,"Missing segmented forward tensor or metadata"};
    }
}
}
std::vector<std::string> observation_names(const model::ModelDefinition& definition, bool prerouter) {
    std::vector<std::string> result;
    for(std::uint64_t layer=0;layer<definition.parameters.at("n_layer");++layer)
        for(const auto* name:{"attn_norm","linear_attn_out","attn_output","attn_residual","ffn_moe_probs","ffn_moe_topk","ffn_moe_weights_norm","ffn_moe_out","ffn_shexp_gated","ffn_out","l_out"})
            result.push_back(std::string(name)+"-"+std::to_string(layer));
    if(prerouter) for(std::uint64_t layer=definition.parameters.at("prerouter_first_owner");layer<definition.parameters.at("n_layer");++layer)
        for(const auto* name:{"prerouter_in","prerouter_out","prerouter_scores","prerouter_ids","executed","executed_scores"})
            result.push_back(std::string(name)+"-"+std::to_string(layer));
    result.push_back("result_norm"); result.push_back("result_output"); return result;
}
Result<std::unique_ptr<ForwardResult>> forward(const std::shared_ptr<model::LoadedModel>& model,
    const ForwardRequest& request,session::KVCache& kv,session::StateStore& states,exec::Budget& budget) {
    if(request.progress) *request.progress={};
    if(model && (!model->pools.empty() || model->options.at("routing")!="actual")) return forward_slots(model,request,kv,states,budget);
    Timing timing(request);
    try {
        if(!model) throw Error{ErrorCategory::invalid_argument,"Missing model"};
        check(model->require_active());
        if(request.step.n<=0 || request.step.n!=std::int64_t(request.tokens.size()) || request.output_positions.empty())
            throw Error{ErrorCategory::invalid_argument,"Invalid forward block or output positions"};
        for(auto token:request.tokens) if(token<0 || std::uint64_t(token)>=model->assets.definition.parameters.at("n_vocab"))
            throw Error{ErrorCategory::invalid_argument,"Token outside vocabulary"};
        for(auto pos:request.output_positions) if(pos>=request.tokens.size()) throw Error{ErrorCategory::invalid_argument,"Output position outside block"};
        const auto declarations=take(describe_states(model->assets.definition));
        for(const auto& entry:declarations.kv) take(kv.history_length(entry.name,request.step));
        const auto mask_values=take(kv.causal_mask(declarations.kv.front().name,request.step));
        const auto n=request.step.n;
        const bool decode=request.phase==exec::Phase::decode && n==1;
        const bool persistent=decode && request.observe.empty() && request.plan_mode==PlanMode::automatic;
        std::string plan_key=request.route_trace.empty()?"aux.decode_plan":"aux.decode_plan.trace";
        if(model->stages.mixed()) plan_key+="."+std::to_string(reinterpret_cast<std::uintptr_t>(execution_backend(model,request).get()));
        auto cached=persistent ? take(states.get<DecodePlan>(plan_key)) : nullptr;
        const auto length=take(kv.history_length(declarations.kv.front().name,request.step));
        bool reused=false;
        std::vector<std::pair<std::string,ggml_tensor*>> observed;
        std::vector<std::string> updated;
        if (cached) {
            std::map<ggml_tensor*,std::int64_t> ranges;
            for (auto* view:cached->ranges) ranges[view]=length;
            auto reuse=exec::check_plan_reuse(*cached->plan,n,ranges);
            if (reuse.decision==exec::ReuseDecision::unsupported) throw Error{ErrorCategory::unsupported,reuse.reason};
            if (reuse.decision==exec::ReuseDecision::rebuild) {
                check(cached->plan->release_plan()); states.remove(plan_key); cached.reset();
            } else {
                reused=true;
                for (const auto& entry:cached->weights) {
                    const auto& weight=model->resident_on(execution_backend(model,request)).at(entry.first);
                    auto view=take(model->weights.acquire_views(weight.storage,weight.binding,execution_backend(model,request)));
                    check(cached->plan->bind_weights(view.tensor,view.use));
                }
                updated=declarations.updated_entries;
            }
        }
        if (!cached) {
        // Fewer than 300 tensors per layer including LoRA, views and observations;
        // reserve 512 per layer plus 256 for inputs/output. No per-token unrolling.
        auto graph=take(exec::create_graph(budget,execution_backend(model,request),request.phase,512*model->assets.definition.parameters.at("n_layer")+256));
                if(model->stages.mixed()) check(graph->coordinate(model->stages));
        check(graph->set_token_count(request.step.n));
        auto* c=graph->context();
        auto* tokens=take(graph->input("tokens",{GGML_TYPE_I32,{n}}));
        auto* positions=take(graph->input("positions",{GGML_TYPE_I32,{n}}));
        auto* outputs=take(graph->input("outputs",{GGML_TYPE_I32,{std::int64_t(request.output_positions.size())}}));
        const auto maximum=decode ? take(kv.history_length(declarations.kv.front().name,{0,kv.capacity()})) : std::int64_t(mask_values.size())/n;
        auto* mask_input=take(graph->input("mask",{GGML_TYPE_F16,{maximum,n}}));
        auto* mask=decode ? ggml_view_2d(c,mask_input,maximum,n,mask_input->nb[1],0) : mask_input;
        auto* rows=decode ? take(graph->input("rows",{GGML_TYPE_I64,{n}})) : nullptr;
        Builder b{model,graph,c,request,kv,states,{}, {},n,float(model->assets.definition.numeric_parameters.at("qwen3next.attention.layer_norm_rms_epsilon")),positions,mask,rows,{}, {}};
        auto* x=ggml_get_rows(c,b.weight("token_embd.weight"),tokens);
        for(int layer=0;layer<b.p("n_layer");++layer) {
            auto prefix="blk."+std::to_string(layer)+".";
            auto* normalized=b.observe(b.norm(x,prefix+"attn_norm.weight"),"attn_norm",layer);
            const auto& attn=edge0_35b::attention_layers(model->assets.definition);
            auto* out=std::find(attn.begin(),attn.end(),layer)!=attn.end()?b.attention(normalized,layer):b.delta(normalized,layer);
            x=b.observe(ggml_add(c,x,out),"attn_residual",layer);
            x=b.observe(ggml_add(c,x,b.ffn(b.norm(x,prefix+"post_attention_norm.weight"),layer)),"l_out",layer);
        }
        x=b.observe(ggml_get_rows(c,b.norm(x,"output_norm.weight"),outputs),"result_norm");
        auto* logits=b.observe(b.mm("output.weight",x),"result_output"); check(graph->output(logits));
        cached=std::make_shared<DecodePlan>();
        if (request.greedy || decode) cached->greedy=take(append_greedy(*graph,logits));
        for(const auto& route:b.trace_routes) check(graph->output(route.second));
        cached->resident_routes=std::move(b.trace_routes);
        cached->plan=take(exec::build_plan(graph));
        cached->output=logits; cached->ranges=std::move(b.ranges); cached->weights=std::move(b.bound);
        observed=std::move(b.observed); updated=std::move(b.updated);
        if (persistent) { states.remove(plan_key); check(states.put(plan_key,cached)); }
        }
        auto plan=cached->plan;
        for (auto* view:cached->ranges) check(plan->set_valid_ranges(view,length));
        if (decode) {
            const std::int64_t row=request.step.e;
            check(plan->set_inputs("rows",&row,sizeof(row)));
        }
        std::vector<std::int32_t> pos(std::size_t(n),0),out;
        std::iota(pos.begin(),pos.end(),std::int32_t(request.step.e));
        for(auto index:request.output_positions) out.push_back(std::int32_t(index));
        check(plan->set_inputs("tokens",request.tokens.data(),request.tokens.size()*sizeof(std::int32_t)));
        check(plan->set_inputs("positions",pos.data(),pos.size()*sizeof(std::int32_t)));
        check(plan->set_inputs("outputs",out.data(),out.size()*sizeof(std::int32_t)));
        check(plan->set_inputs("mask",mask_values.data(),mask_values.size()*sizeof(ggml_fp16_t)));
        const auto submit_started=timing.stamp();
        check(plan->submit()); if(request.progress) request.progress->submitted=true; auto completion=take(plan->collect_completion());
        timing.add(timing.values.submit_seconds,submit_started);
        if(timing.enabled) ++timing.values.segments;
        auto result=std::make_unique<ResultOwner>(); result->model=model; result->plan=plan;
        result->completed=std::move(completion); result->output=cached->output;
        result->observed=std::move(observed); result->registrations=cached->weights.size();
        result->updated=std::move(updated); result->greedy=cached->greedy;
        result->persistent=persistent; result->reused=reused; result->allow_logits=request.retain_logits;
        if(decode && !request.route_trace.empty()) {
            std::ofstream trace(request.route_trace,std::ios::app);
            if(!trace) throw Error{ErrorCategory::invalid_argument,"Cannot open route trace"};
            trace<<"{\"position\":"<<request.step.e<<",\"source_position\":-1,\"top_k\":"
                 <<model->assets.definition.parameters.at("top_k")<<",\"layers\":[";
            bool first=true;
            for(const auto& route:cached->resident_routes) {
                std::vector<std::int32_t> ids(std::size_t(ggml_nelements(route.second)));
                check(plan->read_output(route.second,ids.data(),0,ids.size()*sizeof(std::int32_t)));
                result->bytes_read+=ids.size()*sizeof(std::int32_t);
                if(!first){trace<<',';} first=false;
                trace<<"{\"layer\":"<<route.first;
                for(const auto* name:{"actual","executed"}) {
                    trace<<",\""<<name<<"\":[";
                    for(std::size_t i=0;i<ids.size();++i) {if(i) trace<<',';trace<<ids[i];}
                    trace<<']';
                }
                trace<<",\"head\":[],\"prepared\":[]}";
            }
            trace<<"]}\n";
            if(!trace) throw Error{ErrorCategory::backend_error,"Cannot write route trace"};
        }

        result->timing=timing.finish();
        return std::unique_ptr<ForwardResult>(std::move(result));
    } catch(const Error& error) { return error; }
      catch(const std::out_of_range&) { return Error{ErrorCategory::incompatible,"Missing forward tensor or metadata"}; }
}
}
