#include "app/generation_internal.h"
#include "edge0/forward_check.h"
#include "app/runtime.h"
#include "model/loading.h"
#include "app/adapters.h"
#include <numeric>
namespace edge0 {
namespace {
struct CheckSession {
    const models::ModelCompute* compute = nullptr;
    std::shared_ptr<model::LoadedModel> model;
    std::shared_ptr<session::KVCache> kv;
    session::StateStore states;
    ~CheckSession() {
        if(model && kv) compute->release_session(model,*kv,states);
        kv.reset();
        if(model) model::request_unload(model);
    }
};
}
Result<ForwardCheckReport> forward_check(const RuntimeReference& runtime,const ForwardCheckOptions& options) {
    if(options.input.empty() || options.input.size()>1024 || options.generated.size()>1024-options.input.size())
        return Error{ErrorCategory::invalid_argument,"Invalid forward token sequence length"};
    auto acquired=RuntimeAccess::acquire(runtime); if(!acquired)return acquired.error();
    auto resources=acquired.value();
    RuntimeActivity activity(resources.registry);
    if (auto error = activity.enter()) return *error;
    if(resources.budget.current()!=0) return Error{ErrorCategory::conflict,"Forward check requires an idle runtime"};
    ForwardCheckReport report;
    auto budget=query_runtime_budget(runtime); if(!budget)return budget.error(); report.budget=budget.value();
    {
        CheckSession owner;
        model::LoadRequest request;
        request.directory=options.model_directory; request.device=options.device;
        auto catalog=model::open_catalog(request.directory,app::compiled_adapters()); if(!catalog)return catalog.error();
        auto selected=app::select_compiled(catalog.value()); if(!selected)return selected.error();
        owner.compute=selected.value().compute;
        auto assets=model::check_compatibility(catalog.value(),*selected.value().adapter);
        if(!assets)return assets.error();
        auto declaration=selected.value().adapter->describe_weight_storage(assets.value().indexes,{});
        if(!declaration)return declaration.error();
        request.options=app::apply_reference_values(declaration.value().configuration,options.model_options);
        request.on_close=[compute=owner.compute](model::LoadedModel& model) { compute->close_model(model); };
        // Limit the Result scope: it also owns the loaded model.
        {
            auto loaded=model::load(request,{selected.value().adapter},resources.budget,resources.backends);
            if(!loaded){return loaded.error();} owner.model=loaded.value();
        }
        report.weight_activity.applicable=!owner.model->pools.empty();
        auto declarations=owner.compute->describe_states(owner.model->assets.definition);
        if(!declarations)return declarations.error();
        {
            auto kv=session::KVCache::create(resources.budget,owner.model->stages.prefill,1024,declarations.value().kv);
            if(!kv){return kv.error();} owner.kv=kv.value();
        }
        if(auto error=owner.compute->initialize_session(owner.model,*owner.kv,owner.states,resources.budget,owner.model->stages))return *error;
        report.vocabulary=owner.model->assets.definition.vocabulary_size;
        report.rows=options.input.size()+options.generated.size();
        std::int64_t e=0;
        const bool split=options.split_last && options.input.size()>=2;
        for(int step=split?-2:-1;step<int(options.generated.size());++step) {
            models::ForwardRequest forward;
            forward.tokens=step<0?options.input:std::vector<std::int32_t>{options.generated[std::size_t(step)]};
            if(split && step==-2) forward.tokens={options.input.begin(),options.input.end()-1};
            // assign(), not = {x}: GCC 12 at -O3 reports a false array-bounds error for a one-element initializer list.
            if(split && step==-1) forward.tokens.assign(1,options.input.back());
            forward.step={e,std::int64_t(forward.tokens.size())};
            forward.route_trace=options.route_trace;
            forward.phase=step<0?exec::Phase::prefill:exec::Phase::decode;
            forward.output_positions.resize(forward.tokens.size());
            std::iota(forward.output_positions.begin(),forward.output_positions.end(),0);
            if(options.observe && step<int(options.observe_decode_steps) && !(split && step==-1))forward.observe=owner.compute->observation_names(*owner.model);
            select_cpu_threads(resources, forward.phase);
            forward.backend=owner.model->stages.of(forward.phase);
            if(owner.model->stages.mixed()) if(auto error=app::prepare_stage_forward(*owner.compute,owner.model,*owner.kv,owner.states,forward,resources.registry.get())) return *error;
            auto result=owner.compute->forward(owner.model,forward,*owner.kv,owner.states,resources.budget);
            if(!result) { owner.compute->finish_pending_weights(owner.model,owner.states,models::FinishReason::failure); return result.error(); }
            const auto& output=result.value();
            const auto abandon=[&](const Error& error) {
                owner.compute->finish_pending_weights(owner.model,owner.states,models::FinishReason::failure);
                return error;
            };
            if(!output->completion().success)return abandon(Error{ErrorCategory::backend_error,"Forward execution failed"});
            auto logits=output->logits(); if(!logits)return abandon(logits.error());
            report.logits.insert(report.logits.end(),logits.value().begin(),logits.value().end());
            auto tensors=output->observations(); if(!tensors)return abandon(tensors.error());
            for(const auto& tensor:tensors.value()) {
                ForwardCheckTensor record; record.name=tensor.name; record.step=step==-2?-1:step; record.position=e;
                record.type=int(tensor.layout.type); record.shape.fill(1);
                std::copy(tensor.layout.dimensions.begin(),tensor.layout.dimensions.end(),record.shape.begin());
                record.data=tensor.data; report.layers.push_back(std::move(record));
            }
            report.weight_activity+=output->weight_activity();
            report.weight_registrations.push_back(output->weight_registrations());
            if(auto error=output->end_observation())return abandon(*error);
            e+=forward.step.n;
        }
        auto finished=owner.compute->finish_pending_weights(owner.model,owner.states,models::FinishReason::normal);
        if(!finished) return finished.error();
        report.weight_activity+=finished.value();
    }
    report.budget_peak=resources.budget.peak(); report.budget_current=resources.budget.current();
    if(report.budget_current)return Error{ErrorCategory::backend_error,"Forward check leaked budget reservations"};
    return report;
}
}
