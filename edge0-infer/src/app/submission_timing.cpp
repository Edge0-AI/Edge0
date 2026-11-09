#include "edge0/generation_check.h"
#include "app/runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
namespace edge0 {
Result<TimingDistribution> measure_submission_overhead(const RuntimeReference& runtime,
    const std::string& device, std::size_t samples) {
    if(samples==0) return Error{ErrorCategory::invalid_argument,"Samples must be positive"};
    auto resources=RuntimeAccess::acquire(runtime); if(!resources) return resources.error();
    auto r=resources.value();
    RuntimeActivity activity(r.registry);
    if (auto error = activity.enter()) return *error;
    select_cpu_threads(r, exec::Phase::decode);
    auto backend=r.backends.find(device);
    if(backend==r.backends.end()) return Error{ErrorCategory::invalid_argument,"Device is outside runtime scope"};
    auto graph=exec::create_graph(r.budget,backend->second,exec::Phase::decode);
    if(!graph) return graph.error();
    auto input=graph.value()->input("input",{GGML_TYPE_F32,{1}}); if(!input) return input.error();
    auto* output=ggml_sqr(graph.value()->context(),input.value());
    if(auto error=graph.value()->output(output)) return *error;
    auto plan=exec::build_plan(graph.value()); if(!plan) return plan.error();
    std::vector<double> times; times.reserve(samples);
    const float value=1;
    for(std::size_t i=0;i<samples;++i) {
        if(auto error=plan.value()->set_inputs("input",&value,sizeof(value))) return *error;
        const auto start=std::chrono::steady_clock::now();
        if(auto error=plan.value()->submit()) return *error;
        auto completed=plan.value()->collect_completion(); if(!completed) return completed.error();
        const auto end=std::chrono::steady_clock::now();
        if(!completed.value().success) return Error{ErrorCategory::backend_error,"Submission timing failed"};
        times.push_back(std::chrono::duration<double>(end-start).count());
        if(auto error=plan.value()->end_observation()) return *error;
    }
    std::sort(times.begin(),times.end());
    const auto median=(times[(samples-1)/2]+times[samples/2])/2;
    return TimingDistribution{median,times[std::size_t(std::ceil(.95*static_cast<double>(samples)))-1],times.back(),samples};
}
}
