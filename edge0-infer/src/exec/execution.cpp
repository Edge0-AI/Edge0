#include "exec/execution.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace edge0::exec {
namespace {
Error invalid(const std::string& message) { return {ErrorCategory::invalid_argument, message}; }
Error conflict(const std::string& message) { return {ErrorCategory::conflict, message}; }
Error failed(const std::string& message) { return {ErrorCategory::backend_error, message}; }
Status shape(const Layout& layout, ggml_tensor& tensor) {
    if (layout.dimensions.empty() || layout.dimensions.size() > GGML_MAX_DIMS ||
        layout.type < 0 || layout.type >= GGML_TYPE_COUNT || ggml_type_size(layout.type) == 0)
        return invalid("Invalid tensor layout");
    tensor = {};
    tensor.type = layout.type;
    std::fill(std::begin(tensor.ne), std::end(tensor.ne), 1);
    std::copy(layout.dimensions.begin(), layout.dimensions.end(), tensor.ne);
    const auto block = ggml_blck_size(layout.type);
    if (tensor.ne[0] <= 0 || tensor.ne[0] % block) return invalid("Invalid block dimension");
    tensor.nb[0] = ggml_type_size(layout.type);
    std::size_t bytes = tensor.nb[0];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const auto count = i == 0 ? tensor.ne[i] / block : tensor.ne[i];
        if (count <= 0 || static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / bytes)
            return invalid("Tensor size overflow");
        bytes *= static_cast<std::size_t>(count);
        if (i + 1 < GGML_MAX_DIMS) tensor.nb[i + 1] = bytes;
    }
    return {};
}
ggml_tensor* base(ggml_tensor* tensor) {
    while (tensor && tensor->view_src) tensor = tensor->view_src;
    return tensor;
}
bool same_layout(const ggml_tensor& a, const ggml_tensor& b) {
    return a.type == b.type && a.buffer == b.buffer && a.data == b.data &&
        std::equal(std::begin(a.ne), std::end(a.ne), b.ne) &&
        std::equal(std::begin(a.nb), std::end(a.nb), b.nb);
}
}
Result<Stages> resolve_stages(const std::string& prefill, const std::string& decode,
    const std::shared_ptr<platform::BackendFacility>& facility) {
    if (prefill != decode) return Error{ErrorCategory::unsupported, "Mixed stage devices are unsupported"};
    if (!facility || !*facility || prefill != ggml_backend_dev_name(facility->device()))
        return invalid("Stage device does not match initialized facility");
    static std::mutex mutex;
    static std::map<ggml_backend_t, std::weak_ptr<Backend>> instances;
    std::lock_guard<std::mutex> lock(mutex);
    auto backend = instances[facility->handle()].lock();
    if (!backend) {
        backend = std::make_shared<Backend>();
        backend->facility = facility;
        instances[facility->handle()] = backend;
    }
    return Stages{backend, backend};
}
Buffer::~Buffer() {
    if (buffer && owns_buffer) ggml_backend_buffer_free(buffer);
    if (context) ggml_free(context);
}
Result<std::shared_ptr<Buffer>> allocate_buffer(Budget& budget, const std::shared_ptr<Backend>& backend,
    const std::vector<Layout>& layouts, Purpose purpose, std::size_t extra_tensors) {
    return allocate_buffer_on(budget, backend, layouts, extra_tensors,
        backend ? ggml_backend_get_default_buffer_type(backend->handle()) : nullptr, purpose);
}
Result<std::shared_ptr<Buffer>> allocate_buffer_on(Budget& budget, const std::shared_ptr<Backend>& backend,
    const std::vector<Layout>& layouts, std::size_t extra_tensors, ggml_backend_buffer_type_t type, Purpose purpose) {
    if (!backend || !type || layouts.empty()) return invalid("Missing backend or layouts");
    const auto alignment = ggml_backend_buft_get_alignment(type);
    std::size_t size = 0;
    for (const auto& layout : layouts) {
        ggml_tensor tensor{};
        if (auto error = shape(layout, tensor)) return *error;
        const auto bytes = ggml_backend_buft_get_alloc_size(type, &tensor);
        if (bytes > std::numeric_limits<std::size_t>::max() - alignment ||
            size > std::numeric_limits<std::size_t>::max() - (bytes + alignment)) return invalid("Allocation size overflow");
        size += (bytes + alignment - 1) / alignment * alignment;
    }
    if (size > ggml_backend_buft_get_max_size(type)) return invalid("Storage exceeds backend buffer limit");
    if (extra_tensors > std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead() - layouts.size())
        return invalid("Metadata size overflow");
    const auto metadata = (layouts.size() + extra_tensors) * ggml_tensor_overhead();
    auto reserved = budget.reserve(std::vector<Budget::Charge>{{size, purpose, buffer_device(type)}, {metadata, Purpose::metadata, "host"}});
    if (!reserved) return reserved.error();
    auto result = std::shared_ptr<Buffer>(new Buffer);
    result->backend = backend;
    result->charges = reserved.value();
    result->context = ggml_init({metadata, nullptr, true});
    if (!result->context) return failed("Storage metadata allocation failed");
    for (const auto& layout : layouts)
        result->tensors.push_back(ggml_new_tensor(result->context, layout.type, static_cast<int>(layout.dimensions.size()), layout.dimensions.data()));
    result->buffer = ggml_backend_buft_alloc_buffer(type, size);
    if (!result->buffer) return failed("Backend buffer allocation failed");
    if (auto error = result->charges[0]->resize(ggml_backend_buffer_get_size(result->buffer))) return *error;
    auto allocator = ggml_tallocr_new(result->buffer);
    for (auto* tensor : result->tensors)
        if (ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) return failed("Tensor allocation failed");
    return result;
}
Result<std::shared_ptr<StateStorage>> allocate_state_storage(Budget& budget,
    const std::shared_ptr<Backend>& backend, const Layout& layout, Purpose purpose) {
    auto buffer = allocate_buffer(budget, backend, {layout}, purpose);
    if (!buffer) return buffer.error();
    return std::make_shared<StateStorage>(buffer.value());
}
Graph::~Graph() { release_resources(); }
void Graph::release_resources() {
    for (auto& item : weights_) if (item.second && item.second->valid()) item.second->release();
    weights_.clear(); weight_storage_.clear(); states_.clear();
    inputs_.clear(); bounds_.clear(); outputs_.clear();
    if (context_) { ggml_free(context_); context_ = nullptr; }
    graph_ = nullptr; empty_ = nullptr; metadata_.reset();
}
Result<std::shared_ptr<Graph>> create_graph(Budget& budget, const std::shared_ptr<Backend>& backend,
    Phase phase, std::size_t nodes) {
    if (!backend || nodes == 0 || nodes > 1000000) return invalid("Invalid graph size or backend");
    const auto bytes = nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false) + ggml_graph_overhead_custom(1, false);
    auto reserved = budget.reserve(bytes, Purpose::metadata, "host");
    if (!reserved) return reserved.error();
    auto graph = std::shared_ptr<Graph>(new Graph(budget, backend, phase));
    graph->metadata_ = reserved.value();
    graph->context_ = ggml_init({bytes, nullptr, true});
    if (!graph->context_) return failed("Graph metadata allocation failed");
    graph->graph_ = ggml_new_graph_custom(graph->context_, nodes, false);
    graph->empty_ = ggml_new_graph_custom(graph->context_, 1, false);
    return graph;
}
Status Graph::set_token_count(std::int64_t count) {
    if (built_ || count <= 0) return invalid("Invalid token count or graph already built");
    token_count_ = count;
    return {};
}
ReuseCheck check_plan_reuse(const Plan& plan, std::int64_t token_count,
    const std::map<ggml_tensor*, std::int64_t>& ranges) {
    const auto& graph = plan.graph_;
    if (!graph) return {ReuseDecision::rebuild, "plan_released"};
    {
        std::lock_guard<std::mutex> lock(graph->backend_->mutex);
        if (graph->backend_->failed) return {ReuseDecision::unsupported, "backend_failed"};
    }
    if (plan.running_) return {ReuseDecision::rebuild, "plan_observed"};
    if (token_count != graph->token_count_) return {ReuseDecision::rebuild, "token_count_changed"};
    for (const auto& state : graph->states_) {
        const auto owner = state.second.owner.lock();
        if (!owner || owner->allocation != state.second.allocation ||
            state.second.allocation->backend != graph->backend_ || !same_layout(*state.first, state.second.snapshot))
            return {ReuseDecision::rebuild, "state_storage_changed"};
    }
    if (ranges.size() != graph->bounds_.size()) return {ReuseDecision::rebuild, "range_set_changed"};
    for (const auto& bound : graph->bounds_) {
        const auto it = ranges.find(bound.view);
        if (it == ranges.end()) return {ReuseDecision::rebuild, "range_set_changed"};
        if (it->second <= 0 || it->second > bound.maximum) return {ReuseDecision::rebuild, "range_out_of_bounds"};
    }
    return {ReuseDecision::reuse, "compatible"};
}
Result<ggml_tensor*> Graph::input(const std::string& name, const Layout& layout) {
    if (built_ || inputs_.count(name) || name.empty()) return invalid("Invalid input declaration: " + name);
    ggml_tensor tensor{};
    if (auto error = shape(layout, tensor)) return *error;
    auto* result = ggml_new_tensor(context_, layout.type, static_cast<int>(layout.dimensions.size()), layout.dimensions.data());
    ggml_set_input(result); ggml_set_name(result, name.c_str()); inputs_[name] = result;
    return result;
}
Result<ggml_tensor*> Graph::bind_states(const std::shared_ptr<StateStorage>& storage) {
    if (built_ || !storage || storage->allocation->backend != backend_) return invalid("State backend instance mismatch or graph is built");
    if (storage->replicas) {
        if (runs_ && runs_ != storage->replicas->stages.runs) return invalid("State belongs to another stage group");
        runs_ = storage->replicas->stages.runs;
    }
    auto* tensor = storage->tensor();
    states_.try_emplace(tensor, StateBinding{storage, storage->allocation, *tensor, nullptr, storage->replicas ? storage : nullptr});
    return tensor;
}
Result<ggml_tensor*> Graph::bind_state_write(const std::shared_ptr<StateStorage>& storage, ggml_tensor* write) {
    auto bound = bind_states(storage);
    if (!bound) return bound.error();
    ggml_tensor* destination = nullptr;
    if (write && write->op == GGML_OP_CPY) destination = write->src[1];
    if (write && write->op == GGML_OP_SET_ROWS) destination = write->src[2];
    if (!destination || base(destination) != storage->tensor()) return invalid("Write node does not target state");
    auto& binding = states_.at(storage->tensor());
    if (binding.write && binding.write != write) return invalid("Multiple state writes require separate plans");
    binding.write = write;
    ggml_build_forward_expand(graph_, write);
    return write;
}
Status Graph::bind_weights(ggml_tensor* tensor, const std::shared_ptr<Backend>& backend, std::shared_ptr<Use> use) {
    if (built_ || !tensor || !use || !use->valid() || backend != backend_ || !use->matches(tensor, backend_->handle())) {
        const bool held = std::any_of(weights_.begin(), weights_.end(), [&](const auto& item) { return item.second == use; });
        if (use && !held && use->valid()) use->release();
        return invalid("Weight backend instance mismatch or invalid use");
    }
    if (weights_.count(tensor)) {
        if (weights_.at(tensor) != use) use->release();
        return conflict("Weight already bound");
    }
    weight_storage_[tensor] = use->keep_alive();
    weights_[tensor] = std::move(use);
    return {};
}
Status Graph::bounded_view(ggml_tensor* view, int axis, Strides strides, ggml_tensor* consumer) {
    if (built_ || !view || view->op != GGML_OP_VIEW || axis < 0 || axis >= GGML_MAX_DIMS)
        return invalid("Invalid bounded view");
    if (!consumer || consumer->op != GGML_OP_FLASH_ATTN_EXT ||
        (consumer->src[1] != view && consumer->src[2] != view && consumer->src[3] != view))
        return Error{ErrorCategory::unsupported, "Bounded views require flash_attn_ext K, V or mask consumers"};
    if ((consumer->src[3] == view && axis != 0) || (consumer->src[3] != view && axis != 1))
        return Error{ErrorCategory::unsupported, "Bounded axis must preserve attention output shape"};
    if (std::any_of(bounds_.begin(), bounds_.end(), [view](const Bound& b) { return b.view == view; }))
        return conflict("View already registered");
    bounds_.push_back({view, axis, view->ne[axis], strides, consumer});
    return {};
}
Status Graph::output(ggml_tensor* tensor) {
    if (built_ || !tensor) return invalid("Invalid graph output");
    ggml_set_output(tensor); outputs_.insert(tensor); ggml_build_forward_expand(graph_, tensor); return {};
}
Status Graph::retain_output(ggml_tensor* tensor, const std::shared_ptr<StateStorage>& storage) {
    auto target = bind_states(storage);
    if (!target) return target.error();
    if (!ggml_are_same_shape(tensor, target.value()) || tensor->type != target.value()->type)
        return invalid("Retained output layout mismatch");
    auto write = bind_state_write(storage, ggml_cpy(context_, tensor, target.value()));
    if (!write) return write.error();
    return output(write.value());
}
Result<std::shared_ptr<Plan>> build_plan(const std::shared_ptr<Graph>& graph,
    std::function<bool(ggml_backend_t, const ggml_tensor*)> supports) {
    if (!graph || graph->built_) return invalid("Graph is missing or already built");
    bool committed = false;
    const auto rollback = [&](Graph*) {
        if (!committed) for (auto& item : graph->weights_) {
            if (item.second && item.second->valid()) item.second->release();
            item.second.reset();
        }
    };
    const std::unique_ptr<Graph, decltype(rollback)> guard(graph.get(), rollback);
    if (graph->outputs_.empty()) return invalid("Graph has no outputs");
    for (const auto& weight : graph->weights_)
        if (!weight.second || !weight.second->valid()) return conflict("Missing weight use");
    std::unordered_set<ggml_tensor*> visited;
    std::function<Status(ggml_tensor*)> check = [&](ggml_tensor* tensor) -> Status {
        if (!tensor || !visited.insert(tensor).second) return {};
        if (graph->weights_.count(tensor)) return {};
        if ((tensor->buffer || tensor->data) && !tensor->view_src && !graph->states_.count(tensor) && !graph->weights_.count(tensor))
            return invalid("Unregistered external tensor: " + std::string(tensor->name));
        for (auto* source : tensor->src) if (auto error = check(source)) return error;
        return {};
    };
    // Output roots include leaf-only graphs; recursion also checks every input leaf.
    for (auto* output : graph->outputs_)
        if (auto error = check(output)) return *error;
    for (int i = 0; i < ggml_graph_n_nodes(graph->graph_); ++i) {
        auto* node = ggml_graph_node(graph->graph_, i);
        if (auto error = check(node)) return *error;
        if (!supports(graph->backend_->handle(), node))
            return Error{ErrorCategory::unsupported, std::string(ggml_op_name(node->op)) + "/" + ggml_op_desc(node) + ": " + ggml_type_name(node->src[0] ? node->src[0]->type : node->type)};
        for (const auto& bound : graph->bounds_) {
            for (int j = 0; j < GGML_MAX_SRC; ++j)
                if (node->src[j] == bound.view && (node->op != GGML_OP_FLASH_ATTN_EXT || j < 1 || j > 3))
                    return invalid("Bounded view has an unregistered consumer");
        }
    }
    for (const auto& entry : graph->states_) {
        auto* write = entry.second.write;
        if (!write) continue;
        // A computed ancestor executes before this write and can also feed an
        // output. Pure aliases do not capture the old value or order its reads.
        std::unordered_set<ggml_tensor*> ancestors, computed_before_write;
        std::function<void(ggml_tensor*)> collect_ancestors = [&](ggml_tensor* tensor) {
            if (!tensor || !ancestors.insert(tensor).second) return;
            switch (tensor->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
                break;
            default:
                computed_before_write.insert(tensor);
                break;
            }
            for (auto* source : tensor->src) collect_ancestors(source);
        };
        for (auto* source : write->src) collect_ancestors(source);
        std::unordered_set<ggml_tensor*> reads;
        std::function<Status(ggml_tensor*)> check_read = [&](ggml_tensor* tensor) -> Status {
            if (!tensor || tensor == write || computed_before_write.count(tensor) ||
                !reads.insert(tensor).second) return {};
            if (tensor == entry.first) return invalid("State read bypasses its write node");
            for (auto* source : tensor->src) if (auto error = check_read(source)) return error;
            return {};
        };
        for (auto* output : graph->outputs_)
            if (auto error = check_read(output)) return *error;
        for (const auto& other : graph->states_)
            if (auto error = check_read(other.second.write)) return *error;
    }
    auto plan = std::shared_ptr<Plan>(new Plan(graph));
    plan->allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend_->handle()));
    std::size_t bytes = 0;
    ggml_gallocr_reserve_n_size(plan->allocator_, graph->graph_, nullptr, nullptr, &bytes);
    auto reserved = graph->budget_.reserve(bytes, Purpose::workspace, buffer_device(ggml_backend_get_default_buffer_type(graph->backend_->handle())));
    if (!reserved) return reserved.error();
    plan->workspace_ = reserved.value();
    if (!ggml_gallocr_reserve(plan->allocator_, graph->graph_) ||
        !ggml_gallocr_alloc_graph(plan->allocator_, graph->graph_)) return failed("Graph workspace allocation failed");
    if (auto error = plan->workspace_->resize(ggml_gallocr_get_buffer_size(plan->allocator_, 0))) return *error;
    graph->built_ = true;
    committed = true;
    return plan;
}
Plan::~Plan() {
    if (running_) { collect_completion(); end_observation(); }
    clear_pending();
    if (allocator_) ggml_gallocr_free(allocator_);
    if (graph_ && graph_->built_) graph_->release_resources();
}
void Plan::clear_pending() {
    if (!graph_) return;
    for (auto& item : graph_->weights_) {
        if (item.second && item.second->valid()) item.second->release();
        item.second.reset();
    }
    inputs_set_.clear(); ranges_set_.clear();
}
Status Plan::reject(Error error) { clear_pending(); return error; }
Status Plan::set_inputs(const std::string& name, const void* data, std::size_t bytes) {
    if (!graph_ || running_) return conflict("Plan is released or still observed");
    const auto it = graph_->inputs_.find(name);
    if (it == graph_->inputs_.end() || !data) return reject(invalid("Unknown or empty input: " + name));
    auto expected = ggml_nbytes(it->second);
    for (const auto& bound : graph_->bounds_) {
        if (bound.strides == Strides::packed && base(bound.view) == it->second) {
            if (!ranges_set_.count(bound.view)) return reject(invalid("Set packed range before input: " + name));
            expected = ggml_nbytes(bound.view);
        }
    }
    if (bytes != expected) return reject(invalid("Input byte count mismatch: " + name));
    ggml_backend_tensor_set(it->second, data, 0, bytes); inputs_set_.insert(name); return {};
}
Status Plan::set_valid_ranges(ggml_tensor* view, std::int64_t length) {
    if (!graph_ || running_) return conflict("Plan is released or still observed");
    const auto it = std::find_if(graph_->bounds_.begin(), graph_->bounds_.end(), [view](const Graph::Bound& b) { return b.view == view; });
    if (it == graph_->bounds_.end() || length <= 0 || length > it->maximum)
        return reject(invalid("Valid range exceeds registered bound"));
    view->ne[it->axis] = length;
    if (it->strides == Strides::packed) {
        view->nb[0] = ggml_type_size(view->type);
        view->nb[1] = ggml_row_size(view->type, view->ne[0]);
        for (int i = 2; i < GGML_MAX_DIMS; ++i) view->nb[i] = view->nb[i-1] * static_cast<std::size_t>(view->ne[i-1]);
    }
    ranges_set_.insert(view); return {};
}
Status Plan::bind_weights(ggml_tensor* tensor, std::shared_ptr<Use> use) {
    if (!graph_ || running_ || !graph_->weights_.count(tensor) || !use || !use->valid() || !use->matches(tensor, graph_->backend_->handle())) {
        const bool held = graph_ && std::any_of(graph_->weights_.begin(), graph_->weights_.end(), [&](const auto& item) { return item.second == use; });
        if (use && !held && use->valid()) use->release();
        return conflict("Invalid weight use or observed plan");
    }
    auto& previous = graph_->weights_[tensor];
    if (previous == use) return {};
    if (previous && previous->valid()) previous->release();
    previous = std::move(use); return {};
}
Status Plan::submit() {
    if (!graph_ || running_) return conflict("Plan is released or still observed");
    std::unique_lock<std::mutex> stage_lock;
    if (graph_->runs_) {
        stage_lock = std::unique_lock<std::mutex>(graph_->runs_->mutex);
        if (graph_->runs_->busy) return reject(conflict("Stages have an uncollected run"));
    }
    std::lock_guard<std::mutex> lock(graph_->backend_->mutex);
    if (graph_->backend_->failed) return reject(failed("Backend instance has failed"));
    if (graph_->backend_->busy) return reject(conflict("Backend has an uncollected run"));
    for (const auto& input : graph_->inputs_)
        if (!inputs_set_.count(input.first)) return reject(invalid("Missing input: " + input.first));
    for (const auto& bound : graph_->bounds_)
        if (!ranges_set_.count(bound.view)) return reject(invalid("Missing valid range"));
    for (const auto& state : graph_->states_) {
        const auto owner = state.second.owner.lock();
        if (!owner || owner->allocation != state.second.allocation ||
            state.second.allocation->backend != graph_->backend_ || !same_layout(*state.first, state.second.snapshot))
            return reject(conflict("State storage replaced or released"));
        if (!owner->current()) return reject(conflict("State replica is stale"));
    }
    for (const auto& weight : graph_->weights_)
        if (!weight.second || !weight.second->valid()) return reject(conflict("Missing weight use"));
    graph_->backend_->busy = true;
    if (graph_->runs_) graph_->runs_->busy = true;
    running_ = true; completed_ = false; success_ = false;
    submitted_ = ggml_backend_graph_compute_async(graph_->backend_->handle(), graph_->graph_);
    return {};
}
Result<Completion> Plan::collect_completion() {
    if (!graph_ || !running_) return conflict("No active run");
    if (!completed_) {
        std::unique_lock<std::mutex> stage_lock;
        if (graph_->runs_) stage_lock = std::unique_lock<std::mutex>(graph_->runs_->mutex);
        std::lock_guard<std::mutex> lock(graph_->backend_->mutex);
        const bool confirmed = graph_->backend_->confirm(graph_->backend_->handle(), graph_->empty_);
        success_ = submitted_ == GGML_STATUS_SUCCESS && confirmed;
        completed_ = true; graph_->backend_->busy = false;
        if (!success_) graph_->backend_->failed = true;
        if (success_) for (const auto& entry : graph_->states_) if (entry.second.write) {
            if (auto owner = entry.second.owner.lock()) owner->mark_written();
        }
        if (graph_->runs_) graph_->runs_->busy = false;
    }
    Completion result{success_, !success_, {}};
    for (const auto& entry : graph_->states_) if (entry.second.write) result.written.push_back(entry.second.allocation);
    return result;
}
Status Plan::read_output(ggml_tensor* tensor, void* destination, std::size_t offset, std::size_t bytes) {
    if (!graph_ || !running_ || !completed_ || !success_) return conflict("Output is not successfully completed and observed");
    if (!graph_->outputs_.count(tensor) || !destination || offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - offset)
        return invalid("Invalid output range");
    ggml_backend_tensor_get(tensor, destination, offset, bytes); return {};
}
Status Plan::end_observation() {
    if (!running_ || !completed_) return conflict("Run must complete before ending observation");
    clear_pending(); running_ = false; return {};
}
Status Plan::release_plan() {
    if (running_) return conflict("Run is still observed");
    clear_pending();
    if (allocator_) { ggml_gallocr_free(allocator_); allocator_ = nullptr; }
    workspace_.reset();
    if (graph_) graph_->release_resources();
    graph_.reset(); return {};
}
}

namespace edge0::exec {
Result<Stages> resolve_stages(const std::string& prefill, const std::string& decode,
    const std::map<std::string, std::shared_ptr<platform::BackendFacility>>& facilities) {
    const auto a = facilities.find(prefill), b = facilities.find(decode);
    if (a == facilities.end() || b == facilities.end() || !a->second || !b->second ||
        !*a->second || !*b->second)
        return Error{ErrorCategory::unsupported, "Stage device is outside runtime scope"};
    // Registry aliases identify facilities; the facility retains the native device name.
    auto first = resolve_stages(ggml_backend_dev_name(a->second->device()),
        ggml_backend_dev_name(a->second->device()), a->second);
    if (!first) return first.error();
    if (a->second->handle() == b->second->handle()) return first;
    auto second = resolve_stages(ggml_backend_dev_name(b->second->device()),
        ggml_backend_dev_name(b->second->device()), b->second);
    if (!second) return second.error();
    Stages stages{first.value().prefill, second.value().decode};
    stages.runs = std::make_shared<StageRuns>();
    return stages;
}
Result<StagePolicy> stage_policy(const std::string& name) {
    static const std::map<std::string, StagePolicy> policies{
        {"fixed", {
            [](const Stages& stages) { return std::vector<std::shared_ptr<Backend>>{stages.prefill}; },
            [](const Stages& stages, const StageFacts&) { return stages.prefill; }}}
    };
    const auto found = policies.find(name);
    if (found == policies.end()) return Error{ErrorCategory::invalid_argument, "Unknown stage policy: " + name};
    return found->second;
}
Result<std::shared_ptr<Backend>> choose_prefill_backend(const Stages& stages, const StagePolicy& policy,
    const std::vector<std::shared_ptr<Backend>>& candidates, const StageFacts& facts) {
    if (!policy.choose_prefill || candidates.empty())
        return Error{ErrorCategory::invalid_argument, "Stage policy has no candidates or selection function"};
    for (const auto& candidate : candidates)
        if (!candidate || (candidate != stages.prefill && candidate != stages.decode))
            return Error{ErrorCategory::invalid_argument, "Stage policy candidate is outside configured stages"};
    auto selected = policy.choose_prefill(stages, facts);
    if (!selected || std::find(candidates.begin(), candidates.end(), selected) == candidates.end())
        return Error{ErrorCategory::invalid_argument, "Stage policy selected a backend outside its candidates"};
    return selected;
}
}

namespace edge0::exec {
std::string buffer_device(ggml_backend_buffer_type_t type) {
    auto device = type ? ggml_backend_buft_get_device(type) : nullptr;
    if (!device && type == ggml_backend_cpu_buffer_type())
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return device ? ggml_backend_dev_name(device) : "host";
}
}
