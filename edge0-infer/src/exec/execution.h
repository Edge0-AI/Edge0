#pragma once
#include "exec/budget.h"
#include "platform/backend.h"
#include "ggml-alloc.h"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace edge0::exec {
using Status = std::optional<Error>;
using Consumption = platform::Consumption;
using ConsumptionQuery = platform::ConsumptionQuery;
struct Backend;
Consumption buffer_consumption(ggml_backend_buffer_type_t, const std::shared_ptr<Backend>&, const ConsumptionQuery& = {});
struct Layout {
    ggml_type type = GGML_TYPE_F32;
    std::vector<std::int64_t> dimensions;
};
struct Backend {
    std::shared_ptr<platform::BackendFacility> facility;
    std::mutex mutex;
    bool busy = false, failed = false;
    ggml_backend_t handle() const { return facility->handle(); }
    std::function<bool(ggml_backend_t, ggml_cgraph*)> confirm = platform::confirm_completion;
};
enum class Phase { prefill, decode };
struct TransferRange { std::size_t offset = 0, bytes = 0; };
using TransferHook = std::function<Status(ggml_tensor*, ggml_tensor*, const std::vector<TransferRange>&)>;
struct StageRuns { std::mutex mutex; bool busy = false; };
struct TransferStats { std::size_t count = 0, bytes = 0; double seconds = 0; };
struct Stages {
    std::shared_ptr<Backend> prefill, decode;
    platform::ConsumptionQuery consumption = {};
    TransferHook transfer = {};
    std::shared_ptr<StageRuns> runs = {};
    bool mixed() const { return prefill != decode; }
    const std::shared_ptr<Backend>& of(Phase phase) const { return phase == Phase::prefill ? prefill : decode; }
};
Result<Stages> resolve_stages(const std::string& prefill, const std::string& decode,
    const std::shared_ptr<platform::BackendFacility>& facility);
Result<Stages> resolve_stages(const std::string& prefill, const std::string& decode,
    const std::map<std::string, std::shared_ptr<platform::BackendFacility>>& facilities);

ggml_backend_buffer_type_t storage_home(const std::vector<std::shared_ptr<Backend>>&, const ConsumptionQuery& = {});

Result<Stages> resolve_stage_backends(const std::string&, const std::string&, const std::map<std::string, std::shared_ptr<Backend>>&);

struct StageFacts {
    std::size_t tokens = 0;
    std::shared_ptr<Backend> state_home;
    std::size_t handoff_bytes = 0;
};
struct StagePolicy {
    std::function<std::vector<std::shared_ptr<Backend>>(const Stages&)> candidates;
    std::function<std::shared_ptr<Backend>(const Stages&, const StageFacts&)> choose_prefill;
};
Result<StagePolicy> stage_policy(const std::string& name);
Result<std::shared_ptr<Backend>> choose_prefill_backend(const Stages&, const StagePolicy&,
    const std::vector<std::shared_ptr<Backend>>& candidates, const StageFacts&);

class Buffer {
public:
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor*> tensors;
    std::shared_ptr<Backend> backend;
    std::vector<Budget::Ticket> charges;
    std::shared_ptr<Buffer> home;
    bool owns_buffer = true;
private:
    Buffer() = default;
    friend Result<std::shared_ptr<Buffer>> allocate_buffer(Budget&, const std::shared_ptr<Backend>&,
        const std::vector<Layout>&, Purpose, std::size_t);
    friend Result<std::shared_ptr<Buffer>> allocate_buffer_on(Budget&, const std::shared_ptr<Backend>&,
        const std::vector<Layout>&, std::size_t, ggml_backend_buffer_type_t, Purpose);
    friend Result<std::shared_ptr<Buffer>> consume_buffer(Budget&, const std::shared_ptr<Buffer>&,
        const std::shared_ptr<Backend>&, platform::Consumption, std::size_t);
};
Result<std::shared_ptr<Buffer>> allocate_buffer(Budget& budget, const std::shared_ptr<Backend>& backend,
    const std::vector<Layout>& layouts, Purpose purpose, std::size_t extra_tensors = 0);
Result<std::shared_ptr<Buffer>> allocate_buffer_on(Budget&, const std::shared_ptr<Backend>&,
    const std::vector<Layout>&, std::size_t, ggml_backend_buffer_type_t, Purpose purpose);
Result<std::shared_ptr<Buffer>> consume_buffer(Budget&, const std::shared_ptr<Buffer>&,
    const std::shared_ptr<Backend>&, platform::Consumption, std::size_t extra_tensors = 0);
std::string buffer_device(ggml_backend_buffer_type_t);
struct StateStorage;
struct StateReplicas {
    explicit StateReplicas(Budget value) : budget(std::move(value)) {}
    Budget budget;
    std::mutex mutex;
    Stages stages;
    bool shared = false;
    std::array<std::shared_ptr<Buffer>, 2> buffers;
    std::array<std::weak_ptr<StateStorage>, 2> views;
    std::array<bool, 2> valid{true, true};
    TransferStats stats;
};
struct StateStorage : std::enable_shared_from_this<StateStorage> {
    explicit StateStorage(std::shared_ptr<Buffer> buffer) : allocation(std::move(buffer)) {}
    std::shared_ptr<Buffer> allocation;
    std::shared_ptr<StateReplicas> replicas;
    std::size_t replica_index = 0;
    ggml_tensor* tensor() const { return allocation->tensors.front(); }
    Result<std::shared_ptr<StateStorage>> on(const std::shared_ptr<Backend>&);
    bool current() const;
    void mark_written();
    Status reset_zero();
};
Result<std::shared_ptr<StateStorage>> allocate_state_storage(Budget& budget,
    const std::shared_ptr<Backend>& backend, const Layout& layout, Purpose purpose);
Result<std::shared_ptr<StateStorage>> allocate_state_storage(Budget&, const Stages&, const Layout&, Purpose);
Status transfer(const std::shared_ptr<StateStorage>&, const std::shared_ptr<Backend>& target,
    const std::vector<TransferRange>& ranges = {}, bool make_valid = true);

class Use {
public:
    virtual ~Use() = default;
    virtual Status release() = 0;
    virtual bool valid() const = 0;
    virtual std::string identity() const = 0;
    virtual bool matches(ggml_tensor* tensor, ggml_backend_t backend) const = 0;
    virtual std::shared_ptr<void> keep_alive() const = 0;
};
enum class Strides { preserve, packed };
enum class ReuseDecision { reuse, rebuild, unsupported };
struct ReuseCheck { ReuseDecision decision; std::string reason; };
class Plan;
ReuseCheck check_plan_reuse(const Plan&, std::int64_t token_count,
    const std::map<ggml_tensor*, std::int64_t>& ranges);
class Graph : public std::enable_shared_from_this<Graph> {
public:
    ~Graph();
    Phase phase() const { return phase_; }
    Status set_token_count(std::int64_t count);
    ggml_context* context() const { return context_; }
    Result<ggml_tensor*> input(const std::string& name, const Layout& layout);
    Result<ggml_tensor*> bind_states(const std::shared_ptr<StateStorage>& storage);
    Result<ggml_tensor*> bind_state_write(const std::shared_ptr<StateStorage>& storage, ggml_tensor* write);
    Status bind_weights(ggml_tensor* tensor, const std::shared_ptr<Backend>& backend, std::shared_ptr<Use> use);
    Status bounded_view(ggml_tensor* view, int axis, Strides strides, ggml_tensor* consumer);
    Status output(ggml_tensor* tensor);
    Status retain_output(ggml_tensor* tensor, const std::shared_ptr<StateStorage>& storage);
    Status coordinate(const Stages& stages);
private:
    friend class Plan;
    friend ReuseCheck check_plan_reuse(const Plan&, std::int64_t, const std::map<ggml_tensor*, std::int64_t>&);
    friend Result<std::shared_ptr<Graph>> create_graph(Budget&, const std::shared_ptr<Backend>&, Phase, std::size_t);
    friend Result<std::shared_ptr<Plan>> build_plan(const std::shared_ptr<Graph>&,
        std::function<bool(ggml_backend_t, const ggml_tensor*)>);
    void release_resources();
    struct StateBinding {
        std::weak_ptr<StateStorage> owner;
        std::shared_ptr<Buffer> allocation;
        ggml_tensor snapshot;
        ggml_tensor* write = nullptr;
        std::shared_ptr<StateStorage> replica_owner;
    };
    struct Bound { ggml_tensor* view; int axis; std::int64_t maximum; Strides strides; ggml_tensor* consumer; };
    Graph(Budget budget, std::shared_ptr<Backend> backend, Phase phase)
        : budget_(std::move(budget)), backend_(std::move(backend)), phase_(phase) {}
    Budget budget_;
    std::shared_ptr<Backend> backend_;
    std::shared_ptr<StageRuns> runs_;
    Phase phase_;
    std::int64_t token_count_ = 1;
    Budget::Ticket metadata_;
    ggml_context* context_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_cgraph* empty_ = nullptr;
    bool built_ = false;
    std::map<std::string, ggml_tensor*> inputs_;
    std::map<ggml_tensor*, StateBinding> states_;
    std::map<ggml_tensor*, std::shared_ptr<Use>> weights_;
    std::map<ggml_tensor*, std::shared_ptr<void>> weight_storage_;
    std::vector<Bound> bounds_;
    std::set<ggml_tensor*> outputs_;
};
Result<std::shared_ptr<Graph>> create_graph(Budget& budget, const std::shared_ptr<Backend>& backend,
    Phase phase, std::size_t nodes = 256);
struct Completion {
    bool success = false;
    bool writes_may_be_partial = false;
    std::vector<std::shared_ptr<Buffer>> written;
};
class Plan {
public:
    ~Plan();
    Status set_inputs(const std::string& name, const void* data, std::size_t bytes);
    Status set_valid_ranges(ggml_tensor* view, std::int64_t length);
    Status bind_weights(ggml_tensor* tensor, std::shared_ptr<Use> use);
    Status submit();
    Result<Completion> collect_completion();
    Status read_output(ggml_tensor* tensor, void* destination, std::size_t offset, std::size_t bytes);
    Status end_observation();
    Status release_plan();
private:
    friend Result<std::shared_ptr<Plan>> build_plan(const std::shared_ptr<Graph>&,
        std::function<bool(ggml_backend_t, const ggml_tensor*)>);
    friend ReuseCheck check_plan_reuse(const Plan&, std::int64_t, const std::map<ggml_tensor*, std::int64_t>&);
    explicit Plan(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}
    void clear_pending();
    Status reject(Error error);
    std::shared_ptr<Graph> graph_;
    ggml_gallocr_t allocator_ = nullptr;
    Budget::Ticket workspace_;
    std::set<std::string> inputs_set_;
    std::set<ggml_tensor*> ranges_set_;
    bool running_ = false, completed_ = false, success_ = false;
    ggml_status submitted_ = GGML_STATUS_SUCCESS;
};
Result<std::shared_ptr<Plan>> build_plan(const std::shared_ptr<Graph>& graph,
    std::function<bool(ggml_backend_t, const ggml_tensor*)> supports = ggml_backend_supports_op);
}
