#include "exec/execution.h"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace edge0::exec {
namespace {
Error invalid(const char* message) { return {ErrorCategory::invalid_argument, message}; }
Error conflict(const char* message) { return {ErrorCategory::conflict, message}; }
}
Result<std::shared_ptr<Buffer>> consume_buffer(Budget& budget, const std::shared_ptr<Buffer>& home,
    const std::shared_ptr<Backend>& consumer, platform::Consumption mode, std::size_t extra_tensors) {
    if (!home || !consumer || mode == platform::Consumption::copy)
        return invalid("Buffer cannot be consumed without copying");
    if (consumer == home->backend && mode == platform::Consumption::direct) return home;
    const auto metadata = (home->tensors.size() + extra_tensors) * ggml_tensor_overhead();
    auto charge = budget.reserve(metadata, Purpose::metadata, "host");
    if (!charge) return charge.error();
    auto result = std::shared_ptr<Buffer>(new Buffer);
    result->charges.push_back(charge.value());
    result->backend = consumer; result->home = home;
    result->context = ggml_init({metadata, nullptr, true});
    if (!result->context) return Error{ErrorCategory::backend_error, "Buffer view metadata allocation failed"};
    if (mode == platform::Consumption::direct) {
        result->buffer = home->buffer; result->owns_buffer = false;
    } else {
        std::size_t maximum = 0;
        for (const auto* tensor : home->tensors) maximum = std::max(maximum, ggml_nbytes(tensor));
        result->buffer = platform::wrap_host_buffer(home->buffer, consumer->handle(), maximum);
        if (!result->buffer) return Error{ErrorCategory::unsupported, "Host buffer wrapping failed"};
    }
    for (auto* tensor : home->tensors) {
        auto* alias = ggml_dup_tensor(result->context, tensor);
        if (ggml_backend_tensor_alloc(result->buffer, alias, tensor->data) != GGML_STATUS_SUCCESS)
            return Error{ErrorCategory::backend_error, "Buffer view allocation failed"};
        result->tensors.push_back(alias);
    }
    return result;
}
Result<std::shared_ptr<StateStorage>> allocate_state_storage(Budget& budget, const Stages& stages, const Layout& layout, Purpose purpose) {
    if (!stages.prefill || !stages.decode) return invalid("Missing stage backend");
    if (!stages.mixed()) return allocate_state_storage(budget, stages.prefill, layout, purpose);
    if (!stages.runs) return invalid("Mixed stages have no run coordinator");
    auto group = std::make_shared<StateReplicas>(budget);
    group->stages = stages;
    const auto home = platform::choose_storage_home({stages.prefill->handle(), stages.decode->handle()}, stages.consumption);
    group->shared = home != nullptr;
    const std::array<std::shared_ptr<Backend>, 2> backends{stages.prefill, stages.decode};
    if (home) {
        // The owner is a logical backend. The chosen buffer type determines physical placement.
        auto buffer = allocate_buffer_on(budget, stages.prefill, {layout}, 0, home, purpose);
        if (!buffer) return buffer.error();
        for (std::size_t i = 0; i < backends.size(); ++i) {
            const auto mode = stages.consumption ? stages.consumption(home, backends[i]->handle()) :
                platform::consumption(home, backends[i]->handle());
            // The home may require a wrapper even for the first logical backend.
            auto view = consume_buffer(budget, buffer.value(), backends[i], mode);
            if (!view) return view.error();
            group->buffers[i] = view.value();
        }
        ggml_backend_buffer_clear(buffer.value()->buffer, 0);
    } else {
        for (std::size_t i = 0; i < backends.size(); ++i) {
            auto buffer = allocate_buffer_on(budget, backends[i], {layout}, 0,
                ggml_backend_get_default_buffer_type(backends[i]->handle()), purpose);
            if (!buffer) return buffer.error();
            group->buffers[i] = buffer.value();
            ggml_backend_buffer_clear(buffer.value()->buffer, 0);
        }
    }
    auto result = std::make_shared<StateStorage>(group->buffers[0]);
    result->replicas = group; group->views[0] = result;
    return result;
}
Result<std::shared_ptr<StateStorage>> StateStorage::on(const std::shared_ptr<Backend>& backend) {
    if (!replicas) {
        if (allocation->backend != backend) return invalid("State backend mismatch");
        return shared_from_this();
    }
    std::lock_guard<std::mutex> lock(replicas->mutex);
    for (std::size_t i = 0; i < replicas->buffers.size(); ++i) if (replicas->buffers[i]->backend == backend) {
        if (auto cached = replicas->views[i].lock()) return cached;
        auto result = std::make_shared<StateStorage>(replicas->buffers[i]);
        result->replicas = replicas; result->replica_index = i;
        replicas->views[i] = result;
        return result;
    }
    return invalid("State backend is outside its replica group");
}
bool StateStorage::current() const {
    if (!replicas) return true;
    std::lock_guard<std::mutex> lock(replicas->mutex);
    return replicas->valid[replica_index];
}
void StateStorage::mark_written() {
    if (!replicas || replicas->shared) return;
    std::lock_guard<std::mutex> lock(replicas->mutex);
    replicas->valid = {false, false}; replicas->valid[replica_index] = true;
}
Status StateStorage::reset_zero() {
    if (!replicas) {
        ggml_backend_buffer_clear(allocation->buffer, 0);
        return {};
    }
    const auto& group = *replicas;
    std::lock_guard<std::mutex> stage_lock(group.stages.runs->mutex);
    std::scoped_lock backend_lock(group.stages.prefill->mutex, group.stages.decode->mutex);
    if (group.stages.runs->busy || group.stages.prefill->busy || group.stages.decode->busy)
        return conflict("State reset requires both backends to be idle");
    if (group.stages.prefill->failed || group.stages.decode->failed)
        return Error{ErrorCategory::backend_error, "State reset backend has failed"};
    std::lock_guard<std::mutex> state_lock(replicas->mutex);
    for (const auto& backend : {group.stages.prefill, group.stages.decode})
        ggml_backend_synchronize(backend->handle());
    // Known identical contents are written locally, not transferred from another replica.
    // A shared buffer is cleared once through its physical owner, not through a wrapper.
    for (std::size_t i = 0; i < (group.shared ? 1u : group.buffers.size()); ++i) {
        auto* physical = group.buffers[i].get();
        while (physical->home) physical = physical->home.get();
        ggml_backend_buffer_clear(physical->buffer, 0);
    }
    for (const auto& backend : {group.stages.prefill, group.stages.decode})
        ggml_backend_synchronize(backend->handle());
    replicas->valid = {true, true};
    return {};
}
Status Graph::coordinate(const Stages& stages) {
    if (built_ || (backend_ != stages.prefill && backend_ != stages.decode)) return invalid("Graph stage mismatch");
    if (runs_ && runs_ != stages.runs) return invalid("Graph already belongs to another stage group");
    if (stages.mixed() && !stages.runs) return invalid("Mixed stages have no run coordinator");
    runs_ = stages.runs;
    return {};
}
Status transfer(const std::shared_ptr<StateStorage>& storage, const std::shared_ptr<Backend>& target,
    const std::vector<TransferRange>& ranges, bool make_valid) {
    if (!storage || !target) return invalid("Missing transfer state or target");
    auto target_view = storage->on(target);
    if (!target_view) return target_view.error();
    const auto group = storage->replicas;
    if (!group) return {};
    std::lock_guard<std::mutex> stage_lock(group->stages.runs->mutex);
    std::scoped_lock backend_lock(group->stages.prefill->mutex, group->stages.decode->mutex);
    if (group->stages.runs->busy || group->stages.prefill->busy || group->stages.decode->busy)
        return conflict("Transfer requires both backends to be idle");
    std::lock_guard<std::mutex> lock(group->mutex);
    const auto index = target_view.value()->replica_index;
    if (group->shared || group->valid[index]) return {};
    if (group->stages.prefill->failed || group->stages.decode->failed)
        return Error{ErrorCategory::backend_error, "Transfer backend has failed"};
    const auto source_index = 1 - index;
    if (!group->valid[source_index]) return conflict("No valid state replica");
    auto* source = group->buffers[source_index]->tensors.front();
    auto* destination = target_view.value()->tensor();
    const auto size = ggml_nbytes(source);
    std::size_t bytes = 0;
    for (const auto& range : ranges) {
        if (range.offset > size || range.bytes > size - range.offset || range.bytes > size - bytes)
            return invalid("Transfer range is outside state");
        bytes += range.bytes;
    }
    if (ranges.empty()) bytes = size;
    const auto begin = std::chrono::steady_clock::now();
    ggml_backend_synchronize(group->buffers[source_index]->backend->handle());
    if (group->stages.transfer) {
        if (auto error = group->stages.transfer(source, destination, ranges)) return error;
    } else if (ranges.empty()) {
        ggml_backend_tensor_copy(source, destination);
    } else {
        // Bound staging memory independently of the total state or transfer size.
        const auto scratch_bytes = std::min(bytes, std::size_t(64 * 1024));
        auto charge = group->budget.reserve(scratch_bytes, Purpose::transfer, "host");
        if (!charge) return charge.error();
        std::vector<unsigned char> scratch(scratch_bytes);
        for (const auto& range : ranges) for (std::size_t done = 0; done < range.bytes;) {
            const auto count = std::min(scratch.size(), range.bytes - done);
            ggml_backend_tensor_get(source, scratch.data(), range.offset + done, count);
            ggml_backend_tensor_set(destination, scratch.data(), range.offset + done, count);
            done += count;
        }
    }
    ggml_backend_synchronize(target->handle());
    if (make_valid) group->valid[index] = true;
    if (bytes) {
        ++group->stats.count; group->stats.bytes += bytes;
        group->stats.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    return {};
}
}

namespace edge0::exec {
Consumption buffer_consumption(ggml_backend_buffer_type_t home, const std::shared_ptr<Backend>& backend, const ConsumptionQuery& query) {
    return query ? query(home, backend->handle()) : platform::consumption(home, backend->handle());
}
ggml_backend_buffer_type_t storage_home(const std::vector<std::shared_ptr<Backend>>& consumers, const ConsumptionQuery& query) {
    std::vector<ggml_backend_t> handles;
    for (const auto& consumer : consumers) { if (!consumer) return nullptr; handles.push_back(consumer->handle()); }
    return platform::choose_storage_home(handles, query);
}
}

namespace edge0::exec {
Result<Stages> resolve_stage_backends(const std::string& prefill, const std::string& decode,
    const std::map<std::string, std::shared_ptr<Backend>>& backends) {
    const auto a = backends.find(prefill), b = backends.find(decode);
    if (a == backends.end() || b == backends.end() || !a->second || !b->second)
        return Error{ErrorCategory::unsupported, "Stage backend is outside runtime scope"};
    Stages result{a->second, b->second};
    if (result.mixed()) result.runs = std::make_shared<StageRuns>();
    return result;
}
}
