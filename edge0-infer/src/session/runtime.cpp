#include "session/storage.h"
#include <limits>
#include <algorithm>
#include <utility>

namespace edge0::session {
namespace {
Error invalid(const std::string& reason) { return {ErrorCategory::invalid_argument, reason}; }
ggml_tensor* view(ggml_context* context, ggml_tensor* source, int axis, std::int64_t count, std::size_t offset) {
    auto dimensions = std::array<std::int64_t, 4>{source->ne[0], source->ne[1], source->ne[2], source->ne[3]};
    dimensions[axis] = count;
    return ggml_view_4d(context, source, dimensions[0], dimensions[1], dimensions[2], dimensions[3],
        source->nb[1], source->nb[2], source->nb[3], offset);
}
}
Result<std::shared_ptr<KVCache>> KVCache::create(exec::Budget& budget,
    const std::shared_ptr<exec::Backend>& backend, std::int64_t capacity,
    const std::vector<KVDeclaration>& declarations) {
    if (capacity <= 0) return invalid("KV capacity must be positive");
    auto result = std::make_shared<KVCache>(); result->capacity_ = capacity;
    std::set<std::string> names;
    for (const auto& declaration : declarations) {
        if (declaration.visibility != Visibility::causal)
            return Error{ErrorCategory::unsupported, "Only causal visibility is supported"};
        if (declaration.name.empty() || !names.insert(declaration.name).second ||
            declaration.token_shape.size() >= 4 || declaration.token_axis < 0 ||
            declaration.token_axis > static_cast<int>(declaration.token_shape.size()) || declaration.alignment <= 0 ||
            capacity > std::numeric_limits<std::int64_t>::max() - declaration.alignment)
            return invalid("Invalid KV declaration");
    }
    for (const auto& declaration : declarations) {
        auto dimensions = declaration.token_shape;
        const auto physical = (capacity + declaration.alignment - 1) / declaration.alignment * declaration.alignment;
        dimensions.insert(dimensions.begin() + declaration.token_axis, physical);
        auto allocation = exec::allocate_state_storage(budget, backend, {declaration.type, dimensions}, edge0::exec::Purpose::session_state);
        if (!allocation) return allocation.error();
        // Aligned reads can touch masked padding. Initialize it to finite values
        // so kernels that multiply masked values cannot propagate stale NaNs.
        ggml_backend_buffer_clear(allocation.value()->allocation->buffer, 0);
        result->entries_.emplace(declaration.name, Entry{declaration, allocation.value()});
    }
    return result;
}
exec::Status KVCache::validate(const std::string& name, Step step) const {
    if (!entries_.count(name)) return invalid("Unknown KV entry: " + name);
    if (step.e < 0 || step.n <= 0 || step.e > capacity_ || step.n > capacity_ - step.e)
        return invalid("KV write exceeds logical capacity");
    return {};
}
Result<std::shared_ptr<exec::StateStorage>> KVCache::storage(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) return invalid("Unknown KV entry: " + name);
    return stages_ ? it->second.storage->on(active_) : Result<std::shared_ptr<exec::StateStorage>>(it->second.storage);
}
Result<std::int64_t> KVCache::history_length(const std::string& name, Step step) const {
    if (auto error = validate(name, step)) return *error;
    const auto alignment = entries_.at(name).declaration.alignment;
    return (step.e + step.n + alignment - 1) / alignment * alignment;
}
Result<WriteRange> KVCache::write_range(ggml_context* context, const std::string& name, Step step) const {
    if (auto error = validate(name, step)) return *error;
    if (!context) return invalid("Missing graph context");
    const auto& entry = entries_.at(name);
    auto selected = storage(name); if (!selected) return selected.error();
    auto* tensor = selected.value()->tensor();
    WriteRange result{view(context, tensor, entry.declaration.token_axis, step.n,
        static_cast<std::size_t>(step.e) * tensor->nb[entry.declaration.token_axis]), {}};
    result.rows.reserve(static_cast<std::size_t>(step.n));
    for (std::int64_t i = 0; i < step.n; ++i) result.rows.push_back(step.e + i);
    return result;
}
Result<ggml_tensor*> KVCache::history_view(ggml_context* context, const std::string& name, Step step, ggml_tensor* source) const {
    auto length = history_length(name, step); if (!length) return length.error();
    if (!context) return invalid("Missing graph context");
    const auto& entry = entries_.at(name);
    auto selected = storage(name); if (!selected) return selected.error();
    auto* tensor = selected.value()->tensor();
    if (!source) source = tensor;
    auto* root = source; while (root->view_src) root = root->view_src;
    if (root != tensor) return invalid("History source belongs to another storage");
    auto* result = view(context, tensor, entry.declaration.token_axis, length.value(), 0);
    // A slice copy returns the destination slice (including its offset). History
    // still starts at the allocation base; keep the write as the graph edge,
    // while view_src/data describe the full persistent allocation.
    result->src[0] = source;
    return result;
}
Result<std::vector<ggml_fp16_t>> KVCache::causal_mask(const std::string& name, Step step) const {
    auto length = history_length(name, step); if (!length) return length.error();
    if (static_cast<std::uint64_t>(step.n) > std::numeric_limits<std::size_t>::max() / sizeof(ggml_fp16_t) / static_cast<std::uint64_t>(length.value()))
        return invalid("Mask size overflow");
    std::vector<ggml_fp16_t> result(static_cast<std::size_t>(length.value() * step.n));
    for (std::int64_t row = 0; row < step.n; ++row)
        for (std::int64_t column = 0; column < length.value(); ++column)
            result[static_cast<std::size_t>(row * length.value() + column)] = ggml_fp32_to_fp16(column <= step.e + row ? 0.0f : -std::numeric_limits<float>::infinity());
    return result;
}
std::shared_ptr<void> StateStore::remove(const std::string& name) {
    auto node = objects_.extract(name); return node.empty() ? nullptr : std::move(node.mapped().object);
}
void StateStore::remove_prefix(const std::string& prefix) {
    for (auto it = objects_.lower_bound(prefix); it != objects_.end() && it->first.compare(0, prefix.size(), prefix) == 0;)
        it = objects_.erase(it);
}

}

namespace edge0::session {
exec::Status KVCache::truncate(std::int64_t k, std::int64_t current) {
    if (k < 0 || current < k || current > capacity_)
        return Error{ErrorCategory::invalid_argument, "Invalid KV truncation"};
    if (stages_) for (auto& item : entries_)
        for (auto& position : item.second.synced) position = std::min(position, k);
    return {};
}
exec::Status StateStore::save_checkpoint(std::int64_t boundary, const CheckpointCallback& copy) {
    checkpoint_.reset();
    if (boundary <= 0 || !copy) return Error{ErrorCategory::invalid_argument, "Invalid checkpoint save"};
    if (auto error = copy()) return error;
    checkpoint_ = boundary;
    return {};
}
exec::Status StateStore::restore_checkpoint(const CheckpointCallback& copy) {
    if (!checkpoint_ || !copy) return Error{ErrorCategory::invalid_argument, "No checkpoint to restore"};
    return copy();
}
exec::Status StateStore::discard_checkpoint(const CheckpointCallback& discard) {
    checkpoint_.reset();
    return discard ? discard() : exec::Status{};
}
}

namespace edge0::session {
Result<std::shared_ptr<KVCache>> KVCache::create(exec::Budget& budget, const exec::Stages& stages,
    std::int64_t capacity, const std::vector<KVDeclaration>& declarations) {
    if (!stages.mixed()) return create(budget, stages.prefill, capacity, declarations);
    // Reuse declaration validation without allocating any single-backend KV.
    if (capacity <= 0) return Error{ErrorCategory::invalid_argument, "KV capacity must be positive"};
    std::set<std::string> names;
    for (const auto& declaration : declarations) {
        if (declaration.visibility != Visibility::causal)
            return Error{ErrorCategory::unsupported, "Only causal visibility is supported"};
        if (declaration.name.empty() || !names.insert(declaration.name).second ||
            declaration.token_shape.size() >= 4 || declaration.token_axis < 0 ||
            declaration.token_axis > static_cast<int>(declaration.token_shape.size()) || declaration.alignment <= 0 ||
            capacity > std::numeric_limits<std::int64_t>::max() - declaration.alignment)
            return Error{ErrorCategory::invalid_argument, "Invalid KV declaration"};
    }
    auto result = std::make_shared<KVCache>(); result->capacity_ = capacity;
    result->stages_ = std::make_shared<exec::Stages>(stages); result->active_ = stages.prefill;
    for (const auto& declaration : declarations) {
        auto dimensions = declaration.token_shape;
        const auto physical = (capacity + declaration.alignment - 1) / declaration.alignment * declaration.alignment;
        dimensions.insert(dimensions.begin() + declaration.token_axis, physical);
        auto state = exec::allocate_state_storage(budget, stages, {declaration.type, dimensions}, edge0::exec::Purpose::session_state);
        if (!state) return state.error();
        result->entries_.emplace(declaration.name, Entry{declaration, state.value()});
    }
    return result;
}
Result<std::int64_t> KVCache::synced(const std::string& name, const std::shared_ptr<exec::Backend>& backend) const {
    const auto found = entries_.find(name);
    if (found == entries_.end()) return Error{ErrorCategory::invalid_argument, "Unknown KV entry"};
    if (!stages_ || (backend != stages_->prefill && backend != stages_->decode))
        return Error{ErrorCategory::invalid_argument, "KV replica backend is outside stages"};
    return found->second.synced[backend == stages_->prefill ? 0 : 1];
}
exec::Status KVCache::make_current(const std::shared_ptr<exec::Backend>& backend, std::int64_t position) {
    if (!stages_) return {};
    if (position < 0 || position > capacity_ || (backend != stages_->prefill && backend != stages_->decode))
        return Error{ErrorCategory::invalid_argument, "Invalid KV synchronization target or position"};
    const auto index = backend == stages_->prefill ? 0u : 1u;
    for (auto& item : entries_) {
        auto& entry = item.second;
        if (entry.storage->replicas->shared) continue;
        auto target = entry.storage->on(backend); if (!target) return target.error();
        if (target.value()->current()) continue;
        const auto begin = std::min(entry.synced[index], position);
        auto* tensor = target.value()->tensor();
        const auto axis = entry.declaration.token_axis;
        const auto row_bytes = tensor->nb[axis];
        const auto count = static_cast<std::size_t>(position - begin);
        const auto plane = tensor->nb[axis] * static_cast<std::size_t>(tensor->ne[axis]);
        std::size_t planes = 1;
        for (int i = axis + 1; i < GGML_MAX_DIMS; ++i) planes *= static_cast<std::size_t>(tensor->ne[i]);
        std::vector<exec::TransferRange> ranges;
        for (std::size_t i = 0; i < planes; ++i)
            ranges.push_back({i * plane + static_cast<std::size_t>(begin) * row_bytes, count * row_bytes});
        if (auto error = exec::transfer(entry.storage, backend, ranges)) return error;
        entry.synced[index] = position;
    }
    active_ = backend;
    return {};
}
exec::Status KVCache::commit_rows(Step step, const exec::Completion& completion) {
    if (!stages_) return {};
    if (!completion.success || step.e < 0 || step.n <= 0 || step.e > capacity_ || step.n > capacity_ - step.e)
        return Error{ErrorCategory::invalid_argument, "Invalid KV completed range"};
    const auto index = active_ == stages_->prefill ? 0u : 1u;
    for (const auto& item : entries_) {
        const auto& buffer = item.second.storage->replicas->buffers[index];
        if (std::find(completion.written.begin(), completion.written.end(), buffer) == completion.written.end())
            return Error{ErrorCategory::conflict, "Completed graph did not write every KV entry"};
    }
    for (auto& item : entries_) {
        auto& entry = item.second;
        if (entry.storage->replicas->shared) continue;
        entry.synced[index] = step.e + step.n;
        entry.synced[1 - index] = std::min(entry.synced[1 - index], step.e);
    }
    return {};
}
}

namespace edge0::session {
exec::TransferStats KVCache::transfer_stats() const {
    exec::TransferStats result;
    if (stages_) for (const auto& item : entries_) {
        const auto& stats = item.second.storage->replicas->stats;
        result.count += stats.count; result.bytes += stats.bytes; result.seconds += stats.seconds;
    }
    return result;
}
std::size_t KVCache::handoff_bytes(const std::shared_ptr<exec::Backend>& backend, std::int64_t position) const {
    if (!stages_ || position < 0) return 0;
    const auto index = backend == stages_->prefill ? 0u : 1u;
    std::size_t bytes = 0;
    for (const auto& item : entries_) {
        const auto& entry = item.second;
        if (entry.storage->replicas->shared || entry.storage->replicas->valid[index]) continue;
        auto* tensor = entry.storage->tensor();
        const auto axis = entry.declaration.token_axis;
        const auto per_row = ggml_nbytes(tensor) / static_cast<std::size_t>(tensor->ne[axis]);
        bytes += static_cast<std::size_t>(std::max(std::int64_t(0), position - entry.synced[index])) * per_row;
    }
    return bytes;
}
}
