#pragma once
#include "exec/execution.h"
#include <map>
#include <functional>
#include <optional>
#include <memory>
#include <string>
#include <vector>
#include <typeindex>

namespace edge0::session {
struct Step { std::int64_t e, n; };
enum class Visibility { causal, sliding_window };
struct KVDeclaration {
    std::string name;
    ggml_type type;
    std::vector<std::int64_t> token_shape;
    int token_axis = 1;
    std::int64_t alignment = 1;
    Visibility visibility = Visibility::causal;
};
struct WriteRange {
    ggml_tensor* view;
    std::vector<std::int64_t> rows;
};
class KVCache {
public:
    static Result<std::shared_ptr<KVCache>> create(exec::Budget& budget,
        const std::shared_ptr<exec::Backend>& backend, std::int64_t capacity,
        const std::vector<KVDeclaration>& declarations);
    static Result<std::shared_ptr<KVCache>> create(exec::Budget&, const exec::Stages&, std::int64_t,
        const std::vector<KVDeclaration>&);
    exec::Status make_current(const std::shared_ptr<exec::Backend>&, std::int64_t position);
    exec::Status commit_rows(Step, const exec::Completion&);
    Result<std::int64_t> synced(const std::string&, const std::shared_ptr<exec::Backend>&) const;
    exec::TransferStats transfer_stats() const;
    std::size_t handoff_bytes(const std::shared_ptr<exec::Backend>&, std::int64_t position) const;
    exec::Status truncate(std::int64_t k, std::int64_t current);
    std::int64_t capacity() const { return capacity_; }
    Result<std::shared_ptr<exec::StateStorage>> storage(const std::string& name) const;
    Result<WriteRange> write_range(ggml_context* context, const std::string& name, Step step) const;
    Result<ggml_tensor*> history_view(ggml_context* context, const std::string& name, Step step,
        ggml_tensor* source = nullptr) const;
    Result<std::vector<ggml_fp16_t>> causal_mask(const std::string& name, Step step) const;
    Result<std::int64_t> history_length(const std::string& name, Step step) const;
private:
    struct Entry {
        KVDeclaration declaration;
        std::shared_ptr<exec::StateStorage> storage;
        std::array<std::int64_t, 2> synced{0, 0};
    };
    std::shared_ptr<exec::Stages> stages_;
    std::shared_ptr<exec::Backend> active_;
    std::int64_t capacity_ = 0;
    std::map<std::string, Entry> entries_;
    exec::Status validate(const std::string& name, Step step) const;
};
class StateStore {
public:
    using CheckpointCallback = std::function<exec::Status()>;
    exec::Status save_checkpoint(std::int64_t boundary, const CheckpointCallback&);
    exec::Status restore_checkpoint(const CheckpointCallback&);
    exec::Status discard_checkpoint(const CheckpointCallback& = {});
    std::optional<std::int64_t> checkpoint() const { return checkpoint_; }
    template<class T> exec::Status put(const std::string& name, std::shared_ptr<T> object) {
        if (objects_.count(name)) return Error{ErrorCategory::conflict, "State name already exists: " + name};
        if (name.empty() || !object) return Error{ErrorCategory::invalid_argument, "State requires a name and object"};
        objects_.emplace(name, Entry{std::move(object), std::type_index(typeid(T))});
        return {};
    }
    template<class T> Result<std::shared_ptr<T>> get(const std::string& name) const {
        const auto found=objects_.find(name);
        if (found==objects_.end()) return std::shared_ptr<T>{};
        if (found->second.type!=std::type_index(typeid(T)))
            return Error{ErrorCategory::conflict, "State type mismatch: " + name};
        const auto& object=found->second.object;
        return std::shared_ptr<T>(object, static_cast<T*>(object.get()));
    }
    std::shared_ptr<void> remove(const std::string& name);
    void remove_prefix(const std::string& prefix);
private:
    struct Entry { std::shared_ptr<void> object; std::type_index type; };
    std::map<std::string, Entry> objects_;
    std::optional<std::int64_t> checkpoint_;
};
}
