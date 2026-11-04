#pragma once
#include "session/storage.h"
#include <functional>
#include <mutex>

namespace edge0::session {
struct Identity {
    std::string adapter, template_sha256, tokenizer;
    bool operator==(const Identity& b) const {
        return adapter == b.adapter && template_sha256 == b.template_sha256 && tokenizer == b.tokenizer;
    }
};
enum class ReuseKind { first, keep, truncate, restore, cold };
struct ReusePlan {
    ReuseKind kind = ReuseKind::first;
    std::int64_t k = 0, prefix = 0, e = 0;
    std::uint64_t serial = 0;
    Identity identity;
};
struct Update {
    Step step;
    std::vector<std::int32_t> tokens;
    std::uint64_t serial = 0;
};
struct Snapshot {
    std::int64_t capacity = 0, e = 0;
    std::vector<std::int32_t> tokens;
    bool valid = true, release_requested = false, released = false;
    std::optional<std::int64_t> checkpoint;
    Identity identity;
};
class Session {
public:
    using Initialize = std::function<exec::Status(KVCache&, StateStore&)>;
    using Release = std::function<void(KVCache&, StateStore&)>;
    static Result<std::shared_ptr<Session>> create(exec::Budget&, const std::shared_ptr<exec::Backend>&,
        std::int64_t capacity, const std::vector<KVDeclaration>&, std::vector<std::string> updates);
    static Result<std::shared_ptr<Session>> create(exec::Budget&, const exec::Stages&,
        std::int64_t capacity, const std::vector<KVDeclaration>&, std::vector<std::string> updates);
    ~Session();
    exec::Status initialize(Initialize callback, Release release);
    exec::Status configure_reuse(Identity, bool whole, bool save_restore,
        Initialize save, Initialize restore, Initialize discard);
    Result<ReusePlan> check_reuse(std::int64_t prefix, std::int64_t target, const Identity&) const;
    exec::Status apply_reuse(const ReusePlan&, Initialize reset, bool skip_restore = false);
    exec::Status save_checkpoint();
    Result<Update> begin_update(const std::vector<std::int32_t>& tokens);
    exec::Status commit_update(const Update&, const exec::Completion&, const std::vector<std::string>& updated);
    exec::Status report_interruption(const Update&, bool execution_submitted);
    exec::Status reset(Initialize callback);
    exec::Status begin_use();
    void end_use();
    void invalidate();
    void request_release();
    Snapshot query() const;
    KVCache& kv() { return *kv_; }
    StateStore& states() { return states_; }
private:
    Session() = default;
    void cleanup();
    mutable std::recursive_mutex mutex_;
    Snapshot status_;
    std::shared_ptr<KVCache> kv_;
    StateStore states_;
    std::vector<std::string> kv_names_, updates_;
    Release release_;
    Initialize save_, restore_, discard_;
    bool whole_ = false, save_restore_ = false;
    std::size_t uses_ = 0;
    std::uint64_t serial_ = 0;
    std::vector<std::int32_t> pending_tokens_;
    bool initialized_ = false, pending_ = false;
};
}
