#include "session/session.h"
#include <algorithm>
#include <set>

namespace edge0::session {
namespace {
Error invalid(const char* message) { return {ErrorCategory::invalid_argument, message}; }
}
Result<std::shared_ptr<Session>> Session::create(exec::Budget& budget,
    const std::shared_ptr<exec::Backend>& backend, std::int64_t capacity,
    const std::vector<KVDeclaration>& declarations, std::vector<std::string> updates) {
    std::set<std::string> names(updates.begin(), updates.end());
    if (names.size() != updates.size() || names.count("")) return invalid("Invalid state update declaration");
    auto kv = KVCache::create(budget, backend, capacity, declarations);
    if (!kv) return kv.error();
    auto result = std::shared_ptr<Session>(new Session);
    result->kv_ = kv.value();
    result->status_.capacity = capacity;
    result->updates_ = std::move(updates);
    for (const auto& declaration : declarations) result->kv_names_.push_back(declaration.name);
    return result;
}
Result<std::shared_ptr<Session>> Session::create(exec::Budget& budget,
    const exec::Stages& stages, std::int64_t capacity,
    const std::vector<KVDeclaration>& declarations, std::vector<std::string> updates) {
    std::set<std::string> names(updates.begin(), updates.end());
    if (names.size() != updates.size() || names.count("")) return invalid("Invalid state update declaration");
    auto kv = KVCache::create(budget, stages, capacity, declarations);
    if (!kv) return kv.error();
    auto result = std::shared_ptr<Session>(new Session);
    result->kv_ = kv.value();
    result->status_.capacity = capacity;
    result->updates_ = std::move(updates);
    for (const auto& declaration : declarations) result->kv_names_.push_back(declaration.name);
    return result;
}
Session::~Session() { cleanup(); }
void Session::cleanup() {
    if (status_.released) return;
    if (release_ && kv_) release_(*kv_, states_);
    release_ = {};
    states_ = {};
    kv_.reset();
    status_.released = true;
    pending_ = false;
}
exec::Status Session::initialize(Initialize callback, Release release) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (initialized_ || status_.released || status_.release_requested || !callback || !release)
        return invalid("Session cannot initialize");
    release_ = std::move(release);
    if (auto error = callback(*kv_, states_)) { cleanup(); return error; }
    initialized_ = true;
    return {};
}
exec::Status Session::begin_use() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || status_.release_requested || status_.released) return invalid("Session is released or uninitialized");
    if (!status_.valid) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (uses_) return Error{ErrorCategory::busy, "Session is in use"};
    ++uses_;
    return {};
}
void Session::end_use() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (uses_) --uses_;
    if (!uses_ && !pending_ && status_.release_requested) cleanup();
}
Result<Update> Session::begin_update(const std::vector<std::int32_t>& tokens) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || status_.released || status_.release_requested) return invalid("Session is released or uninitialized");
    if (!status_.valid) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (pending_) return Error{ErrorCategory::busy, "An update is pending"};
    if (tokens.empty()) return invalid("Empty update");
    if (tokens.size() > static_cast<std::uint64_t>(status_.capacity - status_.e))
        return Error{ErrorCategory::capacity_exceeded, "Update exceeds session capacity"};
    pending_tokens_ = tokens;
    pending_ = true;
    return Update{{status_.e, static_cast<std::int64_t>(tokens.size())}, tokens, ++serial_};
}
exec::Status Session::commit_update(const Update& update, const exec::Completion& completion,
    const std::vector<std::string>& updated) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!pending_ || update.tokens != pending_tokens_ || update.serial != serial_ || update.step.e != status_.e ||
        update.step.n != static_cast<std::int64_t>(update.tokens.size()) || update.step.n <= 0 ||
        update.step.n > status_.capacity - status_.e) return invalid("Stale or invalid update");
    if (!completion.success || completion.writes_may_be_partial)
        return Error{ErrorCategory::backend_error, "Update did not complete successfully"};
    for (const auto& name : kv_names_) {
        const auto storage = kv_->storage(name).value();
        if (std::find(completion.written.begin(), completion.written.end(), storage->allocation) == completion.written.end())
            return invalid("Missing KV write completion");
    }
    for (const auto& name : updates_)
        if (std::find(updated.begin(), updated.end(), name) == updated.end()) return invalid("Missing model state update");
    if (auto error = kv_->commit_rows(update.step, completion)) return error;
    status_.tokens.insert(status_.tokens.end(), update.tokens.begin(), update.tokens.end());
    status_.e += update.step.n;
    pending_ = false;
    if (!uses_ && status_.release_requested) cleanup();
    return {};
}
exec::Status Session::report_interruption(const Update& update, bool submitted) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!pending_ || update.serial != serial_) return invalid("Stale update interruption");
    if (submitted) { status_.valid = false; states_.discard_checkpoint([&] { return discard_ ? discard_(*kv_, states_) : exec::Status{}; }); }
    pending_ = false;
    if (!uses_ && status_.release_requested) cleanup();
    return {};
}
exec::Status Session::reset(Initialize callback) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || status_.released || status_.release_requested || !callback) return invalid("Session cannot reset");
    if (!status_.valid) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (uses_ || pending_) return Error{ErrorCategory::busy, "Session is in use"};
    if (auto error = states_.discard_checkpoint([&] { return discard_ ? discard_(*kv_, states_) : exec::Status{}; })) { status_.valid = false; return error; }
    if (auto error = kv_->truncate(0, status_.e)) return error;
    if (auto error = callback(*kv_, states_)) { status_.valid = false; return error; }
    status_.e = 0;
    status_.tokens.clear();
    ++serial_;
    return {};
}
void Session::request_release() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    status_.release_requested = true;
    if (!uses_ && !pending_) cleanup();
}
Snapshot Session::query() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto result = status_;
    result.checkpoint = states_.checkpoint();
    return result;
}
}

namespace edge0::session {
exec::Status Session::configure_reuse(Identity identity, bool whole, bool save_restore,
    Initialize save, Initialize restore, Initialize discard) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || status_.released || uses_ || pending_ || status_.e)
        return invalid("Cannot configure reuse after use");
    if (save_restore && (!whole || !save || !restore || !discard))
        return invalid("Incomplete checkpoint callbacks");
    status_.identity = std::move(identity); whole_ = whole; save_restore_ = save_restore;
    save_ = std::move(save); restore_ = std::move(restore); discard_ = std::move(discard);
    return {};
}
Result<ReusePlan> Session::check_reuse(std::int64_t prefix, std::int64_t target, const Identity& identity) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || status_.released || status_.release_requested || target <= 0 || prefix < 0 || prefix > status_.e || prefix > target)
        return invalid("Invalid reuse request");
    if (!status_.valid) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (pending_) return Error{ErrorCategory::busy, "An update is pending"};
    const bool matches = identity == status_.identity;
    const auto p = matches ? prefix : 0;
    const auto t = std::min(p, target - 1);
    ReusePlan result{ReuseKind::first, 0, p, status_.e, serial_, identity};
    const auto c = states_.checkpoint();
    if (!status_.e) return result;
    if (t >= status_.e) { result.kind = ReuseKind::keep; result.k = status_.e; }
    else if (!whole_) { result.kind = ReuseKind::truncate; result.k = t; }
    else if (matches && save_restore_ && c && *c > 0 && *c <= t) { result.kind = ReuseKind::restore; result.k = *c; }
    else result.kind = ReuseKind::cold;
    return result;
}
exec::Status Session::apply_reuse(const ReusePlan& plan, Initialize reset_callback, bool skip_restore) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!status_.valid) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (status_.released || status_.release_requested || plan.serial != serial_ || plan.e != status_.e || plan.k < 0 || plan.k > status_.e)
        return invalid("Stale reuse plan");
    if (uses_ || pending_) return Error{ErrorCategory::busy, "Session is in use"};
    if (plan.kind == ReuseKind::cold) {
        if (auto error = reset(reset_callback)) return error;
    } else if (plan.kind == ReuseKind::truncate || plan.kind == ReuseKind::restore) {
        if (auto error = kv_->truncate(plan.k, status_.e)) return error;
        status_.e = plan.k; status_.tokens.resize(static_cast<std::size_t>(plan.k));
        if (plan.kind == ReuseKind::restore && !skip_restore) {
            if (auto error = states_.restore_checkpoint([&] { return restore_(*kv_, states_); })) {
                status_.valid = false;
                states_.discard_checkpoint([&] { return discard_ ? discard_(*kv_, states_) : exec::Status{}; });
                return error;
            }
        }
    }
    if (!(plan.identity == status_.identity) || (states_.checkpoint() && plan.k < *states_.checkpoint())) {
        if (auto error = states_.discard_checkpoint([&] { return discard_ ? discard_(*kv_, states_) : exec::Status{}; })) {
            status_.valid = false; return error;
        }
    }
    status_.identity = plan.identity;
    ++serial_;
    return {};
}
exec::Status Session::save_checkpoint() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!save_restore_) return Error{ErrorCategory::unsupported, "Checkpoint saving is unsupported"};
    if (!status_.valid) return Error{ErrorCategory::session_invalid, "Session state is invalid"};
    if (status_.released || status_.release_requested || pending_ || status_.e <= 0)
        return invalid("Session cannot save a checkpoint");
    return states_.save_checkpoint(status_.e, [&] { return save_(*kv_, states_); });
}
}

namespace edge0::session {
void Session::invalidate() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    status_.valid = false;
    states_.discard_checkpoint([&] { return discard_ && kv_ ? discard_(*kv_, states_) : exec::Status{}; });
}
}
