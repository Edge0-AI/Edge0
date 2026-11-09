#include "exec/budget.h"
#include <algorithm>

namespace edge0::exec {
const char* purpose_name(Purpose purpose) {
    switch (purpose) {
    case Purpose::resident_weights: return "resident_weights";
    case Purpose::experts: return "experts";
    case Purpose::session_state: return "session_state";
    case Purpose::workspace: return "workspace";
    case Purpose::transfer: return "transfer";
    case Purpose::staging: return "staging";
    case Purpose::metadata: return "metadata";
    }
    return "unknown";
}

struct Budget::State {
    explicit State(std::size_t maximum) : limit(maximum) {}
    std::mutex mutex;
    std::size_t limit, current = 0, peak = 0;
    std::map<Purpose, std::size_t> purposes;
    std::map<std::string, DeviceMemory> devices;
};
Budget::Budget(std::size_t limit) : state_(std::make_shared<State>(limit)) {}
Budget::Reservation::Reservation(std::shared_ptr<State> state, std::size_t bytes, Purpose purpose, std::string device)
    : state_(std::move(state)), bytes_(bytes), purpose_(purpose), device_(std::move(device)) {}
Budget::Reservation::~Reservation() {
    if (bytes_ == 0) return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->current -= bytes_;
    state_->purposes[purpose_] -= bytes_;
    state_->devices[device_].current -= bytes_;
}
std::optional<Error> Budget::Reservation::resize(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (bytes > bytes_ && bytes - bytes_ > state_->limit - state_->current)
        return Error{ErrorCategory::budget_exceeded, std::string("Budget exceeded: ") + purpose_name(purpose_)};
    state_->current = state_->current - bytes_ + bytes;
    state_->purposes[purpose_] = state_->purposes[purpose_] - bytes_ + bytes;
    state_->peak = std::max(state_->peak, state_->current);
    auto& device = state_->devices[device_];
    device.current = device.current - bytes_ + bytes;
    device.peak = std::max(device.peak, device.current);
    bytes_ = bytes;
    return {};
}
std::size_t Budget::Reservation::size() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return bytes_;
}
Result<std::vector<Budget::Ticket>> Budget::reserve(const std::vector<Charge>& charges) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    std::size_t total = 0;
    for (const auto& charge : charges) {
        if (charge.bytes > state_->limit - state_->current - total)
            return Error{ErrorCategory::budget_exceeded, std::string("Budget exceeded: ") + purpose_name(charge.purpose)};
        total += charge.bytes;
    }
    // Publish all charges atomically, so an insufficient compound request never
    // raises the historical peak or leaves a partial metadata reservation.
    std::vector<Ticket> tickets;
    tickets.reserve(charges.size());
    for (const auto& charge : charges) {
        state_->purposes.try_emplace(charge.purpose, 0);
        state_->devices.try_emplace(charge.device);
        tickets.push_back(Ticket(new Reservation(state_, 0, charge.purpose, charge.device)));
    }
    for (std::size_t i = 0; i < charges.size(); ++i) {
        tickets[i]->bytes_ = charges[i].bytes;
        state_->purposes[charges[i].purpose] += charges[i].bytes;
        auto& device = state_->devices[charges[i].device];
        device.current += charges[i].bytes;
        device.peak = std::max(device.peak, device.current);
    }
    state_->current += total;
    state_->peak = std::max(state_->peak, state_->current);
    return tickets;
}
Result<Budget::Ticket> Budget::reserve(std::size_t bytes, Purpose purpose, const std::string& device) {
    auto result = reserve(std::vector<Charge>{{bytes, purpose, device}});
    if (!result) return result.error();
    return result.value().front();
}
std::size_t Budget::limit() const { return state_->limit; }
std::size_t Budget::current() const {
    std::lock_guard<std::mutex> lock(state_->mutex); return state_->current;
}
std::size_t Budget::current(Purpose purpose) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto it = state_->purposes.find(purpose);
    return it == state_->purposes.end() ? 0 : it->second;
}
std::size_t Budget::peak() const {
    std::lock_guard<std::mutex> lock(state_->mutex); return state_->peak;
}
}

namespace edge0::exec {
BudgetSnapshot Budget::snapshot() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    BudgetSnapshot result{state_->limit, state_->current, state_->peak, state_->purposes, state_->devices};
    for (auto purpose : purposes) result.by_purpose.try_emplace(purpose, 0);
    return result;
}
}
