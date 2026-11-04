#pragma once
#include "edge0/result.h"
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <utility>

namespace edge0::exec {
enum class Purpose { resident_weights, experts, session_state, workspace, transfer, staging, metadata };
inline constexpr std::array<Purpose, 7> purposes{Purpose::resident_weights, Purpose::experts, Purpose::session_state, Purpose::workspace, Purpose::transfer, Purpose::staging, Purpose::metadata};
const char* purpose_name(Purpose);
struct DeviceMemory { std::size_t current = 0, peak = 0; };
struct BudgetSnapshot {
    std::size_t limit = 0, current = 0, peak = 0;
    std::map<Purpose, std::size_t> by_purpose;
    std::map<std::string, DeviceMemory> by_device;
};
class Budget {
    struct State;
public:
    class Reservation {
    public:
        ~Reservation();
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        std::optional<Error> resize(std::size_t bytes);
        std::size_t size() const;
        Purpose purpose() const { return purpose_; }
    private:
        friend class Budget;
        Reservation(std::shared_ptr<State> state, std::size_t bytes, Purpose purpose, std::string device);
        std::shared_ptr<State> state_;
        std::size_t bytes_;
        Purpose purpose_;
        std::string device_;
    };
    using Ticket = std::shared_ptr<Reservation>;
    struct Charge {
        std::size_t bytes;
        Purpose purpose;
        std::string device;
        Charge(std::size_t size, Purpose use, std::string location)
            : bytes(size), purpose(use), device(std::move(location)) {}
    };
    explicit Budget(std::size_t limit);
    Result<Ticket> reserve(std::size_t bytes, Purpose purpose, const std::string& device);
    Result<std::vector<Ticket>> reserve(const std::vector<Charge>& charges);
    std::size_t limit() const;
    std::size_t current() const;
    std::size_t current(Purpose purpose) const;
    std::size_t peak() const;
    BudgetSnapshot snapshot() const;
private:
    std::shared_ptr<State> state_;
};
}
