#pragma once
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace edge0 {
enum class ErrorCategory { conflict, backend_error, invalid_argument, runtime_ended, asset_invalid, unsupported, incompatible, budget_exceeded, busy, capacity_exceeded, session_invalid, unloading };
struct Error {
    ErrorCategory category;
    std::string message;
};

template<class T> class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    Result(Error error) : value_(std::move(error)) {}
    explicit operator bool() const noexcept { return std::holds_alternative<T>(value_); }
    const T& value() const { return std::get<T>(value_); }
    const Error& error() const { return std::get<Error>(value_); }
private:
    std::variant<T, Error> value_;
};
}
