#pragma once
#include "edge0/generation.h"
#include "edge0/diagnostics.h"
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace portal {
using Json = nlohmann::ordered_json;
enum class Protocol { chat, responses };
struct Request {
    Protocol protocol = Protocol::chat;
    edge0::ChatInput input;
    edge0::GenerationOverrides overrides;
    bool stream = false, include_usage = false;
    Json echo = Json::object();
};
struct Failure : std::runtime_error {
    int status;
    std::string code, param, native_detail;
    Failure(int status, std::string message, std::string code = "invalid_parameter", std::string param = "");
    std::string message() const;
    Json body() const;
};
Request parse(const Json&, Protocol, const std::string& alias);
Failure failure(const edge0::Error&, bool shutting_down = false);
std::string identifier(const std::string& prefix);
class Encoder {
public:
    Encoder(const Request&, std::string alias, std::string version);
    std::vector<std::string> begin();
    std::vector<std::string> accept(const edge0::OutputEvent&);
    std::vector<std::string> finish(const edge0::GenerationResult&, const edge0::MemorySnapshot& = {});
    std::vector<std::string> fail(const Failure&);
    Json complete(const edge0::GenerationResult&, const edge0::MemorySnapshot& = {}) const;
private:
    Request request_;
    std::string alias_, fingerprint_, id_;
    std::int64_t created_;
    std::size_t sequence_ = 0;
    std::map<std::size_t,std::size_t> call_indices_;
    Json output_ = Json::array(), pending_probs_ = Json::array();
    int active_ = -1;
    std::string event(const std::string&, Json);
    Json chunk(Json, Json reason = nullptr, Json probs = nullptr) const;
    void close(std::vector<std::string>&);
    void open(std::vector<std::string>&, const edge0::OutputEvent&);
    Json response(const edge0::GenerationResult*, const std::string&, const edge0::MemorySnapshot& = {}) const;
};
}
