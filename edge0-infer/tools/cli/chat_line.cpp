#include "chat_line.h"
#include <cctype>
#include <string>
#include <utility>
namespace cli {
// The first field is the action; message text preserves whitespace except final CR.
std::pair<std::string, std::string> classify_chat_line(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::size_t first = 0, last = line.size();
    while (first < last && std::isspace(static_cast<unsigned char>(line[first]))) ++first;
    while (last > first && std::isspace(static_cast<unsigned char>(line[last - 1]))) --last;
    if (first == last) return {"empty", {}};
    const auto trimmed = line.substr(first, last - first);
    if (trimmed == "/regen" || trimmed == "/new" || trimmed == "/quit") return {trimmed, {}};
    if(trimmed=="/think on" || trimmed=="/think off") return {"/think",trimmed.substr(7)};
    if (trimmed == "\"\"\"") return {"block", {}};
    if (trimmed == "/file") return {"/file", {}};
    if (trimmed.size() > 5 && trimmed.compare(0, 5, "/file") == 0 &&
        std::isspace(static_cast<unsigned char>(trimmed[5]))) {
        auto start = trimmed.find_first_not_of(" \t\r\n", 5);
        return {"/file", start == std::string::npos ? "" : trimmed.substr(start)};
    }
    if (trimmed.front() == '/') return {"unknown", trimmed};
    return {"message", line};
}
}

namespace cli {
std::pair<std::string, std::string> ChatLines::feed(std::string line) {
    const auto classified = classify_chat_line(line);
    if (classified.first == "block") {
        if (!collecting_) { collecting_ = true; return {"empty", {}}; }
        auto message = std::move(block_);
        finish();
        return {message.empty() ? "empty" : "message", std::move(message)};
    }
    if (!collecting_) return classified;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (lines_++) block_ += '\n';
    block_ += line;
    return {"empty", {}};
}
}
