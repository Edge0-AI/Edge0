#pragma once
#include <string>
#include <utility>
namespace cli {
std::pair<std::string, std::string> classify_chat_line(std::string);
class ChatLines {
public:
    std::pair<std::string, std::string> feed(std::string);
    bool collecting() const { return collecting_; }
    void finish() { collecting_ = false; lines_ = 0; block_.clear(); }
private:
    bool collecting_ = false;
    std::size_t lines_ = 0;
    std::string block_;
};
}
