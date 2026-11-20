#pragma once
#include <cstddef>
#include <string>
#include <utility>

namespace delegate {
// Incremental SSE framing. Only a blank line dispatches an event; data fields are joined with LF.
class SseDecoder {
public:
    template <class F> void push(const char* bytes, std::size_t size, F&& consume) {
        for (std::size_t i = 0; i < size; ++i) {
            const char c = bytes[i];
            if (after_cr_) {
                after_cr_ = false;
                if (c == '\n') continue;
            }
            if (c != '\r' && c != '\n') { line_ += c; continue; }
            after_cr_ = c == '\r';
            auto line = std::move(line_);
            line_.clear();
            if (first_line_) {
                first_line_ = false;
                if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
            }
            if (line.empty()) {
                if (data_.empty()) continue;
                auto data = std::move(data_);
                data_.clear();
                data.pop_back();
                consume(data);
            } else if (line == "data" || line.compare(0, 5, "data:") == 0) {
                std::size_t start = line == "data" ? line.size() : 5;
                if (start < line.size() && line[start] == ' ') ++start;
                data_.append(line, start, std::string::npos);
                data_ += '\n';
            }
        }
    }
    bool empty() const { return line_.empty() && data_.empty(); }
private:
    std::string line_, data_;
    bool after_cr_ = false, first_line_ = true;
};
}
