#pragma once
// Minimal JSON value and parser for the CLI input files (turns, replay). Integers only; strict syntax.
#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
namespace cli {
struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, std::int64_t, std::string, Array, Object> value = nullptr;
    const Object& object() const { return std::get<Object>(value); }
    const Array& array() const { return std::get<Array>(value); }
    const std::string& string() const { return std::get<std::string>(value); }
    std::int64_t integer() const { return std::get<std::int64_t>(value); }
    bool boolean() const { return std::get<bool>(value); }
    bool has(const std::string& key) const { return object().count(key); }
    const Json& at(const std::string& key) const { return object().at(key); }
};
class JsonParser {
    const std::string& input;
    std::size_t p = 0;
    [[noreturn]] void error() const { throw std::runtime_error("Invalid turns/replay JSON"); }
    void spaces() { while (p < input.size() && std::isspace(static_cast<unsigned char>(input[p]))) ++p; }
    bool consume(char c) { spaces(); if (p < input.size() && input[p] == c) { ++p; return true; } return false; }
    unsigned hex() {
        unsigned n = 0;
        for (int i = 0; i < 4; ++i) {
            if (p == input.size()) error();
            const char c = input[p++];
            unsigned v = c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 : c >= 'A' && c <= 'F' ? c-'A'+10 : 16;
            if (v == 16) error(); n = n*16+v;
        }
        return n;
    }
    std::string string() {
        if (!consume('"')) error();
        std::string result;
        while (p < input.size()) {
            const char c = input[p++];
            if (c == '"') return result;
            if (static_cast<unsigned char>(c) < 32) error();
            if (c != '\\') { result += c; continue; }
            if (p == input.size()) error();
            const char e = input[p++];
            switch (e) {
            case '"': case '\\': case '/': result += e; break;
            case 'b': result += '\b'; break;
            case 'f': result += '\f'; break;
            case 'n': result += '\n'; break;
            case 'r': result += '\r'; break;
            case 't': result += '\t'; break;
            case 'u': {
                auto cp = hex();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (p+2 > input.size() || input.substr(p,2) != "\\u") error();
                    p += 2; const auto lo = hex(); if (lo < 0xdc00 || lo > 0xdfff) error();
                    cp = 0x10000 + ((cp-0xd800)<<10) + lo-0xdc00;
                } else if (cp >= 0xdc00 && cp <= 0xdfff) error();
                if (cp < 0x80) result += static_cast<char>(cp);
                else {
                    if (cp < 0x800) result += static_cast<char>(0xc0 | (cp>>6));
                    else {
                        if (cp < 0x10000) result += static_cast<char>(0xe0 | (cp>>12));
                        else { result += static_cast<char>(0xf0 | (cp>>18)); result += static_cast<char>(0x80 | ((cp>>12)&63)); }
                        result += static_cast<char>(0x80 | ((cp>>6)&63));
                    }
                    result += static_cast<char>(0x80 | (cp&63));
                }
                break;
            }
            default: error();
            }
        }
        error();
    }
    Json parse(unsigned depth) {
        if (depth > 64) error();
        spaces(); if (p == input.size()) error();
        if (input[p] == '"') return Json{string()};
        if (consume('{')) {
            Json::Object object;
            if (consume('}')) return Json{object};
            do { const auto key = string(); if (!consume(':')) error();
                if (!object.emplace(key,parse(depth+1)).second) error();
            } while (consume(','));
            if (!consume('}')) error(); return Json{object};
        }
        if (consume('[')) {
            Json::Array array;
            if (consume(']')) return Json{array};
            do { array.push_back(parse(depth+1)); } while (consume(','));
            if (!consume(']')) error(); return Json{array};
        }
        for (const auto& item : std::vector<std::pair<std::string, Json>>{{"true",Json{true}},{"false",Json{false}},{"null",Json{nullptr}}}) {
            if (input.compare(p,item.first.size(),item.first)==0) { p += item.first.size(); return item.second; }
        }
        const auto start = p;
        if (input[p]=='-') ++p;
        if (p == input.size() || !std::isdigit(static_cast<unsigned char>(input[p]))) error();
        if (input[p]=='0') ++p;
        else while (p < input.size() && std::isdigit(static_cast<unsigned char>(input[p]))) ++p;
        return Json{static_cast<std::int64_t>(std::stoll(input.substr(start,p-start)))};
    }
public:
    explicit JsonParser(const std::string& text) : input(text) {}
    Json parse() { auto result = parse(0); spaces(); if (p != input.size()) error(); return result; }
};
}
