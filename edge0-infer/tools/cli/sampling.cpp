#include "edge0/generation.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace cli {
const char* sampling_help() { return " [--temperature <t>] [--top-k <k>] [--top-p <p>] [--repetition-penalty <r>] [--seed <uint64>] [--thinking on|off] [--stop <text>] [--presence-penalty <p>] [--frequency-penalty <p>]\nThinking switches model recommendations; explicit parameters take precedence. Defaults include presence penalty; for pure greedy use --temperature 0 --presence-penalty 0."; }
bool sampling_option(const std::string& arg, const std::string& text, edge0::GenerationOverrides& o) {
    std::size_t used = 0;
    if(arg=="--thinking") {
        if(text!="on" && text!="off") throw std::runtime_error("Expected thinking on or off");
        o.thinking=text=="on";
    } else if(arg=="--stop") {
        if(!o.stop) o.stop=std::vector<std::string>{};
        o.stop->push_back(text);
    } else if(arg=="--presence-penalty" || arg=="--frequency-penalty") {
        const auto value=std::stod(text,&used);
        if(used!=text.size() || !std::isfinite(value) || std::abs(value)>2) throw std::runtime_error("Invalid penalty");
        if(arg=="--presence-penalty") o.presence_penalty=value;else o.frequency_penalty=value;
    } else if (arg == "--temperature" || arg == "--top-p" || arg == "--repetition-penalty") {
        const double value = std::stod(text, &used);
        if (used != text.size() || !std::isfinite(value) ||
            (arg == "--temperature" ? value < 0 : arg == "--top-p" ? value <= 0 || value > 1 : value <= 0))
            throw std::runtime_error("Invalid sampling parameter: " + arg);
        if (arg == "--temperature") o.temperature = value;
        else if (arg == "--top-p") o.top_p = value;
        else o.repetition_penalty = value;
    } else if (arg == "--top-k") {
        const auto value = std::stoll(text, &used);
        if (used != text.size() || value < 0) throw std::runtime_error("Invalid top-k");
        o.top_k = value;
    } else if (arg == "--seed") {
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("Invalid seed");
        const auto value = std::stoull(text, &used);
        if (used != text.size()) throw std::runtime_error("Invalid seed");
        o.seed = value;
    } else return false;
    return true;
}
}
