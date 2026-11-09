#include "models/edge0_35b/output.h"
#include "model/json.h"
#include <set>
#include <stdexcept>

namespace edge0::models::edge0_35b {
namespace {
using model::ParsedKind;
using model::ParsedOutput;
using model::ParsedReceiver;
std::size_t character_size(const std::string& text) {
    const auto c=static_cast<unsigned char>(text.front());
    return c<128?1:c<224?2:c<240?3:4;
}
bool whitespace(const std::string& text) {
    unsigned cp=static_cast<unsigned char>(text.front());
    const auto size=character_size(text);
    if(size>1) {
        cp &= size==2?31:size==3?15:7;
        for(std::size_t i=1;i<size;++i) cp=(cp<<6)|(static_cast<unsigned char>(text[i])&63);
    }
    return (cp>=9&&cp<=13)||(cp>=0x1c&&cp<=0x20)||cp==0x85||cp==0xa0||
        cp==0x1680||(cp>=0x2000&&cp<=0x200a)||cp==0x2028||cp==0x2029||
        cp==0x202f||cp==0x205f||cp==0x3000;
}
std::string trim(const std::string& text) {
    std::size_t first=text.size(), last=0;
    for(std::size_t i=0;i<text.size();) {
        const auto suffix=text.substr(i);
        const auto n=character_size(suffix);
        if(!whitespace(suffix)) {if(first==text.size()) first=i;last=i+n;}
        i+=n;
    }
    return first==text.size()?std::string{}:text.substr(first,last-first);
}
class Parser final : public model::OutputParser {
    enum class State { reasoning, answer, function, parameter, value, closed, broken };
    State state_;
    bool tools_, leading_, classification_, string_value_=false, stopped_=false;
    std::string buffer_, spaces_, raw_, arguments_, value_;
    ToolCall call_;
    std::size_t index_=0;
    std::set<std::string> keys_;
    std::vector<model::Json> definitions_;
    static constexpr const char* end_call="</tool_call>";
    bool emit(ParsedKind kind,const std::string& text,const ParsedReceiver& receive,
              const std::string& reason={}) {
        ParsedOutput event{kind,text,index_,call_,reason};
        if(!receive(event)) {stopped_=true;return false;}
        return true;
    }
    std::string consume(std::size_t n,bool in_call=true) {
        auto text=buffer_.substr(0,n);buffer_.erase(0,n);
        if(in_call) raw_+=text;
        return text;
    }
    bool prefix(const std::string& marker) const {
        return buffer_.size()<marker.size() && marker.compare(0,buffer_.size(),buffer_)==0;
    }
    bool starts(const std::string& marker) const {return buffer_.compare(0,marker.size(),marker)==0;}
    bool argument(const std::string& text,const ParsedReceiver& receive) {
        arguments_+=text;return emit(ParsedKind::tool_call_arguments,text,receive);
    }
    bool is_string(const std::string& key) const {
        for(const auto& definition:definitions_) {
            const auto* f=definition.find("function");if(!f) continue;
            const auto* name=f->find("name");
            if(!name || name->kind!=model::Json::Kind::string || name->value!=call_.name) continue;
            const auto* p=f->find("parameters");if(p) p=p->find("properties");if(p) p=p->find(key);if(p) p=p->find("type");
            if(!p) return false;
            return (p->kind==model::Json::Kind::string && p->value=="string") ||
                (p->kind==model::Json::Kind::array && p->elements.size()==1 &&
                 p->elements[0].kind==model::Json::Kind::string && p->elements[0].value=="string");
        }
        return false;
    }
    bool end(bool valid,const ParsedReceiver& receive) {
        call_.arguments_json=arguments_;
        if(!emit(valid?ParsedKind::tool_call_end:ParsedKind::tool_call_rejected,
                 valid?"":raw_,receive,valid?"":"malformed")) return false;
        ++index_;state_=State::answer;leading_=true;spaces_.clear();return true;
    }
    bool process(const ParsedReceiver& receive,bool final) {
        while(!buffer_.empty() && !stopped_) {
            const bool plain=state_==State::answer || state_==State::reasoning;
            if(plain) {
                if(state_==State::reasoning && starts("</think>")) {
                    consume(8,false);spaces_.clear();state_=State::answer;leading_=true;continue;
                }
                if(tools_ && starts("<tool_call>")) {
                    consume(11,false);spaces_.clear();raw_.clear();arguments_.clear();keys_.clear();
                    call_={};state_=State::function;continue;
                }
                if(!final && ((state_==State::reasoning && prefix("</think>")) ||
                              (tools_ && prefix("<tool_call>")))) break;
                const auto n=character_size(buffer_);
                auto text=consume(n,false);
                if(whitespace(text)) {
                    if(!leading_) spaces_+=text;
                    continue;
                }
                leading_=false;
                text=spaces_+text;spaces_.clear();
                if(!emit(state_==State::reasoning?ParsedKind::reasoning:ParsedKind::answer,text,receive)) return false;
                continue;
            }
            if(starts(end_call)) {
                consume(12,false);
                if(!end(state_==State::closed,receive)) return false;
                continue;
            }
            if(!final && prefix(end_call)) break;
            if(state_==State::broken) {consume(character_size(buffer_));continue;}
            if(state_==State::value) {
                const std::string delimiter="\n</parameter>";
                if(starts(delimiter)) {
                    consume(delimiter.size());
                    if(string_value_) {if(!argument("\"",receive)) return false;}
                    else {
                        std::string encoded;
                        try {encoded=model::Json::parse(value_).dump();}
                        catch(const std::invalid_argument&) {encoded=model::json_quote(value_);}
                        if(!argument(encoded,receive)) return false;
                    }
                    state_=State::parameter;continue;
                }
                if(!final && prefix(delimiter)) break;
                auto text=consume(character_size(buffer_));
                if(string_value_) {
                    auto quoted=model::json_quote(text);
                    if(!argument(quoted.substr(1,quoted.size()-2),receive)) return false;
                } else value_+=text;
                continue;
            }
            if(whitespace(buffer_)) {consume(character_size(buffer_));continue;}
            if(state_==State::closed) {state_=State::broken;continue;}
            const std::string open=state_==State::function?"<function=":"<parameter=";
            if(state_==State::parameter && starts("</function>")) {
                consume(11);
                if(!argument(keys_.empty()?"{}":"}",receive)) return false;
                state_=State::closed;continue;
            }
            if(!final && (prefix(open) || (state_==State::parameter && prefix("</function>")))) break;
            if(!starts(open)) {state_=State::broken;continue;}
            const auto close=buffer_.find('>',open.size());
            if(close==std::string::npos) {if(!final) break;state_=State::broken;continue;}
            auto name=trim(buffer_.substr(open.size(),close-open.size()));
            if(name.empty()) {state_=State::broken;continue;}
            if(state_==State::function) {
                consume(close+1);call_.name=name;state_=State::parameter;
                if(!emit(ParsedKind::tool_call_begin,"",receive)) return false;
            } else {
                if(buffer_.size()==close+1 && !final) break;
                if(buffer_.size()==close+1 || buffer_[close+1]!='\n' || keys_.count(name)) {
                    state_=State::broken;continue;
                }
                consume(close+2);
                const auto start=keys_.empty()?"{":", ";
                keys_.insert(name);string_value_=is_string(name);value_.clear();state_=State::value;
                if(!argument(start+model::json_quote(name)+": ",receive)) return false;
                if(string_value_ && !argument("\"",receive)) return false;
            }
        }
        return !stopped_;
    }
public:
    explicit Parser(const model::TextOptions& options):
        state_(options.thinking.value_or(false)?State::reasoning:State::answer),
        tools_(!options.tools.empty()),leading_(options.thinking.value_or(false)),
        classification_(tools_ || leading_) {
        for(const auto& tool:options.tools) definitions_.push_back(model::Json::parse(tool.json));
    }
    ParsedKind section() const override {
        return state_==State::reasoning?ParsedKind::reasoning:
            state_==State::answer?ParsedKind::answer:ParsedKind::tool_call_arguments;
    }
    bool preserve_control_piece(const std::string& piece) const override {
        return classification_ && (piece=="<think>" || piece=="</think>" ||
                                   piece=="<tool_call>" || piece=="</tool_call>");
    }
    bool feed(const std::string& text,const ParsedReceiver& receive) override {
        if(stopped_) return false;
        // Preserve the pre-classification delivery granularity on the legacy path.
        if(!tools_ && state_==State::answer && !leading_ && buffer_.empty() && spaces_.empty())
            return text.empty() || emit(ParsedKind::answer,text,receive);
        buffer_+=text;return process(receive,false);
    }
    bool finish(const ParsedReceiver& receive) override {
        if(stopped_) return false;
        if(!process(receive,true)) return false;
        if(state_!=State::answer && state_!=State::reasoning) {
            raw_+=buffer_;buffer_.clear();
            return emit(ParsedKind::tool_call_rejected,raw_,receive,"incomplete");
        }
        if(state_==State::answer && !spaces_.empty()) {
            auto text=std::move(spaces_);spaces_.clear();
            return emit(ParsedKind::answer,text,receive);
        }
        return true;
    }
};
}
std::unique_ptr<model::OutputParser> output_parser(const model::TextOptions& options) {
    return std::make_unique<Parser>(options);
}
}
