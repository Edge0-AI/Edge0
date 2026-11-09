#include "model/json.h"
#include <cmath>
#include <iomanip>
#include <locale>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace edge0::model {
namespace {
[[noreturn]] void invalid() { throw std::invalid_argument("Invalid JSON"); }
void utf8(std::string& out, unsigned cp) {
    if (cp <= 0x7f) out += static_cast<char>(cp);
    else if (cp <= 0x7ff) {
        out += static_cast<char>(0xc0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 63));
    } else if (cp <= 0xffff) {
        out += static_cast<char>(0xe0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 63));
        out += static_cast<char>(0x80 | (cp & 63));
    } else {
        out += static_cast<char>(0xf0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 63));
        out += static_cast<char>(0x80 | ((cp >> 6) & 63));
        out += static_cast<char>(0x80 | (cp & 63));
    }
}
class Reader {
    std::string_view input;
    std::size_t pos = 0;
    bool take(char c) { if (pos < input.size() && input[pos] == c) { ++pos; return true; } return false; }
    void space() { while (pos < input.size() && (input[pos]==' ' || input[pos]=='\n' || input[pos]=='\r' || input[pos]=='\t')) ++pos; }
    unsigned hex() {
        unsigned result = 0;
        for (int i=0;i<4;++i) {
            if (pos == input.size()) invalid();
            const char c=input[pos++];
            const int n=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;
            if (n<0) invalid();
            result=result*16+static_cast<unsigned>(n);
        }
        return result;
    }
    std::string string() {
        if (!take('"')) invalid();
        std::string out;
        while (pos<input.size()) {
            const auto c=static_cast<unsigned char>(input[pos++]);
            if (c=='"') return out;
            if (c<32) invalid();
            if (c=='\\') {
                if (pos==input.size()) invalid();
                switch(input[pos++]) {
                case '"': out+='"'; break; case '\\': out+='\\'; break; case '/': out+='/'; break;
                case 'b': out+='\b'; break; case 'f': out+='\f'; break; case 'n': out+='\n'; break;
                case 'r': out+='\r'; break; case 't': out+='\t'; break;
                case 'u': {
                    unsigned cp=hex();
                    if (cp>=0xd800 && cp<=0xdbff) {
                        if (!take('\\') || !take('u')) invalid();
                        const unsigned low=hex();
                        if (low<0xdc00 || low>0xdfff) invalid();
                        cp=0x10000+((cp-0xd800)<<10)+(low-0xdc00);
                    } else if (cp>=0xdc00 && cp<=0xdfff) invalid();
                    utf8(out,cp); break;
                }
                default: invalid();
                }
            } else if (c<128) out+=static_cast<char>(c);
            else {
                unsigned cp=0; int count=0; unsigned minimum=0;
                if (c>=0xc2 && c<=0xdf) { cp=c&31;count=1;minimum=0x80; }
                else if(c>=0xe0 && c<=0xef) { cp=c&15;count=2;minimum=0x800; }
                else if(c>=0xf0 && c<=0xf4) { cp=c&7;count=3;minimum=0x10000; }
                else invalid();
                for(int i=0;i<count;++i) {
                    if(pos==input.size()) invalid();
                    const auto b=static_cast<unsigned char>(input[pos++]);
                    if((b&0xc0)!=0x80) invalid();
                    cp=(cp<<6)|(b&63);
                }
                if(cp<minimum || cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff)) invalid();
                utf8(out,cp);
            }
        }
        invalid();
    }
    Json value(unsigned depth) {
        if(depth>256) invalid();
        space();
        Json out;
        if(pos==input.size()) invalid();
        if(input[pos]=='"') { out.kind=Json::Kind::string;out.value=string();return out; }
        if(take('{')) {
            out.kind=Json::Kind::object;space(); if(take('}')) return out;
            std::unordered_set<std::string> keys;
            do {
                space();auto key=string();
                if(!keys.insert(key).second) invalid();
                space();if(!take(':')) invalid();
                out.members.emplace_back(std::move(key),value(depth+1));space();
                if(take('}')) return out;
            } while(take(','));
            invalid();
        }
        if(take('[')) {
            out.kind=Json::Kind::array;space();if(take(']')) return out;
            do { out.elements.push_back(value(depth+1));space();if(take(']')) return out; } while(take(','));
            invalid();
        }
        for(const auto* literal:{"null","true","false"}) {
            const std::string_view word(literal);
            if(input.substr(pos,word.size())==word) {
                pos+=word.size();out.kind=word=="null"?Json::Kind::null:Json::Kind::boolean;
                out.value=word;return out;
            }
        }
        const auto start=pos;
        take('-');
        if(!take('0')) {
            if(pos==input.size() || input[pos]<'1' || input[pos]>'9') invalid();
            while(pos<input.size() && input[pos]>='0' && input[pos]<='9') ++pos;
        }
        auto digits=[&] {
            const auto begin=pos;
            while(pos<input.size() && input[pos]>='0' && input[pos]<='9') ++pos;
            if(pos==begin) invalid();
        };
        if(take('.')) digits();
        if(take('e') || take('E')) { if(!take('+')) take('-'); digits(); }
        out.kind=Json::Kind::number;out.value=input.substr(start,pos-start);
        // Validate overflow even if the caller only inspects the parsed object.
        (void)out.dump();
        return out;
    }
public:
    explicit Reader(std::string_view s):input(s) {}
    Json read() { auto out=value(0);space();if(pos!=input.size()) invalid();return out; }
};
std::string number(const std::string& text) {
    if(text.find_first_of(".eE")==std::string::npos) return text=="-0"?"0":text;
    double d=0;
    std::istringstream input(text);
    input.imbue(std::locale::classic());
    input>>d;
    // num_get sets failbit for underflow as well as overflow. Underflow is a
    // valid binary64 conversion (including signed zero), as in Python.
    if(!std::isfinite(d) || (input.fail() && std::abs(d)==std::numeric_limits<double>::max())) invalid();
    // Find the shortest correctly rounded significant-digit representation that
    // round-trips to this binary64 value, using the classic locale throughout.
    std::string s;
    for(int precision=1;precision<=17;++precision) {
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output<<std::scientific<<std::setprecision(precision-1)<<d;
        s=output.str();
        std::istringstream check(s);
        check.imbue(std::locale::classic());
        double roundtrip=0;
        check>>roundtrip;
        if(roundtrip==d) break;
    }
    const auto e=s.find('e');
    const int exponent=std::stoi(s.substr(e+1));
    if(exponent < -4 || exponent >= 16) return s;
    const bool negative=s[0]=='-';
    std::string digits=s.substr(negative?1:0,e-(negative?1:0));
    const auto dot=digits.find('.');
    if(dot!=std::string::npos) digits.erase(dot,1);
    const int point=exponent+1;
    std::string out=negative?"-":"";
    if(point<=0) out+="0."+std::string(static_cast<std::size_t>(-point),'0')+digits;
    else if(static_cast<std::size_t>(point)>=digits.size())
        out+=digits+std::string(static_cast<std::size_t>(point)-digits.size(),'0')+".0";
    else out+=digits.substr(0,point)+"."+digits.substr(point);
    return out;
}
}
const Json* Json::find(std::string_view key) const {
    for(const auto& item:members) if(item.first==key) return &item.second;
    return nullptr;
}
Json Json::parse(std::string_view s) { return Reader(s).read(); }
std::string json_quote(std::string_view s) {
    std::string out="\"";
    constexpr char hex[]="0123456789abcdef";
    for(unsigned char c:s) {
        switch(c) {
        case '"':out+="\\\"";break;case '\\':out+="\\\\";break;
        case '\n':out+="\\n";break;case '\r':out+="\\r";break;case '\t':out+="\\t";break;
        case '\b':out+="\\b";break;case '\f':out+="\\f";break;
        default:if(c<32) {out+="\\u00";out+=hex[c>>4];out+=hex[c&15];} else out+=static_cast<char>(c);
        }
    }
    return out+'"';
}
std::string Json::dump() const {
    switch(kind) {
    case Kind::null:return "null";
    case Kind::boolean:return value;
    case Kind::number:return number(value);
    case Kind::string:return json_quote(value);
    case Kind::array: {
        std::string out="[";
        for(const auto& item:elements) {if(out.size()>1) out+=", ";out+=item.dump();}
        return out+"]";
    }
    case Kind::object: {
        std::string out="{";
        for(const auto& item:members) {if(out.size()>1) out+=", ";out+=json_quote(item.first)+": "+item.second.dump();}
        return out+"}";
    }
    }
    invalid();
}
} // namespace edge0::model
