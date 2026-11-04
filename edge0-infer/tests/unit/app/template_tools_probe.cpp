#include "model/json.h"
#include "models/edge0_35b/text.h"
#include <iostream>
#include <iterator>
#ifdef _WIN32
// check_layers allows these headers in this file only (binary stdio below).
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#endif
int main() {
#ifdef _WIN32
    // Read and write raw bytes so the JSON input is consumed verbatim and a bare
    // '\n' in the rendered template is not expanded to CRLF; the test compares the
    // output byte for byte against the Python Jinja rendering.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        using edge0::model::Json;
        const auto input=Json::parse(std::string(std::istreambuf_iterator<char>(std::cin),{}));
        edge0::model::TextOptions options;
        options.thinking=input.find("thinking")->value=="true";
        options.preserve_thinking=input.find("preserve")->value=="true";
        if(const auto* tools=input.find("tools"))
            for(const auto& tool:tools->elements) options.tools.push_back({tool.dump()});
        std::vector<edge0::model::Message> messages;
        for(const auto& m:input.find("messages")->elements) {
            edge0::model::Message message{m.find("role")->value,m.find("content")->value};
            if(const auto* r=m.find("reasoning_content")) message.reasoning_content=r->value;
            if(const auto* calls=m.find("tool_calls")) for(const auto& c:calls->elements) {
                const auto* f=c.find("function");if(!f) f=&c;
                message.tool_calls.push_back({"",f->find("name")->value,f->find("arguments")->dump()});
            }
            messages.push_back(std::move(message));
        }
        for(const auto& f:edge0::models::edge0_35b::text_format(messages,options).fragments) std::cout<<f.text;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
