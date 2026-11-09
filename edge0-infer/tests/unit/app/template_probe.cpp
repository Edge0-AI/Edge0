#include "app/text/rules.h"
#include "app/text/unicode.h"
#include "models/edge0_35b/assets.h"
#include <sstream>
#include <iostream>
#include <iterator>
int main(int argc,char** argv) {
    try {
        int preserve=0;std::cin>>preserve;std::string line;std::getline(std::cin,line);
        std::vector<edge0::TextMessage> messages;
        while(std::getline(std::cin,line)) {
            const auto split=line.find(' ');if(split==std::string::npos)throw std::runtime_error("Invalid probe input");
            const auto encoded=line.substr(split+1);if(encoded.size()%2)throw std::runtime_error("Odd hexadecimal input");
            std::string content;
            for(std::size_t i=0;i<encoded.size();i+=2)content+=static_cast<char>(std::stoul(encoded.substr(i,2),nullptr,16));
            messages.push_back({line.substr(0,split),content});
        }
        edge0::PromptOptions options{false,preserve!=0};
        if(argc==2) {
            auto result=edge0::render_prompt(argv[1],messages,options);
            if(!result)throw std::runtime_error(result.error().message);
            for(auto token:result.value().tokens)std::cout<<token<<' ';
            std::cout<<'\n';return 0;
        }
        edge0::text::TokenData data;
        for(int i=0;i<256;++i) {data.tokens.push_back(edge0::text::unicode_byte_to_utf8(static_cast<std::uint8_t>(i)));data.types.push_back(1);}
        const auto& adapter=edge0::models::edge0_35b::adapter();
        edge0::text::Rules rules{edge0::text::Bpe(std::move(data)),adapter.describe_text(),&adapter};
        auto result=edge0::text::render_input(rules,messages,options);
        if(!result)throw std::runtime_error(result.error().message);
        if(result.value().preserve_thinking!=*options.preserve_thinking)throw std::runtime_error("Effective option mismatch");
        std::cout<<result.value().text;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
