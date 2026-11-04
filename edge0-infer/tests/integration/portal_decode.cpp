#include "edge0/text.h"
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::vector<std::int32_t> tokens;std::int32_t token;
    while(std::cin>>token)tokens.push_back(token);
    const auto result=edge0::detokenize(argv[1],tokens);
    if(!result){std::cerr<<result.error().message<<'\n';return 1;}
    std::cout<<result.value().text;
}
