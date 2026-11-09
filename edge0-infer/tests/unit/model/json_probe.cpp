#include "model/json.h"
#include <iostream>
#include <string>
int main() {
    std::string line;
    while(std::getline(std::cin,line)) {
        try { std::cout<<edge0::model::Json::parse(line).dump()<<'\n'; }
        catch(const std::invalid_argument&) { std::cout<<"INVALID\n"; }
    }
}
