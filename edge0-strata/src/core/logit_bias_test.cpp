#include "strata/core/logit_bias.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    std::vector<float> b;
    std::string err;
    auto check = [](bool ok) { if (!ok) { std::fprintf(stderr, "logit_bias parser regression\n"); std::exit(1); } };
    check(strata::core::parse_logit_bias("0:-100,2:1.25,3:-0.5", 4, b, err));
    check(b.size() == 4 && std::isinf(b[0]) && b[0] < 0 && b[1] == 0 && b[2] == 1.25f && b[3] == -0.5f);
    for (const char* text : {"", "0:", "0:NaN", "0:inf", "0:-101", "0:101", "4:0", "-1:0", "x:1",
                            "0:1,", "0:1,0:2", "0:1x", "0:-100,1:-100,2:-100,3:-100", "999999999999999999999:0"}) {
        check(!strata::core::parse_logit_bias(text, 4, b, err));
        check(b.empty());
    }
    std::string large;
    for (int i = 0; i < 103215; ++i) { if (i) large += ','; large += std::to_string(i) + ":-100"; }
    check(strata::core::parse_logit_bias(large, 248320, b, err));
    check(std::isinf(b[103214]) && b[103215] == 0 && b.size() == 248320);
    std::puts("logit_bias_test: passed, including 103215 bans");
}
