// Strata. The DeepSeek V4.1 engine's prompt-path rule (src/dsv41/ds4_pick_rule.hpp): by length only, one number, DS4_PICK_MIN moves it.
#include "ds4_pick_rule.hpp"

#include <cstdio>

using strata::ds4::prompt_flow_min;
using strata::ds4::prompt_uses_flow;

static int fails = 0;
#define CHECK(c)                                                  \
    do {                                                          \
        if (!(c)) {                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                              \
        }                                                         \
    } while (0)

int main() {
    const int m = prompt_flow_min(nullptr);
    CHECK(m == 96);
    CHECK(prompt_flow_min("") == 96);
    CHECK(prompt_flow_min("400") == 400);
    CHECK(prompt_flow_min("2") == 2);
    CHECK(prompt_flow_min("0") == 1);   // never below one token
    CHECK(prompt_flow_min("-5") == 1);
    // the same prompt always takes the same path
    for (int rep = 0; rep < 3; ++rep) {
        CHECK(!prompt_uses_flow(1, m));    // a decode token
        CHECK(!prompt_uses_flow(28, m));   // a short chat turn
        CHECK(!prompt_uses_flow(95, m));
        CHECK(prompt_uses_flow(96, m));
        CHECK(prompt_uses_flow(300, m));
        CHECK(prompt_uses_flow(8192, m));
    }
    CHECK(prompt_uses_flow(100, 2) && !prompt_uses_flow(100, 100000));   // DS4_PICK_MIN=2 / 100000: always flow / always windows
    if (fails == 0) std::printf("prompt path rule: ok\n");
    return fails == 0 ? 0 : 1;
}
