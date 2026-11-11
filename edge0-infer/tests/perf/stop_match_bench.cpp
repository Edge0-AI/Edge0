#include "app/text/stop.h"
#include <chrono>
#include <iostream>
int main() {
    const std::string prefix(4096, 'a');
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 3; ++repeat) {
        edge0::text::StopMatcher matcher({prefix + "b"});
        std::string output;
        for (std::size_t i = 0; i < prefix.size(); ++i) output += matcher.feed("a");
        output += matcher.feed("c");
        output += matcher.finish();
        if (output != prefix + "c" || matcher.matched()) return 1;
    }
    std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
}
