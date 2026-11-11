// src/spec/spec_start_test.cpp - the speculative loop's start position: 0 is a real position (a 1-token prompt on a native pack).
#include "strata/spec/spec_start.hpp"

#include <cstdio>

using namespace strata::spec;

int main() {
    int fail = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::printf("FAIL: %s\n", what); ++fail; }
    };
    check(!spec_loop_runs(kSpecNotStarted, 0, 100, false), "no start position: the loop is not used");
    check(spec_loop_runs(0, 0, 100, false), "position 0 (a 1-token prompt on a native pack) starts the loop");
    check(spec_loop_runs(910, 0, 100, false), "a long prompt's start position");
    check(spec_loop_runs(1, 0, 100, false), "position 1 (a 1-token prompt on the per-token path)");
    check(!spec_loop_runs(0, 100, 100, false), "nothing left to generate");
    check(!spec_loop_runs(0, 3, 100, true), "the text already ended");
    check(kSpecNotStarted < 0, "the sentinel is not a position");
    std::printf("spec_start tests: %s\n", fail ? "FAILED" : "OK");
    return fail ? 1 : 0;
}
