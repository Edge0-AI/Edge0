#include "strata/program/message_boundary.hpp"

#include <cstdio>

int main() {
    using strata::program::message_checkpoint_boundary;
    constexpr int64_t token = 248045;
    std::vector<int64_t> ids(33100, 7);
    ids[0] = ids[12] = ids[32800] = ids[33000] = token;
    int checks = 0;
    auto check = [&](int64_t resume, int64_t turn, int64_t expected) {
        ++checks;
        return message_checkpoint_boundary(ids, resume, turn, token) == expected;
    };
    if (!check(0, 33000, 32800) || !check(22016, 33000, 32800) ||
        !check(32800, 33000, -1) || !check(33000, 33000, -1) ||
        !check(-1, 33000, -1) || !check(0, -1, -1) ||
        !check(0, 33100, -1) || !check(0, 32999, -1) ||
        !check(0, 12, -1) || !check(0, 32800, -1)) return 1;
    for (int tail : {1, 63, 64, 512, 1024, 1025, 2048}) {
        std::vector<int64_t> fixture(8192 + tail + 1, 9);
        fixture[8192] = fixture[8192 + tail] = token;
        if (message_checkpoint_boundary(fixture, 0, 8192 + tail, token) !=
            (tail <= 1024 ? 8192 : -1)) return 1;
        ++checks;
    }
    if (message_checkpoint_boundary({}, 0, 0, token) != -1 ||
        message_checkpoint_boundary(ids, 0, 33000, -1) != -1 ||
        message_checkpoint_boundary(ids, 0, 33000, token, 8192, 0) != -1 ||
        message_checkpoint_boundary(ids, 0, 33000, token, -1) != -1) return 1;
    checks += 4;
    std::printf("message_boundary_test OK (%d checks)\n", checks);
}
