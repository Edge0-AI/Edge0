#include "strata/core/expert_cache.hpp"
#include <cstdio>
#include <vector>

int main() {
    strata::core::ExpertCache cache;
    std::string err;
    auto check = [&](bool ok) {
        if (!ok) std::fprintf(stderr, "%s\n", err.c_str());
        return ok;
    };
    // open_sized first allocates byte slots, then grows to the largest expert.
    // Exercise that growth and repeated reuse with distinct pageable payloads.
    for (int reopen = 0; reopen < 2; ++reopen) {
        if (!check(cache.open_sized({4096, 8192, 16384}, 1, 3, err))) return 1;
        for (int round = 0; round < 8; ++round) {
            for (int slot = 0; slot < 3; ++slot) {
                std::vector<uint8_t> payload((size_t) 4096 << slot);
                for (size_t i = 0; i < payload.size(); ++i)
                    payload[i] = (uint8_t) (i * 17 + round * 11 + slot + reopen);
                if (!check(cache.fill_slot_blocking(slot, payload.data(), err, payload.size())) ||
                    !check(cache.verify_slot(slot, payload.data(), err, payload.size()))) return 1;
            }
        }
        if (cache.fill_slot_blocking(0, nullptr, err)) return 1;
        cache.close();
    }
    if (!check(cache.open(2, 1, 2, 1024, err))) return 1;
    std::vector<uint8_t> payload(1024, 73);
    if (!check(cache.fill_slot_blocking(1, payload.data(), err)) ||
        !check(cache.verify_slot(1, payload.data(), err))) return 1;
    std::puts("HIP expert staging: variable slots, repeated payloads, close/reopen passed");
    return 0;
}
