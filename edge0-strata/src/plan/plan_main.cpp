// src/plan/plan_main.cpp - the `strata-plan` CLI: print the memory plan, or say why it does not close.
//
// The engine must adapt to the GPU it is running on, so the plan is computed from measured free VRAM and the
// requested context rather than configured by hand.  This tool is that computation on its own, which is what
// makes it testable without a GPU.
#include "strata/plan/plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    uint64_t ctx = 20480;
    uint64_t free_vram = 12ull * 1000 * 1000 * 1000; // RTX 5070 nominal; overridden by --vram
    strata::plan::Costs costs;
    bool state_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](uint64_t& out) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            out = std::strtoull(argv[++i], nullptr, 10);
        };
        if (a == "--max-context")
            next(ctx);
        else if (a == "--vram")
            next(free_vram);
        else if (a == "--dense")
            next(costs.dense_bytes);
        else if (a == "--embd")
            next(costs.embd_bytes);
        else if (a == "--workspace")
            next(costs.workspace_bytes);
        else if (a == "--state") {
            next(costs.state_bytes);
            state_given = true;
        } else if (a == "--pool") {
            uint64_t v = 0;
            next(v);
            free_vram = v;
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: strata-plan [--max-context N] [--vram BYTES] [--dense B] [--embd B]\n"
                        "                   [--workspace B] [--state B] [--pool B]\n"
                        "  --state defaults to the geometry's own GDN recurrence + conv history, NOT to 0:\n"
                        "  the flag used to default to zero and nothing passed it, so the planner silently\n"
                        "  planned without 117.7 MB (85 expert slots) of non-evictable state.\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    // DEFAULT THE STATE FROM THE GEOMETRY.  A `--state` that defaults to 0 and is never passed is not a
    // default, it is an omission - and this one was worth 85 cache slots of overcommit.
    if (!state_given) costs.state_bytes = strata::plan::state_bytes(strata::plan::Geometry{});
    // The pool is what the KV cache and the expert cache share.  Derived from the card's free VRAM less the
    // resident dense weights; `--pool` overrides it for reproducing the standard-tier numbers exactly.
    const uint64_t pool = free_vram > costs.dense_bytes + costs.embd_bytes
                              ? free_vram - costs.dense_bytes - costs.embd_bytes
                              : 0;
    try {
        const strata::plan::Plan p = strata::plan::make_plan(ctx, strata::plan::Geometry{}, costs,
                                                             pool ? pool : strata::plan::vram_pool_bytes());
        std::printf("%s", strata::plan::to_string(p).c_str());
        return 0;
    } catch (const strata::plan::DoesNotClose& e) {
        std::fprintf(stderr, "strata-plan: %s\n", e.what());
        return 1;
    }
}
