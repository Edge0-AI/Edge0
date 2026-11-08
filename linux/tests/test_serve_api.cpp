// Public hook/lifecycle checks against the built libllama, without model weights.
#include "mem_budget.h"
#include "prefetch.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("public API assertion failed"); }
size_t count(const char * path) {
    return std::distance(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator());
}
void clear_environment() {
    for (const char * key : {"E0_MEM_BUDGET_MB", "E0_POOL_MB", "E0_PREROUTER", "E0_GGUF", "E0_PREFETCH", "E0_PREF_ASYNC"}) {
        unsetenv(key);
    }
}
void dormant() {
    const size_t threads = count("/proc/self/task"), descriptors = count("/proc/self/fd");
    check(edge0_apply_mem_budget(0));
    check(!edge0_mem_budget_enabled() && edge0_mem_budget_check());
    check(!edge0_pref_init(nullptr, nullptr, 0, nullptr));
    edge0_pref_on_step(nullptr, -1);
    edge0_pref_shutdown();
    edge0_mem_budget_shutdown();
    check(threads == count("/proc/self/task") && descriptors == count("/proc/self/fd"));
}
void budget() {
    setenv("E0_MEM_BUDGET_MB", "256", 1);
    check(edge0_apply_mem_budget(0) && edge0_mem_budget_enabled() && edge0_mem_budget_check());
    edge0_mem_budget_shutdown();
    setenv("E0_MEM_BUDGET_MB", "bad", 1);
    check(edge0_apply_mem_budget(256) && edge0_mem_budget_check());
    edge0_mem_budget_shutdown();
    check(!edge0_apply_mem_budget(0) && !edge0_mem_budget_enabled());
    setenv("E0_MEM_BUDGET_MB", "0", 1);
    check(edge0_apply_mem_budget(0) && !edge0_mem_budget_enabled());
}
void invalid() {
    const size_t threads = count("/proc/self/task"), descriptors = count("/proc/self/fd");
    check(!edge0_apply_mem_budget(-1));
    setenv("E0_POOL_MB", "-1", 1);
    check(!edge0_pref_init(nullptr, nullptr, 0, nullptr));
    unsetenv("E0_POOL_MB");
    setenv("E0_PREROUTER", "missing-sidecar", 1);
    setenv("E0_PREFETCH", "unexpected", 1);
    check(!edge0_pref_init(nullptr, nullptr, 0, nullptr));
    check(threads == count("/proc/self/task") && descriptors == count("/proc/self/fd"));
}
} // namespace

int main(int argc, char ** argv) {
    clear_environment();
    try {
        check(argc == 2);
        const std::string name = argv[1];
        if (name == "api_dormant") dormant();
        else if (name == "api_budget") budget();
        else if (name == "api_invalid") invalid();
        else check(false);
        std::printf("PASS %s\n", name.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
