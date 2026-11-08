// RSS budget polling must not scan all model page tables on each read.
#include "posix_io.h"
#include "mem_budget_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <sys/mman.h>

namespace {
void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Mapping {
    size_t size = size_t(1) << 30;
    void * data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Mapping() {
        check(data != MAP_FAILED, "cannot allocate RSS cost fixture");
        if (madvise(data, size, MADV_NOHUGEPAGE)) {
            munmap(data, size);
            throw std::runtime_error("cannot disable fixture huge pages");
        }
        std::memset(data, 1, size);
    }
    ~Mapping() { if (data != MAP_FAILED) munmap(data, size); }
};

bool reference_rss(size_t & bytes) {
    FILE * file = std::fopen("/proc/self/statm", "r");
    if (!file) return false;
    size_t total = 0, pages = 0;
    const bool ok = std::fscanf(file, "%zu %zu", &total, &pages) == 2;
    std::fclose(file);
    bytes = pages * edge0::page_size();
    return ok;
}

double measure(bool (*reader)(size_t &), size_t minimum) {
    const auto start = std::clock();
    for (int i = 0; i < 32; ++i) {
        size_t rss = 0;
        check(reader(rss) && rss >= minimum, "resident allocation missing from RSS");
    }
    return double(std::clock() - start) / CLOCKS_PER_SEC;
}

void budget_reserve(size_t limit_mib, size_t rss_mib, bool should_reclaim) {
    const size_t mib = size_t(1) << 20;
    size_t rss = rss_mib * mib;
    int discarded = 0;
    edge0::MemoryBudget budget({[&](size_t & value) { value = rss; return true; },
                               [&](void *, size_t) { ++discarded; rss = 0; return 0; }});
    check(budget.start(limit_mib * mib, false), "budget did not start");
    budget.mapped(reinterpret_cast<void *>(edge0::page_size()), edge0::page_size());
    check(budget.check(), "budget reclamation failed");
    check((discarded > 0) == should_reclaim, "budget has insufficient transient reserve");
}
} // namespace

int main() {
    try {
        budget_reserve(4096, 3788, true);
        budget_reserve(12288, 11300, false);
        budget_reserve(12288, 11800, true);
        Mapping memory;
        const double reference = measure(reference_rss, memory.size);
        const double actual = measure(edge0::read_rss, memory.size);
        std::printf("RSS polling: product=%.6f reference=%.6f CPU seconds\n", actual, reference);
        // CPU time excludes host scheduling; a broad ratio allows library/stdio
        // noise while catching repeated walks of a large resident mapping.
        check(actual <= std::max(0.001, reference) * 32, "RSS polling scans model page tables");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
