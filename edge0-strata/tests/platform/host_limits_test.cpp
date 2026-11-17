// Windows HIP auto-chunk cap (#1630) and the commit-limit verdict (#1607): no GPU, no Windows needed.
#include <cstdio>

#include "strata/platform/host_limits.hpp"

using namespace strata::platform::host_limits;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // a bare auto on Windows HIP is capped below the 8192 that crashed; every other build keeps it
    CHECK(auto_chunk_cap(true, true) == 6144);
    CHECK(auto_chunk_cap(true, true) < 8192);
    CHECK(auto_chunk_cap(false, true) == 0);
    CHECK(auto_chunk_cap(true, false) == 0);    // auto:N / STRATA_PREFILL_AUTO_MAX: the operator's pick
    CHECK(auto_chunk_cap(false, false) == 0);
    // a chunk the user names stays, with a warning from 8192 up (7500 was stable)
    CHECK(warn_user_chunk(true, false, 8192));
    CHECK(warn_user_chunk(true, false, 16384));
    CHECK(!warn_user_chunk(true, false, 7500));
    CHECK(!warn_user_chunk(true, true, 8192));  // the bare auto is capped instead
    CHECK(!warn_user_chunk(false, false, 8192));
    // commit: 188 GiB limit with 90 GiB left and a 10 GiB plan: quiet; 95 GiB limit with 6 GiB left: warns
    const uint64_t G = 1ull << 30;
    CHECK(!commit_verdict(90 * G, 188 * G, 10 * G, 4 * G).warn);
    CHECK(!commit_verdict(14 * G, 96 * G, 10 * G, 4 * G).warn);       // exactly enough
    const CommitVerdict v = commit_verdict(6 * G, 96 * G, 10 * G, 4 * G);
    CHECK(v.warn);
    CHECK(v.text.find("1607") != std::string::npos);
    CHECK(!commit_verdict(0, 0, 10 * G, 4 * G).warn);                 // unknown limit (not Windows)
    if (failed == 0) std::printf("host_limits_test: ok\n");
    return failed ? 1 : 0;
}
