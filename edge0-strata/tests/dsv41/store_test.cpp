// Strata. ExpertStore (src/dsv41/ds4_store.hpp) on real expert files, no device: the bytes a slot holds equal the file's, the
// unbuffered reads' rate, the policy's hits on a skewed random access stream.
//   ds4_store_test <gguf> <layer0 gate off> <layer0 up off> <layer0 down off> <expert bytes (gate/up/down)> [budget GB] [policy 0|1|2] [accesses]
// (offsets from the GGUF's tensor directory: tools/dsv41 header scan; the three tensors of a layer are [384][...] so expert e of
// a matrix lies at off + e * bytes)
#include "ds4_store.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

using namespace strata::ds4;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::printf("usage: ds4_store_test <gguf> <gate off> <up off> <down off> <matrix bytes> [budget GB] [policy] [accesses]\n");
        return 2;
    }
    const std::string path = argv[1];
    const uint64_t offs[3] = {std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10), std::strtoull(argv[4], nullptr, 10)};
    const uint64_t mb = std::strtoull(argv[5], nullptr, 10);
    const double gb = argc > 6 ? std::atof(argv[6]) : 1.0;
    const int policy = argc > 7 ? std::atoi(argv[7]) : 0;
    const int n_acc = argc > 8 ? std::atoi(argv[8]) : 2000;
    const os::fd_t fd = os::open_ro(path), dfd = os::open_direct(path), pfd = os::open_par(path);
    if (fd == os::kBadFd || pfd == os::kBadFd) {
        std::printf("cannot open %s\n", path.c_str());
        return 2;
    }
    std::printf("direct handle: %s\n", dfd != os::kBadFd ? "yes" : "NO (buffered)");
    ExpertStore::LayerGeo geo;
    geo.n_exp = 384;
    uint64_t at = 0;
    for (int r = 0; r < 3; ++r) {
        geo.file[r] = 0;
        geo.off[r] = offs[r];
        geo.stride[r] = mb;
        geo.bytes[r] = mb;
        geo.tr[r] = at;   // the engine's blob layout with the store's 8192-byte gaps
        at += ((mb + 255) & ~(uint64_t) 255) + ExpertStore::kSlack;
    }
    geo.blob = at;
    ExpertStore st;
    ExpertStore::Cfg cfg;
    cfg.budget_bytes = (size_t) (gb * 1e9);
    cfg.threads = std::getenv("STORE_TEST_THREADS") ? std::atoi(std::getenv("STORE_TEST_THREADS")) : 8;
    cfg.policy = (ExpertStore::Policy) policy;
    cfg.floor_bytes = 4ull << 30;
    cfg.ram_avail = [] { return os::ram_avail(); };
    if (!st.init(cfg, {geo}, {dfd}, {pfd})) {
        std::printf("init failed\n");
        return 2;
    }
    std::printf("slot %.2f MB, up to %zu slots\n", st.slot_bytes() / 1e6, st.slots_max());
    // a skewed stream: 70% of the accesses from the 64 hottest experts, the rest uniform; 6 a "token"
    std::mt19937_64 rng(7);
    auto pick = [&] { return rng() % 100 < 70 ? (int) (rng() % 64) : (int) (rng() % 384); };
    std::vector<uint8_t> ref(mb);
    int bad = 0;
    long checked = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int a = 0; a < n_acc; ++a) {
        int es[6];
        for (int i = 0; i < 6; ++i) {
            es[i] = pick();
            for (int j = 0; j < i; ++j)
                if (es[j] == es[i]) es[i] = (es[i] + 1 + j) % 384;
        }
        std::vector<char> ok;
        st.ensure(0, es, 6, ok);
        if (a % 17 == 0 && !std::getenv("STORE_TEST_NOCMP"))   // a sample compared with the file read plainly
            for (int i = 0; i < 6; ++i)
                if (ok[(size_t) i])
                    for (int r = 0; r < 3; ++r) {
                        const int s = st.find(0, es[i]);
                        os::pread_full(fd, ref.data(), mb, offs[r] + (uint64_t) es[i] * mb);
                        ++checked;
                        if (s < 0 || std::memcmp(st.mat(s, r), ref.data(), mb) != 0) ++bad;
                    }
        st.release(0, es, 6, ok);
    }
    // guesses queued behind demand, cancelled, and a shrink: the store must still give the file's bytes
    for (int round = 0; round < 3; ++round) {
        for (int e = 100; e < 140; ++e) st.prefetch(0, e);
        st.cancel_stale(0);
        int es[6] = {5 + round, 17, 200, 201, 202, 250 + round};
        std::vector<char> ok;
        st.ensure(0, es, 6, ok);
        for (int i = 0; i < 6; ++i)
            if (ok[(size_t) i]) {
                const int sl = st.find(0, es[i]);
                for (int r = 0; r < 3; ++r) {
                    os::pread_full(fd, ref.data(), mb, offs[r] + (uint64_t) es[i] * mb);
                    ++checked;
                    if (sl < 0 || std::memcmp(st.mat(sl, r), ref.data(), mb) != 0) ++bad;
                }
            }
        st.release(0, es, 6, ok);
        std::this_thread::sleep_for(std::chrono::milliseconds(800));   // (the guesses read by now)
        const size_t g = st.shrink_chunks(1);
        std::printf("round %d: shrank %zu chunk(s), %zu slots left, cancelled %lld\n", round, g, st.slots_allocated(), (long long) st.stats().cancelled.load());
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto& s = st.stats();
    std::printf("%d accesses of 6 experts in %.2f s: hits %lld misses %lld evictions %lld fails %lld; read %.2f GB (%.2f GB/s); waited %.2f s; "
                "compared %ld matrices, %d differ\n",
                n_acc, dt, (long long) s.hits.load(), (long long) s.misses.load(), (long long) s.evictions.load(), (long long) s.fails.load(),
                s.bytes_read.load() / 1e9, s.bytes_read.load() / 1e9 / dt, s.wait_ns.load() / 1e9, checked, bad);
    return bad == 0 ? 0 : 1;
}
