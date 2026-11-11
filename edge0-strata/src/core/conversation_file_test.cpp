// CPU-only tests for the on-disk session format (include/strata/core/conversation_file.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"
#include "strata/core/session_save_reclaim.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <new>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

SavedConversation sample() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(1000, 1);
    s.checkpoints.push_back(checkpoint(10, 2));
    s.checkpoints.push_back(checkpoint(500, 3));
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = layer == 2 ? 0 : 9; kv.idx_dim = 128;
        // > 16 MiB so the buffer spans several segments
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(layer));
        kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(layer == 1 ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(layer == 2 ? 0 : 9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// format v1 golden: the file the fixed sample() gives (size and session_hash64 of all bytes, seed 0)
constexpr size_t kGoldenSize = 17878535;
constexpr uint64_t kGoldenHash = 0x70f812b350360d09ull;

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<char>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize) d.size());
}

// no temporary file (any leftover of a write) in the directory
bool no_temp(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0 && n != "keep.bin.tmp") return false;
    }
    return true;
}

bool rejects(const fs::path& p, const SessionFileIdentity& id, const char* expect = nullptr,
             const SessionReadLimits& limits = {}) {
    SavedConversation out;
    size_t bytes = 0;
    std::string error;
    const bool ok = session_file_read(p.string(), id, out, bytes, error, limits);
    if (!ok && expect && error.find(expect) == std::string::npos) {
        std::fprintf(stderr, "  unexpected error text: %s (wanted %s)\n", error.c_str(), expect);
        return false;
    }
    return !ok && !error.empty();
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-session-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    const SavedConversation original = sample();
    const fs::path good = dir / "good.bin";

    // round trip
    size_t written = 0;
    std::string error;
    check(session_file_write(good.string(), original, id, written, error), "write succeeds");
    check(error.empty(), "write leaves no error");

    // golden: format v1 is frozen. The fixed sample must give these exact bytes (little-endian header fields at
    // fixed offsets, and a fixed hash of the whole file); a format change must bump the version and this test.
    {
        const std::vector<char> g = slurp(good);
        auto u32 = [&](size_t o) { uint32_t v = 0; for (int i = 3; i >= 0; --i) v = v << 8 | uint8_t(g[o + i]); return v; };
        auto u64 = [&](size_t o) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = v << 8 | uint8_t(g[o + i]); return v; };
        check(g.size() >= 80 && std::memcmp(g.data(), "STRSESS\x01", 8) == 0, "golden: magic");
        check(u32(8) == 1 && u32(12) == 64, "golden: version 1, header 64 bytes (little-endian)");
        check(u64(16) == id.model && u64(24) == id.config, "golden: fingerprints at offsets 16 and 24");
        check(u64(32) == g.size() - 64 - 16, "golden: payload length at offset 32");
        check(u64(40) == 0 && u64(48) == 0, "golden: reserved fields are zero");
        check(std::memcmp(g.data() + g.size() - 8, "STRSEND\x01", 8) == 0, "golden: end marker");
        const uint64_t h = session_hash64(g.data(), g.size(), 0);
        const char* golden = std::getenv("STRATA_SESSION_GOLDEN_PRINT");
        if (golden) std::printf("golden size %zu hash %016llx\n", g.size(), (unsigned long long)h);
        check(g.size() == kGoldenSize && h == kGoldenHash, "golden: the fixed sample's file bytes are unchanged");
    }
    check(written == fs::file_size(good), "write reports the file size");
    check(no_temp(dir), "no temporary left behind");
    {
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(good.string(), id, back, read, error), "read succeeds");
        check(read == written, "read reports the file size");
        check(same(original, back), "round trip is identical");
    }
    // writing twice produces the same bytes (deterministic format)
    {
        const fs::path again = dir / "again.bin";
        check(session_file_write(again.string(), original, id, written, error), "second write");
        check(slurp(again) == slurp(good), "format is deterministic");
    }
    // saving over an existing file replaces it (MoveFileExW on Windows, rename elsewhere)
    {
        const fs::path over = dir / "over.bin";
        spit(over, std::vector<char>(100, 'o'));
        check(session_file_write(over.string(), original, id, written, error), "write over an existing file");
        check(slurp(over) == slurp(good), "existing file replaced by the session");
        check(no_temp(dir), "no temporary left behind after replace");
    }
    // buffered I/O (the path taken where direct I/O is refused or unavailable) writes and reads the same bytes
    {
        const char* inherited = std::getenv("STRATA_SESSION_BUFFERED");
        const std::string kept = inherited ? inherited : "";
        set_env("STRATA_SESSION_BUFFERED", "1");
        const fs::path p = dir / "buffered.bin";
        check(session_file_write(p.string(), original, id, written, error), "buffered write");
        check(slurp(p) == slurp(good), "buffered write equals the default write");
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(p.string(), id, back, read, error) && same(original, back), "buffered read");
        set_env("STRATA_SESSION_BUFFERED", inherited ? kept.c_str() : nullptr);   // the caller's choice again
    }

    const std::vector<char> image = slurp(good);
    // truncation at every structural boundary and in the middle
    for (size_t cut : {size_t(0), size_t(7), size_t(63), size_t(64), size_t(65), size_t(200), image.size() / 2,
                       image.size() - 17, image.size() - 16, image.size() - 8, image.size() - 1}) {
        const fs::path p = dir / "cut.bin";
        spit(p, std::vector<char>(image.begin(), image.begin() + (std::ptrdiff_t) cut));
        check(rejects(p, id), "truncated file rejected");
    }
    // trailing garbage
    {
        auto d = image; d.push_back('x');
        const fs::path p = dir / "tail.bin"; spit(p, d);
        check(rejects(p, id, "size"), "file with trailing bytes rejected");
    }
    // flipped payload byte (inside the big K buffer), flipped header byte, flipped trailer
    for (size_t at : {size_t(64 + 300), image.size() / 2, size_t(20), image.size() - 12, image.size() - 3}) {
        auto d = image; d[at] ^= 0x40;
        const fs::path p = dir / "flip.bin"; spit(p, d);
        check(rejects(p, id), "corrupted byte rejected");
    }
    // other model or other configuration
    check(rejects(good, {id.model ^ 1, id.config}, "model"), "different model fingerprint rejected");
    check(rejects(good, {id.model, id.config ^ 1}, "config"), "different configuration rejected");
    // wrong version: header field patched and header hash recomputed so only the version differs
    {
        auto d = image;
        uint32_t v = 2; std::memcpy(d.data() + 8, &v, 4);
        const uint64_t h = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &h, 8);
        const fs::path p = dir / "ver.bin"; spit(p, d);
        check(rejects(p, id, "version"), "unknown version rejected");
    }
    // a corrupted element count with a recomputed payload hash must fail on bounds, not allocate
    {
        auto d = image;
        // payload starts at 64: geometry 18*8, layer_lo, layer_hi, cvec = 21*8 bytes, then live.ids count
        const size_t at = 64 + 21 * 8;
        uint64_t huge = 0x0fffffffffffffffull; std::memcpy(d.data() + at, &huge, 8);
        uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
        const uint64_t h = session_hash64(d.data() + 64, payload, 0);
        std::memcpy(d.data() + 64 + payload, &h, 8);
        const fs::path p = dir / "count.bin"; spit(p, d);
        check(rejects(p, id, "count"), "oversized count rejected before allocation");
    }
    // missing file
    check(rejects(dir / "absent.bin", id), "missing file rejected");
    // a failed read leaves the destination untouched
    {
        SavedConversation keep = sample();
        size_t b = 0;
        check(!session_file_read((dir / "absent.bin").string(), id, keep, b, error), "missing read fails");
        check(same(keep, original), "failed read does not modify the destination");
    }
    // layer-split images are not written; no file appears
    {
        SavedConversation split = sample();
        split.live.stage_parts.push_back(checkpoint(1, 9));
        const fs::path p = dir / "split.bin";
        check(!session_file_write(p.string(), split, id, written, error), "layer-split image refused");
        check(!fs::exists(p) && no_temp(dir), "refused write leaves no file");
    }
    // a write into a missing directory fails cleanly
    check(!session_file_write((dir / "nope" / "x.bin").string(), original, id, written, error),
          "write into a missing directory fails");
    // the hash is position-sensitive and seed-sensitive
    {
        const char a[] = "abcdefghijklmnopqrstuvwxyz0123456789", b[] = "bacdefghijklmnopqrstuvwxyz0123456789";
        check(session_hash64(a, sizeof a, 0) != session_hash64(b, sizeof b, 0), "hash sees swapped bytes");
        check(session_hash64(a, sizeof a, 0) != session_hash64(a, sizeof a, 1), "hash sees the seed");
        SessionHasher h; h.update(a, 5); h.update(a + 5, sizeof a - 5);
        check(h.digest() == session_hash64(a, sizeof a, 0), "streaming hash equals one-shot hash");
    }
    // model fingerprint: changes with content at the head of a file and with the file list
    {
        const fs::path m1 = dir / "m1.bin", m2 = dir / "m2.bin";
        spit(m1, std::vector<char>(3u << 20, 'a'));
        spit(m2, std::vector<char>(100, 'b'));
        uint64_t f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0, f7 = 0, f8 = 0;
        const std::vector<SessionModelFile> both = {{"a", m1.string()}, {"b", m2.string()}};
        check(session_model_fingerprint(both, f1, error), "fingerprint ok");
        const std::vector<SessionModelFile> subset = {{"a", m1.string()}};
        const std::vector<SessionModelFile> roles = {{"b", m1.string()}, {"a", m2.string()}};
        check(session_model_fingerprint(subset, f2, error), "fingerprint subset ok");
        check(session_model_fingerprint(roles, f5, error), "fingerprint roles ok");
        check(f1 != f5, "fingerprint follows the role each file plays");
        // a moved folder: same roles, same bytes, other paths
        const fs::path moved = dir / "moved";
        fs::create_directories(moved);
        fs::copy_file(m1, moved / "m1.bin"); fs::copy_file(m2, moved / "m2.bin");
        const std::vector<SessionModelFile> moved_list = {{"a", (moved / "m1.bin").string()}, {"b", (moved / "m2.bin").string()}};
        check(session_model_fingerprint(moved_list, f6, error) && f6 == f1, "fingerprint does not follow the folder");
        // an optional input that is absent is recorded, not an error; a present one changes the fingerprint
        const std::vector<SessionModelFile> absent = {{"a", m1.string()}, {"b", m2.string()}, {"c", (dir / "nothere").string(), true}};
        const std::vector<SessionModelFile> present = {{"a", m1.string()}, {"b", m2.string()}, {"c", m2.string(), true}};
        check(session_model_fingerprint(absent, f7, error), "absent optional input ok");
        check(session_model_fingerprint(present, f8, error) &&
              f7 != f8 && f7 != f1, "optional input presence enters the fingerprint");
        auto d = slurp(m1); d[10] = 'z'; spit(m1, d);
        check(session_model_fingerprint(both, f3, error), "fingerprint after edit ok");
        check(f1 != f2 && f1 != f3, "fingerprint follows files and content");
        check(!session_model_fingerprint({{"x", (dir / "absent").string()}}, f4, error), "fingerprint of a missing file fails");
    }

    // only the deepest checkpoint is worth a disk write: the next turn resumes from it
    {
        std::vector<ConversationCheckpoint> chain;
        check(session_checkpoints_to_save(chain).empty(), "no checkpoints, none saved");
        chain.push_back(checkpoint(10, 2));
        chain.push_back(checkpoint(700, 3));
        chain.push_back(checkpoint(500, 4));
        const auto kept = session_checkpoints_to_save(chain);
        check(kept.size() == 1 && same_checkpoint(kept[0], chain[1]), "deepest checkpoint kept, alone");
    }
    // K/V streamed from its owner (no host copy): the same bytes as writing the captured image
    {
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> sources;
        for (const auto& k : original.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                return parts[part]->read(dst, offset, n);
            };
            sources.push_back(std::move(s));
        }
        const fs::path p = dir / "streamed.bin";
        check(session_file_write(p.string(), meta, sources, id, written, error), "streamed write succeeds");
        check(slurp(p) == image, "streamed write equals the captured-image write");
        // a failing source leaves no file
        auto broken = sources;
        broken[1].read = [](size_t, size_t, void*, size_t) { return false; };
        const fs::path q = dir / "broken.bin";
        check(!session_file_write(q.string(), meta, broken, id, written, error), "failing source fails the write");
        check(!fs::exists(q) && no_temp(dir), "failed streamed write leaves no file");
        // K/V in the image and as sources at once is ambiguous: refused
        check(!session_file_write((dir / "both.bin").string(), original, sources, id, written, error),
              "image K/V plus sources refused");
    }

    // the configuration fingerprint: every field counts, doubles by their exact bits
    {
        SessionConfig c;
        c.engine_version = "0.1.38"; c.backend = "cuda"; c.kv = "int8"; c.max_context = 65536; c.kv_resident = 0;
        c.mtp_window = 4; c.rope.type = 0; c.rope.freq_base = 1e7; c.rope.factor = 1.0; c.rope.freq_scale = 1.0;
        c.rope.orig_ctx = 262144; c.rope.attn_factor = 1.0; c.rope.beta_fast = 32; c.rope.beta_slow = 1;
        c.switches = {{"STRATA_FAST_GDN", 0}};
        const uint64_t base = session_config_fingerprint(c);
        check(session_config_fingerprint(c) == base, "config fingerprint is deterministic");
        auto differs = [&](auto edit) { SessionConfig d = c; edit(d); return session_config_fingerprint(d) != base; };
        check(differs([](SessionConfig& d) { d.engine_version = "0.1.39"; }), "engine version enters the config");
        check(differs([](SessionConfig& d) { d.backend = "hip"; }), "backend enters the config");
        check(differs([](SessionConfig& d) { d.kv = "fp16"; }), "kv enters the config");
        check(differs([](SessionConfig& d) { d.max_context = 65537; }), "max context enters the config");
        check(differs([](SessionConfig& d) { d.kv_resident = 1; }), "kv residency enters the config");
        check(differs([](SessionConfig& d) { d.mtp_window = 0; }), "mtp window enters the config");
        check(differs([](SessionConfig& d) { d.kv_rot = true; }), "kv rotation enters the config");
        check(differs([](SessionConfig& d) { d.rope.factor = std::nextafter(1.0, 2.0); }),
              "a one-ulp rope change enters the config");
        check(differs([](SessionConfig& d) { d.rope.type = 2; }), "rope type enters the config");
        check(differs([](SessionConfig& d) { d.rope.beta_slow = 2; }), "rope beta enters the config");
        check(differs([](SessionConfig& d) { d.cvec = 7; }), "control vectors enter the config");
        check(differs([](SessionConfig& d) { d.switches[0].second = 1; }), "a switch value enters the config");
        check(differs([](SessionConfig& d) { d.switches.push_back({"X", 0}); }), "the switch list enters the config");
        // field boundaries: ("ab","c") and ("a","bc") hash differently
        SessionIdentityBuilder x(1), y(1);
        x.str("ab", "c"); y.str("a", "bc");
        check(x.digest() != y.digest(), "identity fields are length-delimited");
        SessionIdentityBuilder i(1), u(1);
        i.i64("n", 5); u.u64("n", 5);
        check(i.digest() != u.digest(), "identity fields are typed");
    }
    // bounds a read checks before it allocates, and the caller's admission
    {
        SessionReadLimits l;
        l.max_tokens = 999;   // the live state holds 1000
        check(rejects(good, id, "limit", l), "token count over the limit refused");
        l = {}; l.max_checkpoints = 1;
        check(rejects(good, id, "limit", l), "checkpoint count over the limit refused");
        l = {}; l.max_kv_layers = 2;
        check(rejects(good, id, "limit", l), "K/V layer count over the limit refused");
        l = {}; l.max_file_bytes = image.size() - 1;
        check(rejects(good, id, "limit", l), "file over the size limit refused");
        l = {};
        uint64_t asked = 0;
        l.admit = [&](uint64_t n, std::string& why) { asked = n; why = "no room for it"; return false; };
        check(rejects(good, id, "no room", l) && asked == session_read_peak_bytes(image.size()) &&
              asked > image.size(), "admission sees the parse's peak (more than the file) and can refuse");
        l = {};
        bool admitted = false;
        l.admit = [&](uint64_t, std::string&) { admitted = true; return true; };
        check(rejects(good, {id.model ^ 1, id.config}, "model", l) && !admitted,
              "admission is not asked for a foreign file");
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(good.string(), id, back, n, error, l) && admitted && same(back, original),
              "admitted file reads");
    }
    // free-space reserve: an impossible reserve refuses the write and leaves the existing file untouched
    {
        const fs::path p = dir / "reserve.bin";
        spit(p, std::vector<char>(10, 'r'));
        SessionWriteOptions opt;
        opt.min_free_bytes = UINT64_MAX / 2;
        check(!session_file_write(p.string(), original, id, written, error, opt) &&
              error.find("disk space") != std::string::npos, "write over the free-space reserve refused");
        check(slurp(p) == std::vector<char>(10, 'r') && no_temp(dir), "refused write keeps the old file, no temporary");
        opt.min_free_bytes = 1;
        uint64_t calls = 0;
        opt.progress = [&](uint64_t, uint64_t) { ++calls; };
        check(session_file_write(p.string(), original, id, written, error, opt) && slurp(p) == image,
              "write with a small reserve succeeds");
    }
    // a failed write over an existing session keeps it, and touches no other file (a planted "<name>.tmp", the old
    // fixed temporary pattern included)
    {
        const fs::path p = dir / "keep.bin", planted = dir / "keep.bin.tmp";
        check(session_file_write(p.string(), original, id, written, error), "keep: first write");
        spit(planted, std::vector<char>(5, 'p'));
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> broken(1);
        broken[0].sizes = {100, 0, 0, 0, 0};
        broken[0].read = [](size_t, size_t, void*, size_t) { return false; };
        check(!session_file_write(p.string(), meta, broken, id, written, error), "keep: failing write fails");
        check(slurp(p) == image, "keep: the previous session survives a failed write");
        check(slurp(planted) == std::vector<char>(5, 'p') && no_temp(dir), "keep: no other file touched");
    }
    // R1: the input list carries every expert file the loader resolved, per (layer, role) - a GGUF that only
    // native_experts.txt names changes the fingerprint when it changes; one file in two roles is two entries
    {
        const fs::path g1 = dir / "ext-a.gguf", g2 = dir / "ext-b.gguf";
        spit(g1, std::vector<char>(4096, 'g'));
        spit(g2, std::vector<char>(4096, 'h'));
        SessionInputs in;
        in.native_shards = {g1.string()};
        in.experts = {{"expert blk.0.ffn_gate", g1.string()}, {"expert blk.0.ffn_up", g2.string()},
                      {"expert blk.0.ffn_down", g2.string()}};
        const auto list = session_model_inputs(in);
        check(list.size() == 4, "inputs: shard plus three expert roles, none merged");
        uint64_t a = 0, b = 0, c = 0;
        check(session_model_fingerprint(list, a, error), "inputs: fingerprint");
        auto d = slurp(g2); d[0] = 'X'; spit(g2, d);   // the expert-only file changes; the CLI shard does not
        check(session_model_fingerprint(list, b, error) && a != b, "inputs: an expert-only GGUF enters the fingerprint");
        SessionInputs swapped = in;
        std::swap(swapped.experts[0].second, swapped.experts[1].second);
        check(session_model_fingerprint(session_model_inputs(swapped), c, error) && c != b,
              "inputs: the role each expert file plays enters the fingerprint");
    }
    // R2: runtime limits bound every array before it is allocated, and the file size follows from them
    {
        auto exact = [&] {
            SessionReadLimits l;
            l.max_tokens = 1000; l.max_checkpoints = 2; l.max_kv_layers = 3;
            l.geometry = original.geometry;
            l.layer_range = std::make_pair(original.layer_lo, original.layer_hi);
            l.max_state_bytes = {4099, 77, 301, 64, 8};
            for (const auto& k : original.kv)
                l.max_kv_bytes.push_back({k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()});
            return l;
        };
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(good.string(), id, back, n, error, exact()) && same(back, original),
              "limits: a file within the exact runtime limits reads");
        check(session_read_max_file_bytes(exact()) >= image.size() &&
              session_read_max_file_bytes(exact()) < UINT64_MAX, "limits: the runtime limits bound the file size");
        check(session_read_max_file_bytes(SessionReadLimits{}) == UINT64_MAX, "limits: open limits leave it open");
        auto l = exact(); l.max_state_bytes[0] = 4098;
        check(rejects(good, id, "limit", l), "limits: a GDN state over the runtime size refused");
        l = exact(); l.max_state_bytes[4] = 7;
        check(rejects(good, id, "limit", l), "limits: block_pos over the runtime size refused");
        l = exact(); l.max_kv_bytes[0][0] = original.kv[0].k.size() - 1;
        check(rejects(good, id, "limit", l), "limits: a K buffer over the runtime size refused");
        l = exact(); l.max_kv_bytes[2][1] = original.kv[2].v.size() - 1;
        check(rejects(good, id, "limit", l), "limits: the draft's V over the runtime size refused");
        l = exact(); (*l.geometry)[3] ^= 1;
        check(rejects(good, id, "geometry", l), "limits: another geometry refused before the state is read");
        l = exact(); l.layer_range = std::make_pair(int64_t(0), int64_t(24));
        check(rejects(good, id, "layer", l), "limits: another layer range refused");
        // a huge count in a state array with a recomputed hash: refused by the bound, nothing allocated
        auto d = image;
        // live: ids count (8) + 1000*4, imgs count (8) + 1*16, then the gdn byte count
        const size_t at = 64 + 21 * 8 + 8 + 1000 * 4 + 8 + 16;
        uint64_t was = 0; std::memcpy(&was, d.data() + at, 8);
        check(was == 4099, "limits: test offset finds the GDN count");
        uint64_t huge = 3ull << 30; std::memcpy(d.data() + at, &huge, 8);
        uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
        const uint64_t h = session_hash64(d.data() + 64, payload, 0);
        std::memcpy(d.data() + 64 + payload, &h, 8);
        const fs::path p = dir / "bigstate.bin"; spit(p, d);
        check(rejects(p, id, nullptr, exact()), "limits: an oversized state count refused before allocation");
        // status kinds of a read
        SessionStatus st;
        check(!session_file_read(p.string(), id, back, n, error, exact(), &st) && st.error == SessionError::invalid,
              "status: a bad file is 'invalid'");
        l = exact();
        l.admit = [](uint64_t, std::string& why) { why = "no RAM"; return false; };
        check(!session_file_read(good.string(), id, back, n, error, l, &st) && st.error == SessionError::memory,
              "status: the RAM preflight is 'memory'");
    }
    // R3/R5: injected failures at each write step - kind, publication and the old file
    {
        const fs::path p = dir / "fault.bin";
        const std::vector<char> old(10, 'q');
        struct Case { const char* step; int err; SessionError kind; bool published; };
        for (const Case c : {Case{"write", ENOSPC, SessionError::storage, false},
#ifdef EDQUOT
                             Case{"write", EDQUOT, SessionError::storage, false},
#endif
                             Case{"write", EIO, SessionError::io, false},
                             Case{"file_flush", EIO, SessionError::io, false},
                             Case{"rename", EACCES, SessionError::io, false},
                             Case{"dir_flush", EIO, SessionError::io, true}}) {
            spit(p, old);
            SessionWriteOptions opt;
            const std::string want = c.step;
            const int code = c.err;
            opt.fault = [want, code](const char* step) { return want == step ? code : 0; };
            SessionStatus st;
            const bool ok = session_file_write(p.string(), original, id, written, error, opt, &st);
            check(!ok && st.error == c.kind, "fault: the failure has its kind");
            check(st.published == c.published, "fault: publication reported exactly");
            if (c.published) {
                check(slurp(p) == image && error.find("replaced") != std::string::npos,
                      "fault: after a published failure the new file is there and the error says so");
            } else {
                check(slurp(p) == old, "fault: a failure before the rename keeps the old file");
            }
            check(no_temp(dir), "fault: no temporary left behind");
        }
#ifndef _WIN32
        // a filesystem that cannot flush a folder: saved, and said
        spit(p, old);
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::string(step) == "dir_flush" ? EINVAL : 0; };
        SessionStatus st;
        check(session_file_write(p.string(), original, id, written, error, opt, &st) && st.dir_flush_unsupported &&
              slurp(p) == image, "fault: EINVAL on the folder flush is not a failure, and is reported");
#endif
        // the free-space preflight is 'storage'
        SessionWriteOptions big;
        big.min_free_bytes = UINT64_MAX / 2;
        SessionStatus st2;
        check(!session_file_write(p.string(), original, id, written, error, big, &st2) &&
              st2.error == SessionError::storage, "status: the free-space preflight is 'storage'");
        check(std::string(session_error_name(SessionError::storage)) == "storage" &&
              std::string(session_error_name(SessionError::memory)) == "memory" &&
              std::string(session_error_name(SessionError::invalid)) == "invalid" &&
              std::string(session_error_name(SessionError::io)) == "io", "status: protocol names");
    }
    // B: progress after every block, the last partial one included (a file below 16 MiB reports too), and the
    // blocking steps announced before they start
    {
        const fs::path p = dir / "progress.bin";
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        std::vector<std::string> phases;
        SessionWriteOptions opt;
        opt.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        opt.phase = [&](const char* ph, uint64_t b) { phases.push_back(std::string(ph) + ":" + std::to_string(b)); };
        // a file below one block (the sample without its K/V layers) reports too: at the start and its one block
        SavedConversation small = original;
        small.kv.clear();
        check(session_file_write(p.string(), small, id, written, error, opt), "progress: small write");
        check(written > 0 && written < (16u << 20), "progress: the small file is below one block");
        check(seen.size() == 2 && seen[0] == std::make_pair(uint64_t(0), uint64_t(written)) &&
              seen[1] == std::make_pair(uint64_t(written), uint64_t(written)), "progress: start and the partial block");
        check(phases.size() == 2 && phases[0] == "flush:" + std::to_string(written) &&
              phases[1] == "publish:" + std::to_string(written), "progress: flush and publish announced, in order");
        opt.durable = false;
        phases.clear();
        check(session_file_write(p.string(), small, id, written, error, opt) && phases.size() == 1 &&
              phases[0].rfind("publish:", 0) == 0, "progress: no flush step announced without a flush");
        // a read reports every block, the first (the header's) included
        seen.clear();
        SessionReadLimits l;
        l.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(p.string(), id, back, n, error, l) && seen.size() == 1 &&
              seen[0] == std::make_pair(uint64_t(n), uint64_t(n)), "progress: a small read reports its one block");
        // the fixed sample: one full block and a partial one, both reported on write and on read
        seen.clear();
        opt.durable = true;
        check(session_file_write(p.string(), original, id, written, error, opt) && written > (16u << 20) &&
              written < (32u << 20), "progress: sample write");
        check(seen.size() == 3 && seen[1].first == (16u << 20) && seen[2].first == written,
              "progress: the sample's full block and its partial last block");
        seen.clear();
        check(session_file_read(p.string(), id, back, n, error, l) && seen.size() == 2 &&
              seen[0].first == (16u << 20) && seen[1].first == n, "progress: the sample's read reports both blocks");
        // a multi-block streamed write: 16 MiB blocks then the partial one, never more than a block apart
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> big(1);
        big[0].sizes = {(40u << 20) + 123, 0, 0, 0, 0};
        big[0].read = [](size_t, size_t, void* d, size_t c) { std::memset(d, 0x5a, c); return true; };
        seen.clear();
        opt.durable = true;
        check(session_file_write(p.string(), meta, big, id, written, error, opt), "progress: 40 MiB write");
        bool steps = seen.size() == 4 && seen.front().first == 0 && seen.back().first == written;
        for (size_t i = 1; i < seen.size(); ++i)
            steps = steps && seen[i].first > seen[i - 1].first && seen[i].first - seen[i - 1].first <= (16u << 20);
        check(steps, "progress: every block reported, at most 16 MiB apart, the last one at the end");
        // the allowance of a blocking step: bounded, growing with its size
        check(session_phase_limit_s(0) == 60 && session_phase_limit_s(1198691396) == 60 + (1198691396 >> 22) &&
              session_phase_limit_s(UINT64_MAX) == 3600, "phase limit: 60 s + 1 s per 4 MiB, at most an hour");
    }
    // A: the SAVE's RAM preflight comes BEFORE the checkpoint is copied
    {
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(checkpoint(10, 2));
        chain.push_back(checkpoint(700, 3));
        chain.back().gdn.assign(8u << 20, 0x11);   // the deepest: an 8 MiB running state
        const auto before = chain;
        SessionSaveLive live;
        live.state_bytes = 5000; live.tokens = 1000; live.images = 1; live.kv_layers = 13;
        check(session_deepest_checkpoint(chain) == &chain[1], "save: deepest chosen by reference");
        {   // pin=N: the pinned checkpoint is the one kept, even when a deeper periodic one exists
            auto pinned_chain = chain;
            pinned_chain[0].pinned = true;
            check(session_deepest_checkpoint(pinned_chain) == &pinned_chain[0], "save: a pinned shared prefix beats a deeper checkpoint");
            check(session_deepest_checkpoint(chain) == &chain[1], "save: no pin, the deepest as before");
        }
        const uint64_t need = session_save_peak_bytes(&chain[1], live);
        check(need >= 2 * (uint64_t) chain[1].bytes() + 5000 + 4000 + (16u << 20) &&
              need <= 2 * (uint64_t) chain[1].bytes() + 5000 + 4000 + (18u << 20),
              "save: the peak counts two checkpoint copies, the live state, tokens and the buffer");
        SessionSaveLive huge = live;
        huge.state_bytes = UINT64_MAX;
        check(session_save_peak_bytes(&chain[1], huge) == UINT64_MAX, "save: the peak saturates");
        std::vector<ConversationCheckpoint> out(1);
        std::string why;
        uint64_t asked = 0;
        bool empty_when_asked = false;
        const auto refuse = [&](uint64_t n, std::string& w) { asked = n; empty_when_asked = out.empty(); w = "low RAM"; return false; };
        check(!session_save_checkpoints(chain, live, refuse, out, why) && why == "low RAM" && asked == need,
              "save: the preflight is asked for the peak and can refuse");
        check(empty_when_asked && out.empty(), "save: refused before any copy (nothing copied)");
        bool same_chain = chain.size() == before.size();
        for (size_t i = 0; same_chain && i < chain.size(); ++i) same_chain = same_checkpoint(chain[i], before[i]);
        check(same_chain, "save: a refusal leaves the live checkpoints untouched");
        const auto agree = [&](uint64_t, std::string&) { return true; };
        check(session_save_checkpoints(chain, live, agree, out, why) && out.size() == 1 &&
              same_checkpoint(out[0], chain[1]), "save: admitted, the deepest checkpoint is copied");
        check(session_save_checkpoints({}, live, agree, out, why) && out.empty(), "save: no checkpoints, none copied");
#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__) && !defined(STRATA_NO_RLIMIT_TEST)
        // under a real address-space limit that cannot hold one more copy: the refusal comes first, no bad_alloc;
        // and the same limit with an admitting preflight does fail at the copy (the test can see the difference)
        std::vector<ConversationCheckpoint> fat(1);
        fat[0].ids = {1, 2, 3};
        fat[0].gdn.assign(256u << 20, 0x22);
        const fs::path old_file = dir / "old-session.bin";
        spit(old_file, std::vector<char>(9, 'o'));
        std::fflush(nullptr);
        const pid_t pid = ::fork();
        if (pid == 0) {
            long pages = 0, rss = 0;
            if (FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) pages = 0; std::fclose(f); }
            struct rlimit rl {};
            rl.rlim_cur = rl.rlim_max = (rlim_t) pages * (rlim_t) ::sysconf(_SC_PAGESIZE) + (64u << 20);
            if (pages <= 0 || ::setrlimit(RLIMIT_AS, &rl) != 0) ::_exit(10);
            std::vector<ConversationCheckpoint> o2;
            std::string w;
            const auto low = [](uint64_t n, std::string& m) { m = "need " + std::to_string(n); return false; };
            bool refused = false;
            try { refused = !session_save_checkpoints(fat, live, low, o2, w) && o2.empty(); } catch (...) { ::_exit(11); }
            if (!refused) ::_exit(12);
            try { session_save_checkpoints(fat, live, agree, o2, w); } catch (const std::bad_alloc&) { ::_exit(0); }
            ::_exit(13);   // the copy fitted: the limit did not bite, the test proves nothing
        }
        int status = 0;
        check(pid > 0 && ::waitpid(pid, &status, 0) == pid, "save: low-memory child ran");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "save: under a low address-space limit the preflight refuses before the copy (an admitted copy fails)");
        check(fat[0].gdn.size() == (256u << 20) && fat[0].gdn[12345] == 0x22 &&
              slurp(old_file) == std::vector<char>(9, 'o'), "save: live state and the old file intact");
#endif
    }
    // Opt-in pressure reclamation does not change the selected file bytes, even with pin=N.
    for (bool pin : {false, true}) {
        auto source = sample();
        source.checkpoints[0].pinned = pin;
        const auto live_before = source.live;
        auto expected = sample();
        expected.checkpoints = session_checkpoints_to_save(source.checkpoints);
        const fs::path before_path = dir / "reclaim-before.bin", after_path = dir / "reclaim-after.bin";
        check(session_file_write(before_path.string(), expected, id, written, error), "reclaim: baseline file written");
        ConversationCache parked(1u << 20, 4);
        SessionSaveLive sl; sl.state_bytes = source.live.bytes(); sl.tokens = source.live.ids.size();
        const auto freed = session_save_reclaim(source.checkpoints, parked, sl, 0,
                                                [] { return std::optional<uint64_t>(0); });
        check(freed.checkpoints == 1, "reclaim: a non-selected checkpoint released");
        check(same_checkpoint(source.live, live_before), "reclaim: live state remains intact");
        check(session_file_write(after_path.string(), source, id, written, error), "reclaim: file written after pressure");
        check(slurp(before_path) == slurp(after_path), "reclaim: file byte-identical to original selection");
        const auto old = slurp(after_path);
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::strcmp(step, "write") == 0 ? ENOSPC : 0; };
        SessionStatus status;
        check(!session_file_write(after_path.string(), source, id, written, error, opt, &status) &&
              status.error == SessionError::storage && !status.published, "reclaim: later write failure reported");
        check(slurp(after_path) == old && same_checkpoint(source.live, live_before) && no_temp(dir),
              "reclaim: later failure preserves old file and live state, no temporary left");
    }
    // R6: the folder the free-space query is asked about
    {
        check(session_free_space_dir("\\\\server\\share\\sessions\\chat.bin", true) == "\\\\server\\share\\sessions\\",
              "free space: UNC subfolder keeps its trailing backslash");
        check(session_free_space_dir("\\\\server\\share\\chat.bin", true) == "\\\\server\\share\\",
              "free space: UNC share root keeps its trailing backslash");
        check(session_free_space_dir("//server/share/s/chat.bin", true) == "\\\\server\\share\\s\\",
              "free space: forward slashes become backslashes");
        check(session_free_space_dir("C:\\chat.bin", true) == "C:\\", "free space: drive root");
        check(session_free_space_dir("C:chat.bin", true) == "C:\\", "free space: drive-relative name");
        check(session_free_space_dir("chat.bin", true).empty(), "free space: bare name is the current folder");
        check(session_free_space_dir("/a/b/chat.bin", false) == "/a/b", "free space: POSIX folder");
        check(session_free_space_dir("/chat.bin", false) == "/", "free space: POSIX root");
        check(session_free_space_dir("chat.bin", false) == ".", "free space: POSIX bare name");
    }
#ifndef _WIN32
    // what restore will not open: a symbolic link, a hard-linked file, a FIFO, a directory
    {
        const fs::path link = dir / "link.bin";
        fs::create_symlink(good, link);
        check(rejects(link, id, "symbolic link"), "a symbolic link is not followed");
        const fs::path hard = dir / "hard.bin";
        fs::create_hard_link(dir / "again.bin", hard);
        check(rejects(hard, id, "hard link"), "a hard-linked file is refused");
        fs::remove(hard);
        check(!rejects(dir / "again.bin", id), "the same file with one name again reads");
        const fs::path fifo = dir / "fifo.bin";
        check(::mkfifo(fifo.c_str(), 0600) == 0, "mkfifo");
        check(rejects(fifo, id, "regular"), "a FIFO is refused without blocking");
        check(rejects(dir / "moved", id, "regular"), "a directory is refused");
    }
    // saving over a symbolic link replaces the link, never writes through it; the new file is the owner's only
    {
        const fs::path target = dir / "target.bin", link = dir / "out-link.bin";
        spit(target, std::vector<char>(7, 't'));
        fs::create_symlink(target, link);
        check(session_file_write(link.string(), original, id, written, error), "write at a link name");
        check(!fs::is_symlink(link) && slurp(link) == image, "the link was replaced by the file");
        check(slurp(target) == std::vector<char>(7, 't'), "the link's target is untouched");
        struct stat st {};
        check(::stat(link.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600, "the session file is mode 0600");
    }
#endif

    fs::remove_all(dir);
    std::printf("conversation_file_test: %d checks passed\n", checks);
    return 0;
}
