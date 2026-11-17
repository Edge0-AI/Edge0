// The kept conversations' bookkeeping without a GPU (src/dsv41/ds4_slotmeta.hpp): the .meta file's round trip and
// every way it is refused, and which kept conversation a prompt continues, and from where.
//   ds4_slot_test            exit 0: all passed
#include "ds4_slotmeta.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace sm = strata::ds4::slotmeta;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

static std::vector<int32_t> seq(int a, int n) {
    std::vector<int32_t> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = a + i;
    return v;
}
static std::vector<int> as_int(const std::vector<int32_t>& v) { return std::vector<int>(v.begin(), v.end()); }

static void patch(const std::string& p, size_t off, uint8_t b) {
    std::FILE* f = std::fopen(p.c_str(), "r+b");
    std::fseek(f, (long) off, SEEK_SET);
    std::fwrite(&b, 1, 1, f);
    std::fclose(f);
}

static void meta_tests(const fs::path& dir) {
    const std::string id = "strata-ds4-session-2 engine=test", path = (dir / "slot-1.meta").string();
    const std::vector<int32_t> toks = seq(100, 50);
    const std::vector<int64_t> cks = {40, 33, 7};
    CHECK(sm::write(path, id, toks, 123456789ull, cks));
    CHECK(!fs::exists(path + ".tmp"));   // (renamed over, no half file left)
    std::vector<int32_t> t;
    std::vector<int64_t> c;
    uint64_t bytes = 0;
    CHECK(sm::read(path, id, 1000, t, bytes, c));
    CHECK(t == toks && c == cks && bytes == 123456789ull);
    // no checkpoints is valid (a slot of a state that had none)
    CHECK(sm::write(path, id, toks, 5, {}));
    CHECK(sm::read(path, id, 1000, t, bytes, c) && c.empty() && bytes == 5);
    // another model's / engine's identity: refused, never read
    CHECK(!sm::read(path, id + "x", 1000, t, bytes, c));
    CHECK(!sm::read(path, "other", 1000, t, bytes, c));
    // more tokens than the context: refused
    CHECK(!sm::read(path, id, 49, t, bytes, c));
    CHECK(sm::read(path, id, 50, t, bytes, c));
    // a missing file
    CHECK(!sm::read((dir / "none.meta").string(), id, 1000, t, bytes, c));
    // every cut length, a damaged magic at either end
    CHECK(sm::write(path, id, toks, 9, cks));
    const size_t full = (size_t) fs::file_size(path);
    for (size_t keep = 0; keep < full; ++keep) {
        fs::copy_file(path, dir / "cut.meta", fs::copy_options::overwrite_existing);
        fs::resize_file(dir / "cut.meta", keep);
        if (sm::read((dir / "cut.meta").string(), id, 1000, t, bytes, c)) {
            std::printf("FAIL: a .meta cut to %zu of %zu bytes was read\n", keep, full);
            ++fails;
            break;
        }
    }
    fs::copy_file(path, dir / "bad0.meta", fs::copy_options::overwrite_existing);
    patch((dir / "bad0.meta").string(), 0, 0x55);
    CHECK(!sm::read((dir / "bad0.meta").string(), id, 1000, t, bytes, c));
    fs::copy_file(path, dir / "bad1.meta", fs::copy_options::overwrite_existing);
    patch((dir / "bad1.meta").string(), full - 1, 0x55);
    CHECK(!sm::read((dir / "bad1.meta").string(), id, 1000, t, bytes, c));
    // a checkpoint outside the conversation, a zero one, more than four
    CHECK(sm::write(path, id, toks, 9, {51}));
    CHECK(!sm::read(path, id, 1000, t, bytes, c));
    CHECK(sm::write(path, id, toks, 9, {0}));
    CHECK(!sm::read(path, id, 1000, t, bytes, c));
    CHECK(sm::write(path, id, toks, 9, {4, 3, 2, 1}));
    CHECK(sm::read(path, id, 1000, t, bytes, c) && c.size() == 4);
    CHECK(sm::write(path, id, toks, 9, {1, 2, 3, 4, 5}));
    CHECK(!sm::read(path, id, 1000, t, bytes, c));
    // no tokens at all
    CHECK(sm::write(path, id, {}, 9, {}));
    CHECK(!sm::read(path, id, 1000, t, bytes, c));
    // an unwritable place: false, not a crash
    CHECK(!sm::write((dir / "nodir" / "x.meta").string(), id, toks, 1, {}));
}

static size_t pick(const std::vector<std::vector<int32_t>>& toks, const std::vector<std::vector<int64_t>>& cks,
                   const std::vector<int>& prompt, size_t floor, size_t* len) {
    std::vector<sm::Cand> c;
    for (size_t i = 0; i < toks.size(); ++i) c.push_back({&toks[i], &cks[i]});
    return sm::pick(c, prompt, floor, len);
}

static void pick_tests() {
    size_t len = 99;
    const std::vector<int32_t> a = seq(1000, 3000), b = seq(5000, 2000);
    // the next turn of A: A whole, then new tokens
    std::vector<int32_t> a_turn = a;
    for (int32_t x : seq(9000, 40)) a_turn.push_back(x);
    CHECK(pick({a, b}, {{}, {}}, as_int(a_turn), 0, &len) == 0 && len == a.size());
    CHECK(pick({b, a}, {{}, {}}, as_int(a_turn), 0, &len) == 1 && len == a.size());
    // nothing shared: none, len 0
    CHECK(pick({a, b}, {{}, {}}, as_int(seq(7777, 100)), 0, &len) == SIZE_MAX && len == 0);
    // a prompt equal to the slot's tokens: the last token is still read, so the slot is not taken whole
    CHECK(pick({a}, {{}}, as_int(a), 0, &len) == SIZE_MAX && len == 0);
    CHECK(pick({a}, {{2900}}, as_int(a), 0, &len) == 0 && len == 2900);
    // a shorter prompt than the slot (a prefix of it): taken at a checkpoint within it, never past it
    std::vector<int32_t> head(a.begin(), a.begin() + 2000);
    CHECK(pick({a}, {{2900, 1500}}, as_int(head), 0, &len) == 0 && len == 1500);
    // a retried prompt: A's prompt end (2950) is a checkpoint; the new prompt agrees up to 2960 then differs
    std::vector<int32_t> retry(a.begin(), a.begin() + 2960);
    for (int32_t x : seq(8000, 30)) retry.push_back(x);
    CHECK(pick({a}, {{2950, 2000, 1000}}, as_int(retry), 0, &len) == 0 && len == 2950);
    // diverging before the latest checkpoint: the latest one it still agrees with (the anchor)
    std::vector<int32_t> early(a.begin(), a.begin() + 1200);
    for (int32_t x : seq(8000, 30)) early.push_back(x);
    CHECK(pick({a}, {{2950, 2000, 1000}}, as_int(early), 0, &len) == 0 && len == 1000);
    // diverging before every checkpoint: none
    std::vector<int32_t> first(a.begin(), a.begin() + 500);
    for (int32_t x : seq(8000, 30)) first.push_back(x);
    CHECK(pick({a}, {{2950, 2000, 1000}}, as_int(first), 0, &len) == SIZE_MAX && len == 0);
    // the floor: what the model itself can go back to wins unless the slot goes further
    CHECK(pick({a}, {{}}, as_int(a_turn), 3000, &len) == SIZE_MAX && len == 0);
    CHECK(pick({a}, {{}}, as_int(a_turn), 2999, &len) == 0 && len == 3000);
    // two slots, the longer match wins; a tie keeps the earlier one
    std::vector<int32_t> a_short(a.begin(), a.begin() + 1000);
    CHECK(pick({a_short, a}, {{}, {}}, as_int(a_turn), 0, &len) == 1 && len == 3000);
    CHECK(pick({a, a}, {{}, {}}, as_int(a_turn), 0, &len) == 0 && len == 3000);
    // too short a prompt
    CHECK(pick({a}, {{}}, {1000}, 0, &len) == SIZE_MAX && len == 0);
    CHECK(pick({a}, {{}}, {}, 0, &len) == SIZE_MAX && len == 0);
    // an empty slot list
    CHECK(pick({}, {}, as_int(a_turn), 0, &len) == SIZE_MAX && len == 0);
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "ds4-slot-test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    meta_tests(dir);
    pick_tests();
    fs::remove_all(dir, ec);
    std::printf(fails ? "FAILED: %d\n" : "slot_test: all passed\n", fails);
    return fails ? 1 : 0;
}
