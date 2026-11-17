// read_expert_profile refuses a profile whose header version is not 1, naming the version it found, the one it
// reads, and the file - before the geometry check, since the fields after the version are only known to be a
// version 1 layout.  A version 1 file still loads.  CPU only: no device is touched.
#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
// overwrite the little-endian uint32 `version` field at byte 4 (after "STRP") in place
bool set_version(const std::string& path, uint32_t v) {
    std::FILE* f = std::fopen(path.c_str(), "r+b");
    if (f == nullptr) return false;
    uint8_t le[4];
    for (int b = 0; b < 4; ++b) le[b] = (uint8_t) (v >> (8 * b));
    const bool ok = std::fseek(f, 4, SEEK_SET) == 0 && std::fwrite(le, 1, 4, f) == 4;
    return (std::fclose(f) == 0) && ok;
}
bool has(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }
}  // namespace

int main() {
    using Pair = std::pair<int32_t, int32_t>;
    const int64_t L = 2, E = 3;
    const std::vector<Pair> ranked = {{1, 2}, {0, 0}, {1, 0}};
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_profile_version_test";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "profile.bin").string();
    std::string err;
    std::vector<Pair> back;
    int64_t slots = 0;

    // version 1, as --expert-profile-save and tools/make_profile.py write it: loads
    check(strata::core::write_expert_profile(path, L, E, ranked, err), "written");
    check(strata::core::read_expert_profile(path, L, E, back, slots, err) && back == ranked && slots == 3,
          "a version 1 profile loads");

    // the same bytes with the version field at 2: refused, naming 2, 1 and the file
    check(set_version(path, 2), "version field rewritten to 2");
    err.clear();
    check(!strata::core::read_expert_profile(path, L, E, back, slots, err), "a version 2 profile is refused");
    check(has(err, "read_expert_profile: ") && has(err, path), "the refusal names the file");
    check(has(err, "version 2") && has(err, "version 1"), "the refusal names the version found and the one read");

    // version 0 (a zeroed header) is not version 1 either
    check(set_version(path, 0), "version field rewritten to 0");
    check(!strata::core::read_expert_profile(path, L, E, back, slots, err) && has(err, "version 0"),
          "a version 0 profile is refused by its number");

    // the version is checked before the geometry: a version 2 file for another model is refused as version 2
    check(set_version(path, 2), "version field rewritten to 2 again");
    check(!strata::core::read_expert_profile(path, L + 1, E, back, slots, err) && has(err, "version 2"),
          "a version 2 profile of another geometry is refused by its version, not its geometry");

    // and back to 1: it loads again, so the refusal was the version field and nothing else
    check(set_version(path, 1), "version field rewritten to 1");
    check(strata::core::read_expert_profile(path, L, E, back, slots, err) && back == ranked, "version 1 loads again");

    std::filesystem::remove_all(dir);
    if (fails == 0) std::puts("expert_profile_version_test: OK");
    return fails == 0 ? 0 : 1;
}
