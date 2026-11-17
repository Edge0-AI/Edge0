#include "strata/core/expert_source.hpp"
#include "strata/core/foresight_swap.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
#endif

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

#if defined(_WIN32)
void check_mapped_release(strata::core::FileExpertSource& source, int64_t layer, int64_t expert, uint64_t bytes) {
    const uint8_t* p = source.blob(layer, expert);
    require(p != nullptr, "release fixture has no mapped blob");
    const std::vector<uint8_t> saved(p, p + (size_t) bytes);
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const uintptr_t page = info.dwPageSize;
    const uintptr_t first = ((uintptr_t) p + page - 1) / page * page;
    const uintptr_t end = ((uintptr_t) p + bytes) / page * page;
    require(end > first, "release fixture has no full interior page");
    auto resident = [&](uintptr_t address) {
        PSAPI_WORKING_SET_EX_INFORMATION state{};
        state.VirtualAddress = (void*) address;
        require(QueryWorkingSetEx(GetCurrentProcess(), &state, sizeof state) != 0,
                "QueryWorkingSetEx failed for the release fixture");
        return state.VirtualAttributes.Valid != 0;
    };
    const char* setting = std::getenv("STRATA_FILE_RELEASE");
    const bool enabled = setting != nullptr && std::strcmp(setting, "1") == 0;
    for (int round = 0; round < 3; ++round) {
        require(std::memcmp(p, saved.data(), saved.size()) == 0, "release changed the mapped expert bytes");
        require(resident(first), "reading the expert did not bring its interior page into the working set");
        const uint64_t released = source.release(layer, expert);
        require(released == (enabled ? end - first : 0), "release did not honor full pages or its opt-in switch");
        // Off: the page stays.  On: Windows may keep a trimmed page on its standby list and map it back at once, so a
        // page that is still valid is reported, not failed (the byte count above is the contract).
        if (!enabled) require(resident(first), "release trimmed pages with the switch off");
        else if (resident(first)) std::fprintf(stderr, "note: the trimmed page is already back in the working set\n");
        if ((uintptr_t) p < first)
            require(resident((uintptr_t) p), "release trimmed the expert's shared first page");
        if (end < (uintptr_t) p + bytes)
            require(resident(end), "release trimmed the expert's shared last page");
    }
    require(std::memcmp(p, saved.data(), saved.size()) == 0, "the expert changed after repeated release/read cycles");
    require(source.release(-1, expert) == 0 && source.release(layer, -1) == 0 &&
                source.release(std::numeric_limits<int64_t>::max(), expert) == 0 &&
                source.release(layer, std::numeric_limits<int64_t>::max()) == 0,
            "release accepted an invalid layer or expert");
}
#endif

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");
#if defined(_WIN32)
    check_mapped_release(source, 0, 0, BLOB);
    check_mapped_release(source, 0, 1, BLOB);
#endif

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
    require(source.release(0, 0) == 0, "a closed file source released pages");
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");
#if defined(_WIN32)
    check_mapped_release(source, 0, 1, first_fmt.bytes);
    check_mapped_release(source, 1, 1, second_fmt.bytes);
#endif

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

void test_complement_plan() {
    using namespace strata::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
}

void test_resident_lend_region() {
    using namespace strata::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_split_lend_regions() {
    using namespace strata::core::detail;
    // two layers, three experts; a layer-0 blob is 5 bytes and a layer-1 blob 3.  The plan under
    // construction already holds the (0,1) expert (an expert no cache holds); stage 0's region lends
    // (0,2) and (0,1) from its highest slots, stage 1's lends (1,2), (1,0), (1,1) in that order.
    const std::vector<uint64_t> blobs{5, 3};
    const std::vector<std::vector<std::pair<int32_t, int32_t>>> regions = {{{0, 2}, {0, 1}}, {{1, 2}, {1, 0}, {1, 1}}};
    std::string err;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;

    // room for everything: (0,1) is already in the copy and is passed over, not counted twice
    offsets = {kNoCacheComplement, 0, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement};
    bytes = 5;
    require(append_stage_lend_regions(2, 3, blobs, offsets, bytes, 100, regions, err) == 4,
            "both stage lend regions must be kept");
    require(bytes == 19 &&
                offsets == std::vector<uint64_t>{kNoCacheComplement, 0, 5, 13, 16, 10},
            "the stage lend regions got wrong compact offsets");

    // the cap is the whole copy's: the stage-0 walk fits (0,2) and stage 1 stops at the first pair that
    // does not fit; that stage's rest stays on the file
    offsets = {kNoCacheComplement, 0, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement};
    bytes = 5;
    require(append_stage_lend_regions(2, 3, blobs, offsets, bytes, 10, regions, err) == 1, "cap 10 keeps only (0,2)");
    require(bytes == 10 && offsets[2] == 5 && offsets[3] == kNoCacheComplement && offsets[5] == kNoCacheComplement,
            "the walk placed experts past the cap");

    // (0,2)'s 5 bytes do not fit in the 4 left: stage 0 keeps nothing, and stage 1's own walk still runs -
    // its (1,2) fits, its (1,0) stops that stage
    offsets = {kNoCacheComplement, 0, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement};
    bytes = 5;
    require(append_stage_lend_regions(2, 3, blobs, offsets, bytes, 9, regions, err) == 1,
            "a stage refused at the cap must not stop the next stage's walk");
    require(bytes == 8 && offsets[2] == kNoCacheComplement && offsets[5] == 5 && offsets[3] == kNoCacheComplement,
            "the refused stage's experts entered the copy, or the next stage's did not");

    // a pair outside the geometry is an error, and the plan is left untouched: validate before appending
    offsets = {kNoCacheComplement, 0, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement, kNoCacheComplement};
    bytes = 5;
    require(append_stage_lend_regions(2, 3, blobs, offsets, bytes, 100, {{{0, 2}, {2, 0}}}, err) == -1 &&
                !err.empty() && bytes == 5 && offsets[2] == kNoCacheComplement,
            "an out-of-range lend-region pair was accepted, or a rejected walk touched the plan");
}

void test_resident_exchange() {
    using namespace strata::core;
    using namespace strata::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

// #4051: the wait for the RAM an arena needs (an engine still exiting gives it back a few seconds late)
void test_wait_for_ram() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    double waited = -1.0;
    uint64_t avail = 0;
    int slept = 0, noted = 0;
    const auto sleeper = [&](double) { ++slept; };
    const auto note = [&](double, uint64_t) { ++noted; };
    // enough at once: no wait, no sleep, no note
    require(wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [](uint64_t& a) { a = 50 * GiB; return true; }, sleeper, note, waited, avail) &&
                waited == 0.0 && slept == 0 && noted == 0 && avail == 50 * GiB, "RAM that was there was waited for");
    // 10 GiB free for a 40 GiB arena, the exiting engine gives its 50 GiB back after 3 s (12 steps of 0.25 s)
    int reads = 0;
    slept = noted = 0;
    require(wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [&](uint64_t& a) { a = ++reads > 12 ? 60 * GiB : 10 * GiB; return true; },
                         sleeper, note, waited, avail) && waited == 3.0 && slept == 12 && noted == 1 && avail == 60 * GiB,
            "RAM that came back after 3 s was not waited for");
    // it never comes: the wait ends at the limit and says so (the load then goes on and warns)
    slept = noted = 0;
    require(!wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [](uint64_t& a) { a = 10 * GiB; return true; }, sleeper, note, waited, avail) &&
                waited == 20.0 && slept == 80 && noted == 4 && avail == 10 * GiB, "the wait did not end at its limit");
    // no wait asked for: one reading
    slept = 0;
    require(!wait_for_ram(40 * GiB, 0.0, 0.25, 5.0, [](uint64_t& a) { a = 10 * GiB; return true; }, sleeper, note, waited, avail) &&
                waited == 0.0 && slept == 0, "a zero wait slept");
    // the RAM cannot be read: the wait ends at once
    slept = 0;
    require(!wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [](uint64_t&) { return false; }, sleeper, note, waited, avail) && slept == 0,
            "an unreadable RAM was waited for");
}

// #1250: when the page-locked complement is registered in steps, and what a meminfo reading says
void test_pin_pacing() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    // the reporter's reading: 54 GiB available but 16 GiB really free for a 26 GiB complement
    require(pin_depends_on_reclaim(16 * GiB, 26 * GiB, 4 * GiB), "a complement bigger than the free pages was not flagged");
    require(!pin_depends_on_reclaim(40 * GiB, 26 * GiB, 4 * GiB), "a complement the free pages cover was flagged");
    require(pin_depends_on_reclaim(30 * GiB, 26 * GiB, 4 * GiB) == false, "exactly free = bytes + reserve is enough");
    require(!pin_depends_on_reclaim(0, 0, 4 * GiB), "no complement, nothing to pin");
    require(pin_step_fits(6 * GiB, 1 * GiB, 4 * GiB) && !pin_step_fits(4 * GiB, 1 * GiB, 4 * GiB),
            "a step must leave the reserve");
    const std::string info =
        "MemTotal:       65011484 kB\nMemFree:         1284096 kB\nMemAvailable:   33463296 kB\nCached:         54507520 kB\n";
    require(meminfo_bytes(info, "MemFree") == 1284096ull * 1024, "MemFree was misread");
    require(meminfo_bytes(info, "MemAvailable") == 33463296ull * 1024, "MemAvailable was misread");
    require(meminfo_bytes(info, "Mem") == 0 && meminfo_bytes(info, "Shmem") == 0, "an absent key was not 0");
    // 0.1.42 (dual-socket report): node 1 full of another model's cache, node 0 empty of it
    require(numa_pin_free(100 * GiB, {}) == 100 * GiB && numa_pin_free(100 * GiB, {100 * GiB}) == 100 * GiB,
            "a single-node host's figure changed");
    require(numa_pin_free(90 * GiB, {88 * GiB, 2 * GiB}) == 4 * GiB, "an emptied node did not lower the figure");
    require(numa_pin_free(40 * GiB, {20 * GiB, 20 * GiB}) == 40 * GiB, "balanced nodes changed the figure");
    require(numa_pin_free(10 * GiB, {50 * GiB, 50 * GiB}) == 10 * GiB, "the total must still bound it");
    require(pin_depends_on_reclaim(numa_pin_free(90 * GiB, {88 * GiB, 2 * GiB}), 26 * GiB, 4 * GiB),
            "an unbalanced two-node host was not paced");
    require(page_cache_hint(info).find("drop_caches") != std::string::npos &&
            page_cache_hint(info).find("STRATA_DROP_CACHE_ON_EXIT") != std::string::npos &&
            page_cache_hint(info).find("52.0 GiB") != std::string::npos, "the cache hint is incomplete");
    require(page_cache_hint("MemFree: 100 kB\nCached: 1000 kB\n").empty(), "a small cache was reported");
}

// NUMA split: per-slice byte ranges of one complement slot (page rule: a page holding a slice-0 byte is slice 0's)
void test_numa_slice_ranges() {
    using strata::core::detail::numa_slice_ranges;
    constexpr uint64_t P = 4096;
    const uint64_t roles[3] = {870400, 870400, 921600};
    const uint64_t slot_bytes = roles[0] + roles[1] + roles[2];
    const uint64_t offs[3] = {0, 2662400ull * 7, P * 3 + 100};
    for (uint64_t off : offs) {
        const auto r0 = numa_slice_ranges(off, roles, 0), r1 = numa_slice_ranges(off, roles, 1);
        for (int r = 0; r < 3; ++r) {
            require(r0[r].first % P == 0 && r0[r].second % P == 0 && r1[r].first % P == 0 && r1[r].second % P == 0,
                    "a slice range boundary was not page aligned");
            require(r0[r].first <= r0[r].second && r1[r].first <= r1[r].second, "a slice range was inverted");
        }
        // every page fully inside the slot is in exactly one of the six ranges (and none is in two)
        for (uint64_t pg = (off + P - 1) / P * P; pg + P <= off + slot_bytes; pg += P) {
            int hits = 0;
            for (int r = 0; r < 3; ++r) {
                hits += (pg >= r0[r].first && pg + P <= r0[r].second);
                hits += (pg >= r1[r].first && pg + P <= r1[r].second);
            }
            require(hits == 1, "a page was in " + std::to_string(hits) + " slice ranges, not exactly one");
        }
    }
    // (b) no slice-1 range intersects a slice-0 range, within a slot and across two consecutive slots
    const auto meet = [](const std::pair<uint64_t, uint64_t>& x, const std::pair<uint64_t, uint64_t>& y) {
        return x.first < y.second && y.first < x.second;
    };
    for (uint64_t off : offs) {
        const auto s0 = numa_slice_ranges(off, roles, 0), s1 = numa_slice_ranges(off, roles, 1);
        const auto n0 = numa_slice_ranges(off + slot_bytes, roles, 0), n1 = numa_slice_ranges(off + slot_bytes, roles, 1);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                require(!meet(s0[i], s1[j]), "a slice-1 range intersects a slice-0 range of the same slot");
                require(!meet(s1[i], n0[j]) && !meet(n1[i], s0[j]),
                        "a slice-1 range intersects a slice-0 range of the neighbouring slot");
            }
    }
    const auto a = numa_slice_ranges(0, roles, 0), b = numa_slice_ranges(0, roles, 1);
    require(a[0].first <= 435199 && 435199 < a[0].second, "last slice-0 gate byte not in slice 0's gate range");
    require(b[0].first <= 435200 + P && 435200 + P < b[0].second, "a slice-1 gate byte not in slice 1's gate range");
    require(a[1].first <= 870400 && 870400 < a[1].second, "the gate/up boundary page was not slice 0's up range");
    require(!(b[0].first <= 870400 && 870400 < b[0].second), "the gate/up boundary page was in slice 1's gate range");
}

void test_resident_memory_budget() {
    using strata::core::detail::clamp_resident_budget;
    constexpr uint64_t GiB = 1ull << 30, margin = 256ull << 20, headroom = 4 * GiB;
    constexpr uint64_t unlimited = std::numeric_limits<uint64_t>::max();
    // #730: RAM can hold the requested cache, but Windows cannot commit it.
    require(clamp_resident_budget(66 * GiB, 69 * GiB, 47 * GiB, headroom) == 43 * GiB - margin,
            "a RAM budget exceeded available commit capacity");
    require(clamp_resident_budget(66 * GiB, 47 * GiB, 69 * GiB, headroom) == 43 * GiB - margin,
            "a RAM budget exceeded available physical memory");
    require(clamp_resident_budget(40 * GiB, 69 * GiB, 47 * GiB, headroom) == 40 * GiB,
            "a smaller explicit budget was changed");
    require(clamp_resident_budget(unlimited, 69 * GiB, 47 * GiB, headroom) == 43 * GiB - margin,
            "the what-fits budget ignored commit capacity");
    require(clamp_resident_budget(66 * GiB, 47 * GiB, unlimited, headroom) == 43 * GiB - margin,
            "a platform without a commit reading lost its RAM limit");
    require(clamp_resident_budget(43 * GiB, 69 * GiB, 47 * GiB, headroom) == 43 * GiB,
            "a budget that exactly fits received the clamping margin");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, 0, headroom) == 0,
            "exhausted commit capacity was treated as unlimited");
    require(clamp_resident_budget(66 * GiB, 0, 69 * GiB, headroom) == 0,
            "exhausted RAM produced a positive budget");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, headroom - 1, headroom) == 0 &&
                clamp_resident_budget(66 * GiB, 69 * GiB, headroom, headroom) == 0,
            "subtracting headroom underflowed");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, headroom + margin - 1, headroom) == 0 &&
                clamp_resident_budget(66 * GiB, 69 * GiB, headroom + margin, headroom) == 0,
            "subtracting the clamping margin underflowed");
}

void test_cgroup_memory_budget() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    // the kernel keeps min(half the clean cache, limit / 64) of it: 56 GiB / 64 = 0.875 GiB here
    const uint64_t keep56 = 56 * GiB / 64;
    CgroupMemoryStat clean_cache{40 * GiB, 32 * GiB, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB - keep56, "clean inactive file cache was not credited against the cgroup cap");

    CgroupMemoryStat dirty_cache{40 * GiB, 32 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, dirty_cache, bytes), "valid dirty-cache counters were rejected");
    require(bytes == 42 * GiB - keep56, "dirty/writeback pages were incorrectly counted as reclaimable");

    CgroupMemoryStat oversized_inactive{10 * GiB, 20 * GiB, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_inactive, bytes),
            "valid oversized inactive-file counter was rejected");
    require(bytes == 12 * GiB - 12 * GiB / 64, "inactive-file accounting exceeded charged current usage");

    // 149: the ACTIVE page cache is reclaimable too (memory.stat "file" = active + inactive), as MemAvailable counts
    // it: a 32 GiB group whose own reads left 26 GiB of cache, 25 GiB of it active, and 4.5 GiB of anonymous memory
    CgroupMemoryStat active_cache{30 * GiB + GiB / 2, 1 * GiB, 0, 0, true, 26 * GiB};
    require(cgroup_available_bytes(32 * GiB, active_cache, bytes), "a memory.stat with file was rejected");
    require(bytes == 32 * GiB - (4 * GiB + GiB / 2) - 32 * GiB / 64, "the active page cache was not credited");
    // without "file" (an old kernel's stat): inactive_file alone, as before
    CgroupMemoryStat no_file{30 * GiB + GiB / 2, 1 * GiB, 0, 0, true};
    require(cgroup_available_bytes(32 * GiB, no_file, bytes) && bytes == 32 * GiB - (29 * GiB + GiB / 2) - GiB / 2,
            "a stat without file did not fall back to inactive_file");
    // dirty and writeback pages of the active cache stay charged; a small cache keeps half of itself
    CgroupMemoryStat dirty_active{20 * GiB, 1 * GiB, 2 * GiB, 1 * GiB, true, 10 * GiB};
    require(cgroup_available_bytes(32 * GiB, dirty_active, bytes) && bytes == 32 * GiB - (20 * GiB - 7 * GiB) - GiB / 2,
            "dirty / writeback pages of the active cache were counted as reclaimable");
    CgroupMemoryStat tiny_cache{8 * GiB, 0, 0, 0, true, 256ull << 20};
    require(cgroup_available_bytes(32 * GiB, tiny_cache, bytes) && bytes == 32 * GiB - 8 * GiB + (128ull << 20),
            "a small clean cache was not kept at half");
    // "file" larger than what is charged (a racy read) is bounded by memory.current
    CgroupMemoryStat racy{4 * GiB, 1 * GiB, 0, 0, true, 9 * GiB};
    require(cgroup_available_bytes(32 * GiB, racy, bytes) && bytes == 32 * GiB - GiB / 2,
            "a file counter above memory.current escaped the bound");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes),
            "large memory.stat counters were rejected");
    require(bytes == 0, "dirty/writeback subtraction overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes),
            "valid over-limit memory counters were rejected");
    require(bytes == 0, "over-limit cgroup usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// #286: the unbuffered file tier (Windows: FILE_FLAG_NO_BUFFERING; Linux: O_DIRECT + io_submit) hands out the same
// bytes as the mapping - one blob at a time, a batch of adjacent experts (merged into one request), and copy_blob - and
// never falls back to the mapping.  The pack lives in the working directory, not the temp directory: a tmpfs /tmp has
// no O_DIRECT, and there the test says so and skips.
void test_unbuffered_reads() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    std::vector<uint8_t> bytes((size_t) total);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint8_t& b : bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "could not map the synthetic pack: " + err);
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    std::string why;
    const bool unbuffered = source.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
#if defined(_WIN32) || defined(__linux__)
    if (!unbuffered) {
        std::cout << "file_expert_source_test: unbuffered reads skipped (" << why << ")\n";
        source.close();
        return;
    }
#else
    require(!unbuffered, "unbuffered reads on a platform without them");
    source.close();
    return;
#endif
    require(source.unbuffered(), "set_unbuffered succeeded but the source is not unbuffered");
    const int64_t batch[experts] = {0, 1, 2};
    source.prefetch(1, batch, experts);   // layer 1 in one batch: three adjacent blobs
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < experts; ++e) {
            const uint8_t* b = source.blob(l, e);
            require(b != nullptr, "an unbuffered blob lookup failed");
            require(std::memcmp(b, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB),
                                (size_t) BLOB) == 0,
                    "an unbuffered blob differs from the file");
        }
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(0, 2, copy.data()) &&
                std::memcmp(copy.data(), bytes.data() + (size_t) (2 * BLOB), (size_t) BLOB) == 0,
            "an unbuffered copy_blob differs from the file");
    require(source.direct_fallbacks() == 0, "an unbuffered read fell back to the mapping");
    source.close();
}

// #633: the host RAM probe with fake /proc and cgroup trees: v2 (a limit, "max", a missing or malformed limit), v1
// (a limit, unlimited), and no cgroup line at all.  Elsewhere than Linux it reads the machine's RAM.
void test_host_memory() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    HostMemory m;
#if defined(__linux__)
    TempDirectory t;
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((t.path / rel).parent_path());
        std::ofstream(t.path / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (t.path / "meminfo").string(), cg = (t.path / "cgroup").string(),
                      root = (t.path / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        m.commit = 0;
        const bool ok = host_available_memory(m, mi, cg, root);
        require(m.commit == ~0ull, "Linux memory probe retained a commit limit");
        return ok;
    };
    // no cgroup line at all: MemAvailable alone (before #633: "cannot determine")
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 36 * GiB - 48 * GiB / 64 && m.cgroup_limit == 48 * GiB, "v2 limit");
    // 149: with "file" (active + inactive cache) the active part counts too: 12 GiB of cache, 4 GiB anonymous
    put("fs/box/memory.stat", "file " + std::to_string(12 * GiB) + "\ninactive_file " + std::to_string(4 * GiB) +
                              "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 44 * GiB - 48 * GiB / 64 && m.cgroup_limit == 48 * GiB,
            "v2 limit, the active page cache credited");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n"), "v2 malformed limit accepted");
    fs::remove(t.path / "fs/box/memory.max");
    require(!probe("0::/box\n"), "v2 group without memory.max accepted");
    // v1: the memory controller's group (a hybrid line list), a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v1 unlimited");
    // a v1 group not visible here (another namespace): MemAvailable alone
    require(probe("4:memory:/elsewhere\n") && m.available == 100 * GiB, "v1 group not mounted");
#else
    require(host_available_memory(m) && m.available > 0 && m.cgroup_limit == ~0ull, "this PC's RAM");
#if defined(_WIN32)
    require(m.commit != ~0ull, "Windows memory probe did not report commit capacity");
#else
    require(m.commit == ~0ull, "a platform without commit accounting imposed a limit");
#endif
    (void) GiB;
#endif
}

// STRATA_IO_PREFETCH: the experts.bin file tier hands out the same bytes with the Linux I/O path on (whole-blob
// preads, the read-ahead workers, the mapping for cached blobs), however the reads and the predictions interleave.
void test_io_prefetch() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 3;
    constexpr int64_t experts = 8;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    std::vector<uint8_t> bytes((size_t) total);
    uint64_t x = 0xD1B54A32D192ED03ull;
    for (uint8_t& b : bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "could not map the synthetic pack: " + err);
    source.set_io_prefetch(true, 3);
#if defined(__linux__)
    require(source.io_prefetch() && source.warms(), "io prefetch did not turn on");
    // predictions for the next layers, some of them wrong or repeated, while the layers are read
    for (int round = 0; round < 20; ++round)
        for (int64_t l = 0; l < layers; ++l) {
            const int64_t pred[5] = {(round + l) % experts, (round * 3 + 1) % experts, (round + 2 * l + 5) % experts,
                                     (round + l) % experts, 7};
            source.warm((l + 1) % layers, pred, 5);
            source.begin_layer(l, nullptr, 0);
            std::vector<int64_t> want = {(round + l) % experts, (round * 5 + 2) % experts, (l * 3 + round) % experts};
            source.prefetch(l, want.data(), (int64_t) want.size());
            for (int64_t e : want) {
                const uint8_t* b = source.blob(l, e);
                require(b != nullptr, "an io-prefetch blob lookup failed");
                require(std::memcmp(b, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB), (size_t) BLOB) == 0,
                        "an io-prefetch blob differs from the file");
                const uint8_t* s2 = source.blob_stable(l, e);
                require(s2 != nullptr && std::memcmp(s2, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB), (size_t) BLOB) == 0,
                        "a blob_stable pointer differs from the file");
            }
        }
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(1, 6, copy.data()) &&
                std::memcmp(copy.data(), bytes.data() + (size_t) (layer_bytes + 6 * BLOB), (size_t) BLOB) == 0,
            "an io-prefetch copy_blob differs from the file");
    const FileExpertSource::IoCounters c = source.io_counters();
    require(c.cached_bytes + c.uncached_bytes > 0, "no residency counted");
    std::cout << "io prefetch: " << c.pf_read_blobs << " read ahead, " << c.pf_used << " used, " << c.pf_resident_skips
              << " skipped, pread " << c.pread_bytes << " B: bytes PASS\n";
#else
    require(!source.io_prefetch(), "io prefetch on a platform without it");
#endif
    source.close();
}

void test_rotating_source(bool rotate, bool pin) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("STRATA_EXCHANGE_ROTATE", rotate ? "1" : "0");
#else
    setenv("STRATA_EXCHANGE_ROTATE", rotate ? "1" : "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t)BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t)((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*)truth[e].data(), (std::streamsize)bytes);
        }
        require((bool)out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    // Keep expert 4 on the file tier to check that the fallback survives rotation.
    require(src.pin_cache_complement(cache, err, pin, {}, -1, 0, 2 * bytes, &rank), err);
    require(src.reserve_exchanges(2, err), err);
    require(src.exchange_rotation() == (rotate && pin), "rotation activation/fallback wrong");
    require(src.reserve_exchanges(1, err), "smaller capacity rejected");
    if (rotate && pin) require(!src.reserve_exchanges(3, err), "live arena growth accepted");
    int incoming[2] = {0, 1}, outgoing[2] = {2, 3};
    std::vector<uint8_t> actual(bytes);
    double commit_ms = 0;
    for (int round = 0; round < 32; ++round) {
        const uint8_t* prior[2]{};
        uint8_t* eviction[2]{};
        for (int q = 0; q < 2; ++q) {
            prior[q] = src.blob(0, incoming[q]); eviction[q] = src.exchange_buffer(q);
            require(prior[q] && !std::memcmp(prior[q], truth[incoming[q]].data(), bytes), "wrong incoming bytes");
            require(cudaMemcpy(eviction[q], cache.device_slot(q), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                    "D2H eviction failed");
            require(src.stage_exchange(0, incoming[q], outgoing[q], q), "stage rejected");
            require(!std::memcmp(src.blob(0, outgoing[q]), truth[outgoing[q]].data(), bytes), "staged override wrong");
            require(cache.fill_slot_blocking(q, prior[q], err), err); // H2D completed before commit
        }
        require(!src.stage_exchange(0, incoming[0], outgoing[0], 0), "duplicate stage accepted");
        require(!src.stage_exchange(0, incoming[0], 5, 0), "invalid expert accepted");
        const auto t0 = std::chrono::steady_clock::now();
        require(src.commit_exchanges() == 2, "commit count wrong");
        commit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        require(src.commit_exchanges() == 0, "empty commit not empty");
        for (int q = 0; q < 2; ++q) {
            require(!src.has_resident(0, incoming[q]) && src.has_resident(0, outgoing[q]), "residency wrong");
            const uint8_t* held = src.blob(0, outgoing[q]);
            require(!std::memcmp(held, truth[outgoing[q]].data(), bytes), "evicted bytes corrupted");
            require(held == ((rotate && pin) ? eviction[q] : prior[q]), "wrong storage selected");
            if (rotate && pin) require(src.exchange_buffer(q) == prior[q], "old input not recycled");
            require(src.copy_blob(0, outgoing[q], actual.data()) && actual == truth[outgoing[q]], "copy_blob mismatch");
            require(!src.transient(0, outgoing[q]), "resident became transient");
            if (pin) {
                require(src.pinned(0, outgoing[q]) && src.device_alias(0, outgoing[q]), "mapping lost");
                require(cudaMemcpy(actual.data(), src.device_alias(0, outgoing[q]), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                        "device alias read failed");
                require(actual == truth[outgoing[q]], "CUDA alias points at stale expert");
            }
            require(cache.verify_slot(q, truth[incoming[q]].data(), err), err);
            std::swap(incoming[q], outgoing[q]);
        }
        require(!src.has_resident(0, 4) && src.copy_blob(0, 4, actual.data()) && actual == truth[4], "file fallback changed");
    }
    require(src.exchanges() == 64, "exchange accounting wrong");
    require(src.rotated_exchanges() == ((rotate && pin) ? 64u : 0u), "rotation accounting wrong");
    require(src.avoided_exchange_copy_bytes() == ((rotate && pin) ? 64 * bytes : 0u), "avoided bytes wrong");
    std::cout << "rotation integration: rotate=" << rotate << " pinned=" << pin
              << " exchanges=64 commit_ms=" << commit_ms << " exact_bytes=PASS\n";
    src.close();
    require(!src.exchange_rotation() && !src.exchange_buffer(0), "close retained storage");
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(!src.has_resident(0, 0) && !src.exchange_rotation(), "reopen retained residency");
}

// #2134: a Foresight slot copied from an expert's RAM place before an exchange rewrote that place is never served; the
// experts an exchange moves get no copy between commit_copies and commit_flip; and a dropped slot that a window may
// still read waits `depth` finished windows before it is refilled.
void test_foresight_exchange(bool two_phase) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("STRATA_EXCHANGE_ROTATE", "0");
#else
    setenv("STRATA_EXCHANGE_ROTATE", "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t) BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t) ((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*) truth[e].data(), (std::streamsize) bytes);
        }
        require((bool) out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    require(src.pin_cache_complement(cache, err, true, {}, -1, 0, 2 * bytes, &rank), err);
    require(src.reserve_exchanges(1, err), err);
    require(src.pinned(0, 0), "expert 0 is not in the pinned RAM copy");
    int dev = 0;
    cudaGetDevice(&dev);
    ForesightSwap fs;
    require(fs.init(1, 1, 5, {0}, {dev}, err), err);   // one slot, depth 4
    fs.verify = true;
    src.set_foresight_swap(&fs);
    unsigned long long ptr = 0;
    auto landed = [&](int32_t e) {
        for (int i = 0; i < 5000; ++i) {
            if (fs.take(0, e, ptr)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    };
    fs.note_miss(&src, 0, 0);                                    // expert 0 -> the slot, from its RAM place
    require(landed(0) && fs.verify_ok == 1, "the first copy did not land");
    // the adaptive tier swaps expert 0 into cache slot 0 and evicts expert 2 into the exchange buffer
    uint8_t* eviction = src.exchange_buffer(0);
    const uint8_t* prior = src.blob(0, 0);
    require(cudaMemcpy(eviction, cache.device_slot(0), bytes, cudaMemcpyDeviceToHost) == cudaSuccess, "D2H eviction failed");
    require(src.stage_exchange(0, 0, 2, 0), "stage rejected");
    require(cache.fill_slot_blocking(0, prior, err), err);
    if (two_phase) {
        src.commit_copies();                                     // 0's place holds 2's bytes from here on
        const uint64_t c0 = fs.copies;
        fs.completed.fetch_add(8);
        fs.note_miss(&src, 0, 2);
        fs.predict(&src, 0, 0);
        require(fs.copies == c0, "an expert was copied while its exchange was half done");
        require(!fs.take(0, 0, ptr), "a slot dropped by commit_copies was served");
        require(src.commit_flip() == 1, "flip count wrong");
    } else {
        require(src.commit_exchanges() == 1, "commit count wrong");
    }
    require(fs.dropped == 1, "the exchange did not drop the slot");
    require(!fs.take(0, 0, ptr), "a slot copied before the exchange was served");
    if (!two_phase) {   // the slot was served at window 0: no refill before window 0 + depth
        const uint64_t c0 = fs.copies, b0 = fs.busy;
        fs.note_miss(&src, 0, 2);
        require(fs.copies == c0 && fs.busy == b0 + 1, "a dropped slot was refilled while a window may still read it");
        fs.completed.fetch_add(4);
    }
    require(src.has_resident(0, 2) && !src.has_resident(0, 0), "residency wrong");
    fs.note_miss(&src, 0, 2);                                    // expert 2, now at 0's old place
    require(landed(2), "the copy after the exchange did not land");
    std::vector<uint8_t> back(bytes);
    require(cudaMemcpy(back.data(), (const void*) ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess && back == truth[2],
            "the slot does not hold expert 2's bytes");
    require(fs.verify_bad == 0 && fs.verify_ok == 2, "verify counted a bad slot");
    src.set_foresight_swap(nullptr);
    std::cout << "foresight exchange (" << (two_phase ? "commit_copies + commit_flip" : "commit_exchanges")
              << "): dropped " << fs.dropped << ", " << fs.report() << ": PASS\n";
    src.close();
}

// STRATA_RAM_ADAPT: an expert read from the file takes the RAM place of another of its layer.
void test_replace_resident(bool pin) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("STRATA_EXCHANGE_ROTATE", "0");
#else
    setenv("STRATA_EXCHANGE_ROTATE", "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t)BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t)((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*)truth[e].data(), (std::streamsize)bytes);
        }
        require((bool)out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    // experts 2 and 3 are in VRAM, the RAM budget holds 0 and 1, expert 4 stays on the file tier
    require(src.pin_cache_complement(cache, err, pin, {}, -1, 0, 2 * bytes, &rank), err);
    require(src.reserve_exchanges(1, err), err);
    require(src.has_resident(0, 0) && src.has_resident(0, 1) && !src.has_resident(0, 4), "fixture residency wrong");
    std::vector<uint8_t> actual(bytes);
    auto holds = [&](int e) {
        const uint8_t* b = src.blob(0, e);
        return b != nullptr && !std::memcmp(b, truth[(size_t)e].data(), bytes);
    };
    require(!src.replace_resident(0, 1, 0), "an expert already in RAM accepted");
    require(!src.replace_resident(0, 4, 2), "a place the RAM does not hold accepted");
    require(!src.replace_resident(0, 4, 4) && !src.replace_resident(0, 5, 0) && !src.replace_resident(1, 4, 0),
            "invalid arguments accepted");
    require(src.ram_adapts() == 0 && holds(0) && holds(1) && holds(4), "a refused call changed the copy");
    // a staged exchange owns the tables until it commits
    require(cudaMemcpy(src.exchange_buffer(0), cache.device_slot(0), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
            "D2H eviction failed");
    require(src.stage_exchange(0, 0, 2, 0), "stage rejected");
    require(!src.replace_resident(0, 4, 1), "accepted beside a staged exchange");
    require(cache.fill_slot_blocking(0, truth[0].data(), err), err);
    require(src.commit_exchanges() == 1, "commit count wrong");   // RAM: 2 (in 0's place) and 1; VRAM: 0 and 3
    const uint8_t* place = src.blob(0, 1);
    require(src.replace_resident(0, 4, 1), "replace rejected");
    require(src.ram_adapts() == 1 && src.has_resident(0, 4) && !src.has_resident(0, 1), "residency wrong");
    require(src.blob(0, 4) == place && holds(4), "the incoming expert is not in the outgoing one's place");
    require(holds(1) && holds(2), "another expert's bytes changed");
    require(src.copy_blob(0, 1, actual.data()) && actual == truth[1], "the outgoing expert is not read from the file");
    require(!src.transient(0, 4), "resident became transient");
    if (pin) {
        require(src.pinned(0, 4) && src.device_alias(0, 4), "mapping lost");
        require(cudaMemcpy(actual.data(), src.device_alias(0, 4), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                "device alias read failed");
        require(actual == truth[4], "CUDA alias points at the old expert");
    }
    // and back, then an exchange with the VRAM tier still works on the moved place
    require(src.replace_resident(0, 1, 4) && holds(1) && src.blob(0, 1) == place && !src.has_resident(0, 4),
            "replace back wrong");
    require(cudaMemcpy(src.exchange_buffer(0), cache.device_slot(1), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
            "D2H eviction failed");
    require(src.stage_exchange(0, 1, 3, 0), "stage after replace rejected");
    require(cache.fill_slot_blocking(1, truth[1].data(), err), err);
    require(src.commit_exchanges() == 1 && src.has_resident(0, 3) && holds(3) && !src.has_resident(0, 1),
            "exchange after replace wrong");
    require(src.ram_adapts() == 2, "accounting wrong");
    // a staged incoming blob: RAM {2,3}, VRAM {0,1}, 4 on the file tier
    require(!src.replace_resident_staged(0, 4, 3) && src.has_resident(0, 3) && src.ram_adapts() == 2,
            "an unstaged expert moved");
    std::string why;
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    const bool unbuffered = src.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
    if (unbuffered && src.stages_blobs()) {
        require(!src.replace_resident_staged(0, 4, 3) && holds(3), "moved before any demand read");
        require(!src.has_resident(0, 4) && holds(4), "demand read wrong");   // leaves 4 in a stage buffer
        require(!src.replace_resident_staged(0, 4, 0) && !src.replace_resident_staged(0, 2, 3),
                "invalid staged pair accepted");
        require(src.replace_resident_staged(0, 4, 3) && src.has_resident(0, 4) && !src.has_resident(0, 3),
                "staged replacement residency wrong");
        require(src.resident_blob(0, 4) != nullptr && !std::memcmp(src.resident_blob(0, 4), truth[4].data(), bytes),
                "staged replacement bytes wrong");
        require(holds(2) && holds(3) && src.ram_adapts() == 3 && src.ram_staged() == 1,
                "staged replacement changed another expert");
        std::cout << "replace_resident_staged: pinned=" << pin << " exact_bytes=PASS\n";
    } else {
        std::cout << "replace_resident_staged: pinned=" << pin << " unbuffered unavailable here (" << why << "), positive case SKIPPED\n";
    }
    std::cout << "replace_resident: pinned=" << pin << " exact_bytes=PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        test_complement_plan();
        test_resident_lend_region();
        test_split_lend_regions();
        test_resident_exchange();
        test_resident_memory_budget();
        test_pin_pacing();
        test_wait_for_ram();
        test_numa_slice_ranges();
        test_cgroup_memory_budget();
        test_host_memory();
        test_canonical_layout();
        test_unbuffered_reads();
        test_io_prefetch();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
#endif
        if (argc == 2 && std::string(argv[1]) == "--rotation-gpu") {
            test_rotating_source(false, true);
            test_rotating_source(true, true);
            test_rotating_source(true, false);
            test_foresight_exchange(false);
            test_foresight_exchange(true);
            test_replace_resident(true);
            test_replace_resident(false);
        }
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
