// The resident RAM mode with a layer split: the compact copy holds the experts no stage's cache holds AND each
// stage's LEND REGION - the experts in the cache slots the prompt path borrows - so a prompt streams the
// borrowed experts from RAM instead of reading the model files through page faults.  Runs on a CPU-only
// PC: the device calls are wrapped at the link (`--wrap` in CMakeLists.txt, conversation_transfer_test's
// pattern), the memory is ordinary memory, and `pin = false` keeps the copy pageable.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// ---- the wrapped device calls: memory is ordinary memory and every operation completes at once.
extern "C" cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    *p = std::malloc(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFree(void* p) { std::free(p); return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t st) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t st) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaDeviceSynchronize(void) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaGetLastError(void) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaSetDevice(int d) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    *free_b = 64ull << 30;
    *total_b = 128ull << 30;
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned f) {
    *p = std::malloc(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }

namespace fs = std::filesystem;

namespace {

constexpr int64_t kLayers = 2;      // stage 0 owns layer 0, stage 1 owns layer 1 (the split's `claimed` layers)
constexpr int64_t kExperts = 4;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-split-lend-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const uint8_t truth_byte(int64_t l, int64_t e, size_t i) {
    return (uint8_t) ((i * 131u + (size_t) (l * kExperts + e) * 37u) ^ (i >> 8));
}

// one fully-built fixture: experts.bin written, the source open, both stage caches open with their
// own layer's experts admitted and filled (highest slot = highest expert, as the
// cache fills from slot 0 up)
struct Rig {
    std::vector<std::vector<uint8_t>> truth;   // [layer * kExperts + expert]
    TempDirectory dir;
    strata::core::FileExpertSource src;
    strata::core::ExpertCache stage0, stage1;   // the two split stages' caches
    size_t blob = 0;

    explicit Rig(const char* tag) {
        std::string err;
        require(strata::kernels::cpu::expert_layout_load(dir.path.string(), kLayers, kExperts, err), err);
        blob = (size_t) strata::kernels::cpu::BLOB;
        truth.resize((size_t) kLayers * kExperts);
        {
            std::ofstream out(dir.path / "experts.bin", std::ios::binary);
            for (int64_t l = 0; l < kLayers; ++l)
                for (int64_t e = 0; e < kExperts; ++e) {
                    std::vector<uint8_t>& t = truth[(size_t) l * kExperts + e];
                    t.resize(blob);
                    for (size_t i = 0; i < blob; ++i) t[i] = truth_byte(l, e, i);
                    out.write((const char*) t.data(), (std::streamsize) blob);
                }
            require((bool) out, std::string(tag) + ": fixture write failed");
        }
        require(src.open(dir.path.string(), kLayers, kExperts, err), err);
        // kExperts slots each: every layer of the stage, and the tail the prompt path borrows
        require(stage0.open(kExperts, kLayers, kExperts, blob, err), err);
        require(stage1.open(kExperts, kLayers, kExperts, blob, err), err);
        for (int64_t e = 0; e < kExperts; ++e) {
            const int32_t s0 = stage0.admit(0, e);
            require(s0 >= 0 && stage0.fill_slot_blocking(s0, truth[(size_t) e].data(), err), err);
            const int32_t s1 = stage1.admit(1, e);
            require(s1 >= 0 && stage1.fill_slot_blocking(s1, truth[(size_t) kExperts + e].data(), err), err);
        }
    }

    // every expert of stage 1's cache, the way generate hands the later stages to pin_cache_complement
    std::vector<std::pair<int32_t, int32_t>> stage1_pairs() const {
        std::vector<std::pair<int32_t, int32_t>> pairs;
        for (int64_t e = 0; e < kExperts; ++e) pairs.emplace_back(1, (int32_t) e);
        return pairs;
    }
};

// both stage caches hold everything of their layer: the complement - the experts NO cache
// holds - is empty, so before this feature the copy kept no bytes at all and every expert a prompt borrowed
// came back through page faults from the mapped files.
void test_split_lend_region_served_from_ram() {
    Rig rig("split lend region");
    std::string err;

    // each stage lends its two highest slots to the prompt path: pairs listed highest slot first,
    // as the borrowing takes them
    rig.src.stage_lend_regions({{{0, 3}, {0, 2}}, {{1, 3}, {1, 2}}});
    require(rig.src.pin_cache_complement(rig.stage0, err, false, rig.stage1_pairs(), -1, 0, 0), err);

    require(rig.src.resident_bytes() == 4 * (uint64_t) rig.blob,
            "the copy is the empty complement plus the two stages' lend regions");
    require(rig.src.resident_lent_slots() == 4, "the copy reports both stages' lend regions");
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const bool in_a_cache = true;   // every pair is held by some stage's cache: none is in the complement
            require(rig.src.has_resident(l, e) == (e >= 2), "the copy holds exactly the two stages' lent experts");
        }

    // the borrowed experts are served from the RAM copy: the lookup resolves into the copy, the bytes match
    // the file's, and not one read touched the file source
    std::vector<uint8_t> dst(rig.blob);
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 2; e < kExperts; ++e) {
            require(rig.src.blob(l, e) == rig.src.resident_blob(l, e), "a lent expert is not served from the copy");
            require(rig.src.copy_blob(l, e, dst.data()) &&
                        dst == rig.truth[(size_t) l * kExperts + e],
                    "the copy serves wrong bytes for a lent expert");
        }
    require(rig.src.file_reads() == 0 && rig.src.file_read_bytes() == 0,
            "the prompt path's borrowed experts came from the file source, not the RAM copy");
}

// a --resident-budget-gib smaller than the regions: the copy keeps the bytes it can - the same budget
// caps the region walk - and the experts past it keep the file fallback that reads the same bytes
void test_split_lend_region_budget() {
    Rig rig("split lend region budget");
    std::string err;

    rig.src.stage_lend_regions({{{0, 3}, {0, 2}}, {{1, 3}, {1, 2}}});
    const uint64_t budget = 2 * (uint64_t) rig.blob;   // room for one stage's region, not both
    require(rig.src.pin_cache_complement(rig.stage0, err, false, rig.stage1_pairs(), -1, 0, budget), err);

    require(rig.src.resident_bytes() == budget, "the budget did not cap the stage lend regions");
    require(rig.src.resident_lent_slots() == 2, "the clamped regions kept the wrong count");
    require(rig.src.has_resident(0, 3) && rig.src.has_resident(0, 2), "the first stage's region was not kept");
    require(!rig.src.has_resident(1, 3) && !rig.src.has_resident(1, 2), "bytes past the budget entered the copy");

    std::vector<uint8_t> dst(rig.blob);
    require(rig.src.copy_blob(0, 3, dst.data()) && dst == rig.truth[3], "wrong bytes from the region inside the cap");
    require(rig.src.file_reads() == 0, "an expert the copy holds was read from the file");
    require(rig.src.copy_blob(1, 3, dst.data()) && dst == rig.truth[kExperts + 3],
            "the expert past the budget lost its file fallback");
    require(rig.src.file_reads() == 0 && rig.src.file_read_bytes() == (uint64_t) rig.blob,
            "the expert past the budget was served by the file source, exactly one blob read");
}

// the #848 shape, with no stage regions set: with the later stage's pairs handed over as
// `additional_gpu_pairs` the copy keeps NOTHING (the complement is empty and the single lend region is off), so
// this is the state that measured 3.7 GB of NVMe reads over one 73K-token prompt on 2026-10-06
void test_split_without_regions_keeps_nothing() {
    Rig rig("split without regions");
    std::string err;

    require(rig.src.pin_cache_complement(rig.stage0, err, false, rig.stage1_pairs(), -1, 0, 0), err);
    require(rig.src.complement_ready(), "the empty complement did not complete");
    require(rig.src.resident_bytes() == 0 && rig.src.resident_lent_slots() == 0,
            "without stage regions the copy should hold no expert at all");
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e)
            require(!rig.src.has_resident(l, e), "an expert neither in a cache's complement region entered the copy");
}

}  // namespace

int main() {
    try {
        test_split_lend_region_served_from_ram();
        test_split_lend_region_budget();
        test_split_without_regions_keeps_nothing();
        std::cout << "split_lend_complement_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "split_lend_complement_test: " << error.what() << '\n';
        return 1;
    }
}
