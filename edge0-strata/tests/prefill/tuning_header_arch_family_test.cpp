// TuningTable::load header gate, synthetic tables: no model, no GPU.  gfx1200 and gfx1201 are the same RDNA4
// (gfx12) family and share hipBLASLt solution IDs, so a table calibrated on one must load on the other: a card
// reporting the sibling architecture - an R9700 under HSA_OVERRIDE_GFX_VERSION=12.0.0 reports gfx1200 - otherwise
// rejects its own calibration and loses the tuned prefill route entirely.  The per-solution runtime gate
// (getAlgosFromIndex + matmulIsAlgoSupported, hipBLASEx fallback) is what keeps a wrong solution ID from running,
// so the header gate can be family-wide over the SIBLING SETS the gate names: RDNA4 (gfx1200/gfx1201) and RDNA3
// (gfx1100/gfx1101/gfx1102).  Every other pair stays rejected, including architectures that merely share a
// prefix - gfx1010 (RDNA1) and gfx1030 (RDNA2) are not siblings and must not pass a table to each other.
#include "hipblaslt_tuning.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

using strata::prefill::hipblaslt::TuningTable;

namespace {

// A portable scratch directory.  This test is host-only but builds on Windows/MSVC too, so it cannot use
// <unistd.h>/getpid() or a hard-coded /tmp: std::filesystem::temp_directory_path plus a steady-clock token is
// the shape tests/core/gguf_split_test.cpp uses.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-tuning-header-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

bool load(const fs::path& dir, const std::string& file_arch, int file_version, const std::string& runtime_arch,
          int runtime_version, std::string& err, size_t& rows) {
    const fs::path path = dir / ("case-" + file_arch + "-" + std::to_string(file_version) + "-" + runtime_arch +
                                 "-" + std::to_string(runtime_version) + ".txt");
    {
        std::ofstream out(path);
        out << "# solution IDs are scoped to this hipBLASLt version and device architecture\n"
            << "STRATA_HIPBLASLT_TUNING_V1 " << file_arch << " " << file_version << "\n"
            << "bf16 1 2560 1 4096 110508\n"
            << "bf16 2560 320 2560 8192 135699\n";
        if (!out) {   // otherwise a write failure reads as a gate rejection and the case "passes" for the wrong reason
            err = "cannot write the synthetic table at " + path.string();
            rows = 0;
            return false;
        }
    }
    TuningTable table;
    const bool ok = table.load(path.string(), runtime_arch, runtime_version, err);
    rows = table.rows().size();
    std::error_code ignored;
    fs::remove(path, ignored);
    return ok;
}

struct Expect {
    const char* file_arch;
    int file_version;
    const char* runtime_arch;
    int runtime_version;
    bool accept;
    const char* why;
};

}  // namespace

int main() {
    TempDir dir;
    const Expect cases[] = {
        {"gfx1201", 100500, "gfx1201", 100500, true, "exact match"},
        {"gfx1200", 100500, "gfx1200", 100500, true, "exact match"},
        {"gfx1201", 100500, "gfx1200", 100500, true, "RDNA4 sibling: gfx1201 table on a card reporting gfx1200"},
        {"gfx1200", 100500, "gfx1201", 100500, true, "RDNA4 sibling, symmetric direction"},
        {"gfx1201", 100202, "gfx1201", 100500, false, "the version gate rejects on its own"},
        {"gfx1200", 100202, "gfx1201", 100500, false, "cross-family and wrong version"},
        {"gfx1100", 100202, "gfx1201", 100202, false, "RDNA3 table on RDNA4"},
        {"gfx1201", 100202, "gfx1100", 100202, false, "RDNA4 table on RDNA3"},
        {"gfx1100", 100202, "gfx1101", 100202, true, "RDNA3 siblings share the family rule"},
        {"gfx1102", 100202, "gfx1101", 100202, true, "RDNA3 siblings, the third member of the set"},
        {"gfx1030", 100202, "gfx1201", 100202, false, "RDNA2 table on RDNA4"},
        // The pair a five-character prefix match would wrongly group: both are "gfx10", neither is a sibling.
        {"gfx1010", 100202, "gfx1030", 100202, false, "RDNA1 table on RDNA2: a shared prefix is not a family"},
        {"gfx1030", 100202, "gfx1010", 100202, false, "RDNA2 table on RDNA1, symmetric direction"},
    };
    int failures = 0;
    for (const auto& c : cases) {
        std::string err;
        size_t rows = 0;
        const bool ok = load(dir.path, c.file_arch, c.file_version, c.runtime_arch, c.runtime_version, err, rows);
        const bool good = ok == c.accept && (!ok || rows == 2);
        if (!good) ++failures;
        std::printf("%s %s/%d vs %s/%d rows=%zu  %s\n", good ? "ok  " : "FAIL", c.file_arch, c.file_version,
                    c.runtime_arch, c.runtime_version, rows, c.why);
        if (!ok) std::printf("      err: %s\n", err.c_str());
    }
    const int total = int(sizeof(cases) / sizeof(cases[0]));
    std::printf("tuning_header_arch_family_test: %d cases, %d failures\n", total, failures);
    return failures == 0 ? 0 : 1;
}