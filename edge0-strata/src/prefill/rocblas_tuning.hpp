// src/prefill/rocblas_tuning.hpp - the rocBLAS solution table of the prompt path's FP16-out GEMMs (HIP, gfx103x).
//
// rocBLAS picks a kernel per GEMM shape from its own table, and on gfx1030 that pick is up to 6x slower than the best
// kernel the library holds for the same shape (N=10240 K=2560 below T=1152: tools/hip/tune_rocblas.cpp).  A table
// made by that tool names, per shape and token bucket, the solution index to run instead; Gemm::f16_inplace reads it
// through STRATA_ROCBLAS_TUNING.  Indices are valid for one architecture and one rocBLAS build (its full version
// string), and the table carries both; any other pair is refused and the default kernels run.
#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::prefill::rocblas_tuning {

struct Row {
    int n = 0;
    int k = 0;
    int ldy = 0;
    int t_bucket = 0;
    bool tuned = false;   // false: the default kernel measured best at this bucket (the row bounds its neighbours' reach)
    int solution = 0;     // rocBLAS solution indices can be negative
};

class Table {
public:
    /// `expected_version`: rocblas_get_version_string() of the library the engine runs with.
    bool load(const std::string& path, const std::string& expected_arch, const std::string& expected_version,
              std::string& err) {
        std::ifstream input(path);
        if (!input) {
            err = "cannot open rocBLAS tuning file";
            return false;
        }
        bool saw_header = false;
        std::string line;
        size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            std::istringstream row(line.substr(first));
            if (!saw_header) {
                std::string magic, arch, version, extra;
                if (!(row >> magic >> arch >> version) || (row >> extra) || magic != "STRATA_ROCBLAS_TUNING_V1") {
                    err = "invalid rocBLAS tuning header at line " + std::to_string(line_number);
                    return false;
                }
                if (arch != expected_arch) {
                    err = "rocBLAS tuning architecture mismatch: file=" + arch + " runtime=" + expected_arch;
                    return false;
                }
                if (version != expected_version) {
                    err = "rocBLAS version mismatch: file=" + version + " runtime=" + expected_version;
                    return false;
                }
                arch_ = std::move(arch);
                version_ = std::move(version);
                saw_header = true;
                continue;
            }
            std::string type, solution, extra;
            Row parsed;
            if (!(row >> type >> parsed.n >> parsed.k >> parsed.ldy >> parsed.t_bucket >> solution) || (row >> extra) ||
                type != "f16o" || parsed.n <= 0 || parsed.k <= 0 || parsed.ldy < parsed.n || parsed.t_bucket <= 0) {
                err = "invalid rocBLAS tuning row at line " + std::to_string(line_number);
                return false;
            }
            if (solution != "default") {
                char* end = nullptr;
                const long value = std::strtol(solution.c_str(), &end, 10);
                if (!end || *end || value < INT_MIN || value > INT_MAX) {
                    err = "invalid rocBLAS solution index at line " + std::to_string(line_number);
                    return false;
                }
                parsed.tuned = true;
                parsed.solution = (int) value;
            }
            for (const auto& existing : rows_) {
                if (existing.n == parsed.n && existing.k == parsed.k && existing.ldy == parsed.ldy &&
                    existing.t_bucket == parsed.t_bucket) {
                    err = "duplicate rocBLAS tuning key at line " + std::to_string(line_number);
                    return false;
                }
            }
            rows_.push_back(parsed);
        }
        if (!saw_header) {
            err = "rocBLAS tuning file has no header";
            return false;
        }
        return true;
    }

    /// The row of this shape whose token bucket is nearest to t (the smaller bucket on a tie); null when the shape
    /// has no row at all (never measured).  A `default` row says the default kernel measured best there, so a tuned
    /// neighbour's solution reaches no further than halfway to it.
    const Row* closest(int n, int k, int ldy, int t) const {
        const Row* best = nullptr;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (const auto& row : rows_) {
            if (row.n != n || row.k != k || row.ldy != ldy) continue;
            const int64_t distance = row.t_bucket >= t ? int64_t(row.t_bucket) - t : int64_t(t) - row.t_bucket;
            if (distance < best_distance || (distance == best_distance && row.t_bucket < best->t_bucket)) {
                best = &row;
                best_distance = distance;
            }
        }
        return best;
    }

    const std::vector<Row>& rows() const { return rows_; }
    const std::string& arch() const { return arch_; }
    const std::string& version() const { return version_; }

private:
    std::string arch_;
    std::string version_;
    std::vector<Row> rows_;
};

}  // namespace strata::prefill::rocblas_tuning
