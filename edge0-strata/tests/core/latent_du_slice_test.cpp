// Exercise the production uploader and the same global-layer lookup used by prefill and verify.
// No model required. A single device suffices to test a nonzero stage origin.
#include "strata/core/latent_du_upload.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const char* fmt = std::getenv("STRATA_LATENT_DENSE_FMT");
    const size_t bytes = 32 * (fmt && fmt[0] == '4' ? 18 : 34); // one group, 32 x 32 weights
    constexpr int layers = 7;
    const auto dir = fs::temp_directory_path() / ("strata-du-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!fs::create_directory(dir)) return 1;
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{dir};
    const auto path = dir / "dense.bin";
    {
        std::ofstream file(path, std::ios::binary);
        std::vector<char> payload(bytes);
        for (int block = 0; block < 2 * layers; ++block) {
            std::fill(payload.begin(), payload.end(), static_cast<char>(block + 1)); // distinct D/U per layer
            file.write(payload.data(), payload.size());
        }
        file.close();
        if (!file) return 1;
    }
    strata::core::LatentDense pack;
    std::string err;
    if (!pack.open(path.string(), layers, 1, 32, 32, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto check = [&](int device, int first, int end) {
        strata::core::LatentDeviceDU du;
        if (cudaSetDevice(device) != cudaSuccess || !upload_latent_du(pack, first, end, du, err)) return false;
        std::vector<unsigned char> row(bytes);
        bool ok = du.d_at(first, pack) == du.d && du.u_at(first, pack) == du.u;
        for (int l = first; ok && l < end; ++l) {
            for (int u = 0; u < 2; ++u) {
                const auto* data = u ? du.u_at(l, pack) : du.d_at(l, pack);
                ok = cudaMemcpy(row.data(), data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                     std::all_of(row.begin(), row.end(), [=](unsigned char v) { return v == 2 * l + u + 1; });
                if (!ok) break;
            }
        }
        cudaFree(du.d);
        cudaFree(du.u);
        return ok;
    };
    if (!check(0, 0, layers) || !check(0, 0, 2) || !check(devices > 1 ? 1 : 0, 2, 6) || !check(0, 6, 7)) {
        std::fprintf(stderr, "latent D/U range readback failed: %s\n", err.c_str());
        return 1;
    }
    std::puts("PASS: full pair, first stage, nonzero middle stage, one-layer tail");
}
