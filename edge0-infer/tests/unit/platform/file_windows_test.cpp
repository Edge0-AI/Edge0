#include "platform/file.h"
#include "tests/support/check.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;
using edge0::platform::ReadFile;
using edge0::platform::ReadMode;

// Value at position i, so any slice read back is checkable against its offset.
unsigned char pattern_at(std::uint64_t i) {
    return static_cast<unsigned char>((i * 131u + 7u) & 0xFFu);
}

void write_pattern(const fs::path &path, std::size_t size) {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> buffer(size);
    for (std::size_t i = 0; i < size; ++i) buffer[i] = static_cast<char>(pattern_at(i));
    out.write(buffer.data(), static_cast<std::streamsize>(size));
}
} // namespace

int main() {
    using edge0::platform::split_file_offset;
    // (1) Pure offset split; high word non-zero without a >4 GiB file.
    CHECK(split_file_offset(0).low == 0u && split_file_offset(0).high == 0u);
    CHECK(split_file_offset(0xFFFFFFFFULL).low == 0xFFFFFFFFu && split_file_offset(0xFFFFFFFFULL).high == 0u);
    CHECK(split_file_offset(0x100000000ULL).low == 0u && split_file_offset(0x100000000ULL).high == 1u);
    CHECK(split_file_offset(0x123456789ULL).low == 0x23456789u && split_file_offset(0x123456789ULL).high == 1u);

    // A CJK-named directory exercises the UTF-8 -> UTF-16 widen() path. The name
    // is built from universal-character escapes to keep this source ASCII.
    const auto stamp = std::to_wstring(static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path dir = fs::temp_directory_path() / (std::wstring(L"\u6a21\u578b\u8bfb\u53d6_") + stamp);
    fs::create_directories(dir);
    const fs::path file = dir / L"data.bin";
    constexpr std::size_t size = std::size_t(1) << 20; // 1 MiB
    write_pattern(file, size);
    const std::string utf8 = file.u8string(); // C++17: UTF-8 std::string

    // (2) Open through the UTF-8 (CJK) path and read a slice back.
    auto opened = ReadFile::open(utf8);
    CHECK(bool(opened));
    auto handle = opened.value();
    auto id = handle->identity();
    CHECK(bool(id) && id.value().size == size);
    {
        std::vector<unsigned char> dst(4096);
        auto err = handle->read_exact(1000, dst.data(), dst.size());
        CHECK(!err);
        bool ok = true;
        for (std::size_t i = 0; i < dst.size(); ++i) ok = ok && dst[i] == pattern_at(1000 + i);
        CHECK(ok);
    }

    // (3) Many threads read random slices through the SAME shared handle; the
    // overlapped reads must each land at their own offset without interfering.
    {
        std::atomic<bool> good{true};
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t) {
            workers.emplace_back([&, t]() {
                std::uint64_t seed = 2654435761ull * static_cast<std::uint64_t>(t + 1);
                for (int r = 0; r < 300; ++r) {
                    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                    const std::size_t len = 1 + static_cast<std::size_t>((seed >> 40) % 8192);
                    const std::uint64_t off = (seed >> 16) % (size - len);
                    std::vector<unsigned char> dst(len);
                    if (handle->read_exact(off, dst.data(), len)) { good = false; return; }
                    for (std::size_t i = 0; i < len; ++i)
                        if (dst[i] != pattern_at(off + i)) { good = false; return; }
                }
            });
        }
        for (auto &w : workers) w.join();
        CHECK(good.load());
    }

    // (4) Direct mode is rejected as unsupported on Windows.
    {
        auto direct = ReadFile::open(utf8, ReadMode::direct);
        CHECK(!direct && direct.error().category == edge0::ErrorCategory::unsupported);
    }

    // Record-only: serial vs parallel wall time for the same total read volume
    // (no threshold asserted; printed for evidence 1.3-parallel-read.md).
    {
        constexpr int rounds = 64; // read the whole file this many times total
        std::vector<unsigned char> sink(size);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < rounds; ++r)
            (void)handle->read_exact(0, sink.data(), size);
        const auto t1 = std::chrono::steady_clock::now();
        std::atomic<bool> ok{true};
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t)
            workers.emplace_back([&]() {
                std::vector<unsigned char> local(size);
                for (int r = 0; r < rounds / 8; ++r)
                    if (handle->read_exact(0, local.data(), size)) ok = false;
            });
        for (auto &w : workers) w.join();
        const auto t2 = std::chrono::steady_clock::now();
        CHECK(ok.load());
        const auto serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const auto parallel_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "timing serial_ms=" << serial_ms << " parallel_ms=" << parallel_ms
                  << " total_bytes=" << (static_cast<std::uint64_t>(size) * rounds) << '\n';
    }

    handle.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::cout << "windows file: offset split, CJK path, shared-handle parallel reads, direct rejection passed\n";
}
