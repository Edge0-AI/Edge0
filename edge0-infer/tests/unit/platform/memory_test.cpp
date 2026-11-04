#include "platform/memory.h"
#include "tests/support/check.h"
#include <memory>
int main() {
    using edge0::platform::parse_process_memory;
    auto parsed = parse_process_memory("VmRSS: 12 kB\nVmHWM: 18 kB\n", "Pss: 9 kB\n");
    CHECK(parsed.rss.source == "proc_status" && parsed.physical.source == "pss");
    CHECK(parsed.rss.values->current_bytes == 12*1024 && parsed.rss.values->peak_bytes == 18*1024);
    CHECK(parsed.physical.values->current_bytes == 9*1024 && !parsed.physical.values->peak_bytes);
    CHECK(!parsed.physical.values->peak_reason.empty());
    auto unreadable_status = parse_process_memory("", "Pss: 9 kB");
    CHECK(!unreadable_status.rss.values->current_bytes && !unreadable_status.rss.values->peak_bytes);
    CHECK(unreadable_status.rss.values->current_reason == "Cannot read /proc/self/status");
    CHECK(unreadable_status.rss.values->peak_reason == "Cannot read /proc/self/status");
    CHECK(unreadable_status.physical.values->current_bytes == 9*1024);
    auto unreadable_smaps = parse_process_memory("VmRSS: 12 kB\nVmHWM: 18 kB", "");
    CHECK(!unreadable_smaps.physical.values->current_bytes);
    CHECK(unreadable_smaps.physical.values->current_reason == "Cannot read /proc/self/smaps_rollup");
    CHECK(unreadable_smaps.rss.values->current_bytes == 12*1024);
    CHECK(unreadable_smaps.rss.values->peak_bytes == 18*1024);
    auto absent_fields = parse_process_memory("Name: test", "Rss: 9 kB");
    CHECK(absent_fields.rss.values->current_reason == "Missing field: VmRSS");
    CHECK(absent_fields.rss.values->peak_reason == "Missing field: VmHWM");
    CHECK(absent_fields.physical.values->current_reason == "Missing field: Pss");
    for (const auto& text : {"", "VmRSS: -1 kB", "VmRSS: 1 MB", "VmRSS: 1 kB junk", "VmRSS: 99999999999999999999999 kB", "VmRSS: 1 kB\nVmRSS: 2 kB"}) {
        auto missing = parse_process_memory(text, "Pss: broken");
        CHECK(!missing.rss.values->current_bytes && !missing.rss.values->current_reason.empty());
        CHECK(!missing.physical.values->current_bytes && !missing.physical.values->current_reason.empty());
    }
    auto current = parse_process_memory("VmRSS: 0 kB", "");
    CHECK(current.rss.values->current_bytes == 0 && !current.rss.values->peak_bytes);
    auto peak = parse_process_memory("VmHWM: 20 kB", "");
    CHECK(!peak.rss.values->current_bytes && peak.rss.values->peak_bytes == 20*1024);
    const auto before=edge0::platform::process_memory();
#if defined(__APPLE__)
    CHECK(before.rss.values && before.physical.values);
    CHECK(before.physical.source == "phys_footprint");
    for(const auto& metric:{before.rss,before.physical}) {
        CHECK(metric.values->current_bytes>0);
        CHECK(metric.values->peak_bytes>=metric.values->current_bytes);
        CHECK(metric.reason.empty());
    }
    constexpr std::size_t size=64*1024*1024;
    auto allocation=std::make_unique<unsigned char[]>(size);
    volatile unsigned char* bytes=allocation.get();
    for(std::size_t i=0;i<size;i+=4096) bytes[i]=static_cast<unsigned char>(i/4096);
    const auto after=edge0::platform::process_memory();
    CHECK(after.rss.values && after.physical.values);
    CHECK(after.rss.values->current_bytes>=*before.rss.values->current_bytes+size/2);
    CHECK(after.rss.values->peak_bytes>=after.rss.values->current_bytes);
    CHECK(after.physical.values->peak_bytes>=after.physical.values->current_bytes);
#elif defined(_WIN32)
    CHECK(before.rss.values && before.physical.values);
    CHECK(before.rss.source == "working_set" && before.physical.source == "private_usage");
    CHECK(before.rss.values->current_bytes > 0);
    CHECK(before.rss.values->peak_bytes >= before.rss.values->current_bytes);
    CHECK(before.physical.values->current_bytes > 0);
    // Windows exposes no private-bytes peak counter, so the physical peak is
    // absent and must carry a reason.
    CHECK(!before.physical.values->peak_bytes && !before.physical.values->peak_reason.empty());
    CHECK(before.rss.reason.empty() && before.physical.reason.empty());
    constexpr std::size_t size=64*1024*1024;
    auto allocation=std::make_unique<unsigned char[]>(size);
    volatile unsigned char* bytes=allocation.get();
    for(std::size_t i=0;i<size;i+=4096) bytes[i]=static_cast<unsigned char>(i/4096);
    const auto after=edge0::platform::process_memory();
    CHECK(after.rss.values && after.physical.values);
    CHECK(after.rss.values->current_bytes >= before.rss.values->current_bytes);
    CHECK(after.rss.values->peak_bytes >= after.rss.values->current_bytes);
    CHECK(after.physical.values->current_bytes >= before.physical.values->current_bytes);
    CHECK(!after.physical.values->peak_bytes && !after.physical.values->peak_reason.empty());
#else
    CHECK(!before.rss.values && !before.rss.reason.empty());
    CHECK(!before.physical.values && !before.physical.reason.empty());
#endif
    std::cout<<"process memory availability, distinct metrics and touched allocation passed\n";
}
