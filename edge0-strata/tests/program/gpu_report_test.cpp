// tests/program/gpu_report_test.cpp - #3797: the out-of-memory report at start (pure text; the NVML query is only
// checked to be safe to call: it must return, not throw, and leave the list empty when it answers false).
#include "strata/program/gpu_report.hpp"

#include <cstdio>
#include <string>

using namespace strata::program::gpu_report;

namespace {
int g_fail = 0;
void check(bool ok, const char* what, const std::string& got = {}) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s%s%s\n", what, got.empty() ? "" : "\n  got: ", got.c_str());
    ++g_fail;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    {   // the case of #3797: 320 MiB free of a 22 GB card, another program holds the rest
        Card c;
        c.index = 1;
        c.name = "NVIDIA GeForce RTX 2080 Ti";
        c.free_mib = 320;
        c.total_mib = 21998;
        c.processes_known = true;
        c.processes = {{4242, "/usr/bin/python3.12", 20100}, {777, "/opt/strata/strata", 900}, {1, "Xorg", 12}, {5, "", 0}};
        const std::string r = format_gpu_memory_report({c}, 777);
        check(has(r, "CUDA1 (NVIDIA GeForce RTX 2080 Ti): 320 MiB free of 21998 MiB"), "free and total", r);
        check(has(r, "python3.12 (pid 4242) 19.6 GiB"), "the largest other program, by file name", r);
        check(!has(r, "strata (pid 777)"), "this process is not listed as another program", r);
        check(has(r, "Xorg (pid 1) 12 MiB"), "a small one is still named", r);
        check(has(r, "1 program whose memory the driver does not report"), "a process of unknown size is counted", r);
        check(r.find("python3.12") < r.find("Xorg"), "largest first", r);
    }
    {   // more than three: the rest is counted
        Card c;
        c.index = 0;
        c.free_mib = 100;
        c.total_mib = 12288;
        c.processes_known = true;
        for (uint32_t i = 1; i <= 6; ++i) c.processes.push_back({i, "p" + std::to_string(i), 100u * i});
        const std::string r = format_gpu_memory_report({c}, 0);
        check(has(r, "p6 (pid 6) 600 MiB, p5 (pid 5) 500 MiB, p4 (pid 4) 400 MiB, and 3 more"), "three named, three counted", r);
        check(!has(r, "p3 (pid 3)"), "the fourth is not named", r);
    }
    {   // the driver would not say: no claim about other programs
        Card c;
        c.index = 0;
        c.free_mib = 512;
        c.total_mib = 8192;
        c.processes_known = false;
        const std::string r = format_gpu_memory_report({c}, 1);
        check(has(r, "512 MiB free of 8192 MiB; the driver does not say which programs use it here"), "unknown processes", r);
        check(!has(r, "other programs on it"), "no list", r);
    }
    {   // the driver answered with nothing but ourselves: said carefully (Windows often lists no program at all)
        Card c;
        c.index = 0;
        c.free_mib = 700;
        c.total_mib = 12288;
        c.processes_known = true;
        c.processes = {{99, "strata.exe", 11000}};
        const std::string r = format_gpu_memory_report({c}, 99);
        check(has(r, "the driver lists no other program using it"), "only ourselves", r);
    }
    {   // two cards, a pid with no name, a Windows path
        Card a, b;
        a.index = 0; a.free_mib = 2000; a.total_mib = 16000; a.processes_known = true;
        b.index = 1; b.free_mib = 50; b.total_mib = 12000; b.processes_known = true;
        b.processes = {{31, "", 11000}, {32, "C:\\Program Files\\Obs\\obs64.exe", 300}};
        const std::string r = format_gpu_memory_report({a, b}, 5);
        check(has(r, "CUDA0: 2000 MiB free of 16000 MiB; the driver lists no other program using it\n"), "first card line", r);
        check(has(r, "CUDA1: 50 MiB free of 12000 MiB; other programs on it: pid 31 10.7 GiB, obs64.exe (pid 32) 300 MiB"), "second card line", r);
    }
    check(format_gpu_memory_report({}, 1).empty(), "no cards, no text");
    {   // planned versus free
        check(planned_vs_free_text(796, 68) ==
                  "the layer split planned to keep 796 MiB free here after loading; 68 MiB are free, so 728 MiB went to something the plan did not count",
              "planned versus free", planned_vs_free_text(796, 68));
        check(planned_vs_free_text(796, 796).empty() && planned_vs_free_text(796, 900).empty(), "enough free: nothing to say");
        check(planned_vs_free_text(0, 10).empty() && planned_vs_free_text(-5, 10).empty(), "no plan, no text");
        check(has(planned_vs_free_text(300, -4), "0 MiB are free, so 300 MiB"), "a negative free count is read as 0");
    }
    {   // the query never throws and never leaves a half-filled list behind a false
        std::vector<Process> p{{1, "x", 1}};
        const bool ok = query_other_processes("", p);
        check(!ok && p.empty(), "an empty bus id is refused");
        std::vector<Process> q{{1, "x", 1}};
        const bool r = query_other_processes("ffff:ff:ff.f", q);   // no such card: false, or true with a list; both are fine
        check(r || q.empty(), "a failed query leaves the list empty");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
