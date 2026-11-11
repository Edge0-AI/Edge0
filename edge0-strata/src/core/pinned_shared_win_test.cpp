// Windows selftest for the named-section backing of PinnedArena (the replicas' shared expert arena).
// Parent: two arenas on one section name (same bytes, header refused for another pack / a bigger size, the load state word),
// then a CHILD PROCESS that opens the same section, reads what the parent wrote and writes back; the parent closes its first
// arena while the second still holds the section (the pages must stay), and a third arena joins the live section.  After the
// last close, a new arena with the same name starts empty (the section is gone with its last handle).
#ifdef _WIN32
#include "strata/core/pinned.hpp"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kBytes = 64u << 10;
constexpr uint64_t kPackHash = 0x0123456789abcdefull;
}  // namespace

int main(int argc, char** argv) {
    const std::vector<uint64_t> no_bounds;
    if (argc >= 3 && std::strcmp(argv[1], "child") == 0) {
        const std::string name = argv[2];
        strata::core::PinnedArena a(kBytes, no_bounds, 0, name, kPackHash);
        if (!a.valid()) { std::fprintf(stderr, "child: cannot open: %s\n", a.note.c_str()); return 2; }
        if (a.data()[123] != 0x5a) { std::fprintf(stderr, "child: did not see the parent's byte\n"); return 3; }
        if (a.shared_state == nullptr || a.shared_state->load() != 4242u) { std::fprintf(stderr, "child: state word is not the parent's\n"); return 4; }
        a.data()[456] = 0xa5;
        a.shared_state->store(~uint64_t{0});     // (filled)
        return 0;
    }

    int fail = 0;
    const std::string name = "win:Local\\strata-pinned-test-" + std::to_string((unsigned long long) GetCurrentProcessId());
    {
        auto* first = new strata::core::PinnedArena(kBytes, no_bounds, 0, name, kPackHash);
        if (!first->valid()) {
            std::fprintf(stderr, "first section failed: %s\n", first->note.c_str());
            return 1;
        }
        first->data()[123] = 0x5a;
        if (first->shared_state == nullptr) { std::fprintf(stderr, "no state word\n"); fail = 1; }
        else if (first->shared_state->load() != 0) { std::fprintf(stderr, "a new section's state word is not 0\n"); fail = 1; }
        else first->shared_state->store(4242u);

        auto* second = new strata::core::PinnedArena(kBytes, no_bounds, 0, name, kPackHash);
        if (!second->valid()) { std::fprintf(stderr, "second section failed: %s\n", second->note.c_str()); fail = 1; }
        else {
            if (second->data()[123] != 0x5a) { std::fprintf(stderr, "same name did not expose the same bytes\n"); fail = 1; }
            if (second->shared_state == nullptr || second->shared_state->load() != 4242u) { std::fprintf(stderr, "state word not shared\n"); fail = 1; }
        }
        {
            strata::core::PinnedArena wrong_pack(kBytes, no_bounds, 0, name, kPackHash + 1);
            if (wrong_pack.valid() || wrong_pack.note.find("pack hash") == std::string::npos) {
                std::fprintf(stderr, "pack mismatch was not refused: %s\n", wrong_pack.note.c_str());
                fail = 1;
            }
            strata::core::PinnedArena wrong_size(kBytes * 2, no_bounds, 0, name, kPackHash);
            if (wrong_size.valid()) { std::fprintf(stderr, "a bigger size was not refused\n"); fail = 1; }
        }

        // a child process opens the live section
        char exe[MAX_PATH];
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        std::string cmd = std::string("\"") + exe + "\" child \"" + name + "\"";
        STARTUPINFOA si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        std::vector<char> cl(cmd.begin(), cmd.end());
        cl.push_back('\0');
        if (!CreateProcessA(nullptr, cl.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            std::fprintf(stderr, "cannot start the child (error %lu)\n", GetLastError());
            fail = 1;
        } else {
            WaitForSingleObject(pi.hProcess, 120000);
            DWORD code = 99;
            GetExitCodeProcess(pi.hProcess, &code);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            if (code != 0) { std::fprintf(stderr, "the child failed with %lu\n", code); fail = 1; }
            if (first->data()[456] != 0xa5) { std::fprintf(stderr, "the child's byte did not arrive\n"); fail = 1; }
            if (first->shared_state->load() != ~uint64_t{0}) { std::fprintf(stderr, "the child's state word did not arrive\n"); fail = 1; }
        }

        delete first;                           // the creator is gone; the second still holds the section
        if (second != nullptr && second->valid()) {
            if (second->data()[123] != 0x5a || second->data()[456] != 0xa5) { std::fprintf(stderr, "bytes lost when the first arena closed\n"); fail = 1; }
            strata::core::PinnedArena third(kBytes, no_bounds, 0, name, kPackHash);
            if (!third.valid() || third.data()[123] != 0x5a) { std::fprintf(stderr, "a late arena did not join the live section\n"); fail = 1; }
        }
        delete second;
    }
    {
        strata::core::PinnedArena again(kBytes, no_bounds, 0, name, kPackHash);   // (every handle was closed: a new, empty section)
        if (!again.valid() || again.data()[123] != 0 || again.shared_state->load() != 0) {
            std::fprintf(stderr, "the section survived its last handle\n");
            fail = 1;
        }
    }
    std::printf("pinned_shared_win_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
#else
int main() { return 77; }   // (skipped: the Linux build has pinned_shared_test)
#endif
