// tools/vision/gpu_arch.h - which GPU the HIP picture reader (strata-vision-hip) was compiled for, and whether the GPU it
// found is one of them.  ggml-hip prints "  Device 0: AMD Radeon RX 9070 XT, gfx1201:sramecc-:xnack- (0x1201), ..." when it
// starts; a card whose architecture is not in the build would fail at its first kernel ("no kernel image", an abort), so
// the reader reads that line and refuses before any kernel runs.  The server then starts the CPU reader (see
// serve/server.py, Vision._start).  Header only, so gpu_arch_test.cpp can check it without a GPU.
#pragma once

#include <cctype>
#include <string>
#include <vector>

// "gfx1201" out of ggml's device line (or a gcnArchName: "gfx1201:sramecc-:xnack-"); "" when there is none
inline std::string strata_gfx_in(const std::string& text) {
    for (size_t at = text.find("gfx"); at != std::string::npos; at = text.find("gfx", at + 1)) {
        size_t end = at + 3;
        while (end < text.size() && std::isxdigit((unsigned char) text[end])) ++end;
        if (end > at + 3) return text.substr(at, end - at);
    }
    return "";
}

// "gfx1200,gfx1201" (the build's list; ',' or ';' between) holds `arch`
inline bool strata_arch_listed(const std::string& list, const std::string& arch) {
    if (arch.empty()) return false;
    size_t start = 0;
    while (start <= list.size()) {
        size_t end = list.find_first_of(",; ", start);
        if (end == std::string::npos) end = list.size();
        if (list.compare(start, end - start, arch) == 0 && end - start == arch.size()) return true;
        start = end + 1;
    }
    return false;
}
