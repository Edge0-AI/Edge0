#include "edge0/generation.h"
#include "console_input.h"
namespace cli {
const char* sampling_help();
bool sampling_option(const std::string&, const std::string&, edge0::GenerationOverrides&);
}
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
// Console and stdio setup has to run in the process itself; check_layers allows
// these headers in this file only.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif
int serve_command(int argc, char **argv);
int chat_command(int argc, char **argv);
int bench_command(int argc, char **argv);
int describe_command(int argc, char **argv);
int run_command(int argc, char **argv);
int forward_command(int argc, char **argv);
int info_command(int argc, char **argv);
int inspect_command(int argc, char **argv);
int text_command(int argc, char **argv);

namespace cli {
bool read_input_line(std::string& line) {
#ifdef _WIN32
    const HANDLE console = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    // GetConsoleMode succeeds only for a real console; pipes keep getline so tests
    // that feed stdin still work.
    if (console != INVALID_HANDLE_VALUE && GetConsoleMode(console, &mode)) {
        // Cooked input accepts at most this many UTF-16 units, minus CR/LF, and
        // drops the rest instead of leaving it for a later read. A 1024-unit read
        // therefore kept only 1022 characters of a longer paste. Ask for a large
        // chunk and keep reading until the newline.
        constexpr DWORD kChunk = 16384;
        std::wstring wide;
        wide.reserve(kChunk);
        std::vector<wchar_t> chunk(kChunk);
        for (;;) {
            DWORD count = 0;
            if (!ReadConsoleW(console, chunk.data(), kChunk, &count, nullptr)) return false;
            if (count == 0) return false;
            wide.append(chunk.data(), chunk.data() + count);
            const auto newline = wide.find(L'\n');
            if (newline != std::wstring::npos) {
                wide.resize(newline);
                break;
            }
        }
        while (!wide.empty() && wide.back() == L'\r') wide.pop_back();
        // getline treated a leading Ctrl+Z as EOF. ReadConsoleW returns it as U+001A.
        if (!wide.empty() && wide.front() == L'\x1A') return false;
        if (wide.empty()) {
            line.clear();
            return true;
        }
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                                              nullptr, 0, nullptr, nullptr);
        if (bytes <= 0) return false;
        line.resize(static_cast<std::size_t>(bytes));
        WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                            line.data(), bytes, nullptr, nullptr);
        return true;
    }
#endif
    return static_cast<bool>(std::getline(std::cin, line));
}
}

int main(int argc, char **argv) {
#ifdef _WIN32
    // Switch the console to UTF-8 so the engine's UTF-8 output renders correctly;
    // interactive chat input uses ReadConsoleW (see read_input_line) because the
    // CRT narrow stdin path still corrupts IME text under CP_UTF8. The manifest
    // (edge0-infer.manifest) covers argv and narrow Win32 APIs.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Write stdout in binary mode so a bare '\n' stays one byte instead of the
    // CRLF the CRT's text mode would emit. The reference answers and the exact
    // byte comparisons in the tests require identical output on every platform
    // (decision 2); the console renders a lone '\n' correctly.
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    struct Command {
        const char *name;
        int (*run)(int, char **);
    };
    const Command commands[] = {{"serve", serve_command}, {"chat", chat_command}, {"bench", bench_command}, {"describe", describe_command}, {"run", run_command}, {"forward", forward_command}, {"info", info_command}, {"inspect", inspect_command}, {"tokenize", text_command}, {"prompt", text_command}};
    if (argc >= 2)
        for (const auto &command : commands)
            if (argv[1] == std::string(command.name))
                return command.run(argc, argv);
    std::cerr << "Usage: edge0-infer <command> [options]\n"
                 "Commands: info, describe, inspect, tokenize, prompt, run, chat, serve, bench, forward\n"
                 "Run edge0-infer <command> --help for the options of a command.\n";
    return 2;
}
