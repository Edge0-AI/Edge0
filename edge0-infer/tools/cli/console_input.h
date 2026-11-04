#pragma once
#include <string>

namespace cli {
// One line of interactive input as UTF-8 (no trailing CR/LF). False on EOF/error.
// On Windows, when stdin is a real console, uses ReadConsoleW so IME text is not
// corrupted by the CRT narrow path (SetConsoleCP(UTF-8) alone is not enough).
// A console line is accumulated until the newline. Cooked input drops characters
// past the pending read, so each read is large enough for a long paste. A line
// that starts with Ctrl+Z (U+001A) is end of input, matching getline.
bool read_input_line(std::string& line);
}
