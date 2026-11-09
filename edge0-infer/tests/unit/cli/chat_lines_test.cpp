#include "tools/cli/chat_line.h"
#include "tests/support/check.h"
void test_chat_lines() {
    cli::ChatLines lines;
    CHECK(lines.feed(" \"\"\"\r").first == "empty" && lines.collecting());
    CHECK(lines.feed(" first\r").first == "empty");
    CHECK(lines.feed("/quit").first == "empty");
    CHECK(lines.feed("").first == "empty");
    CHECK(lines.feed("\"\"\"").second == " first\n/quit\n");
    CHECK(!lines.collecting());
    lines.feed("\"\"\""); CHECK(lines.feed("\"\"\"").first == "empty");
    lines.feed("\"\"\""); lines.feed("discard"); lines.finish();
    CHECK(!lines.collecting() && lines.feed("new").second == "new");
    CHECK(lines.feed(" /file  a b.txt \r").second == "a b.txt");
    CHECK(lines.feed("/file").first == "/file");
    CHECK(lines.feed("/filename").first == "unknown");
}
