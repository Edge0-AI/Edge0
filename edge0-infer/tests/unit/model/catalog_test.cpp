#include "tests/support/model_fixture.h"
int main() {
    using namespace fixture;
    using edge0::ErrorCategory;
    const std::pair<std::string, std::string> vectors[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmno"
         "pqrstnopqrstu",
         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"}};
    for (const auto &v : vectors) {
        Sha256 h;
        h.update(v.first);
        CHECK(h.finish() == v.second);
    }
    std::string data(1 << 20, 'a');
    Sha256 one, parts;
    one.update(data);
    for (std::size_t i = 0; i < data.size(); i += 117)
        parts.update(data.data() + i, std::min<std::size_t>(117, data.size() - i));
    CHECK(one.finish() == parts.finish());
    Temp t;
    auto setup = [&]() {
        write(t.path / "model.json",
              "{\"format_version\":1,\"architecture\":\"test\",\"other\":{\"bool\":true}}");
        write(t.path / "base.gguf", "abc");
        catalog(t.path, {{"base", "base.gguf"}});
    };
    setup();
    CHECK(open_catalog(t.path.string()));
    auto digest = open_catalog(t.path.string(), Integrity::digest);
    CHECK(digest);
    CHECK(digest.value().files.at("base").digest_verified);
    auto bad = [&](const std::string &name) {
        error(open_catalog(t.path.string()), ErrorCategory::asset_invalid, name);
    };
    {
        // Restricted to known architectures: a foreign one is refused (unsupported, naming the known
        // ones) before assets.tsv is read; a known one opens as before.
        const std::vector<std::string> known{"test"}, foreign{"edge0_35b", "other"};
        CHECK(open_catalog(t.path.string(), Integrity::size_only, &known));
        fs::remove(t.path / "assets.tsv");
        error(open_catalog(t.path.string(), Integrity::size_only, &foreign), ErrorCategory::unsupported,
              "architecture test is not compiled into this library (compiled: edge0_35b, other)");
        error(open_catalog(t.path.string(), Integrity::size_only, &foreign), ErrorCategory::unsupported, "edge0-infer serve");
        bad("assets.tsv");
    }
    setup();
    write(t.path / "model.json",
          R"({"format_version":1,"architecture":"test","ignored_note":"\ud83d\ude00"})");
    CHECK(open_catalog(t.path.string()));
    write(t.path / "model.json", R"({"format_version":1,"architecture":"\ud83d\ude00"})");
    CHECK(open_catalog(t.path.string()).value().architecture == "\xf0\x9f\x98\x80");
    for (const auto *escape : {"\\ud83d", "\\ude00", "\\ud83d\\u0041", "\\ud83d\\ud800"}) {
        write(t.path / "model.json",
              std::string("{\"format_version\":1,\"architecture\":\"") + escape + "\"}");
        bad("model.json");
    }
    setup();
    const std::string valid_json = R"({"format_version":1,"architecture":"test"})";
    write(t.path / "model.json", " \t\r\n" + valid_json + " \t\r\n");
    CHECK(open_catalog(t.path.string()));
    write(t.path / "model.json", "{ \t\r\n\"format_version\" \t\r\n:1,\"architecture\":\"test\"}");
    CHECK(open_catalog(t.path.string()));
    for (char whitespace : {'\v', '\f'}) {
        for (const auto &invalid : {std::string(1, whitespace) + valid_json, valid_json + whitespace,
                                    std::string("{") + whitespace + valid_json.substr(1)}) {
            write(t.path / "model.json", invalid);
            bad("model.json");
        }
    }
    const std::vector<std::string> unicode = {"\xc2\xa2", "\xe2\x82\xac", "\xf0\x9f\x98\x80",
                                              "\xf4\x8f\xbf\xbf"};
    for (const auto &raw : unicode) {
        write(t.path / "model.json", "{\"format_version\":1,\"architecture\":\"" + raw + "\",\"" + raw +
                                         "\":{\"ignored\":\"" + raw + "\"}}");
        CHECK(open_catalog(t.path.string()).value().architecture == raw);
    }
    write(t.path / "model.json",
          R"({"format_version":1,"architecture":"\b\f\n\r\t\u0000","ignored":"\ud83d\ude00"})");
    CHECK(open_catalog(t.path.string()).value().architecture == std::string("\b\f\n\r\t\0", 6));
    const std::vector<std::string> invalid_utf8 = {"\xff",
                                                   "\x80",
                                                   "\xc0\x80",
                                                   "\xe0\x80\x80",
                                                   "\xf0\x80\x80\x80",
                                                   "\xed\xa0\x80",
                                                   "\xf4\x90\x80\x80",
                                                   "\xf5\x80\x80\x80",
                                                   "\xe2\x82",
                                                   "\xc2\x41"};
    for (const auto &raw : invalid_utf8) {
        for (const auto &invalid :
             {"{\"format_version\":1,\"architecture\":\"" + raw + "\"}",
              "{\"format_version\":1,\"architecture\":\"test\",\"ignored\":\"" + raw + "\"}",
              "{\"format_version\":1,\"architecture\":\"test\",\"" + raw + "\":true}",
              "{\"format_version\":1,\"architecture\":\"test\",\"nested\":{\"ignored\":\"" + raw + "\"}}",
              valid_json + raw}) {
            write(t.path / "model.json", invalid);
            bad("model.json");
        }
    }
    setup();
    fs::remove(t.path / "assets.tsv");
    bad("assets.tsv");
    setup();
    fs::remove(t.path / "model.json");
    bad("model.json");
    setup();
    for (const auto *json :
         {"{\"format_version\":2,\"architecture\":\"x\"}", "{\"format_version\":1,\"architecture\":\"\"}",
          "[]", "{\"format_version\":01,\"architecture\":\"x\"}"}) {
        write(t.path / "model.json", json);
        bad("model.json");
    }
    setup();
    auto original = read(t.path / "assets.tsv");
    for (const auto &table : {std::string("bad\n"), std::string("role\tbytes\tsha256\tfilename\nwrong\n"),
                              original + original.substr(original.find('\n') + 1)}) {
        write(t.path / "assets.tsv", table);
        bad("assets.tsv");
    }
    for (const auto *name : {"a/b", "..", "a\\b", "/absolute"}) {
        write(t.path / "assets.tsv",
              "role\tbytes\tsha256\tfilename\nbase\t3\t" + std::string(64, 'a') + "\t" + name + "\n");
        bad("assets.tsv");
    }
    setup();
    {
        auto nul = read(t.path / "assets.tsv");
        nul.insert(nul.size() - 1, std::string("\0suffix", 7));
        write(t.path / "assets.tsv", nul);
        bad("assets.tsv");
    }
    setup();
    fs::remove(t.path / "base.gguf");
    bad("base.gguf");
    setup();
    write(t.path / "base.gguf", "abcd");
    bad("base.gguf");
    setup();
    write(t.path / "base.gguf", "abd");
    auto size = open_catalog(t.path.string());
    CHECK(size);
    CHECK(size.value().integrity == Integrity::size_only);
    CHECK(!size.value().files.at("base").digest_verified);
    error(open_catalog(t.path.string(), Integrity::digest), ErrorCategory::asset_invalid, "base.gguf");
    std::cout << "sha256 vectors, streaming and catalog rejection matrix passed\n";
}
