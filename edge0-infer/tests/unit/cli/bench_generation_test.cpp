#include "tests/support/bench_fixture.h"
#include "edge0/engine.h"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace {
std::string quote(const std::string& value) {
#ifdef _WIN32
    CHECK(value.find('"') == std::string::npos);
    return '"' + value + '"';
#else
    std::string out = "'";
    for (char c : value) out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + "'";
#endif
}
}

int main(int argc, char** argv) {
    CHECK(argc == 3 || argc == 4); // Optional backend; no external model download.
    using namespace fixture;
    using Json = nlohmann::ordered_json;
    const std::string backend = argc == 4 ? argv[3] : "cpu";
    CHECK(backend == "cpu" || backend == "metal");
    std::string selected_device;
    for (const auto& device : edge0::query_capabilities().devices)
        if (device.backend == backend) { selected_device = device.name; break; }
    if (selected_device.empty()) {
        std::cout << "Not executed: requested backend unavailable: " << backend << '\n';
        return 77;
    }
    Temp temporary;
    bench_assets(temporary.path, argv[2]);
    const auto run = [&](const std::string& extra) {
        const auto output = temporary.path / "output.jsonl";
        const auto errors = temporary.path / "stderr.txt";
        const std::string command = quote(argv[1]) + " bench --model " + quote(temporary.path.string()) +
            " --device " + backend + " --context 128 --prefill-chunk 32 --tokens " + quote((temporary.path / "tokens.json").string()) +
            " --model-option routing=actual --model-option prefill_ahead_min_tokens=32 --warmup 1 --ntok 2 --runs 2 " + extra +
            " > " + quote(output.string()) + " 2> " + quote(errors.string());
        const int code = std::system(command.c_str());
        if (code != 0) std::cerr << read(errors);
        CHECK(code == 0);
        std::istringstream stream(read(output));
        std::vector<Json> rows;
        for (std::string line; std::getline(stream, line);) rows.push_back(Json::parse(line));
        CHECK(rows.size() == 3 && rows.back()["type"] == "summary");
        CHECK(rows.back().contains("generation"));
        CHECK(rows.back()["model"]["loading"]["device"] == selected_device);
        CHECK(rows.back()["model"]["loading"]["prefill_device"] == selected_device);
        for (std::size_t i = 0; i < 2; ++i) {
            CHECK(rows[i]["type"] == "run");
            CHECK(rows[i].contains("generation"));
            CHECK(rows[i]["request_summary"]["prefill_device"] == selected_device);
            CHECK(rows[i]["generation"] == rows.back()["generation"]);
        }
        return rows.back();
    };
    const auto greedy = run("");
    CHECK(greedy["generation"]["temperature"] == 0.0);
    CHECK(greedy["generation"]["seed"].is_null());
    CHECK(greedy["generation"]["top_k"] == 20);
    CHECK(greedy["generation"]["top_p"] == 0.8);
    CHECK(greedy["generation"]["max_tokens"] == 4);
    CHECK(greedy["model"]["recommended"]["temperature"] == 0.7);
    CHECK(greedy["generation"]["sources"]["temperature"] == "application");
    CHECK(greedy["generation"]["sources"]["top_p"] == "model");
    const auto sampled = run("--temperature 0.8 --top-p 0.75 --seed 7");
    CHECK(sampled["generation"]["temperature"] == 0.8);
    CHECK(sampled["generation"]["top_p"] == 0.75);
    CHECK(sampled["generation"]["seed"] == 7);
    CHECK(sampled["generation"]["sources"]["top_p"] == "application");
    CHECK(sampled["model"]["loading"] == greedy["model"]["loading"]);
    CHECK(sampled["model"]["recommended"] == greedy["model"]["recommended"]);
    std::cout << "native bench effective generation: two runs and summary, defaults, inherited values and overrides passed on "
              << backend << " (" << selected_device << ")\n";
}
