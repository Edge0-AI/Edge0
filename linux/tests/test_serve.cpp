// Synthetic GGUF/safetensors fixtures, real POSIX mappings, and injected failures.
// Run in Docker after vendor-build.sh; no model weights or test framework needed.
#include "mem_budget_internal.h"
#include "prefetch_internal.h"
#include "expert_pool.h"
#include "ggml.h"
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace {
using namespace edge0;
using Json = nlohmann::json;
constexpr size_t MiB = 1024 * 1024;

void check(bool condition, const char * message = "assertion failed") {
    if (!condition) throw std::runtime_error(message);
}

std::string capture_errors(void (*test)()) {
    FILE * capture = std::tmpfile();
    check(capture != nullptr);
    const int saved = dup(STDERR_FILENO);
    check(saved >= 0);
    std::fflush(stderr);
    check(dup2(fileno(capture), STDERR_FILENO) >= 0);
    const auto restore = [&] { std::fflush(stderr); dup2(saved, STDERR_FILENO); ::close(saved); };
    try { test(); }
    catch (...) { restore(); std::fclose(capture); throw; }
    restore();
    std::rewind(capture);
    std::string output;
    char buffer[512];
    while (const size_t count = std::fread(buffer, 1, sizeof(buffer), capture)) output.append(buffer, count);
    std::fclose(capture);
    return output;
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        std::string pattern = (std::filesystem::temp_directory_path() / "edge0-serve-XXXXXX").string();
        check(mkdtemp(pattern.data()) != nullptr);
        path = pattern;
    }
    ~TemporaryDirectory() { std::filesystem::remove_all(path); }
};

struct Mapping {
    void * data = nullptr;
    size_t size;
    Mapping(int fd, size_t length, int protection = PROT_READ, int flags = MAP_SHARED) : size(length) {
        data = mmap(nullptr, size, protection, flags, fd, 0);
        check(data != MAP_FAILED, "fixture mmap failed");
    }
    ~Mapping() { munmap(data, size); }
};

struct Fixture {
    TemporaryDirectory temp;
    std::string gguf = (temp.path / "model.gguf").string();
    std::string sidecar = (temp.path / "heads.safetensors").string();
    ModelShape shape;
    int owner, width = 4;
    Json header = Json::object();
    std::vector<unsigned char> payload;
    Fixture(ModelShape s = {8, 8, 2, 4, 2, 1, true}, int layer = 1, bool missing_gate = false, bool bad_experts = false)
        : shape(s), owner(layer) {
        make_model(missing_gate, bad_experts);
        make_sidecar();
    }
    void make_model(bool missing_gate, bool bad_experts) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({ggml_tensor_overhead() * 16 + 4096, nullptr, true}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
        const std::string arch = shape.sigmoid ? "bailingmoe3" : "qwen3next";
        gguf_set_val_str(meta.get(), "general.architecture", arch.c_str());
        for (auto entry : std::vector<std::pair<std::string, int>>{
            {"embedding_length", shape.hidden}, {"expert_count", shape.experts},
            {"expert_used_count", shape.topk}, {"block_count", shape.layers},
            {"expert_group_count", shape.groups}, {"expert_group_used_count", shape.kept_groups}}) {
            gguf_set_val_u32(meta.get(), (arch + "." + entry.first).c_str(), entry.second);
        }
        std::vector<std::vector<float>> buffers;
        auto tensor = [&](std::string name, int64_t x, int64_t y, int64_t z, bool gate) {
            ggml_tensor * value = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, x, y, z);
            ggml_set_name(value, name.c_str());
            buffers.emplace_back(size_t(x * y * z), 0);
            auto & data = buffers.back();
            if (gate) for (int e = 0; e < shape.experts; ++e) data[size_t(e) * x] = e * 0.125f;
            else for (size_t i = 0; i < data.size(); ++i) data[i] = float(i % 251);
            value->data = data.data();
            gguf_add_tensor(meta.get(), value);
        };
        for (int layer : {owner, owner + 1}) {
            if (!missing_gate || layer == owner) tensor("blk." + std::to_string(layer) + ".ffn_gate_inp.weight",
                                                       shape.hidden, shape.experts, 1, true);
        }
        for (const char * name : {"ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"}) {
            tensor("blk." + std::to_string(owner + 1) + "." + name + ".weight", 32, 1,
                   shape.experts + int(bad_experts), false);
        }
        check(gguf_write_to_file(meta.get(), gguf.c_str(), false));
    }
    void make_sidecar(const std::string & dtype = "F16") {
        header = Json::object();
        payload.clear();
        const size_t features = shape.hidden + 2 * shape.experts;
        for (const std::string name : {"fc1", "fc2", "linear_init"}) {
            const size_t rows = name == "fc1" ? width : shape.experts;
            const size_t columns = name == "fc2" ? width : features;
            const size_t element = dtype == "F32" ? 4 : 2;
            const size_t start = payload.size();
            payload.resize(start + rows * columns * element, 0);
            if (name == "linear_init") {
                for (size_t e = 0; e < rows; ++e) {
                    const float value = e * 0.125f;
                    const size_t position = start + e * columns * element;
                    if (dtype == "F32") std::memcpy(payload.data() + position, &value, 4);
                    else {
                        const uint16_t bits = dtype == "F16" ? ggml_fp32_to_fp16(value) : ggml_fp32_to_bf16(value).bits;
                        std::memcpy(payload.data() + position, &bits, 2);
                    }
                }
            }
            header["layers." + std::to_string(owner) + "." + name + ".weight"] = {
                {"dtype", dtype}, {"shape", {rows, columns}}, {"data_offsets", {start, payload.size()}}};
        }
        save();
    }
    void save(std::string text = {}) {
        if (text.empty()) text = header.dump();
        while (text.size() % 8) text += ' ';
        const uint64_t length = text.size();
        std::ofstream file(sidecar, std::ios::binary);
        file.write(reinterpret_cast<const char *>(&length), 8);
        file.write(text.data(), text.size());
        file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
        check(bool(file));
    }
    std::vector<float> rows() const {
        std::vector<float> result(size_t(shape.hidden) * 2, 0);
        result[0] = result[shape.hidden] = 1;
        return result;
    }
    std::unique_ptr<PrefetchEngine> load(bool advice = false, PrefetchEngine::Advice callback = {}) const {
        auto engine = std::make_unique<PrefetchEngine>(std::move(callback));
        check(engine->load(gguf.c_str(), sidecar.c_str(), advice), "fixture did not load");
        return engine;
    }
};

void mib_inputs() {
    size_t bytes = 42;
    for (const char * value : {static_cast<const char *>(nullptr), "", "0", "0000"}) check(parse_mib(value, bytes) && !bytes);
    check(parse_mib("4096", bytes) && bytes == 4096 * MiB);
    for (const char * value : {"-1", "+1", " 2", "2 ", "2MB", "1.5", "18446744073709551616", "18446744073709551615"}) {
        check(!parse_mib(value, bytes), "invalid MiB accepted");
    }
    const size_t maximum = SIZE_MAX / MiB;
    check(parse_mib(std::to_string(maximum).c_str(), bytes) && bytes == maximum * MiB);
    check(!parse_mib(std::to_string(maximum + 1).c_str(), bytes));
}
void mib_precedence() {
    size_t bytes;
    check(resolve_mib(8, "bad", bytes) && bytes == 8 * MiB);
    check(resolve_mib(0, "16", bytes) && bytes == 16 * MiB);
    check(resolve_mib(0, nullptr, bytes) && bytes == 0);
    check(!resolve_mib(-1, "16", bytes));
    check(!resolve_mib(0, "-1", bytes));
}
void page_boundaries() {
    ByteRange result{};
    const size_t p = page_size();
    check(page_range(p * 2 + 9, {p + 1, p + 8}, result));
    check(result.offset == p && result.length == p * 2);
    check(page_range(p, {0, p}, result) && result.length == p);
    check(!page_range(p, {p, 1}, result));
    check(!page_range(p, {0, 0}, result));
    check(!page_range(p, {1, p}, result));
    check(!page_range(SIZE_MAX, {SIZE_MAX - 1, 1}, result));
}
void expert_bounds() {
    ByteRange result{};
    check(expert_range(120, 20, 10, 10, 9, result) && result.offset == 110 && result.length == 10);
    check(!expert_range(120, 20, 10, 10, -1, result));
    check(!expert_range(120, 20, 10, 10, 10, result));
    check(!expert_range(120, 20, 10, 11, 0, result));
    check(!expert_range(120, SIZE_MAX, 10, 10, 0, result));
    check(!expert_range(SIZE_MAX, 1, SIZE_MAX, 2, 1, result));
    check(!expert_range(120, 0, 0, 1, 0, result));
}
void file_io() {
    TemporaryDirectory temp;
    const std::string path = (temp.path / "unaligned bytes").string();
    const std::string data(page_size() + 17, 'x');
    { std::ofstream file(path); file << data; }
    ReadOnlyFile file;
    check(!file.open((temp.path / "absent").c_str(), false));
    check(file.open(path.c_str(), true));
    std::string copy(data.size(), '\0');
    check(file.read(0, copy.data(), copy.size()) && copy == data);
    check(!file.read(data.size(), copy.data(), 1));
    check(!file.read(SIZE_MAX, copy.data(), 1));
    check(file.advise({1, data.size() - 1}));
    check(!file.advise({data.size(), 1}));
    file.close();
    check(!file.read(0, copy.data(), 1));
    check(!file.advise({0, 1}));
    std::ifstream original(path);
    check(std::string(std::istreambuf_iterator<char>(original), {}) == data);
}
void rss_accounting() {
    size_t before, reserved, touched;
    check(read_rss(before));
    Mapping mapping(-1, 64 * MiB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS);
    check(read_rss(reserved));
    check(reserved < before + 8 * MiB, "virtual reservation counted as RSS");
    std::memset(mapping.data, 1, mapping.size);
    check(read_rss(touched) && touched >= before + 60 * MiB, "anonymous overhead missing from RSS");
}
void budget_dormant() {
    int reads = 0, discards = 0;
    MemoryBudget budget({[&](size_t & rss) { ++reads; rss = 2 * MiB; return true; },
                         [&](void *, size_t) { ++discards; return 0; }});
    check(!budget.start(0));
    check(!budget.enabled() && budget.check() && !reads && !discards);
}
void budget_reclaim() {
    TemporaryDirectory temp;
    const std::string path = (temp.path / "mapped-model").string();
    const int fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    check(fd >= 0 && ftruncate(fd, 32 * MiB) == 0);
    Mapping model(fd, 32 * MiB);
    ::close(fd);
    Mapping live(-1, MiB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS);
    std::memset(live.data, 0x5a, live.size);
    size_t baseline, resident;
    check(read_rss(baseline));
    volatile unsigned char sink = 0;
    for (size_t i = 0; i < model.size; i += page_size()) sink ^= static_cast<unsigned char *>(model.data)[i];
    check(read_rss(resident) && resident > baseline + 28 * MiB);
    MemoryBudget budget;
    check(budget.start(baseline + 4 * MiB, false));
    budget.mapped(model.data, model.size);
    check(budget.check(), "real mapped pages were not reclaimed");
    check(read_rss(resident) && resident <= baseline + 4 * MiB);
    for (size_t i = 0; i < live.size; ++i) check(static_cast<unsigned char *>(live.data)[i] == 0x5a, "live memory discarded");
    check(static_cast<unsigned char *>(model.data)[0] == 0 && sink == 0, "model bytes changed");
}
void budget_overhead() {
    MemoryBudget budget({[](size_t & rss) { rss = 9 * MiB; return true; }, {}});
    check(budget.start(4 * MiB, false));
    check(!budget.check(), "impossible budget reported success");
}
void budget_unmap() {
    const size_t p = page_size();
    Mapping mapping(-1, p * 3, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS);
    const uintptr_t base = reinterpret_cast<uintptr_t>(mapping.data);
    size_t rss = p * 10;
    std::vector<ByteRange> discarded;
    MemoryBudget budget({[&](size_t & out) { out = rss; return true; }, [&](void * address, size_t length) {
        discarded.push_back({reinterpret_cast<uintptr_t>(address), length});
        rss = p * 2;
        return 0;
    }});
    check(budget.start(p * 4, false));
    budget.mapped(mapping.data, mapping.size);
    check(budget.unmap(reinterpret_cast<void *>(base + p), p) == 0);
    check(budget.check());
    rss = p * 10;
    check(budget.check() && discarded.size() == 2);
    for (ByteRange range : discarded) check(range.length == p && (range.offset == base || range.offset == base + p * 2));
    check(budget.unmap(mapping.data, p) == 0);
    check(budget.unmap(reinterpret_cast<void *>(base + p * 2), p) == 0);
    rss = p * 10;
    check(!budget.check() && discarded.size() == 2, "forgotten pages were advised");
}
void budget_syscall_failure() {
    Mapping mapping(-1, page_size(), PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS);
    MemoryBudget budget({[](size_t & rss) { rss = 8 * MiB; return true; },
                         [](void *, size_t) { errno = EIO; return -1; }});
    check(budget.start(4 * MiB, false));
    budget.mapped(mapping.data, mapping.size);
    check(!budget.check());
}
void budget_rss_failure() {
    bool works = false;
    MemoryBudget budget({[&](size_t & rss) { rss = MiB; return works; }, {}});
    check(!budget.start(4 * MiB, false) && !budget.enabled());
    works = true;
    check(budget.start(4 * MiB, false));
    works = false;
    check(!budget.check());
}
void budget_worker() {
    std::atomic<int> calls{0};
    MemoryBudget budget({[&](size_t & rss) { ++calls; rss = MiB; return true; }, {}});
    for (int round = 0; round < 3; ++round) {
        check(budget.start(4 * MiB));
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        budget.stop();
        check(!budget.enabled());
    }
    check(calls >= 6);
}
struct ScriptedBudget {  // a 4 MiB budget over one anonymous page with scripted RSS and clock; counts discards
    Mapping mapping;
    size_t rss;
    int discards = 0;
    std::chrono::steady_clock::time_point clock{};
    MemoryBudget budget;
    explicit ScriptedBudget(size_t start_rss)
        : mapping(-1, page_size(), PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS), rss(start_rss),
          budget({[this](size_t & out) { out = rss; return true; }, [this](void *, size_t) { ++discards; return 0; },
                  [this] { return clock; }}) {
        check(budget.start(4 * MiB, false));
        budget.mapped(mapping.data, mapping.size);
    }
};
void budget_backoff() {
    // Given RSS stays over budget after a full pass, a new pass waits for 64 MiB of growth or a 2 s pause.
    ScriptedBudget fake(9 * MiB);
    check(!fake.budget.check() && fake.discards == 1, "first reclaim pass did not run");
    check(!fake.budget.check() && fake.discards == 1, "reclaim pass repeated without growth or pause");
    fake.rss += 64 * MiB;
    check(!fake.budget.check() && fake.discards == 2, "RSS growth did not retry");
    fake.clock += std::chrono::seconds(3);
    check(!fake.budget.check() && fake.discards == 3, "pause did not retry");
    fake.rss = MiB;
    check(fake.budget.check() && fake.discards == 3, "recovered RSS still reported as failing");
}
void budget_backoff_in_band() {
    // Given a pass leaves RSS above its target but within budget, checks pass and rescan only after a pause.
    ScriptedBudget fake(3 * MiB + MiB / 2);
    check(fake.budget.check() && fake.discards == 1, "in-band reclaim pass did not run");
    check(fake.budget.check() && fake.discards == 1, "in-band pass repeated without growth or pause");
    fake.clock += std::chrono::seconds(3);
    check(fake.budget.check() && fake.discards == 2, "pause did not retry an in-band pass");
}
void shape_8b() {
    Fixture fixture({1536, 128, 8, 24, 8, 4, true}, 7);
    auto engine = fixture.load();
    check(engine->shape().hidden == 1536 && engine->shape().experts == 128 && engine->shape().topk == 8);
    check(engine->capture_layers() == std::vector<int>({7, 8}));
    check(engine->step(fixture.rows()) && engine->step(fixture.rows()));
    check(engine->stats().heads == 2 && engine->stats().predicted == 16 && engine->stats().compared == 8);
}
void shape_35b() {
    Fixture fixture({2048, 256, 4, 40, 1, 1, false}, 6);
    auto engine = fixture.load();
    check(engine->shape().hidden == 2048 && engine->shape().experts == 256 && engine->shape().topk == 4);
    check(engine->capture_layers() == std::vector<int>({6, 7}));
    check(engine->step(fixture.rows()) && engine->step(fixture.rows()));
    check(engine->stats().predicted == 8 && engine->stats().hits == 4);
}
void head_math() {
    for (const std::string dtype : {"F16", "BF16", "F32"}) {
        Fixture fixture;
        fixture.make_sidecar(dtype);
        auto engine = fixture.load();
        check(engine->step(fixture.rows()) && engine->step(fixture.rows()));
        check(engine->stats().hits == uint64_t(fixture.shape.topk));
    }
    Fixture fixture({8, 8, 1, 4, 1, 1, false});
    fixture.make_sidecar("F32");
    std::fill(fixture.payload.begin(), fixture.payload.end(), 0);
    auto weight = [&](const std::string & name, size_t index, float value) {
        const size_t start = fixture.header["layers.1." + name + ".weight"]["data_offsets"][0].get<size_t>();
        std::memcpy(fixture.payload.data() + start + index * 4, &value, 4);
    };
    weight("fc1", 0, 1);
    weight("fc2", 7 * fixture.width, 1);
    // Exact erf GELU(1) > 0.8413; the common tanh approximation is below it.
    weight("linear_init", 6 * (fixture.shape.hidden + 2 * fixture.shape.experts), 0.8413f);
    fixture.save();
    auto engine = fixture.load();
    check(engine->step(fixture.rows()) && engine->step(fixture.rows()) && engine->stats().hits == 1,
          "fc1/GELU/fc2 + linear head math changed");
}
void history_math() {
    Fixture fixture({8, 8, 1, 4, 1, 1, false});
    const auto key = "layers.1.linear_init.weight";
    const size_t start = fixture.header[key]["data_offsets"][0].get<size_t>();
    const size_t columns = fixture.shape.hidden + 2 * fixture.shape.experts;
    std::fill(fixture.payload.begin(), fixture.payload.end(), 0);
    // Predict expert 0 from the previous teacher's expert-7 history feature.
    const uint16_t one = ggml_fp32_to_fp16(1);
    std::memcpy(fixture.payload.data() + start + (fixture.shape.hidden + fixture.shape.experts + 7) * 2, &one, 2);
    // Before history is available, the direct branch predicts expert 7.
    const uint16_t half = ggml_fp32_to_fp16(0.5f);
    std::memcpy(fixture.payload.data() + start + 7 * columns * 2, &half, 2);
    fixture.save();
    auto engine = fixture.load();
    check(engine->step(fixture.rows()) && engine->step(fixture.rows()) && engine->step(fixture.rows()));
    check(engine->stats().compared == 2 && engine->stats().hits == 1, "history feature ordering changed");
}
void account_only() {
    Fixture fixture;
    int calls = 0;
    auto engine = fixture.load(false, [&](const ReadOnlyFile &, ByteRange) { ++calls; return true; });
    check(engine->step(fixture.rows()) && engine->stats().predicted > 0);
    check(!calls && !engine->stats().advice);
}
void advice_on() {
    Fixture fixture;
    ReadOnlyFile original;
    check(original.open(fixture.gguf.c_str(), false));
    std::vector<unsigned char> before(original.size()), after(original.size());
    check(original.read(0, before.data(), before.size()));
    auto engine = fixture.load(true, [&](const ReadOnlyFile & file, ByteRange range) {
        check(range.offset < file.size() && range.length <= file.size() - range.offset);
        return file.advise(range);
    });
    check(engine->step(fixture.rows()) && engine->stats().advice == uint64_t(fixture.shape.topk * 3));
    check(original.read(0, after.data(), after.size()) && before == after, "advice changed model bytes");
}
void advice_failure() {
    Fixture fixture;
    auto engine = fixture.load(true, [](const ReadOnlyFile &, ByteRange) { return false; });
    check(engine->step(fixture.rows()) && engine->step(fixture.rows()));
    check(engine->stats().advice_failures == uint64_t(fixture.shape.topk * 6) && engine->stats().hits > 0);
}
void sidecar_corruption() {
    Fixture fixture;
    const Json valid = fixture.header;
    const std::string key = "layers.1.fc1.weight";
    const std::vector<std::function<void(Json &)>> changes = {
        [&](Json & j) { j[key]["dtype"] = "I64"; },
        [&](Json & j) { j[key]["shape"][1] = 0; },
        [&](Json & j) { j[key]["shape"][0] = UINT64_MAX; },
        [&](Json & j) { j[key]["shape"][0] = -1; },
        [&](Json & j) { j[key]["shape"] = {1}; },
        [&](Json & j) { j[key]["data_offsets"][0] = UINT64_MAX; },
        [&](Json & j) { j[key]["data_offsets"][0] = 1; },
        [&](Json & j) { j[key]["data_offsets"][1] = UINT64_MAX; },
        [&](Json & j) { j["layers.1.fc2.weight"]["data_offsets"] = {0, 64}; },
        [&](Json & j) { j["layers.0.fc1.weight"] = j[key]; j.erase(key); },
        [&](Json & j) { j["layers.2147483647.fc1.weight"] = j[key]; j.erase(key); },
        [&](Json & j) { j["layers.18446744073709551616.fc1.weight"] = j[key]; j.erase(key); },
        [&](Json & j) { j.erase("layers.1.fc2.weight"); },
        [&](Json & j) { j[key]["shape"] = {1, 96}; },
    };
    for (const auto & change : changes) {
        fixture.header = valid;
        change(fixture.header);
        fixture.save();
        PrefetchEngine engine;
        check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false), "corrupt sidecar accepted");
    }
    fixture.header = valid;
    fixture.save("{\"duplicate\":{},\"duplicate\":{}}");
    { PrefetchEngine engine; check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false)); }
    fixture.save("{broken JSON");
    { PrefetchEngine engine; check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false)); }
    fixture.save();
    std::filesystem::resize_file(fixture.sidecar, 5);
    { PrefetchEngine engine; check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false)); }
    fixture.save();
    std::filesystem::resize_file(fixture.sidecar, std::filesystem::file_size(fixture.sidecar) - 1);
    { PrefetchEngine engine; check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false)); }
    fixture.save();
    const uint16_t infinity = 0x7c00;
    std::memcpy(fixture.payload.data(), &infinity, 2);
    fixture.save();
    { PrefetchEngine engine; check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false)); }
    { PrefetchEngine engine; check(!engine.load(fixture.gguf.c_str(), (fixture.temp.path / "absent").c_str(), false)); }
}
void missing_gate() {
    Fixture fixture({8, 8, 2, 4, 2, 1, true}, 1, true);
    PrefetchEngine engine;
    check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false));
}
void invalid_expert_tensor() {
    Fixture fixture({8, 8, 2, 4, 2, 1, true}, 1, false, true);
    PrefetchEngine engine;
    check(!engine.load(fixture.gguf.c_str(), fixture.sidecar.c_str(), false));
}
void bad_rows() {
    Fixture fixture;
    auto engine = fixture.load();
    check(!engine->step({}));
    auto rows = fixture.rows();
    rows[0] = std::numeric_limits<float>::quiet_NaN();
    check(!engine->step(rows) && !engine->stats().steps);
    Fixture wide({8, 16, 2, 4, 2, 1, true});
    auto overflow = wide.load();
    rows = wide.rows();
    rows[0] = std::numeric_limits<float>::max();
    check(!overflow->step(rows), "overflowing gate logits accepted");
}
void worker_mailbox() {
    Fixture fixture;
    std::promise<void> entered, release;
    auto go = release.get_future().share();
    std::atomic<int> calls{0};
    auto engine = fixture.load(true, [&](const ReadOnlyFile &, ByteRange) {
        if (calls++ == 0) { entered.set_value(); go.wait(); }
        return true;
    });
    PrefetchWorker worker(std::move(engine), true);
    check(worker.submit(fixture.rows()));
    check(entered.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    check(!worker.submit(fixture.rows()), "busy mailbox accepted a second snapshot");
    release.set_value();
    worker.wait_idle();
    check(worker.engine().stats().steps == 1);
    check(worker.submit(fixture.rows()));
    worker.wait_idle();
    check(worker.engine().stats().steps == 2);
}
void worker_shutdown() {
    Fixture fixture;
    for (int round = 0; round < 8; ++round) {
        PrefetchWorker worker(fixture.load(), true);
        check(worker.submit(fixture.rows()));
    }
    PrefetchWorker synchronous(fixture.load(), false);
    check(synchronous.submit(fixture.rows()) && synchronous.engine().stats().steps == 1);
}
void worker_failure() {
    Fixture fixture;
    auto engine = fixture.load(true, [](const ReadOnlyFile &, ByteRange) -> bool { throw std::runtime_error("injected advice exception"); });
    PrefetchWorker worker(std::move(engine), true);
    check(worker.submit(fixture.rows()));
    worker.wait_idle();
    // An advisory exception must not terminate the worker or hang shutdown.
    check(worker.submit(fixture.rows()));
    worker.wait_idle();
}
size_t descriptors() {
    return std::distance(std::filesystem::directory_iterator("/proc/self/fd"), std::filesystem::directory_iterator());
}
void descriptor_cleanup() {
    Fixture fixture;
    const size_t before = descriptors();
    for (int round = 0; round < 16; ++round) { auto engine = fixture.load(true); }
    for (int round = 0; round < 16; ++round) {
        PrefetchEngine engine;
        check(!engine.load(fixture.gguf.c_str(), (fixture.temp.path / "absent").c_str(), true));
    }
    check(descriptors() == before, "file descriptors leaked");
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) check(line.find(fixture.gguf) == std::string::npos, "file mapping leaked");
}

struct PoolFixture {
    Fixture fixture;
    std::unique_ptr<PrefetchEngine> engine = fixture.load();
    Mapping model{engine->file().descriptor(), engine->file().size()};
    ExpertPool pool{engine->file()};
    ExpertTensor tensor = engine->tensors()[0];
    const void * identity = static_cast<char *>(model.data) + tensor.offset;
    explicit PoolFixture(size_t pages = 8) {
        check(pool.start(page_size() * pages));
        check(pool.add(identity, tensor));
    }
};
void pool_bytes() {
    PoolFixture fixture;
    const char * data = fixture.pool.resolve(fixture.identity, 3, fixture.tensor.stride);
    check(data && std::memcmp(data, static_cast<const char *>(fixture.identity) + 3 * fixture.tensor.stride,
                             fixture.tensor.stride) == 0, "pool bytes differ from mmap");
    check(fixture.pool.resolve(fixture.identity, 3, fixture.tensor.stride) == data, "immutable slot moved");
    check(fixture.pool.prefetch({fixture.tensor.offset + fixture.tensor.stride * 4, fixture.tensor.stride}));
}
void pool_capacity() {
    PoolFixture fixture(1);
    const char * first = fixture.pool.resolve(fixture.identity, 0, fixture.tensor.stride);
    check(first && fixture.pool.used() == page_size());
    check(!fixture.pool.resolve(fixture.identity, 1, fixture.tensor.stride));
    check(fixture.pool.resolve(fixture.identity, 0, fixture.tensor.stride) == first);
    check(fixture.pool.used() <= page_size());
}
void pool_bounds() {
    PoolFixture fixture;
    check(!fixture.pool.resolve(fixture.identity, -1, fixture.tensor.stride));
    check(!fixture.pool.resolve(fixture.identity, fixture.tensor.experts, fixture.tensor.stride));
    check(!fixture.pool.resolve(fixture.identity, 0, SIZE_MAX));
    check(!fixture.pool.resolve(nullptr, 0, fixture.tensor.stride));
    check(!fixture.pool.prefetch({SIZE_MAX, SIZE_MAX}));
    check(!fixture.pool.used());
}
void pool_concurrency() {
    PoolFixture fixture;
    std::vector<std::future<const char *>> readers;
    for (int i = 0; i < 16; ++i) readers.push_back(std::async(std::launch::async, [&] {
        return fixture.pool.resolve(fixture.identity, 2, fixture.tensor.stride);
    }));
    const char * first = readers[0].get();
    check(first);
    for (size_t i = 1; i < readers.size(); ++i) check(readers[i].get() == first);
    check(fixture.pool.used() == page_size());
}
void pool_layout() {
    PoolFixture fixture;
    std::vector<char> repacked(fixture.tensor.stride * fixture.tensor.experts);
    check(!fixture.pool.add(repacked.data(), fixture.tensor), "repacked layout enrolled");
    Mapping writable(fixture.engine->file().descriptor(), fixture.engine->file().size(), PROT_READ | PROT_WRITE, MAP_PRIVATE);
    check(!fixture.pool.add(static_cast<char *>(writable.data) + fixture.tensor.offset, fixture.tensor), "mutable mapping enrolled");
    ExpertTensor wrong = fixture.tensor;
    wrong.offset += 1;
    check(!fixture.pool.add(fixture.identity, wrong));
}
void pool_read_failure() {
    PoolFixture fixture;
    // A truncation after registration must not publish partially filled slots.
    std::filesystem::resize_file(fixture.fixture.gguf, fixture.tensor.offset + 1);
    check(!fixture.pool.resolve(fixture.identity, 0, fixture.tensor.stride));
    check(!fixture.pool.used());
}

} // namespace

int main(int argc, char ** argv) {
    const std::map<std::string, void (*)()> tests = {
        {"mib_inputs", mib_inputs}, {"mib_precedence", mib_precedence}, {"page_boundaries", page_boundaries},
        {"expert_bounds", expert_bounds}, {"file_io", file_io}, {"rss_accounting", rss_accounting},
        {"budget_dormant", budget_dormant}, {"budget_reclaim", budget_reclaim}, {"budget_overhead", budget_overhead},
        {"budget_unmap", budget_unmap}, {"budget_syscall_failure", budget_syscall_failure},
        {"budget_rss_failure", budget_rss_failure}, {"budget_worker", budget_worker},
        {"budget_backoff", budget_backoff}, {"budget_backoff_in_band", budget_backoff_in_band},
        {"shape_8b", shape_8b}, {"shape_35b", shape_35b}, {"head_math", head_math}, {"history_math", history_math},
        {"account_only", account_only}, {"advice_on", advice_on}, {"advice_failure", advice_failure},
        {"sidecar_corruption", sidecar_corruption}, {"missing_gate", missing_gate},
        {"invalid_expert_tensor", invalid_expert_tensor}, {"bad_rows", bad_rows},
        {"worker_mailbox", worker_mailbox}, {"worker_shutdown", worker_shutdown}, {"descriptor_cleanup", descriptor_cleanup},
        {"worker_failure", worker_failure},
        {"pool_bytes", pool_bytes}, {"pool_capacity", pool_capacity}, {"pool_bounds", pool_bounds},
        {"pool_concurrency", pool_concurrency}, {"pool_layout", pool_layout}, {"pool_read_failure", pool_read_failure},
    };
    try {
        check(argc == 2 && tests.count(argv[1]), "provide a test case name");
        const std::string name = argv[1];
        const std::map<std::string, std::string> expected_errors = {
            {"budget_overhead", "[mem-budget] enforcement failed:"},
            {"budget_syscall_failure", "[mem-budget] enforcement failed:"},
            {"budget_rss_failure", "[mem-budget] enforcement failed:"},
            {"budget_backoff", "[mem-budget] enforcement failed:"},
            {"sidecar_corruption", "[pref-trace] init failed:"},
            {"missing_gate", "[pref-trace] init failed:"},
            {"invalid_expert_tensor", "[pref-trace] init failed:"},
            {"advice_failure", "page advice failed; inference continues"},
            {"worker_failure", "advisory step failed:"},
        };
        if (expected_errors.count(name)) {
            const std::string output = capture_errors(tests.at(name));
            check(output.find(expected_errors.at(name)) != std::string::npos, "required failure log missing");
            std::fputs(output.c_str(), stderr);
        } else tests.at(name)();
        std::printf("PASS %s\n", argv[1]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s: %s\n", argc > 1 ? argv[1] : "arguments", error.what());
        return 1;
    }
}
