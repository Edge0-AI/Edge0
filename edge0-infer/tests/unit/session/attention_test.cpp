#include "session/storage.h"
#include "tests/support/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
using namespace edge0;
using namespace edge0::session;
using test::take;
namespace {
constexpr int width = 64;
float key(int token, int column) { return static_cast<float>((token * 3 + column * 7) % 31 - 15) / 32; }
float value(int token, int column) { return static_cast<float>((token * 11 + column * 5) % 29 - 14) / 32; }
float query(int token, int column) { return static_cast<float>((token * 7 + column * 3) % 23 - 11) / 32; }
struct Attention {
    std::shared_ptr<exec::Graph> graph;
    std::shared_ptr<exec::Plan> plan;
    ggml_tensor *k, *v, *mask, *output;
    bool rows;
};
Attention build(exec::Budget& budget, const std::shared_ptr<exec::Backend>& backend,
    const std::shared_ptr<KVCache>& cache, Step step, bool reusable, int* builds = nullptr) {
    if (builds) ++*builds;
    auto graph = take(exec::create_graph(budget, backend, exec::Phase::prefill));
    auto* context = graph->context();
    auto* q = take(graph->input("q", {GGML_TYPE_F32, {width, step.n, 1}}));
    auto* ki = take(graph->input("k", {GGML_TYPE_F16, {width, step.n, 1}}));
    auto* vi = take(graph->input("v", {GGML_TYPE_F16, {width, step.n, 1}}));
    auto ks = take(cache->storage("k")), vs = take(cache->storage("v"));
    auto* kt = take(graph->bind_states(ks)); auto* vt = take(graph->bind_states(vs));
    ggml_tensor *kw, *vw;
    if (reusable) {
        auto* rows = take(graph->input("rows", {GGML_TYPE_I64, {step.n}}));
        kw = ggml_set_rows(context, kt, ki, rows); vw = ggml_set_rows(context, vt, vi, rows);
    } else {
        kw = ggml_cpy(context, ki, take(cache->write_range(context, "k", step)).view);
        vw = ggml_cpy(context, vi, take(cache->write_range(context, "v", step)).view);
    }
    take(graph->bind_state_write(ks, kw)); take(graph->bind_state_write(vs, vw));
    const Step extent = reusable ? Step{0, 300} : step;
    auto* kh = take(cache->history_view(context, "k", extent, kw));
    auto* vh = take(cache->history_view(context, "v", extent, vw));
    const auto maximum = take(cache->history_length("k", extent));
    auto* mi = take(graph->input("mask", {GGML_TYPE_F16, {maximum, step.n}}));
    auto* mask = ggml_view_2d(context, mi, maximum, step.n, mi->nb[1], 0);
    auto* output = ggml_flash_attn_ext(context, q, kh, vh, mask, 0.125f, 0, 0);
    CHECK(!graph->bounded_view(kh, 1, exec::Strides::preserve, output));
    CHECK(!graph->bounded_view(vh, 1, exec::Strides::preserve, output));
    CHECK(!graph->bounded_view(mask, 0, exec::Strides::packed, output));
    CHECK(!graph->output(output));
    return {graph, take(exec::build_plan(graph)), kh, vh, mask, output, reusable};
}
std::vector<float> run(Attention& attention, const std::shared_ptr<KVCache>& cache, Step step) {
    const auto length = take(cache->history_length("k", step));
    CHECK(!attention.plan->set_valid_ranges(attention.k, length));
    CHECK(!attention.plan->set_valid_ranges(attention.v, length));
    CHECK(!attention.plan->set_valid_ranges(attention.mask, length));
    std::vector<float> q(static_cast<std::size_t>(step.n * width));
    std::vector<ggml_fp16_t> k(q.size()), v(q.size()); std::vector<std::int64_t> rows;
    for (int row = 0; row < step.n; ++row) {
        const auto token = static_cast<int>(step.e) + row; rows.push_back(token);
        for (int col = 0; col < width; ++col) {
            const auto index = static_cast<std::size_t>(row * width + col);
            q[index] = query(token, col); k[index] = ggml_fp32_to_fp16(key(token, col)); v[index] = ggml_fp32_to_fp16(value(token, col));
        }
    }
    auto mask = take(cache->causal_mask("k", step));
    CHECK(!attention.plan->set_inputs("q", q.data(), q.size() * sizeof(float)));
    CHECK(!attention.plan->set_inputs("k", k.data(), k.size() * sizeof(ggml_fp16_t)));
    CHECK(!attention.plan->set_inputs("v", v.data(), v.size() * sizeof(ggml_fp16_t)));
    CHECK(!attention.plan->set_inputs("mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)));
    if (attention.rows) CHECK(!attention.plan->set_inputs("rows", rows.data(), rows.size() * sizeof(std::int64_t)));
    CHECK(!attention.plan->submit());
    auto completion = take(attention.plan->collect_completion()); CHECK(completion.success && completion.written.size() == 2);
    std::vector<float> output(q.size()); CHECK(!attention.plan->read_output(attention.output, output.data(), 0, output.size() * sizeof(float)));
    CHECK(!attention.plan->end_observation()); return output;
}
double error(const std::vector<float>& actual, Step step) {
    double maximum = 0;
    for (int row = 0; row < step.n; ++row) {
        const auto token = static_cast<int>(step.e) + row;
        std::vector<double> scores(static_cast<std::size_t>(token + 1)); double largest = -1e100;
        for (int previous = 0; previous <= token; ++previous) {
            double dot = 0;
            for (int col = 0; col < width; ++col)
                dot += static_cast<double>(query(token, col)) * ggml_fp16_to_fp32(ggml_fp32_to_fp16(key(previous, col)));
            scores[previous] = dot * 0.125; largest = std::max(largest, scores[previous]);
        }
        double sum = 0; for (auto& score : scores) { score = std::exp(score - largest); sum += score; }
        for (int col = 0; col < width; ++col) {
            double expected = 0;
            for (int previous = 0; previous <= token; ++previous)
                expected += scores[previous] / sum * ggml_fp16_to_fp32(ggml_fp32_to_fp16(value(previous, col)));
            const auto observed = actual[static_cast<std::size_t>(row * width + col)];
            CHECK(std::isfinite(observed) && std::isfinite(expected));
            const auto difference = std::abs(expected - observed);
            CHECK(std::isfinite(difference));
            maximum = std::max(maximum, difference);
        }
    }
    return maximum;
}
}
int main(int argc, char** argv) {
    auto backend = test::backend(argc, argv); exec::Budget budget(128 * 1024 * 1024);
    double maximum = 0; int comparisons = 0;
    for (const int alignment : {1, 32}) {
        auto make = [&] { return take(KVCache::create(budget, backend, 300,
            {{"k", GGML_TYPE_F16, {64, 1}, 1, alignment}, {"v", GGML_TYPE_F16, {64, 1}, 1, alignment}})); };
        {
            int persistent_allocations = 0, persistent_builds = 0;
            auto allocate_persistent = [&] { ++persistent_allocations; return make(); };
            auto persistent = allocate_persistent(), fresh = make();
            auto reused = build(budget, backend, persistent, {0, 1}, true, &persistent_builds);
            auto initial_plan = reused.plan;
            auto* initial_v = take(persistent->storage("v"))->tensor();
            const auto initial_v_buffer = initial_v->buffer;
            auto* initial_k = take(persistent->storage("k"))->tensor();
            const auto initial_buffer = initial_k->buffer;
            for (int token = 0; token < 40; ++token) {
                auto rebuilt = build(budget, backend, fresh, {token, 1}, false);
                auto a = run(reused, persistent, {token, 1}), b = run(rebuilt, fresh, {token, 1});
                CHECK(a.size() == b.size());
                if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                    std::cerr << "T4 bitwise mismatch: alignment=" << alignment << " length=" << token + 1 << '\n'; return 1;
                }
                ++comparisons; maximum = std::max(maximum, error(a, {token, 1}));
                CHECK(take(persistent->storage("k"))->tensor() == initial_k && initial_k->buffer == initial_buffer);
                CHECK(take(persistent->storage("v"))->tensor() == initial_v && initial_v->buffer == initial_v_buffer);
                CHECK(reused.plan == initial_plan && persistent_allocations == 1 && persistent_builds == 1);
            }
        }
        CHECK(budget.current() == 0);
        {
            auto a = make(), b = make();
            auto reusable = build(budget, backend, a, {0, 288}, true);
            auto rebuilt = build(budget, backend, b, {0, 288}, false);
            auto output = run(reusable, a, {0, 288}), expected = run(rebuilt, b, {0, 288});
            CHECK(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0);
            ++comparisons; maximum = std::max(maximum, error(output, {0, 288}));
        }
        CHECK(budget.current() == 0);
    }
    CHECK(maximum <= 5e-3);
    std::cout << "attention L2: exact/aligned(32), 40 appends each, 288-token prefill; bitwise comparisons=" << comparisons
              << "; max_abs_error=" << maximum << "; one persistent plan and unchanged storage per append sequence; ledger=0\n";
}
