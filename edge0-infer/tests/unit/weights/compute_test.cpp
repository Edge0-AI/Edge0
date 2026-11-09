#include "weights/storage.h"
#include "tests/support/runtime.h"
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
using namespace edge0;
using namespace edge0::weights;
using test::take;
int main(int argc, char** argv) {
    auto backend = test::backend(argc, argv);
    exec::Budget budget(64 * 1024 * 1024);
    auto path = std::filesystem::current_path() / "build" / ("weights-compute-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::array<float, 16> matrix{};
    for (std::size_t i = 0; i < matrix.size(); ++i) matrix[i] = (static_cast<int>(i % 7) - 3) * 0.125f;
    auto save = [&] { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(matrix.data()), sizeof(matrix)); };
    save(); double maximum = 0;
    {
        Weights weights(budget);
        auto a = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4, 4}}, 1}}, edge0::exec::Purpose::resident_weights));
        auto b = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4, 4}}, 1}}, edge0::exec::Purpose::resident_weights));
        auto item = [&](const std::string& id) { return LoadItem{0, id, {path.string(), 0, sizeof(matrix), {GGML_TYPE_F32, {4, 4}}}}; };
        auto publish = [&](const std::shared_ptr<Storage>& storage, const std::string& id) {
            auto planned = take(weights.plan_resident_load(storage, {item(id)}));
            auto write = take(weights.begin_write(storage, planned)); CHECK(!weights.fill_regions(write));
            auto binding = take(weights.complete(write)).front();
            std::array<float, 16> actual{}; ggml_backend_tensor_get(storage->tensor(0), actual.data(), 0, sizeof(actual));
            CHECK(std::memcmp(actual.data(), matrix.data(), sizeof(matrix)) == 0);
            return binding;
        };
        auto binding = publish(a, "a");
        auto state = take(exec::allocate_state_storage(budget, backend, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        ggml_backend_buffer_clear(state->allocation->buffer, 0);
        auto graph = take(exec::create_graph(budget, backend, exec::Phase::decode));
        auto view = take(weights.acquire_views(a, binding)); CHECK(!graph->bind_weights(view.tensor, backend, view.use));
        auto* input = take(graph->input("x", {GGML_TYPE_F32, {4}}));
        auto* old = take(graph->bind_states(state));
        auto* mixed = ggml_add(graph->context(), old, input);
        take(graph->bind_state_write(state, ggml_cpy(graph->context(), mixed, old)));
        auto* result = ggml_add(graph->context(), ggml_silu(graph->context(), ggml_mul_mat(graph->context(), view.tensor, input)), mixed);
        CHECK(!graph->output(result));
        auto plan = take(exec::build_plan(graph));
        std::array<double, 4> accumulated{};
        for (int step = 0; step < 8; ++step) {
            if (step == 4) {
                const auto stale = binding;
                for (auto& value : matrix) value *= -0.5f;
                save(); binding = publish(a, "replacement");
                CHECK(!weights.acquire_views(a, stale));
            }
            if (step != 0) {
                auto current = take(weights.acquire_views(a, binding)); CHECK(!plan->bind_weights(current.tensor, current.use));
            }
            std::array<float, 4> input_values{0.125f, -0.25f, 0.375f, -0.5f};
            CHECK(!plan->set_inputs("x", input_values.data(), sizeof(input_values)));
            std::promise<void> entered, resume;
            std::shared_ptr<Write> parallel;
            if (step == 0) {
                auto gate = resume.get_future().share();
                weights.before_read = [&entered, gate](std::size_t) { entered.set_value(); gate.wait(); };
                parallel = take(weights.begin_write(b, {item("parallel")}));
                CHECK(!weights.fill_regions(parallel)); entered.get_future().wait();
                CHECK(b->state(0) == RegionState::writing);
            }
            CHECK(!plan->submit());
            const auto occupied = budget.current();
            auto forbidden = weights.begin_write(a, {item("forbidden")});
            CHECK(!forbidden && forbidden.error().category == ErrorCategory::conflict);
            CHECK(a->state(0) == RegionState::published && budget.current() == occupied);
            if (step == 0) {
                CHECK(b->state(0) == RegionState::writing);
                resume.set_value();
                CHECK(take(weights.complete(parallel)).size() == 1);
                weights.before_read = {};
                std::array<float, 16> actual{};
                ggml_backend_tensor_get(b->tensor(0), actual.data(), 0, sizeof(actual));
                CHECK(std::memcmp(actual.data(), matrix.data(), sizeof(matrix)) == 0);
            }
            auto completion = take(plan->collect_completion()); CHECK(completion.success && completion.written.size() == 1);
            CHECK(!weights.begin_write(a, {item("still-observed")}));
            std::array<float, 4> actual{}, updated{}; CHECK(!plan->read_output(result, actual.data(), 0, sizeof(actual)));
            ggml_backend_tensor_get(state->tensor(), updated.data(), 0, sizeof(updated));
            for (int row = 0; row < 4; ++row) {
                accumulated[row] += input_values[row]; double dot = 0;
                CHECK(std::isfinite(updated[row]) && updated[row] == accumulated[row]);
                for (int column = 0; column < 4; ++column) dot += static_cast<double>(matrix[row * 4 + column]) * input_values[column];
                const auto expected = dot / (1 + std::exp(-dot)) + accumulated[row];
                CHECK(std::isfinite(actual[row]) && std::isfinite(expected));
                const auto difference = std::abs(actual[row] - expected);
                CHECK(std::isfinite(difference));
                maximum = std::max(maximum, difference);
            }
            CHECK(!plan->end_observation());
        }
        CHECK(maximum <= 1e-4);
        CHECK(!plan->release_plan()); graph.reset();
        CHECK(!weights.release_storage(a)); CHECK(!weights.release_storage(b));
    }
    CHECK(budget.current() == 0); std::filesystem::remove(path);
    std::cout << "weights L2: 8 runs, one plan, replacement and parallel publication; max_abs_error=" << maximum << "; ledger=0\n";
}
