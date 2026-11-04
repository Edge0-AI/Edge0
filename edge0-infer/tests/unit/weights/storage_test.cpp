#include "weights/storage.h"
#include "tests/support/check.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
using namespace edge0;
using namespace edge0::weights;
template<class T> T take(const Result<T>& result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); } return result.value();
}
int main() {
    auto created = platform::BackendFacility::create("CPU");
    auto facility = std::make_shared<platform::BackendFacility>(std::move(created.first));
    auto backend = take(exec::resolve_stages("CPU", "CPU", facility)).decode;
    auto path = std::filesystem::current_path() / "build" / ("weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    float data[4]{1, 2, 3, 4};
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<char*>(data), sizeof(data)); }
    auto item = [&](std::size_t region, const std::string& id) { return LoadItem{region, id, {path.string(), 0, sizeof(data), {GGML_TYPE_F32, {4}}}}; };
    exec::Budget budget(64 * 1024 * 1024);
    {
        Weights weights(budget);
        auto storage = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 8}}, edge0::exec::Purpose::resident_weights));
        CHECK(budget.current(edge0::exec::Purpose::resident_weights) == ggml_backend_buffer_get_size(storage->allocation()->buffer));
        CHECK(budget.current() == budget.current(edge0::exec::Purpose::resident_weights) + storage->allocation()->charges[1]->size());
        auto publish = [&](const std::shared_ptr<Storage>& target, const std::vector<LoadItem>& items) {
            auto plan = take(weights.plan_resident_load(target, items));
            auto write = take(weights.begin_write(target, plan)); CHECK(!weights.fill_regions(write));
            return take(weights.complete(write));
        };
        auto bindings = publish(storage, {item(0, "a")});
        // Retain the graph after rejection: cleanup must not depend on its destructor.
        for (int failure = 0; failure < 4; ++failure) {
            auto graph = take(exec::create_graph(budget, backend, exec::Phase::decode));
            auto view = take(weights.acquire_views(storage, bindings[0]));
            CHECK(!graph->bind_weights(view.tensor, backend, view.use));
            if (failure != 3) CHECK(!graph->output(ggml_scale(graph->context(), ggml_cont(graph->context(), view.tensor), 2)));
            std::shared_ptr<exec::StateStorage> external;
            exec::Budget::Ticket pressure;
            if (failure == 1) {
                external = take(exec::allocate_state_storage(budget, backend, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
                CHECK(!graph->output(external->tensor()));
            }
            if (failure == 2) pressure = take(budget.reserve(budget.limit() - budget.current(), edge0::exec::Purpose::metadata, "host"));
            const auto occupied = budget.current();
            auto result = exec::build_plan(graph, [failure](ggml_backend_t handle, const ggml_tensor* tensor) {
                return failure != 0 && ggml_backend_supports_op(handle, tensor);
            });
            const auto category = failure == 0 ? ErrorCategory::unsupported :
                failure == 2 ? ErrorCategory::budget_exceeded : ErrorCategory::invalid_argument;
            CHECK(!result && result.error().category == category);
            CHECK(!view.use->valid() && storage->state(0) == RegionState::published);
            CHECK(budget.current() == occupied);
            {
                auto next = weights.begin_write(storage, {item(0, "retry")});
                CHECK(next && storage->state(0) == RegionState::writing);
            }
            CHECK(storage->state(0) == RegionState::published && budget.current() == occupied);
        }
        float actual[4]{}; ggml_backend_tensor_get(storage->tensor(0), actual, 0, sizeof(actual)); CHECK(std::memcmp(actual, data, sizeof(data)) == 0);
        const auto occupied = budget.current();
        auto v1 = take(weights.acquire_views(storage, bindings[0])); auto v2 = take(weights.acquire_views(storage, bindings[0])); auto v3 = take(weights.acquire_views(storage, bindings[0]));
        CHECK(budget.current() == occupied);
        CHECK(!weights.begin_write(storage, {item(0, "b")})); CHECK(weights.release_storage(storage));
        CHECK(v1.use->valid() && v2.use->valid() && v3.use->valid());
        CHECK(!weights.release_use(v1.use)); CHECK(weights.release_use(v1.use));
        CHECK(!weights.release_use(v2.use)); CHECK(!weights.release_use(v3.use));
        auto before = weights.read_count(); auto reused = take(weights.acquire_views(storage, bindings[0]));
        CHECK(weights.read_count() == before); CHECK(!weights.release_use(reused.use));
        auto duplicate = weights.begin_write(storage, {item(1, "same"), item(2, "same")});
        CHECK(!duplicate && duplicate.error().category == ErrorCategory::conflict);
        {
            Weights another_model(budget);
            auto target = take(another_model.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 1}}, edge0::exec::Purpose::resident_weights));
            auto write = take(another_model.begin_write(target, {item(0, "a")}));
            CHECK(!another_model.fill_regions(write)); auto own = take(another_model.complete(write));
            CHECK(!another_model.acquire_views(target, bindings[0]));
            CHECK(!another_model.release_use(take(another_model.acquire_views(target, own[0])).use));
        }
        auto other = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 3}}, edge0::exec::Purpose::resident_weights));
        auto collision = weights.begin_write(other, {item(0, "x"), item(1, "a"), item(2, "z")});
        CHECK(!collision && collision.error().category == ErrorCategory::conflict && collision.error().message.find("region 0") != std::string::npos);
        for (int i = 0; i < 3; ++i) CHECK(other->state(i) == RegionState::empty);
        CHECK(!weights.acquire_views(other, bindings[0]));
        auto newer = publish(storage, {item(0, "b")}); CHECK(!weights.acquire_views(storage, bindings[0]));
        CHECK(!weights.release_use(take(weights.acquire_views(storage, newer[0])).use));
        {
            auto write = take(weights.begin_write(other, {item(0, "not-direct")}));
            weights.direct = [](ggml_backend_t, ggml_backend_buffer_t) { return false; };
            CHECK(!weights.fill_regions(write));
            CHECK(take(weights.complete(write)).size() == 1);
            CHECK(std::memcmp(other->tensor(0)->data, data, sizeof(data)) == 0);
            CHECK(budget.current(edge0::exec::Purpose::staging) == 4 * std::size_t(4 * 1024 * 1024));
            weights.direct = platform::directly_writable;
            CHECK(!weights.evict(other, {0}));
        }
        CHECK(other->state(0) == RegionState::empty);
        {
            std::promise<void> entered, resume; auto wait = resume.get_future().share();
            weights.before_read = [&](std::size_t region) { if (region == 2) { entered.set_value(); wait.wait(); } };
            auto write = take(weights.begin_write(other, {item(0, "x"), item(1, "y"), item(2, "z")}));
            CHECK(!weights.fill_regions(write)); entered.get_future().wait();
            CHECK(!weights.acquire_views(other, {other->identity(), 0, "x", 1}));
            resume.set_value(); auto ready = take(weights.complete(write)); CHECK(ready.size() == 3);
            weights.before_read = {};
        }
        {
            std::promise<void> entered, resume; auto wait = resume.get_future().share();
            weights.before_read = [&](std::size_t) { entered.set_value(); wait.wait(); };
            auto write = take(weights.begin_write(other, {item(0, "changed")}));
            CHECK(!weights.fill_regions(write)); entered.get_future().wait();
            { std::ofstream out(path, std::ios::app); out << 'x'; }
            resume.set_value(); CHECK(!weights.complete(write)); weights.before_read = {};
            CHECK(other->state(0) == RegionState::isolated);
            CHECK(!weights.begin_write(other, {item(0, "again")}));
            CHECK(!weights.acquire_views(other, {other->identity(), 0, "changed", 2}));
        }
        {
            auto a = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 1}}, edge0::exec::Purpose::resident_weights));
            auto b = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 1}}, edge0::exec::Purpose::resident_weights));
            std::shared_ptr<Write> w1, w2;
            std::promise<void> gate; auto start = gate.get_future().share();
            std::thread t1([&] { start.wait(); auto result = weights.begin_write(a, {item(0, "race")}); if (result) w1 = result.value(); });
            std::thread t2([&] { start.wait(); auto result = weights.begin_write(b, {item(0, "race")}); if (result) w2 = result.value(); });
            gate.set_value(); t1.join(); t2.join(); CHECK(static_cast<bool>(w1) != static_cast<bool>(w2));
        }
        {
            auto group = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 2}}, edge0::exec::Purpose::resident_weights, Protection::group));
            auto ready = publish(group, {item(0, "g0"), item(1, "g1")});
            auto use = take(weights.acquire_views(group, ready[0]));
            CHECK(!weights.begin_write(group, {item(1, "g2")}));
            CHECK(use.use->valid()); CHECK(!weights.release_use(use.use));
            {
                auto write = take(weights.begin_write(group, {item(1, "g2")}));
                CHECK(!weights.acquire_views(group, ready[0]));
            }
            CHECK(!weights.release_use(take(weights.acquire_views(group, ready[1])).use));
            CHECK(!weights.release_storage(group));
        }
        CHECK(!weights.release_storage(other)); CHECK(!weights.release_storage(storage));
    }
    CHECK(budget.current() == 0);
    {
        exec::Budget tiny(1); Weights weights(tiny);
        auto rejected = weights.create_storage(backend, {{{GGML_TYPE_F32, {4}}, 8}}, edge0::exec::Purpose::resident_weights);
        CHECK(!rejected && rejected.error().category == ErrorCategory::budget_exceeded);
        CHECK(tiny.current() == 0 && tiny.peak() == 0);
    }
    std::filesystem::remove(path);
    std::cout << "weights: direct content, atomic IDs, publication barrier, isolation, stale bindings, protection, reuse and concurrent writers passed\n";
}
