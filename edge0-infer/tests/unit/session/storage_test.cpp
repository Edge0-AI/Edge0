#include "session/storage.h"
#include "tests/support/check.h"
#include <cmath>
using namespace edge0;
using namespace edge0::session;
template<class T> T take(const Result<T>& result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); } return result.value();
}
int main() {
    {
        StateStore store;auto object=std::make_shared<int>(7);CHECK(!store.put("aux.checkpoint",object));
        CHECK(!store.save_checkpoint(2,[]{return exec::Status{};}));CHECK(store.checkpoint()==2);
        CHECK(store.save_checkpoint(3,[&]()->exec::Status{CHECK(!store.checkpoint());return Error{ErrorCategory::backend_error,"copy failed"};}));
        CHECK(!store.checkpoint());CHECK(!store.save_checkpoint(4,[]{return exec::Status{};}));
        CHECK(!store.restore_checkpoint([]{return exec::Status{};}));CHECK(store.checkpoint()==4);
        CHECK(!store.discard_checkpoint());CHECK(!store.checkpoint() && take(store.get<int>("aux.checkpoint"))==object);
    }
    auto created = platform::BackendFacility::create("CPU");
    auto facility = std::make_shared<platform::BackendFacility>(std::move(created.first));
    auto backend = take(exec::resolve_stages("CPU", "CPU", facility)).decode;
    exec::Budget budget(64 * 1024 * 1024);
    {
        auto cache = take(KVCache::create(budget, backend, 257, {{"k", GGML_TYPE_F32, {256, 2}, 1, 256}}));
        CHECK(!cache->truncate(100,110));CHECK(cache->truncate(111,110));CHECK(cache->truncate(-1,110));
        auto state = take(cache->storage("k")); auto* tensor = state->tensor();
        CHECK(tensor->ne[0] == 256 && tensor->ne[1] == 512 && tensor->ne[2] == 2);
        CHECK(ggml_nbytes(tensor) == 1048576 && tensor->nb[2] == 524288);
        CHECK(budget.current(edge0::exec::Purpose::session_state) == ggml_backend_buffer_get_size(state->allocation->buffer));
        auto graph = take(exec::create_graph(budget, backend, exec::Phase::decode));
        auto range = take(cache->write_range(graph->context(), "k", {100, 10}));
        CHECK(range.rows.front() == 100 && range.rows.back() == 109);
        CHECK(range.view->view_offs == 100 * tensor->nb[1] && range.view->nb[2] == 524288);
        auto mask = take(cache->causal_mask("k", {100, 10}));
        for (int row = 0; row < 10; ++row) for (int column = 0; column < 256; ++column) {
            auto value = ggml_fp16_to_fp32(mask[static_cast<std::size_t>(row * 256 + column)]);
            CHECK(column <= 100 + row ? value == 0 : std::isinf(value) && value < 0);
        }
        auto history = take(cache->history_view(graph->context(), "k", {256, 1}));
        CHECK(history->ne[1] == 512);
        auto padded = take(cache->causal_mask("k", {256, 1})); CHECK(padded.size() == 512);
        for (int i = 0; i < 512; ++i) CHECK(i <= 256 ? padded[i] == 0 : std::isinf(ggml_fp16_to_fp32(padded[i])));
        const auto occupied = budget.current();
        auto overflow = cache->write_range(graph->context(), "k", {257, 1});
        CHECK(!overflow && overflow.error().category == ErrorCategory::invalid_argument);
        CHECK(budget.current() == occupied);
        auto rejected = KVCache::create(budget, backend, 32, {{"k", GGML_TYPE_F16, {64}, 1, 1, Visibility::sliding_window}});
        CHECK(!rejected && rejected.error().category == ErrorCategory::unsupported);
        CHECK(budget.current() == occupied);
    }
    CHECK(budget.current() == 0);
    {
        auto cache = take(KVCache::create(budget, backend, 200, {{"outer", GGML_TYPE_F16, {64, 2}, 2, 1}, {"middle", GGML_TYPE_F16, {64, 2}, 1, 1}}));
        auto graph = take(exec::create_graph(budget, backend, exec::Phase::decode));
        for (const auto& name : {"outer", "middle"}) {
            auto range = take(cache->write_range(graph->context(), name, {100, 10}));
            auto* tensor = take(cache->storage(name))->tensor();
            const int axis = std::string(name) == "outer" ? 2 : 1;
            CHECK(range.view->view_offs == tensor->nb[axis] * 100 && range.view->ne[axis] == 10);
            for (int i = 0; i < 4; ++i) CHECK(range.view->nb[i] == tensor->nb[i]);
            CHECK(take(cache->history_length(name, {100, 10})) == 110);
        }
    }
    CHECK(budget.current() == 0);
    {
        StateStore store;
        auto object = std::make_shared<int>(7); CHECK(!store.put("value", object));
        const auto occupied_before = budget.current();
        auto duplicate = store.put("value", std::make_shared<int>(9));
        CHECK(duplicate && duplicate->category == ErrorCategory::conflict);
        CHECK(budget.current() == occupied_before);
        CHECK(take(store.get<int>("value")) == object && !take(store.get<int>("missing")));
        auto wrong=store.get<double>("value");
        CHECK(!wrong && wrong.error().category==ErrorCategory::conflict);
        CHECK(wrong.error().message.find("value")!=std::string::npos);
        CHECK(take(store.get<int>("value"))==object);
        CHECK(store.remove("value") == object && !store.remove("value"));
        auto state = take(exec::allocate_state_storage(budget, backend, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        CHECK(!store.put("state", state));
        auto graph = take(exec::create_graph(budget, backend, exec::Phase::decode));
        auto* tensor = take(graph->bind_states(state));
        CHECK(!graph->output(ggml_scale(graph->context(), tensor, 2)));
        auto plan = take(exec::build_plan(graph));
        const auto occupied = budget.current(); state.reset();
        { auto removed = store.remove("state"); CHECK(removed); }
        CHECK(budget.current() == occupied);
        CHECK(plan->submit()); CHECK(!plan->release_plan()); graph.reset();
    }
    CHECK(budget.current() == 0);
    std::cout << "session: aligned and exact layouts, causal masks, bounds, both token axes and state ownership passed\n";
}
