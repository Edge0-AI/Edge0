#include "session/storage.h"
#include "tests/support/check.h"
#include <cstring>
using namespace edge0;
using namespace edge0::session;
template<class T> T take(Result<T> result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); } return result.value();
}
int main() {
    auto a = platform::BackendFacility::create("CPU"), b = platform::BackendFacility::create("CPU");
    const auto stages = take(exec::resolve_stages("a", "b", std::map<std::string, std::shared_ptr<platform::BackendFacility>>{
        {"a", std::make_shared<platform::BackendFacility>(std::move(a.first))},
        {"b", std::make_shared<platform::BackendFacility>(std::move(b.first))}}));
    for (bool copy : {false, true}) {
        auto pair = stages;
        if (copy) pair.consumption = [](auto, auto) { return platform::Consumption::copy; };
        exec::Budget budget(1024 * 1024);
        auto kv = take(KVCache::create(budget, pair, 8, {{"k", GGML_TYPE_F32, {2, 3}, 1, 1}}));
        auto write = [&](Step step) {
            const auto storage = take(kv->storage("k"));
            auto* tensor = storage->tensor();
            for (std::size_t head = 0; head < 3; ++head) {
                std::vector<float> rows(static_cast<std::size_t>(step.n) * 2);
                for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<float>(head * 100 + step.e * 2 + i + 1);
                ggml_backend_tensor_set(tensor, rows.data(), head * tensor->nb[2] + static_cast<std::size_t>(step.e) * tensor->nb[1], rows.size() * sizeof(float));
            }
            storage->mark_written();
            CHECK(!kv->commit_rows(step, {true, false, {storage->allocation}}));
        };
        auto first = take(kv->storage("k"));
        write({0, 3});
        if (copy) {
            CHECK(take(kv->synced("k", pair.prefill)) == 3);
            CHECK(take(kv->synced("k", pair.decode)) == 0);
        }
        CHECK(!kv->make_current(pair.decode, 3));
        auto second = take(kv->storage("k"));
        CHECK(second->allocation->backend == pair.decode && second->current());
        const auto row_bytes = 2 * 3 * sizeof(float);
        CHECK(first->replicas->stats.bytes == (copy ? 3 * row_bytes : 0));
        std::vector<unsigned char> left(ggml_nbytes(first->tensor())), right(left.size());
        ggml_backend_tensor_get(first->tensor(), left.data(), 0, left.size());
        ggml_backend_tensor_get(second->tensor(), right.data(), 0, right.size());
        CHECK(left == right);
        write({3, 1});
        if (copy) {
            CHECK(take(kv->synced("k", pair.decode)) == 4);
            CHECK(take(kv->synced("k", pair.prefill)) == 3);
        }
        CHECK(!kv->make_current(pair.prefill, 4));
        CHECK(first->replicas->stats.bytes == (copy ? 4 * row_bytes : 0));
        ggml_backend_tensor_get(first->tensor(), left.data(), 0, left.size());
        ggml_backend_tensor_get(second->tensor(), right.data(), 0, right.size());
        CHECK(left == right);
        const auto prior = first->replicas->stats.bytes;
        CHECK(!kv->truncate(2, 4));
        if (copy) {
            CHECK(take(kv->synced("k", pair.prefill)) == 2);
            CHECK(take(kv->synced("k", pair.decode)) == 2);
        }
        CHECK(!kv->make_current(pair.decode, 2));
        CHECK(first->replicas->stats.bytes == prior);
        CHECK(kv->make_current(pair.decode, 9));
    }
    std::cout << "mixed KV: multi-head row ranges, reverse increments, truncation and sharing passed\n";
}
