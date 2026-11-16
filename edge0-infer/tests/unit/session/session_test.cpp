#include "session/session.h"
#include "tests/support/runtime.h"
using namespace edge0;
using namespace edge0::session;
int main(int argc, char** argv) {
    auto backend = test::backend(argc, argv);
    exec::Budget budget(64 * 1024 * 1024);
    int released = 0, resets = 0;
    auto make = [&] {
        return test::take(Session::create(budget, backend, 8,
            {{"k", GGML_TYPE_F32, {4}, 1, 1, Visibility::causal}}, {"state"}));
    };
    auto initialize = [](KVCache&, StateStore&) -> exec::Status { return {}; };
    auto release = [&](KVCache&, StateStore&) { ++released; };
    {
        auto session = make();
        CHECK(session->initialize([](KVCache&, StateStore&) -> exec::Status {
            return Error{ErrorCategory::backend_error, "injected"};
        }, release));
        CHECK(released == 1 && budget.current() == 0);
    }
    {
        auto session = make();
        CHECK(!session->initialize(initialize, release));
        CHECK(!session->begin_update(std::vector<std::int32_t>(9, 1)));
        CHECK(session->query().e == 0);
        auto update = test::take(session->begin_update({1, 2}));
        auto buffer = test::take(session->kv().storage("k"))->allocation;
        exec::Completion good{true, false, {buffer}};
        CHECK(session->commit_update(update, {false, false, {buffer}}, {"state"}));
        CHECK(session->commit_update(update, {true, true, {buffer}}, {"state"}));
        CHECK(session->commit_update(update, {true, false, {}}, {"state"}));
        CHECK(session->commit_update(update, good, {}));
        CHECK(session->query().e == 0 && session->query().tokens.empty());
        CHECK(!session->commit_update(update, good, {"state"}));
        CHECK(session->query().e == 2 && session->query().tokens == update.tokens);
        CHECK(session->commit_update(update, good, {"state"}));
        update = test::take(session->begin_update({3}));
        CHECK(!session->report_interruption(update, false));
        CHECK(session->query().valid && session->query().e == 2);
        CHECK(!session->reset([&](KVCache&, StateStore&) -> exec::Status { ++resets; return {}; }));
        CHECK(resets == 1 && session->query().e == 0 && session->query().tokens.empty());
        update = test::take(session->begin_update({3}));
        CHECK(!session->report_interruption(update, true));
        CHECK(!session->query().valid && session->query().e == 0);
        CHECK(!session->begin_update({4}));
        buffer.reset(); good.written.clear();
        session->request_release();
        CHECK(session->query().released && released == 2 && budget.current() == 0);
        session->request_release(); CHECK(released == 2);
    }
    {
        auto session = make(); CHECK(!session->initialize(initialize, release));
        CHECK(!session->begin_use());
        const auto occupied = budget.current();
        session->request_release();
        CHECK(!session->query().released && budget.current() == occupied);
        CHECK(!session->begin_update({1}));
        session->end_use();
        CHECK(session->query().released && released == 3 && budget.current() == 0);
        CHECK(session->begin_use());
    }
    for (bool use : {false, true}) for (int completion = 0; completion < 3; ++completion) {
        auto session = make(); CHECK(!session->initialize(initialize, release));
        const auto before = released;
        if (use) CHECK(!session->begin_use());
        auto update = test::take(session->begin_update({1}));
        session->request_release();
        if (use) session->end_use();
        // The pending transaction still owns KV storage after the outer use ends.
        CHECK(!session->query().released && released == before && budget.current() > 0);
        if (completion == 0) {
            auto buffer = test::take(session->kv().storage("k"))->allocation;
            CHECK(!session->commit_update(update, {true, false, {buffer}}, {"state"}));
        } else {
            CHECK(!session->report_interruption(update, completion == 2));
        }
        CHECK(session->query().released && released == before + 1 && budget.current() == 0);
        session->request_release();
        CHECK(released == before + 1);
    }
    std::cout << "session transactions, failure validity, reset and deferred release passed\n";
}
