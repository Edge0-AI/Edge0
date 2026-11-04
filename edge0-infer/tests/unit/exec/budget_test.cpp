#include "exec/budget.h"
#include "tests/support/check.h"
#include <random>
#include <thread>
using edge0::exec::Purpose;
#include <vector>
int main() {
    edge0::exec::Budget budget(100);
    {
        auto a = budget.reserve(40, Purpose::session_state, "CPU"); CHECK(a);
        CHECK(budget.current() == 40 && budget.peak() == 40 && budget.current(edge0::exec::Purpose::session_state) == 40);
        auto rejected = budget.reserve(61, Purpose::metadata, "host");
        CHECK(!rejected && rejected.error().category == edge0::ErrorCategory::budget_exceeded);
        CHECK(budget.current() == 40 && budget.peak() == 40);
        CHECK(a.value()->resize(101));
        CHECK(a.value()->size() == 40 && budget.peak() == 40);
        CHECK(!a.value()->resize(60));
        CHECK(budget.current() == 60 && budget.peak() == 60);
        CHECK(!a.value()->resize(20));
        CHECK(budget.current() == 20 && budget.peak() == 60);
        auto group = budget.reserve(std::vector<edge0::exec::Budget::Charge>{{30, Purpose::session_state, "CPU"}, {51, Purpose::metadata, "host"}});
        CHECK(!group && budget.current() == 20 && budget.peak() == 60);
    }
    CHECK(budget.current() == 0);
    std::mt19937 random(731);
    std::vector<edge0::exec::Budget::Ticket> tickets;
    std::map<Purpose, std::size_t> expected;
    std::map<std::string, std::size_t> peaks;
    for (int step = 0; step < 3000; ++step) {
        const auto before = budget.snapshot();
        if (tickets.empty() || random() % 3 == 0) {
            const auto purpose = edge0::exec::purposes[random() % 7];
            const std::string device = random() % 2 ? "CPU" : "MTL0";
            const auto bytes = random() % 80;
            auto ticket = budget.reserve(bytes, purpose, device);
            if (ticket) tickets.push_back(ticket.value());
            else {
                const auto after = budget.snapshot();
                CHECK(after.current == before.current && after.peak == before.peak);
                CHECK(after.by_purpose == before.by_purpose);
                CHECK(after.by_device.size() == before.by_device.size());
                for (const auto& item : before.by_device) {
                    CHECK(after.by_device.at(item.first).current == item.second.current);
                    CHECK(after.by_device.at(item.first).peak == item.second.peak);
                }
            }
        } else {
            auto index = random() % tickets.size();
            if (random() % 2) tickets.erase(tickets.begin() + index);
            else {
                const auto old = tickets[index]->size();
                if (tickets[index]->resize(random() % 120)) CHECK(tickets[index]->size() == old);
            }
        }
        const auto now = budget.snapshot();
        std::size_t sum = 0, devices = 0;
        expected.clear();
        for (const auto& ticket : tickets) expected[ticket->purpose()] += ticket->size();
        CHECK(now.by_purpose.size() == 7);
        for (auto purpose : edge0::exec::purposes) {
            CHECK(now.by_purpose.at(purpose) == expected[purpose]);
            sum += now.by_purpose.at(purpose);
        }
        for (const auto& item : now.by_device) {
            devices += item.second.current;
            peaks[item.first] = std::max(peaks[item.first], item.second.current);
            // CPU already reached 60 during the deterministic resize sequence.
            if (item.first == "CPU") peaks[item.first] = std::max(peaks[item.first], std::size_t(60));
            CHECK(item.second.peak == peaks[item.first]);
        }
        CHECK(sum == now.current && devices == now.current);
    }
    tickets.clear();
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) workers.emplace_back([&] {
        for (int j = 0; j < 1000; ++j) { auto ticket = budget.reserve(10, Purpose::workspace, "CPU"); CHECK(ticket); }
    });
    for (auto& worker : workers) worker.join();
    CHECK(budget.current() == 0 && budget.current(Purpose::workspace) == 0);
    CHECK(budget.peak() <= budget.limit());
    std::cout << "budget: lifetime, per-purpose counts, atomic rejection, resize, peak and 8000 concurrent reservations passed\n";
}
