#include "tools/cli/serve_status.h"
#include "tests/support/check.h"
int main() {
    CHECK(portal::authorized("192.0.2.1","key","","key"));
    CHECK(!portal::authorized("192.0.2.1","key","","wrong"));
    CHECK(!portal::authorized("192.0.2.1","key","",""));
    CHECK(portal::authorized("192.0.2.1","key","Bearer wrong","key"));
    CHECK(portal::authorized("192.0.2.1","key","Bearer key","wrong"));
    CHECK(portal::authorized("127.0.0.1","key","wrong","wrong"));

    using namespace portal;
    for (const auto& address : {"127.0.0.1","127.5.0.1","::1","::ffff:127.0.0.1"})
        for (const auto& header : {"","Bearer wrong","Bearer secre","Bearer secret","Bearer secret-more"})
            CHECK(authorized(address,"secret",header));
    for (const auto& address : {"192.168.1.2","127.0.0.999","127.0.0.1.attacker","::ffff:192.168.1.2"}) {
        CHECK(!authorized(address,"",""));
        for (const auto& header : {"","Bearer wrong","Bearer secre","Bearer secret-more"}) CHECK(!authorized(address,"secret",header));
        CHECK(authorized(address,"secret","Bearer secret"));
    }
    History h;
    CHECK(!totals_json(h.totals).contains("experts"));
    CHECK(!totals_json(h.totals).contains("prefill_tokens_per_second"));
    CHECK(!totals_json(h.totals).contains("decode_tokens_per_second"));
    edge0::GenerationResult result;
    result.input.resize(12); result.reused_tokens = 4; result.computed_tokens = 11; result.generated_count = 4;
    result.request_summary = {2.0,3.0,1.0,3,5.0,{},{}};
    result.text = "PRIVATE_TOKEN_TEXT"; result.reasoning = "PRIVATE_TOKEN_TEXT";
    result.reuse_method = edge0::ReuseMethod::restore; result.checkpoint_restored = true;
    result.weight_activity.applicable = true; result.weight_activity.prefill = {10,3,7,70}; result.weight_activity.decode = {5,1,4,40};
    RequestRecord base; base.endpoint = "/v1/chat/completions"; base.source = "127.0.0.1"; base.time = "2026-11-03T00:00:00.000Z"; base.http_status = 200;
    auto record = completed_record(base,result);
    CHECK(record_json(record).dump().find("PRIVATE_TOKEN_TEXT") == std::string::npos);
    {
        // A failure is recorded with the status the client received: the mapped status before a stream opens, 200 after.
        const auto before = record_json(failed_record(base,400,"context_length_exceeded","Input exceeds context capacity","input",false));
        CHECK(before["http_status"] == 400 && before["code"] == "context_length_exceeded" && before["param"] == "input");
        const auto after = record_json(failed_record(base,503,"shutting_down","Server shutting down","",true));
        CHECK(after["http_status"] == 200 && after["code"] == "shutting_down");
    }
    h.add(record); h.add(record);
    for (int status : {429,400,401}) { base.http_status = status; h.add(base); }
    const auto totals = totals_json(h.totals);
    CHECK(totals["requests"] == 2 && totals["rejected_busy"] == 1 && totals["rejected_invalid"] == 1 && totals["unauthorized"] == 1);
    CHECK(totals["input_tokens"] == 24 && totals["reused_tokens"] == 8 && totals["computed_tokens"] == 22 && totals["generated_tokens"] == 8);
    CHECK(totals["checkpoint_restores"] == 2 && totals["continued_turns"] == 2);
    CHECK(totals["first_token_seconds_mean"] == 2 && totals["prefill_tokens_per_second"] == 4);
    CHECK(totals["decode_seconds"] == 6 && totals["decode_tokens_per_second"] == 1);
    CHECK(totals["experts"] == Json({{"demand",30},{"hits",8},{"loads",22},{"bytes",220}}));
    for (int i=0;i<101;++i) { base.time = std::to_string(i); h.add(base); }
    CHECK(h.recent.size() == 100 && h.recent.front().time == "100" && h.recent.back().time == "1");
    CHECK(h.last && h.last->entered && h.last->http_status == 200);
    History zero;
    result.request_summary = {}; result.weight_activity.applicable = false;
    zero.add(completed_record(base,result));
    CHECK(!totals_json(zero.totals).contains("experts") && !totals_json(zero.totals).contains("first_token_seconds_mean"));
    result.request_summary.first_token_seconds = 0; result.request_summary.decode_seconds = 0; result.request_summary.decode_forwards = 0;
    zero.add(completed_record(base,result));
    CHECK(!totals_json(zero.totals).contains("prefill_tokens_per_second") && !totals_json(zero.totals).contains("decode_tokens_per_second"));
    History failed;
    RequestRecord early; early.entered = true; early.result_available = false; early.http_status = 500;
    early.input_tokens = 12; early.stop_reason = "failed";
    failed.add(early);
    CHECK(totals_json(failed.totals)["requests"] == 1 && totals_json(failed.totals)["input_tokens"] == 12);
    CHECK(record_json(early)["generated_tokens"].is_null() && record_json(early)["reused_tokens"].is_null());
    edge0::MemorySnapshot memory;
    memory.ledger.by_device["CPU"] = {12,20,edge0::MemoryLocation::system};
    memory.ledger.by_device["GPU"] = {24,40,edge0::MemoryLocation::dedicated};
    CHECK(memory_json(memory)["ledger"]["by_device"]["CPU"]["memory"] == "system");
    CHECK(memory_json(memory)["ledger"]["by_device"]["GPU"]["memory"] == "dedicated");
    StatusInput input; input.history = h;
    CHECK(status_json(input)["current"].is_null());
    input.progress = {edge0::GenerationPhase::prefill,13,0,4,9,0,std::chrono::steady_clock::time_point{}, {}};
    input.now = input.progress.started + std::chrono::seconds(5);
    CHECK(status_json(input)["current"]["pending_tokens"] == 9);
    CHECK(!status_json(input)["current"].contains("first_token_seconds"));
    input.progress.phase = edge0::GenerationPhase::decode; input.progress.computed_tokens = 13;
    input.progress.pending_tokens = 0; input.progress.generated_tokens = 4;
    input.progress.first_token = input.progress.started + std::chrono::seconds(2);
    const auto current = status_json(input)["current"];
    CHECK(current["first_token_seconds"] == 2 && current["prefill_tokens_per_second"] == 6.5);
    CHECK(current["decode_elapsed_seconds"] == 3 && current["decode_tokens_per_second"] == 1);
    std::cout << "serve status: aggregate definitions, optional rates, auth matrix, bounded records and progress JSON passed\n";
}
