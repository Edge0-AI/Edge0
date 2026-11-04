#include "session/session.h"
#include "tests/support/runtime.h"
using namespace edge0;
using namespace edge0::session;
using test::take;
int main(int argc, char** argv) {
    auto backend=test::backend(argc,argv);exec::Budget budget(64*1024*1024);
    const Identity identity{"adapter","template","tokenizer"}, other{"adapter","other","tokenizer"};
    for (bool whole : {false,true}) for (bool supports : {false,true}) {
        if (!whole && supports) continue;
        auto s=take(Session::create(budget,backend,32,{},{}));
        std::vector<std::string> calls;bool save_failure=false,restore_failure=false;
        CHECK(!s->initialize([](KVCache&,StateStore&)->exec::Status{return {};},[](KVCache&,StateStore&){}));
        CHECK(!s->configure_reuse(identity,whole,supports,
            [&](KVCache&,StateStore& store)->exec::Status{calls.push_back("save");CHECK(!store.checkpoint());if(save_failure)return Error{ErrorCategory::backend_error,"save failure"};return {};},
            [&](KVCache&,StateStore&)->exec::Status{calls.push_back("restore");if(restore_failure)return Error{ErrorCategory::backend_error,"restore failure"};return {};},
            [&](KVCache&,StateStore&)->exec::Status{calls.push_back("discard");return {};}));
        auto reset=[&](KVCache&,StateStore&)->exec::Status{calls.push_back("reset");return {};};
        auto compute=[&](std::size_t n) {auto u=take(s->begin_update(std::vector<std::int32_t>(n,1)));CHECK(!s->commit_update(u,{true,false,{}},{}));};
        auto assert_plan=[&](int p,int n,ReuseKind method,int k,const Identity& id) {
            const auto before=s->query();const auto old_calls=calls;
            auto plan=take(s->check_reuse(p,n,id));
            CHECK(plan.kind==method && plan.k==k);
            const auto after=s->query();CHECK(before.e==after.e && before.tokens==after.tokens && before.checkpoint==after.checkpoint && calls==old_calls);
            return plan;
        };
        assert_plan(0,7,ReuseKind::first,0,identity);
        compute(2);
        if(supports)CHECK(!s->save_checkpoint());
        compute(4);
        assert_plan(6,9,ReuseKind::keep,6,identity);
        assert_plan(4,7,!whole?ReuseKind::truncate:supports?ReuseKind::restore:ReuseKind::cold,!whole?4:supports?2:0,identity);
        assert_plan(6,6,!whole?ReuseKind::truncate:supports?ReuseKind::restore:ReuseKind::cold,!whole?5:supports?2:0,identity);
        auto mismatched=assert_plan(6,9,whole?ReuseKind::cold:ReuseKind::truncate,0,other);
        CHECK(!s->apply_reuse(mismatched,reset));CHECK(s->query().e==0 && !s->query().checkpoint);
        compute(6);
        auto early=assert_plan(1,7,whole?ReuseKind::cold:ReuseKind::truncate,whole?0:1,other);
        CHECK(!s->apply_reuse(early,reset));
        CHECK(!s->reset(reset));
        if(!supports)continue;
        compute(4);CHECK(!s->save_checkpoint());compute(3);
        for(int repetition=0;repetition<2;++repetition) {
            auto plan=assert_plan(5,5,ReuseKind::restore,4,other);
            CHECK(!s->apply_reuse(plan,reset));CHECK(s->query().checkpoint==4);compute(3);
        }
        CHECK(!s->reset(reset));compute(4);
        auto interrupted=assert_plan(4,5,ReuseKind::keep,4,other);
        CHECK(!s->apply_reuse(interrupted,reset));CHECK(!s->save_checkpoint());CHECK(s->query().checkpoint==4);
        save_failure=true;CHECK(s->save_checkpoint());CHECK(!s->query().checkpoint);save_failure=false;
        CHECK(!s->save_checkpoint());compute(2);restore_failure=true;
        auto plan=assert_plan(5,5,ReuseKind::restore,4,other);
        CHECK(s->apply_reuse(plan,reset));CHECK(!s->query().valid && !s->query().checkpoint);
        CHECK(s->reset(reset));
    }
    std::cout<<"exact reuse table, identities, checkpoint lifecycle and restore failure passed\n";
}
