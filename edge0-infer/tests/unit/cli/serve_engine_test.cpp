#include "tools/cli/serve_engine.h"
#include "tests/support/check.h"
#include <thread>
#include <atomic>
#include <chrono>
int main() {
    {
        portal::StreamHandoff<int> handoff;
        handoff.worker=std::thread([&]{handoff.finish(std::make_exception_ptr(portal::Failure(500,"early failure","backend_error")));});
        try {handoff.wait_ready();CHECK(false);}catch(const portal::Failure& f){CHECK(f.status==500);}
        handoff.abort();
    }
    {
        portal::StreamHandoff<int> handoff;int sink=0;
        handoff.worker=std::thread([&]{auto* value=handoff.await_sink();if(value)*value=1;handoff.finish(nullptr);});
        handoff.wait_ready();CHECK(sink==0);handoff.attach(sink);handoff.worker.join();CHECK(sink==1);
    }
    {
        portal::StreamHandoff<int> handoff;
        handoff.worker=std::thread([&]{CHECK(handoff.await_sink()==nullptr);handoff.finish(nullptr);});
        handoff.wait_ready();handoff.abort();
    }
    int calls=0,creates=0,releases=0;std::string order;bool valid=true,invalid=false,fail=false;
    portal::EngineHooks hooks;
    hooks.create=[&]{++creates;};hooks.release=[&]{++releases;order+='r';};
    hooks.unload=[&]{order+='u';if(fail)throw std::runtime_error("unknown");};hooks.finish=[&]{order+='f';};hooks.cancel=[]{};
    hooks.generate=[&](const portal::Request&,edge0::OutputReceiver)->edge0::Result<edge0::GenerationResult> {
        ++calls;if(invalid)return edge0::Error{edge0::ErrorCategory::session_invalid,"invalid"};
        edge0::GenerationResult result;result.state_valid=valid;return result;
    };
    portal::Engine engine(hooks);portal::Request request;
    auto first=engine.acquire();engine.generate(request,{});
    bool busy=false;std::thread second([&]{try {auto lease=engine.acquire();engine.generate(request,{});}catch(const portal::Failure& f){busy=f.status==429;}});second.join();
    CHECK(busy && calls==1);first.reset();
    try {portal::parse({{"model","x"},{"unknown",1}},portal::Protocol::chat,"x");CHECK(false);}catch(const portal::Failure&){}
    {auto lease=engine.acquire();valid=false;engine.generate(request,{});}
    {auto lease=engine.acquire();CHECK(creates==1 && releases==1);valid=true;invalid=true;engine.generate(request,{});}
    {auto lease=engine.acquire();CHECK(creates==2 && releases==2);invalid=false;engine.generate(request,{});}
    // Ensure a measurable idle gap: steady_clock granularity (QPC, ~100ns on
    // Windows) can make two back-to-back reads equal, leaving elapsed at 0.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    engine.expire(1e-12);CHECK(releases==3);
    {auto lease=engine.acquire();CHECK(creates==3);}
    order.clear();CHECK(engine.close());CHECK(order=="ruf");
    try {auto lease=engine.acquire();CHECK(false);}catch(const portal::Failure& f){CHECK(f.status==503);}
    std::atomic<bool> cancelled{false},released{false};
    auto waiting_hooks=hooks;
    waiting_hooks.cancel=[&]{cancelled=true;};waiting_hooks.release=[&]{released=true;};
    portal::Engine waiting(waiting_hooks);auto held=waiting.acquire();
    bool closed=false;std::thread closing([&]{closed=waiting.close();});
    while(!cancelled)std::this_thread::yield();
    CHECK(!released);held.reset();closing.join();CHECK(closed && released);
    portal::Engine broken(hooks);fail=true;order.clear();CHECK(!broken.close());CHECK(order=="ru");
    // MSVC 19.44 leaves a garbage process status when this main falls off the
    // end. Failed checks still exit 1. An explicit success status is required.
    return 0;
}
