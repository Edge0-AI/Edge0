#include "edge0/generation_check.h"
#include "tests/support/check.h"
#include <cstdlib>
#include <memory>
#include <new>

namespace allocation_probe {
thread_local bool active=false;
thread_local std::size_t bytes=0, copies=0;
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void release(void* pointer) { std::free(pointer); }
}
void* operator new(std::size_t bytes) {
    if(allocation_probe::active && bytes==allocation_probe::bytes) ++allocation_probe::copies;
    if(auto* pointer=std::malloc(bytes?bytes:1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { allocation_probe::release(pointer); }
void operator delete(void* pointer,std::size_t) noexcept { allocation_probe::release(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { allocation_probe::release(pointer); }
void operator delete[](void* pointer,std::size_t) noexcept { allocation_probe::release(pointer); }

template<class T> T take(const edge0::Result<T>& result) {
    if(!result) { std::cerr<<result.error().message<<'\n';std::exit(1); }
    return result.value();
}
int main() {
    {
        edge0::Result<std::vector<std::int32_t>> stored(std::vector<std::int32_t>(262144,7));
        const auto* original=stored.value().data();
        allocation_probe::bytes=stored.value().size()*sizeof(std::int32_t);
        allocation_probe::active=true;
        auto transferred=std::move(stored).value();
        allocation_probe::active=false;
        CHECK(transferred.data()==original && transferred.size()==262144 && transferred.back()==7);
        CHECK(allocation_probe::copies==0);
        edge0::Result<std::unique_ptr<int>> owner(std::make_unique<int>(42));
        auto pointer=std::move(owner).value();
        CHECK(pointer && *pointer==42);
    }
    const auto* fixtures=std::getenv("MOE_TINY");if(!fixtures)return 77;
    using namespace edge0;
    RuntimeOptions options;options.devices={"CPU"};options.cpu_threads=2;options.cpu_prefill_threads=2;
    auto runtime=take(open_runtime(options));
    const std::string directory=std::string(fixtures)+"/tiny-a";
    LoadOverrides load;load.context=4096;load.prefill_chunk=512;load.model_options={{"routing","actual"}};
    auto model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory,load))));
    auto session=take(create_session(model));
    std::string prompt;
    for(int i=0;i<256;++i) prompt+="hello ";
    const ChatInput chat{{{"user",prompt}}, {}};
    auto rendered=take(render_prompt(directory,chat));
    CHECK(rendered.tokens.size()>256);
    auto answer=take(tokenize(directory,"a"));
    CHECK(answer.tokens.size()==1);
    GenerationOverrides overrides;
    overrides.temperature=0;overrides.presence_penalty=0;overrides.max_tokens=3;
    overrides.logit_bias=std::map<std::int32_t,double>{{answer.tokens.front(),100}};
    overrides.logprobs=true;overrides.top_logprobs=1;
    const auto configuration=take(resolve_generation_configuration(model,overrides));
    allocation_probe::bytes=rendered.tokens.size()*sizeof(std::int32_t);
    std::size_t logprobs=0;
    std::string delivered;
    auto result=generate(session,chat,configuration,{},[&](const OutputEvent& event)->std::optional<Error> {
        if(event.kind==OutputKind::answer) delivered+=event.text;
        if(event.kind==OutputKind::token_logprob && ++logprobs==3) allocation_probe::active=true;
        return {};
    });
    allocation_probe::active=false;
    CHECK(result);
    const auto& completed=result.value();
    CHECK(!completed.error && completed.state_valid && completed.stop_reason==StopReason::length);
    CHECK(completed.input==rendered.tokens && completed.generated.size()==3 && completed.generated_count==3);
    CHECK(completed.text=="aaa" && completed.text==delivered && logprobs==3 && completed.logprobs.size()==3);
    std::cout<<"completed_input_copies="<<allocation_probe::copies
             <<" bytes="<<allocation_probe::copies*allocation_probe::bytes<<'\n';
    CHECK(allocation_probe::copies==0);
    auto checked=take(generate_chat_check(session,chat,configuration));
    CHECK(checked.result.input==completed.input && checked.result.generated==completed.generated);
    CHECK(checked.result.text==completed.text && checked.result.logprobs.size()==completed.logprobs.size());
    CHECK(checked.replay.tokens.size()==completed.input.size()+2 && !checked.replay.blocks.empty());
    take(release_session(session));take(unload_model(model));take(finish_runtime_use(runtime));
}
