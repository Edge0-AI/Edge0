#include "edge0/generation.h"
#include <iostream>
#include <stdexcept>
template<class T>T take(const edge0::Result<T>& r) {if(!r)throw std::runtime_error(r.error().message);return r.value();}
int main(int argc,char** argv) {
    if(argc<2 || argc>3){std::cerr<<"Usage: moe_example_native_chat <model-directory> [cpu|metal]\n";return 2;}
    try {
        const std::string device=argc==3?argv[2]:"cpu";
        std::string selected;
        for(const auto& item:edge0::query_capabilities().devices)if(item.backend==device){selected=item.name;break;}
        if(selected.empty())throw std::runtime_error("Requested device unavailable");
        auto runtime=take(edge0::open_runtime({{selected},{}}));
        edge0::ModelReference model;edge0::SessionReference session;
        auto close=[&] {
            if(session){take(edge0::release_session(session));session.reset();}
            if(model){take(edge0::unload_model(model));model.reset();}
            take(edge0::finish_runtime_use(runtime));
        };
        try {
            edge0::LoadOverrides loading;loading.device=selected;
            model=take(edge0::load_model(runtime,take(edge0::resolve_load_configuration(runtime,argv[1],loading))));
            session=take(edge0::create_session(model));
            edge0::GenerationOverrides options;options.temperature=0;options.max_tokens=32;
            auto configuration=take(edge0::resolve_generation_configuration(model,options));
            edge0::ChatInput conversation;
            for(const auto* text:{"Hello!", "Please tell me more."}) {
                conversation.messages.push_back({"user",text});
                auto result=take(edge0::generate(session,conversation,configuration,{},[](const edge0::OutputEvent& event)->std::optional<edge0::Error>{
                    if(event.kind==edge0::OutputKind::answer)std::cout<<event.text<<std::flush;
                    if(!std::cout)return edge0::Error{edge0::ErrorCategory::backend_error,"Output failed"};return {};
                }));
                if(result.error || result.stop_reason==edge0::StopReason::cancelled || result.stop_reason==edge0::StopReason::failed)throw std::runtime_error("Generation failed");
                std::cout<<'\n';std::cerr<<"reused_tokens="<<result.reused_tokens<<'\n';
                conversation.messages.push_back({"assistant",result.text, result.tool_calls,result.reasoning});
            }
        } catch(...) {close();throw;}
        close();return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
