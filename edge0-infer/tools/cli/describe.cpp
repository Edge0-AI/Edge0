#include "runtime_options.h"
#include "edge0/generation.h"
#include <iostream>
#include <stdexcept>
namespace cli {
void model_option(std::map<std::string,std::string>&,const std::string&);
void model_description(std::ostream&,const edge0::ModelDescription&);
}
edge0::LogReceiver cli_log_receiver(bool);
namespace {
template<class T> T take(const edge0::Result<T>& result) {
    if(!result) throw std::runtime_error(result.error().message);
    return result.value();
}
}
int describe_command(int argc,char** argv) {
    try {
        edge0::RuntimeOptions runtime_options;
        std::string directory,backend="cpu";
        edge0::LoadOverrides overrides;
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if(i+1>=argc) throw std::runtime_error("Missing describe argument: "+arg);
            const std::string value=argv[++i];
            if(cli::runtime_option(arg,value,runtime_options)) {}
            else if(arg=="--model") directory=value;
            else if(arg=="--device") backend=value;
            else if(arg=="--prefill-chunk") {
                std::size_t used=0;overrides.prefill_chunk=std::stoll(value,&used);
                if(used!=value.size()) throw std::runtime_error("Invalid prefill chunk");
            }
            else if(arg=="--model-option") cli::model_option(overrides.model_options,value);
            else throw std::runtime_error("Invalid describe argument: "+arg);
        }
        if(directory.empty()||!cli::device_family(backend))
            throw std::runtime_error("Usage: describe --model <directory> [--device <cpu|metal|vulkan>] [--threads <n>] [--prefill-threads <n>] [--memory-budget <bytes>] [--prefill-chunk <n>] [--model-option <name=value>]");
        std::string device;
        const auto logger=cli_log_receiver(false);
        for(const auto& item:edge0::query_capabilities(false,logger).devices) if(item.backend==backend) {device=item.name;break;}
        if(device.empty()) {std::cerr<<"SKIP: requested device unavailable\n";return 77;}
        auto runtime=take(cli::open_runtime(runtime_options,device,logger));
        struct Close {edge0::RuntimeReference runtime;~Close(){edge0::finish_runtime_use(runtime);}} close{runtime};
        auto model=take(edge0::load_model(runtime,take(edge0::resolve_load_configuration(runtime,directory,overrides))));
        cli::model_description(std::cout,take(edge0::query_model(model)));std::cout<<'\n';
        take(edge0::unload_model(model));
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
