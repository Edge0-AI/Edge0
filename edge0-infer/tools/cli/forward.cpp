#include "runtime_options.h"
#include "edge0/weight_activity.h"
#include <map>
#include <ostream>
#include <string>
namespace cli {
void model_option(std::map<std::string,std::string>&,const std::string&);
void weight_activity(std::ostream&,const edge0::WeightActivity&);
}
#include "edge0/forward_check.h"
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
edge0::LogReceiver cli_log_receiver(bool verbose);
namespace {
std::vector<std::int32_t> tokens(const std::string& json,const std::string& name) {
    const auto key=json.find('"'+name+'"');
    if(key==std::string::npos)throw std::runtime_error("Missing token array: "+name);
    auto index=json.find(':',key+name.size()+2);
    auto spaces=[&] { while(index<json.size()&&std::isspace(static_cast<unsigned char>(json[index])))++index; };
    if(index==std::string::npos)throw std::runtime_error("Invalid token JSON");
    ++index; spaces(); if(index==json.size()||json[index++]!='[')throw std::runtime_error("Expected token array");
    std::vector<std::int32_t> result; spaces();
    if(index<json.size()&&json[index]==']')return result;
    while(index<json.size()) {
        spaces(); auto start=index;
        while(index<json.size()&&std::isdigit(static_cast<unsigned char>(json[index])))++index;
        if(start==index)throw std::runtime_error("Expected nonnegative token ID");
        const auto value=std::stoull(json.substr(start,index-start));
        if(value>std::uint64_t(std::numeric_limits<std::int32_t>::max()))throw std::runtime_error("Token ID out of range");
        result.push_back(std::int32_t(value)); spaces();
        if(index<json.size()&&json[index]==']')return result;
        if(index==json.size()||json[index++]!=',')throw std::runtime_error("Expected comma in token array");
    }
    throw std::runtime_error("Unterminated token array");
}
void write(const std::filesystem::path& path,const void* data,std::size_t size) {
    std::ofstream stream(path,std::ios::binary); stream.write(static_cast<const char*>(data),std::streamsize(size));
    if(!stream)throw std::runtime_error("Cannot write output: "+path.string());
}
void text(const std::filesystem::path& path,const std::string& content) {write(path,content.data(),content.size());}
void array(std::ostream& stream,const std::vector<std::int32_t>& values) {
    stream<<'['; for(std::size_t i=0;i<values.size();++i)stream<<(i?",":"")<<values[i]; stream<<']';
}
}
int forward_command(int argc,char** argv) {
    try {
        edge0::RuntimeOptions runtime_options;
        bool verbose=false;
        edge0::ForwardCheckOptions options; std::string device="cpu",input,out;
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--verbose")verbose=true;
            else if(arg=="--split-last")options.split_last=true;
            else if(arg=="--observe")options.observe=true;
            else if(i+1<argc && (arg=="--threads"||arg=="--prefill-threads"||arg=="--memory-budget"||arg=="--model"||arg=="--device"||arg=="--tokens"||arg=="--out"||arg=="--model-option"||arg=="--route-trace"||arg=="--observe-steps")) {
                const std::string value=argv[++i];
                if(cli::runtime_option(arg,value,runtime_options)) {}
                else if(arg=="--route-trace")options.route_trace=value;
                else if(arg=="--observe-steps")options.observe_decode_steps=std::stoull(value);
                else if(arg=="--model")options.model_directory=value;
                else if(arg=="--model-option")cli::model_option(options.model_options,value);
                if(arg=="--device")device=value;
                if(arg=="--tokens")input=value;
                if(arg=="--out")out=value;
            } else throw std::runtime_error("Invalid forward argument: "+arg);
        }
        if(options.model_directory.empty()||input.empty()||out.empty()||!cli::device_family(device))
            throw std::runtime_error("Usage: forward --model <dir> --device <cpu|metal|vulkan> --tokens <json> --out <dir> [--model-option <name=value>] [--observe] [--split-last] [--verbose]");
        if(std::filesystem::exists(out))throw std::runtime_error("Output directory already exists");
        std::ifstream source(input); if(!source)throw std::runtime_error("Cannot read tokens");
        const std::string json((std::istreambuf_iterator<char>(source)),{});
        options.input=tokens(json,"input"); options.generated=tokens(json,"generated");
        const auto log_receiver=cli_log_receiver(verbose);
        for(const auto& item:edge0::query_capabilities(false,log_receiver).devices)if(item.backend==device){options.device=item.name;break;}
        if(options.device.empty()) {std::cerr<<"SKIP: requested device unavailable\n";return 77;}
        auto runtime=cli::open_runtime(runtime_options,options.device,log_receiver);
        if(!runtime)throw std::runtime_error(runtime.error().message);
        auto result=edge0::forward_check(runtime.value(),options);
        std::ostringstream runtime_summary;cli::runtime_json(runtime_summary,runtime.value());
        auto ended=edge0::finish_runtime_use(runtime.value());
        if(!result)throw std::runtime_error(result.error().message);
        if(!ended)throw std::runtime_error(ended.error().message);
        const auto& report=result.value(); const std::filesystem::path root(out);
        std::filesystem::create_directories(root/"layers");
        write(root/"logits.f32",report.logits.data(),report.logits.size()*sizeof(float));
        std::ostringstream metadata;
        metadata<<"{\"input\":";array(metadata,options.input); metadata<<",\"generated\":";array(metadata,options.generated);
        if(options.split_last) metadata<<",\"chunking\":\"split-last\"";
        metadata<<",\"logits_shape\":["<<report.rows<<','<<report.vocabulary<<"],\"row_semantics\":\"prefill positions, then logits after feeding each generated token\"}\n";
        text(root/"tokens.json",metadata.str());
        std::ostringstream index;index<<'[';
        for(std::size_t i=0;i<report.layers.size();++i) {
            const auto& record=report.layers[i];const auto file=std::to_string(i)+".bin";
            write(root/"layers"/file,record.data.data(),record.data.size());
            index<<(i?",":"")<<"{\"name\":\""<<record.name<<"\",\"file\":\""<<file<<"\",\"step\":"<<record.step<<",\"position\":"<<record.position<<",\"type\":"<<record.type<<",\"shape\":[";
            for(int j=0;j<4;++j)index<<(j?",":"")<<record.shape[j];
            index<<"]}";
        }
        index<<"]\n";text(root/"layers/index.json",index.str());
        std::ostringstream summary;
        summary<<"{\"runtime\":"<<runtime_summary.str()<<",\"budget_limit\":"<<report.budget.limit_bytes<<",\"budget_source\":\""<<(report.budget.source==edge0::BudgetSource::device_default?"device_default":"explicit_value")<<"\",\"budget_peak\":"<<report.budget_peak<<",\"budget_current\":"<<report.budget_current<<",\"weight_registrations\":[";
        for(std::size_t i=0;i<report.weight_registrations.size();++i)summary<<(i?",":"")<<report.weight_registrations[i];
        summary<<"],\"weight_activity\":";cli::weight_activity(summary,report.weight_activity);summary<<"}\n";text(root/"report.json",summary.str());std::cout<<summary.str();return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
