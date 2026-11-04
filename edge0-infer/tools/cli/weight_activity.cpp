#include "edge0/weight_activity.h"
#include <ostream>
#include <map>
#include <stdexcept>
#include <string>
namespace cli {
void model_option(std::map<std::string,std::string>& options,const std::string& value) {
    const auto equals=value.find('=');
    if(equals==std::string::npos || !equals || equals+1==value.size()) throw std::runtime_error("Expected model option name=value");
    // A repeated name would silently keep only the last value.
    if(!options.emplace(value.substr(0,equals),value.substr(equals+1)).second)
        throw std::runtime_error("Model option given more than once: "+value.substr(0,equals));
}
void weight_activity(std::ostream& out,const edge0::WeightActivity& activity) {
    if(!activity.applicable) { out<<"null"; return; }
    // Serialize hit_rate at a fixed high precision regardless of the caller's stream
    // state, so every consumer (run's result.json, forward's report.json, bench, serve)
    // emits byte-identical floating-point output on every platform. The surrounding
    // summary writers already use setprecision(17); this makes the forward report path,
    // which uses a default-precision stream, agree with them. Restore on return.
    const auto previous=out.precision(17);
    auto counts=[&](const edge0::ExpertActivity& a) {
        out<<"{\"demand\":"<<a.demand<<",\"hits\":"<<a.hits<<",\"loads\":"<<a.loads<<",\"bytes\":"<<a.bytes
           <<",\"hit_rate\":"<<(a.demand?double(a.hits)/double(a.demand):0.0)<<'}';
    };
    out<<"{\"decode\":"; counts(activity.decode); out<<",\"prefill\":"; counts(activity.prefill); out<<",\"prefetch\":{\"initiated\":"<<activity.prefetch.initiated
       <<",\"late\":"<<activity.prefetch.late<<",\"bytes\":"<<activity.prefetch.bytes
       <<",\"tail_bytes\":"<<activity.prefetch.tail_bytes<<",\"discarded\":"<<activity.prefetch.discarded<<"}}";
    out.precision(previous);
}
}
