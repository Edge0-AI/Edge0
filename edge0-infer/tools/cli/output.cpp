#include "edge0/generation.h"
#include <iostream>
namespace cli {
void finish_thinking(bool& open) {
    if(open) {std::cerr<<"\n[/thinking]\n"<<std::flush;open=false;}
}
std::optional<edge0::Error> output_event(const edge0::OutputEvent& event,bool& thinking,bool answer) {
    if(event.kind==edge0::OutputKind::reasoning) {
        if(!thinking && !event.text.empty()) {std::cerr<<"[thinking]\n";thinking=true;}
        std::cerr<<event.text<<std::flush;
        if(!std::cerr) return edge0::Error{edge0::ErrorCategory::backend_error,"Cannot write reasoning"};
    } else if(event.kind!=edge0::OutputKind::token_logprob) {
        finish_thinking(thinking);
        if(answer && event.kind==edge0::OutputKind::answer) {
            std::cout<<event.text<<std::flush;
            if(!std::cout) return edge0::Error{edge0::ErrorCategory::backend_error,"Cannot write answer"};
        }
    }
    return {};
}
}
