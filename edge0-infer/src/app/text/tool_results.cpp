#include "app/text/tool_results.h"
#include "model/json.h"
#include <set>
#include <stdexcept>
namespace edge0::text {
std::optional<edge0::Error> accept_tool_results(const ChatInput& input) {
    const auto invalid=[](const char* message) {return edge0::Error{ErrorCategory::invalid_argument,message};};
    try {
        for(const auto& tool:input.tools)
            if(model::Json::parse(tool.json).kind!=model::Json::Kind::object) return invalid("Tool definition must be an object");
        if(input.tool_choice.mode==ToolChoiceMode::required || input.tool_choice.mode==ToolChoiceMode::function)
            return edge0::Error{ErrorCategory::unsupported,"Required and named tool choices require constrained decoding"};
        if(input.tool_choice.mode!=ToolChoiceMode::automatic && input.tool_choice.mode!=ToolChoiceMode::none)
            return invalid("Invalid tool choice");
        std::set<std::string> pending;
        for(const auto& message:input.messages) {
            if(message.role!="system" && message.role!="user" && message.role!="assistant" && message.role!="tool")
                return invalid("Unknown message role");
            if(message.role!="assistant" && (!message.tool_calls.empty() || message.reasoning_content))
                return invalid("Only assistant messages may carry reasoning or tool calls");
            if(message.role=="tool") {
                if(message.tool_call_id.empty() || !pending.erase(message.tool_call_id))
                    return invalid("Unknown or duplicate tool result");
            } else {
                if(!message.tool_call_id.empty()) return invalid("Only tool messages may carry a tool call ID");
                if(!pending.empty()) return invalid("Missing tool results");
                std::set<std::string> ids;
                for(const auto& call:message.tool_calls) {
                    if(call.id.empty() || call.name.empty() || !ids.insert(call.id).second)
                        return invalid("Empty or duplicate tool call identity");
                    if(model::Json::parse(call.arguments_json).kind!=model::Json::Kind::object)
                        return invalid("Tool arguments must be an object");
                }
                pending=std::move(ids);
            }
        }
        if(!pending.empty()) return invalid("Missing tool results");
        return {};
    } catch(const std::invalid_argument& e) {return edge0::Error{ErrorCategory::invalid_argument,e.what()};}
}
}
