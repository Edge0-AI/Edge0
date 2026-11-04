#include "models/edge0_35b/text.h"
#include "model/json.h"
#include <cstdint>
#include <stdexcept>
namespace edge0::models::edge0_35b {
namespace {
bool space(uint32_t c) {
  return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 ||
         c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
         c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f ||
         c == 0x3000;
}
std::string trim(const std::string &s) {
  size_t first = s.size(), end = 0;
  for (size_t p = 0; p < s.size();) {
    size_t start = p;
    auto c = static_cast<unsigned char>(s[p++]);
    uint32_t cp = c;
    size_t n = 0;
    if (c >= 0xc2 && c <= 0xdf) {
      cp = c & 31;
      n = 1;
    } else if (c >= 0xe0 && c <= 0xef) {
      cp = c & 15;
      n = 2;
    } else if (c >= 0xf0 && c <= 0xf4) {
      cp = c & 7;
      n = 3;
    } else if (c >= 0x80)
      throw std::invalid_argument("invalid UTF-8");
    for (size_t k = 0; k < n; ++k) {
      if (p == s.size())
        throw std::invalid_argument("invalid UTF-8");
      auto b = static_cast<unsigned char>(s[p++]);
      if (b < 0x80 || b > 0xbf)
        throw std::invalid_argument("invalid UTF-8");
      cp = (cp << 6) | (b & 63);
    }
    if ((n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff))
      throw std::invalid_argument("invalid UTF-8");
    if (!space(cp)) {
      if (first == s.size())
        first = start;
      end = p;
    }
  }
  return first == s.size() ? "" : s.substr(first, end - first);
}
std::string strip_newlines(const std::string &s) {
  auto a = s.find_first_not_of('\n');
  auto b = s.find_last_not_of('\n');
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
} // namespace
TextDescription describe_text() {
  TextDescription result{"e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
          "gpt2", "qwen35", false, true, true};
  result.thinking=true; result.thinking_enabled=true;
  // https://huggingface.co/Qwen/Qwen3.5-35B-A3B : general-task thinking/instruct recommendations.
  result.sampling_thinking_off=model::SamplingRecommendation{0.7,0.8,1,20,32768,1.5,0};
  result.sampling_thinking_on=model::SamplingRecommendation{1.0,0.95,1,20,32768,1.5,0};
  result.recommended_context=16384;
  return result;
}
TextResult text_format(const std::vector<Message>& messages,const TextOptions& options) {
  if(messages.empty()) throw std::invalid_argument("no messages");
  size_t last=messages.size();
  std::vector<std::string> contents;
  for(size_t i=0;i<messages.size();++i) {
    const auto& m=messages[i];
    if((m.role!="system" && m.role!="user" && m.role!="assistant" && m.role!="tool") ||
       (m.role=="system" && i!=0) || (!m.tool_calls.empty() && m.role!="assistant"))
      throw std::invalid_argument("unsupported message");
    contents.push_back(trim(m.content));
    if(m.role=="user" && !(contents.back().rfind("<tool_response>",0)==0 && contents.back().size()>=16 &&
      contents.back().compare(contents.back().size()-16,16,"</tool_response>")==0)) last=i;
  }
  if(last==messages.size()) throw std::invalid_argument("no user query");
  TextResult result;result.thinking=options.thinking.value_or(false);
  auto add=[&](std::string text,bool protocol) {result.fragments.push_back({std::move(text),protocol});};
  if(!options.tools.empty()) {
    add("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",true);
    for(const auto& tool:options.tools) {
      const auto object=model::Json::parse(tool.json);
      if(object.kind!=model::Json::Kind::object) throw std::invalid_argument("Tool definition must be an object");
      add("\n",true);add(object.dump(),false);
    }
    add("\n</tools>",true);
    add(R"TOOLS(

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>)TOOLS",true);
    if(messages[0].role=="system" && !contents[0].empty()) {add("\n\n",true);add(contents[0],false);}
    add("<|im_end|>\n",true);
  }
  for(size_t i=0;i<messages.size();++i) {
    const auto& m=messages[i];
    if(m.role=="system" && !options.tools.empty()) continue;
    std::string content=contents[i], reasoning;
    if(m.role=="tool") {
      if(i>0 && messages[i-1].role!="tool") add("<|im_start|>user",true);
      add("\n<tool_response>\n",true);add(content,false);add("\n</tool_response>",true);
      if(i+1==messages.size() || messages[i+1].role!="tool") add("<|im_end|>\n",true);
      continue;
    }
    add("<|im_start|>"+m.role+"\n",true);
    if(m.role=="assistant") {
      if(m.reasoning_content) reasoning=trim(*m.reasoning_content);
      else {
        const auto split=content.find("</think>");
        if(split!=std::string::npos) {
          reasoning=content.substr(0,split);
          const auto open=reasoning.rfind("<think>");
          if(open!=std::string::npos) reasoning=reasoning.substr(open+7);
          reasoning=trim(strip_newlines(reasoning));
          content=content.substr(content.rfind("</think>")+8);
          const auto p=content.find_first_not_of('\n');content=p==std::string::npos?"":content.substr(p);
        }
      }
      if(options.preserve_thinking.value_or(true) || i>last) {
        add("<think>\n",true);add(reasoning,false);add("\n</think>\n\n",true);
      }
    }
    add(content,false);
    for(size_t j=0;j<m.tool_calls.size();++j) {
      const auto& call=m.tool_calls[j];
      const auto arguments=model::Json::parse(call.arguments_json);
      if(arguments.kind!=model::Json::Kind::object) throw std::invalid_argument("Tool arguments must be an object");
      add(j?"\n<tool_call>\n<function=":trim(content).empty()?"<tool_call>\n<function=":"\n\n<tool_call>\n<function=",true);
      add(call.name,false);add(">\n",true);
      for(const auto& item:arguments.members) {
        add("<parameter=",true);add(item.first,false);add(">\n",true);
        add(item.second.kind==model::Json::Kind::string?item.second.value:item.second.dump(),false);
        add("\n</parameter>\n",true);
      }
      add("</function>\n</tool_call>",true);
    }
    add("<|im_end|>\n",true);
  }
  add(result.thinking?"<|im_start|>assistant\n<think>\n":"<|im_start|>assistant\n<think>\n\n</think>\n\n",true);
  return result;
}
} // namespace edge0::models::edge0_35b
