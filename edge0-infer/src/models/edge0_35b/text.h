#pragma once
#include "model/text.h"
namespace edge0::models::edge0_35b {
using model::Message;
using model::TextOptions;
using model::TextFragment;
using model::TextDescription;
using model::TextResult;
TextDescription describe_text();
TextResult text_format(const std::vector<Message> &messages, const TextOptions &options = {});
} // namespace edge0::models::edge0_35b
