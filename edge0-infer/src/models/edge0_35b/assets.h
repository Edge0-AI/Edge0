#pragma once
#include "model/adapter.h"
namespace edge0::models::edge0_35b {
const model::Adapter &adapter();
bool use_ahead_prefill(const model::ModelOptions&, std::int64_t block_tokens);
}
