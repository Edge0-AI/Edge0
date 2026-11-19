#pragma once
#include "edge0/generation.h"
#include <string>

namespace cli {
// The effective configuration of a measured request, not model recommendations.
std::string generation_json(const edge0::GenerationConfiguration&);
}
