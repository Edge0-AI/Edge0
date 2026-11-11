// Strata. The pack report (ds4_packinfo.cpp): `strata-dsv41 --pack-info --pack <dir>`.  Host only: no device, no weights.
#pragma once

#include <cstdio>
#include <string>

namespace strata::ds4 {

/// Prints the pack's type map, its bytes an expert / a token, its tier sizes and the tensors whose type has no device
/// kernel.  0 when the pack's description was read, non-zero (and a line saying what is missing) otherwise.
int pack_info(const std::string& dir, std::FILE* out);

}  // namespace strata::ds4
