// Compiles only if the public headers get the C++17 mode that edge0::edge0 requires.
#include "edge0/engine.h"

int probe_main() { return edge0::query_capabilities().models.empty() ? 1 : 0; }
