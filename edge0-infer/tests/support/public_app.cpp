#include "edge0/engine.h"

int main() {
    auto runtime = edge0::open_runtime({{"CPU"}, {}});
    if (!runtime) return 1;
    return edge0::finish_runtime_use(runtime.value()) ? 0 : 1;
}
