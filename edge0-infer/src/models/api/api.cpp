#include "models/api/compute.h"
namespace edge0::models {
Result<GreedyTensors> append_greedy(exec::Graph& graph, ggml_tensor* logits) {
    if (!logits || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits))
        return Error{ErrorCategory::invalid_argument, "Greedy output requires contiguous F32 logits"};
    auto* c = graph.context();
    auto* row = ggml_view_1d(c, logits, logits->ne[0],
        (static_cast<std::size_t>(ggml_nelements(logits) / logits->ne[0]) - 1) * logits->nb[1]);
    GreedyTensors result{ggml_argmax(c, row), ggml_sum(c, row)};
    if (auto error = graph.output(result.token)) return *error;
    if (auto error = graph.output(result.sum)) return *error;
    return result;
}
}
