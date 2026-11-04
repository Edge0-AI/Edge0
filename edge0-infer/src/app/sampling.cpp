#include "app/sampling.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace edge0::app {
Result<std::int32_t> sample_logits(const std::vector<float>& logits, const GenerationConfiguration& c,
    const SamplingHistory& history, const std::function<double()>& uniform, const GeneratedCounts& generated) {
    const auto invalid = [](const char* message) { return Error{ErrorCategory::invalid_argument, message}; };
    if (logits.empty() || logits.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        return invalid("Invalid logits row length");
    struct Candidate { double score; std::int32_t id; };
    const auto better = [](const Candidate& a, const Candidate& b) {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    };
    const auto count = c.temperature == 0 ? std::size_t{1} :
        c.top_k == 0 ? logits.size() : std::min(logits.size(), static_cast<std::size_t>(c.top_k));
    std::vector<Candidate> candidates;
    candidates.reserve(count);
    // Preserve the double-precision penalty arithmetic without a hash lookup per vocabulary entry.
    std::vector<double> penalized;
    if (c.repetition_penalty != 1 || c.presence_penalty != 0 || c.frequency_penalty != 0 || !c.logit_bias.empty()) {
        penalized.assign(logits.begin(), logits.end());
        for(const auto& bias:c.logit_bias) {
            if(bias.first<0 || static_cast<std::size_t>(bias.first)>=penalized.size()) return invalid("Bias token outside vocabulary");
            auto& score=penalized[static_cast<std::size_t>(bias.first)];
            score+=bias.second;if(!std::isfinite(score)) return invalid("Non-finite biased logits");
        }
        for (const auto id : history) {
            if (id < 0 || static_cast<std::size_t>(id) >= penalized.size()) continue;
            auto& score = penalized[static_cast<std::size_t>(id)];
            score = score > 0 ? score / c.repetition_penalty : score * c.repetition_penalty;
            if(!std::isfinite(score)) return invalid("Non-finite repetition-penalized logits");
        }
        for(const auto& item:generated) {
            if(item.first<0 || static_cast<std::size_t>(item.first)>=penalized.size()) continue;
            if(item.second) {
                auto& score=penalized[static_cast<std::size_t>(item.first)];
                score-=c.frequency_penalty*static_cast<double>(item.second)+c.presence_penalty;
                if(!std::isfinite(score)) return invalid("Non-finite frequency-penalized logits");
            }
        }
    }
    // A worst-first heap bounds work to O(V log k); only retained candidates are sorted.
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) return invalid("Non-finite logits");
        const double score = penalized.empty() ? static_cast<double>(logits[i]) : penalized[i];
        if (!std::isfinite(score)) return invalid("Non-finite penalized logits");
        Candidate value{score, static_cast<std::int32_t>(i)};
        if (candidates.size() < count) {
            candidates.push_back(value);
            std::push_heap(candidates.begin(), candidates.end(), better);
        } else if (better(value, candidates.front())) {
            std::pop_heap(candidates.begin(), candidates.end(), better);
            candidates.back() = value;
            std::push_heap(candidates.begin(), candidates.end(), better);
        }
    }
    std::sort(candidates.begin(), candidates.end(), better);
    if (c.temperature == 0) return candidates.front().id;
    const double maximum = candidates.front().score;
    double total = 0;
    for (auto& candidate : candidates) {
        candidate.score = std::exp((candidate.score - maximum) / c.temperature);
        total += candidate.score;
    }
    if (!std::isfinite(total) || total <= 0) return invalid("Invalid sampling distribution");
    std::size_t kept = candidates.size();
    if (c.top_p < 1) {
        double cumulative = 0;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            cumulative += candidates[i].score / total;
            if (cumulative >= c.top_p) { kept = i + 1; break; }
        }
    }
    total = 0;
    for (std::size_t i = 0; i < kept; ++i) total += candidates[i].score;
    const double u = uniform();
    double cumulative = 0;
    for (std::size_t i = 0; i < kept; ++i) {
        cumulative += candidates[i].score / total;
        if (u < cumulative) return candidates[i].id;
    }
    return candidates[kept - 1].id;
}
TokenLogprob raw_logprob(const std::vector<float>& logits,std::int32_t token,std::size_t count) {
    const double maximum=*std::max_element(logits.begin(),logits.end());
    double sum=0;
    for(const auto value:logits) sum+=std::exp(static_cast<double>(value)-maximum);
    const double normalizer=std::log(sum);
    TokenLogprob result;result.token=token;
    result.logprob=(static_cast<double>(logits.at(static_cast<std::size_t>(token)))-maximum)-normalizer;
    std::vector<std::int32_t> order(logits.size());
    for(std::size_t i=0;i<order.size();++i) order[i]=static_cast<std::int32_t>(i);
    count=std::min(count,order.size());
    std::partial_sort(order.begin(),order.begin()+count,order.end(),[&](auto a,auto b) {
        return logits[a]>logits[b] || (logits[a]==logits[b] && a<b);
    });
    for(std::size_t i=0;i<count;++i)
        result.top.push_back({order[i],{},(static_cast<double>(logits[order[i]])-maximum)-normalizer});
    return result;
}
}
