#include "prefetch_internal.h"
#include "ggml.h"
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace edge0 {
namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int metadata_int(const gguf_context * gguf, const std::string & key, int fallback = 0) {
    const int64_t index = gguf_find_key(gguf, key.c_str());
    if (index < 0) return fallback;
    require(gguf_get_kv_type(gguf, index) == GGUF_TYPE_UINT32, "invalid GGUF integer metadata");
    const uint32_t value = gguf_get_val_u32(gguf, index);
    require(value <= INT32_MAX, "GGUF integer metadata overflow");
    return static_cast<int>(value);
}

float dot(const float * weights, const float * features, size_t length) {
    float sum = 0;
    for (size_t i = 0; i < length; ++i) sum += weights[i] * features[i];
    return sum;
}

std::vector<int> top_indices(const std::vector<float> & values, int count) {
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&](int a, int b) {
        return values[a] == values[b] ? a < b : values[a] > values[b];
    });
    indices.resize(count);
    return indices;
}

using Json = nlohmann::json;
size_t unsigned_size(const Json & value) {
    require(value.is_number_unsigned(), "sidecar dimensions/offsets must be unsigned integers");
    const uint64_t number = value.get<uint64_t>();
    require(number <= SIZE_MAX, "sidecar integer overflow");
    return static_cast<size_t>(number);
}

struct SideTensor {
    size_t rows, columns, offset, bytes;
    std::string dtype;
};

std::vector<float> read_weights(const ReadOnlyFile & file, const SideTensor & tensor, size_t base) {
    std::vector<unsigned char> raw(tensor.bytes);
    require(file.read(base + tensor.offset, raw.data(), raw.size()), "truncated sidecar weights");
    std::vector<float> result(tensor.rows * tensor.columns);
    for (size_t i = 0; i < result.size(); ++i) {
        if (tensor.dtype == "F32") std::memcpy(&result[i], raw.data() + i * 4, 4);
        else {
            uint16_t bits;
            std::memcpy(&bits, raw.data() + i * 2, 2);
            result[i] = tensor.dtype == "F16" ? ggml_fp16_to_fp32(bits) : ggml_bf16_to_fp32({bits});
        }
        require(std::isfinite(result[i]), "non-finite sidecar weight");
    }
    return result;
}

} // namespace

PrefetchEngine::PrefetchEngine(Advice advice) : advice_(std::move(advice)) {
    if (!advice_) advice_ = [](const ReadOnlyFile & file, ByteRange range) { return file.advise(range); };
}

bool PrefetchEngine::load(const char * path, const char * sidecar, bool issue_advice) {
    try {
        require(heads_.empty() && tensors_.empty(), "prefetch engine already loaded");
        require(file_.open(path, issue_advice), "cannot open/map GGUF for prefetch");
        std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
            gguf_init_from_file(path, {true, nullptr}), gguf_free);
        require(bool(gguf), "cannot read GGUF metadata");
        const int64_t architecture_index = gguf_find_key(gguf.get(), "general.architecture");
        require(architecture_index >= 0 && gguf_get_kv_type(gguf.get(), architecture_index) == GGUF_TYPE_STRING,
                "missing GGUF architecture");
        const std::string architecture = gguf_get_val_str(gguf.get(), architecture_index);
        require(architecture == "bailingmoe3" || architecture == "qwen3next", "unsupported prerouter architecture");
        const std::string prefix = architecture + ".";
        shape_.hidden = metadata_int(gguf.get(), prefix + "embedding_length");
        shape_.experts = metadata_int(gguf.get(), prefix + "expert_count");
        shape_.topk = metadata_int(gguf.get(), prefix + "expert_used_count");
        shape_.layers = metadata_int(gguf.get(), prefix + "block_count");
        shape_.groups = metadata_int(gguf.get(), prefix + "expert_group_count", 1);
        shape_.kept_groups = metadata_int(gguf.get(), prefix + "expert_group_used_count", shape_.groups);
        shape_.sigmoid = architecture == "bailingmoe3";
        require(shape_.hidden > 0 && shape_.hidden <= 16384 && shape_.experts > 0 && shape_.experts <= 4096 &&
                shape_.topk > 0 && shape_.topk <= shape_.experts && shape_.layers > 1 && shape_.layers <= 1024,
                "invalid GGUF model shape");
        require(shape_.groups > 0 && shape_.experts % shape_.groups == 0 && shape_.kept_groups > 0 &&
                shape_.kept_groups <= shape_.groups && shape_.topk <= shape_.kept_groups * (shape_.experts / shape_.groups),
                "invalid expert group shape");
        const size_t data_offset = gguf_get_data_offset(gguf.get());
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); ++i) {
            const std::string name = gguf_get_tensor_name(gguf.get(), i);
            const size_t relative = gguf_get_tensor_offset(gguf.get(), i);
            const size_t bytes = gguf_get_tensor_size(gguf.get(), i);
            require(data_offset <= file_.size() && relative <= file_.size() - data_offset &&
                    bytes <= file_.size() - data_offset - relative, "GGUF tensor outside file");
            const size_t offset = data_offset + relative;
            const int64_t * ne = gguf_get_tensor_ne(gguf.get(), i);
            for (int layer = 0; layer < shape_.layers; ++layer) {
                const std::string block = "blk." + std::to_string(layer) + ".";
                if (name == block + "ffn_gate_inp.weight") {
                    require(gguf_get_tensor_type(gguf.get(), i) == GGML_TYPE_F32 && ne[0] == shape_.hidden &&
                            ne[1] == shape_.experts && ne[2] == 1 && ne[3] == 1, "invalid F32 routing gate");
                    auto & gate = gates_[layer];
                    gate.resize(bytes / sizeof(float));
                    require(file_.read(offset, gate.data(), bytes), "cannot read routing gate");
                } else if (name == block + "exp_probs_b.bias") {
                    require(gguf_get_tensor_type(gguf.get(), i) == GGML_TYPE_F32 && ne[0] == shape_.experts &&
                            ne[1] == 1 && ne[2] == 1 && ne[3] == 1, "invalid routing bias");
                    auto & bias = biases_[layer];
                    bias.resize(shape_.experts);
                    require(file_.read(offset, bias.data(), bytes), "cannot read routing bias");
                } else if (name == block + "ffn_up_exps.weight" || name == block + "ffn_gate_exps.weight" ||
                           name == block + "ffn_down_exps.weight") {
                    require(ne[2] == shape_.experts && ne[3] == 1 && bytes > 0 &&
                            bytes % shape_.experts == 0, "invalid expert tensor shape");
                    tensors_.push_back({name, layer, offset, bytes / static_cast<size_t>(shape_.experts),
                                        static_cast<size_t>(shape_.experts)});
                }
            }
        }
        require(!tensors_.empty(), "GGUF has no expert tensors");
        if (sidecar && *sidecar) load_sidecar(sidecar);
        std::set<int> layers;
        for (const auto & head : heads_) { layers.insert(head.owner); layers.insert(head.owner + 1); }
        for (int layer : layers) {
            require(gates_.count(layer) != 0, "sidecar owner/consumer has no routing gate");
            require(std::all_of(gates_[layer].begin(), gates_[layer].end(), [](float f) { return std::isfinite(f); }),
                    "non-finite routing gate");
            require(std::all_of(biases_[layer].begin(), biases_[layer].end(), [](float f) { return std::isfinite(f); }),
                    "non-finite routing bias");
            history_[layer].assign(shape_.experts, 0);
        }
        capture_.assign(layers.begin(), layers.end());
        for (const auto & head : heads_) {
            const size_t count = std::count_if(tensors_.begin(), tensors_.end(), [&](const ExpertTensor & tensor) {
                return tensor.layer == head.owner + 1;
            });
            require(count == 3, "sidecar consumer needs up/gate/down expert tensors");
        }
        issue_advice_ = issue_advice;
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[pref-trace] init failed: %s; advisory disabled\n", error.what());
        heads_.clear(); gates_.clear(); biases_.clear(); tensors_.clear(); capture_.clear(); history_.clear();
        file_.close();
        return false;
    }
}

void PrefetchEngine::load_sidecar(const char * path) {
    ReadOnlyFile sidecar;
    require(sidecar.open(path, false), "cannot open sidecar");
    uint64_t header_length = 0;
    require(sidecar.read(0, &header_length, sizeof(header_length)) && header_length > 0 &&
            header_length <= (16 << 20) && sidecar.size() >= 8 && header_length <= sidecar.size() - 8,
            "invalid/truncated sidecar header length");
    std::string header(static_cast<size_t>(header_length), '\0');
    require(sidecar.read(8, header.data(), header.size()), "truncated sidecar header");
    std::vector<std::set<std::string>> keys;
    auto callback = [&](int depth, Json::parse_event_t event, Json & parsed) {
        require(depth <= 8, "sidecar JSON nesting too deep");
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        if (event == Json::parse_event_t::key) require(keys.back().insert(parsed.get<std::string>()).second,
                                                      "duplicate sidecar JSON key");
        if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    };
    const Json json = Json::parse(header, callback);
    require(json.is_object() && json.size() <= 4096, "invalid sidecar tensor table");
    const size_t base = 8 + static_cast<size_t>(header_length);
    std::map<std::string, SideTensor> entries;
    std::set<int> owners;
    std::vector<ByteRange> ranges;
    size_t allocation = 0;
    for (auto it = json.begin(); it != json.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto & entry = it.value();
        const auto & shape = entry.at("shape");
        const auto & offsets = entry.at("data_offsets");
        require(shape.is_array() && shape.size() == 2 && offsets.is_array() && offsets.size() == 2,
                "sidecar tensor must have rank two and two offsets");
        const size_t rows = unsigned_size(shape[0]), columns = unsigned_size(shape[1]);
        require(rows > 0 && columns > 0 && rows <= 16384 && columns <= 24576, "invalid sidecar dimensions");
        const std::string dtype = entry.at("dtype").get<std::string>();
        require(dtype == "F16" || dtype == "BF16" || dtype == "F32", "unsupported sidecar dtype");
        const size_t start = unsigned_size(offsets[0]), end = unsigned_size(offsets[1]);
        const size_t element_size = dtype == "F32" ? 4 : 2;
        require(start <= end && end <= sidecar.size() - base && end - start == rows * columns * element_size,
                "sidecar tensor offsets/size mismatch");
        require(rows * columns <= ((size_t(512) << 20) - allocation) / sizeof(float), "sidecar allocation limit exceeded");
        allocation += rows * columns * sizeof(float);
        entries.emplace(it.key(), SideTensor{rows, columns, start, end - start, dtype});
        ranges.push_back({start, end - start});
        const std::string & key = it.key();
        const size_t separator = key.find('.', 7);
        require(key.compare(0, 7, "layers.") == 0 && separator != std::string::npos, "invalid sidecar tensor name");
        int owner = -1;
        const auto parsed = std::from_chars(key.data() + 7, key.data() + separator, owner);
        require(parsed.ec == std::errc{} && parsed.ptr == key.data() + separator &&
                owner >= 0 && owner < shape_.layers - 1, "invalid sidecar owner layer");
        const std::string suffix = key.substr(separator + 1);
        require((suffix == "fc1.weight" || suffix == "fc2.weight" || suffix == "linear_init.weight") &&
                key == "layers." + std::to_string(owner) + "." + suffix, "unknown sidecar tensor");
        owners.insert(owner);
    }
    std::sort(ranges.begin(), ranges.end(), [](ByteRange a, ByteRange b) { return a.offset < b.offset; });
    size_t end = 0;
    for (ByteRange range : ranges) {
        require(range.offset == end, "sidecar has overlapping or missing tensor data");
        end += range.length;
    }
    require(!owners.empty() && end == sidecar.size() - base, "incomplete sidecar tensor data");
    for (int owner : owners) {
        const std::string prefix = "layers." + std::to_string(owner) + ".";
        const auto & first = entries.at(prefix + "fc1.weight");
        const auto & second = entries.at(prefix + "fc2.weight");
        const auto & linear = entries.at(prefix + "linear_init.weight");
        const size_t features = shape_.hidden + 2 * shape_.experts;
        require(first.columns == features && second.rows == size_t(shape_.experts) && second.columns == first.rows &&
                linear.rows == size_t(shape_.experts) && linear.columns == features, "sidecar/model shape mismatch");
        heads_.push_back({owner, static_cast<int>(first.rows), read_weights(sidecar, first, base),
                          read_weights(sidecar, second, base), read_weights(sidecar, linear, base)});
    }
}

std::vector<int> PrefetchEngine::select(std::vector<float> logits, const std::vector<float> & bias) const {
    if (shape_.sigmoid) {
        for (int i = 0; i < shape_.experts; ++i) logits[i] = 1.f / (1.f + std::exp(-logits[i])) + (bias.empty() ? 0 : bias[i]);
        if (shape_.kept_groups < shape_.groups) {
            const int width = shape_.experts / shape_.groups;
            std::vector<float> scores(shape_.groups);
            for (int group = 0; group < shape_.groups; ++group) {
                std::vector<float> values(logits.begin() + group * width, logits.begin() + (group + 1) * width);
                const auto best = top_indices(values, std::min(width, 2));
                for (int index : best) scores[group] += values[index];
            }
            const auto groups = top_indices(scores, shape_.kept_groups);
            for (int group = 0; group < shape_.groups; ++group) {
                if (std::find(groups.begin(), groups.end(), group) == groups.end()) {
                    std::fill(logits.begin() + group * width, logits.begin() + (group + 1) * width,
                              -std::numeric_limits<float>::infinity());
                }
            }
        }
    }
    return top_indices(logits, shape_.topk);
}

bool PrefetchEngine::step(const std::vector<float> & rows) {
    if (heads_.empty() || rows.size() != capture_.size() * shape_.hidden ||
        !std::all_of(rows.begin(), rows.end(), [](float f) { return std::isfinite(f); })) return false;
    std::map<int, std::vector<int>> teacher;
    for (size_t i = 0; i < capture_.size(); ++i) {
        const int layer = capture_[i];
        const auto & gate = gates_.at(layer);
        std::vector<float> logits(shape_.experts);
        for (int expert = 0; expert < shape_.experts; ++expert) {
            logits[expert] = dot(gate.data() + size_t(expert) * shape_.hidden,
                                 rows.data() + i * shape_.hidden, shape_.hidden);
        }
        if (!std::all_of(logits.begin(), logits.end(), [](float f) { return std::isfinite(f); })) return false;
        teacher[layer] = select(std::move(logits), biases_[layer]);
    }
    ++stats_.steps;
    for (const auto & head : heads_) {
        const int consumer = head.owner + 1;
        const auto previous = previous_predictions_.find(consumer);
        if (previous != previous_predictions_.end()) {
            stats_.compared += shape_.topk;
            for (int expert : previous->second) {
                stats_.hits += std::count(teacher[consumer].begin(), teacher[consumer].end(), expert);
            }
        }
        const size_t row = std::lower_bound(capture_.begin(), capture_.end(), head.owner) - capture_.begin();
        std::vector<float> features(shape_.hidden + 2 * shape_.experts, 0);
        std::copy_n(rows.data() + row * shape_.hidden, shape_.hidden, features.data());
        for (int expert : teacher[head.owner]) features[shape_.hidden + expert] = 1;
        std::copy(history_[head.owner].begin(), history_[head.owner].end(), features.begin() + shape_.hidden + shape_.experts);
        std::vector<float> activation(head.width), logits(shape_.experts);
        for (int i = 0; i < head.width; ++i) {
            const float value = dot(head.fc1.data() + size_t(i) * features.size(), features.data(), features.size());
            activation[i] = 0.5f * value * (1.f + std::erf(value * 0.7071067811865475f));
        }
        if (!std::all_of(activation.begin(), activation.end(), [](float f) { return std::isfinite(f); })) return false;
        for (int expert = 0; expert < shape_.experts; ++expert) {
            logits[expert] = dot(head.fc2.data() + size_t(expert) * head.width, activation.data(), head.width) +
                             dot(head.linear.data() + size_t(expert) * features.size(), features.data(), features.size());
        }
        if (!std::all_of(logits.begin(), logits.end(), [](float f) { return std::isfinite(f); })) return false;
        const auto predicted = select(std::move(logits), biases_[consumer]);
        previous_predictions_[consumer] = predicted;
        ++stats_.heads;
        stats_.predicted += predicted.size();
        if (issue_advice_) {
            for (const auto & tensor : tensors_) {
                if (tensor.layer != consumer) continue;
                for (int expert : predicted) {
                    ByteRange range{};
                    if (expert_range(file_.size(), tensor.offset, tensor.stride, tensor.experts, expert, range) &&
                        advice_(file_, range)) ++stats_.advice;
                    else {
                        if (!stats_.advice_failures) std::fprintf(stderr, "[pref-trace] page advice failed; inference continues\n");
                        ++stats_.advice_failures;
                    }
                }
            }
        }
    }
    for (const auto & entry : teacher) {
        auto & history = history_[entry.first];
        std::fill(history.begin(), history.end(), 0);
        for (int expert : entry.second) history[expert] = 1;
    }
    return true;
}

void PrefetchEngine::trace() const {
    std::fprintf(stderr, "[pref-trace] steps=%llu heads=%llu experts=%llu head=%llu/%llu advice=%llu failures=%llu\n",
                 (unsigned long long) stats_.steps, (unsigned long long) stats_.heads,
                 (unsigned long long) stats_.predicted, (unsigned long long) stats_.hits,
                 (unsigned long long) stats_.compared, (unsigned long long) stats_.advice,
                 (unsigned long long) stats_.advice_failures);
}

PrefetchWorker::PrefetchWorker(std::unique_ptr<PrefetchEngine> engine, bool asynchronous) : engine_(std::move(engine)) {
    if (asynchronous) worker_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            wake_.wait(lock, [this] { return busy_ || quit_; });
            if (quit_) break;
            lock.unlock();
            try { engine_->step(rows_); }
            catch (const std::exception & error) {
                std::fprintf(stderr, "[pref-trace] advisory step failed: %s; inference continues\n", error.what());
            }
            if (engine_->stats().steps % 32 == 0) engine_->trace();
            lock.lock();
            busy_ = false;
            wake_.notify_all();
        }
    });
}

PrefetchWorker::~PrefetchWorker() {
    { std::lock_guard<std::mutex> lock(mutex_); quit_ = true; }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    engine_->trace();
}

bool PrefetchWorker::submit(const std::vector<float> & rows) {
    if (!worker_.joinable()) return engine_->step(rows);
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_ || quit_) return false;
    rows_ = rows;
    busy_ = true;
    wake_.notify_one();
    return true;
}

void PrefetchWorker::wait_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    wake_.wait(lock, [this] { return !busy_; });
}

} // namespace edge0
