#pragma once
#include "posix_io.h"

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace edge0 {

struct ModelShape {
    int hidden = 0, experts = 0, topk = 0, layers = 0;
    int groups = 1, kept_groups = 1;
    bool sigmoid = false;
};
struct ExpertTensor {
    std::string name;
    int layer;
    size_t offset, stride, experts;
};
struct PrefetchStats {
    uint64_t steps = 0, heads = 0, predicted = 0, compared = 0, hits = 0;
    uint64_t advice = 0, advice_failures = 0;
};

class PrefetchEngine {
public:
    using Advice = std::function<bool(const ReadOnlyFile &, ByteRange)>;
    explicit PrefetchEngine(Advice advice = {});
    bool load(const char * gguf, const char * sidecar, bool issue_advice);
    bool step(const std::vector<float> & rows);
    const ModelShape & shape() const { return shape_; }
    const std::vector<int> & capture_layers() const { return capture_; }
    const std::vector<ExpertTensor> & tensors() const { return tensors_; }
    const PrefetchStats & stats() const { return stats_; }
    const ReadOnlyFile & file() const { return file_; }
    void trace() const;
private:
    struct Head { int owner, width; std::vector<float> fc1, fc2, linear; };
    void load_sidecar(const char * path);
    std::vector<int> select(std::vector<float> logits, const std::vector<float> & bias) const;
    ReadOnlyFile file_;
    ModelShape shape_;
    std::vector<Head> heads_;
    std::map<int, std::vector<float>> gates_, biases_, history_;
    std::map<int, std::vector<int>> previous_predictions_;
    std::vector<int> capture_;
    std::vector<ExpertTensor> tensors_;
    PrefetchStats stats_;
    Advice advice_;
    bool issue_advice_ = false;
};

// A single bounded snapshot mailbox. Busy workers drop advisory snapshots;
// readers never wait for predictions and shutdown always joins the worker.
class PrefetchWorker {
public:
    explicit PrefetchWorker(std::unique_ptr<PrefetchEngine> engine, bool asynchronous);
    ~PrefetchWorker();
    bool submit(const std::vector<float> & rows);
    void wait_idle();
    PrefetchEngine & engine() { return *engine_; }
private:
    std::unique_ptr<PrefetchEngine> engine_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    std::vector<float> rows_;
    bool busy_ = false, quit_ = false;
};

} // namespace edge0
