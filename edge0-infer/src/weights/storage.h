#pragma once
#include "exec/execution.h"
#include "platform/file.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>

namespace edge0::weights {
using ReadMode = platform::ReadMode;
bool supports_direct_read();
enum class Priority { urgent, background };
enum class Protection { region, group };
enum class RegionState { empty, writing, published, isolated };
struct Array { exec::Layout layout; std::size_t regions = 1; };
struct Binding { std::uint64_t storage; std::size_t region; std::string id; std::uint64_t generation; };
struct Source { std::string path; std::uint64_t offset; std::size_t bytes; exec::Layout layout; ReadMode read_mode = ReadMode::cached; };
struct LoadItem { std::size_t region; std::string id; Source source; std::shared_ptr<void> metadata = {}; };
struct Registry;
class Storage;
class Write;
class Usage;
class Writer;
struct RegionInfo {
    Binding binding; RegionState state; std::size_t uses;
    std::shared_ptr<void> metadata; std::shared_ptr<Write> pending;
};
struct View { ggml_tensor* tensor; std::shared_ptr<exec::Use> use; };
class Storage {
public:
    ~Storage();
    ggml_tensor* tensor(std::size_t region) const;
    std::shared_ptr<exec::Buffer> allocation() const { return allocation_; }
    RegionState state(std::size_t region) const;
    std::uint64_t identity() const { return identity_; }
private:
    friend class Weights; friend class Write; friend class Usage; friend class Writer;
    struct Region {
        ggml_tensor* tensor;
        RegionState state = RegionState::empty;
        std::string id;
        std::uint64_t generation = 0;
        std::size_t uses = 0;
        std::shared_ptr<void> metadata;
        std::weak_ptr<Write> pending;
    };
    std::shared_ptr<Registry> registry_;
    std::shared_ptr<exec::Buffer> allocation_;
    struct ConsumerView { std::shared_ptr<exec::Buffer> allocation; std::vector<ggml_tensor*> regions; };
    std::map<exec::Backend*, ConsumerView> consumers_;
    std::vector<Region> regions_;
    std::vector<std::pair<std::size_t, std::size_t>> arrays_;
    bool closing_ = false;
    Protection protection_;
    std::uint64_t identity_;
};
class Write {
public:
    ~Write();
private:
    friend class Weights; friend class Writer;
    std::mutex operation_;
    std::vector<Binding> bindings_;
    exec::Status error_;
    std::vector<std::vector<std::pair<std::size_t,std::size_t>>> ranges_;
    std::vector<std::shared_future<exec::Status>> writers_;
    std::shared_ptr<Storage> storage_;
    std::vector<LoadItem> items_;
    std::vector<bool> touched_;
    std::vector<Storage::Region> previous_;
    std::vector<std::shared_ptr<platform::ReadFile>> files_;
    std::vector<platform::FileIdentity> identities_;
    std::vector<std::size_t> item_files_;
    std::vector<std::pair<std::string,ReadMode>> file_keys_;
    std::vector<std::future<exec::Status>> futures_;
    bool started_ = false, finished_ = false;
    void rollback(bool isolate_all = false);
};
// A writer owns an exclusive byte range. Completion is explicit; dropping a
// writer reports failure. No unchecked target pointer escapes the module.
class Writer {
public:
    ~Writer();
    exec::Status write(std::size_t offset, const void* data, std::size_t bytes);
    exec::Status finish(exec::Status error = {});
private:
    friend class Weights;
    std::shared_ptr<Storage> storage_;
    std::size_t region_ = 0, offset_ = 0, size_ = 0;
    std::vector<std::pair<std::size_t,std::size_t>> filled_;
    std::promise<exec::Status> completion_;
    bool finished_ = false;
};
class Weights {
public:
    explicit Weights(exec::Budget budget, std::size_t threads = 4);
    ~Weights();
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;
    Result<std::shared_ptr<Storage>> create_storage(const std::shared_ptr<exec::Backend>& backend,
        const std::vector<Array>& arrays, exec::Purpose purpose, Protection protection = Protection::region);
    Result<std::vector<std::shared_ptr<Storage>>> create_storage(
        const std::vector<std::shared_ptr<exec::Backend>>& consumers, const std::vector<Array>& arrays, exec::Purpose purpose,
        Protection protection = Protection::region, const platform::ConsumptionQuery& query = {});
    Result<std::vector<LoadItem>> plan_resident_load(const std::shared_ptr<Storage>& storage, const std::vector<LoadItem>& items) const;
    Result<std::shared_ptr<Write>> begin_write(const std::shared_ptr<Storage>& storage, const std::vector<LoadItem>& items);
    Result<RegionInfo> query_state(const std::shared_ptr<Storage>& storage, std::size_t region) const;
    Result<RegionInfo> query_state(const std::string& id) const;
    Result<std::shared_ptr<Writer>> writable_regions(const std::shared_ptr<Write>& write,
        std::size_t region, std::size_t offset, std::size_t bytes);
    Result<std::vector<Binding>> await_write(const std::shared_ptr<Write>& write);
    exec::Status abandon(const std::shared_ptr<Write>& write);
    exec::Status evict(const std::shared_ptr<Storage>& storage, const std::vector<std::size_t>& regions);
    exec::Status close(const std::shared_ptr<Storage>& storage);
    Result<View> acquire_views(const std::shared_ptr<Storage>& storage, std::size_t array, const std::vector<Binding>& bindings);
    exec::Status fill_regions(const std::shared_ptr<Write>& write, Priority priority = Priority::urgent);
    exec::Status promote(const std::shared_ptr<Write>& write);
    bool ready(const std::shared_ptr<Write>& write) const;
    Result<std::vector<Binding>> complete(const std::shared_ptr<Write>& write);
    // Reserved before any weight read when a fill target is not directly writable.
    std::size_t staging_bytes() const;
    exec::Status ensure_staging();
    Result<View> acquire_views(const std::shared_ptr<Storage>& storage, const Binding& binding);
    Result<View> acquire_views(const std::shared_ptr<Storage>&, const Binding&, const std::shared_ptr<exec::Backend>&);
    Result<View> acquire_views(const std::shared_ptr<Storage>&, const std::shared_ptr<exec::Backend>&,
        std::size_t array, const std::vector<Binding>&);
    exec::Status release_use(const std::shared_ptr<exec::Use>& use);
    exec::Status release_storage(const std::shared_ptr<Storage>& storage);
    std::size_t read_count() const { return reads_.load(); }
    std::function<bool(ggml_backend_t, ggml_backend_buffer_t)> direct = platform::directly_writable;
    std::function<void(std::size_t)> before_read;
    std::function<void()> after_promote;
private:
    Result<std::shared_ptr<Storage>> create_storage_on(const std::shared_ptr<exec::Backend>&,
        const std::vector<Array>&, Protection, ggml_backend_buffer_type_t, exec::Purpose);
    exec::Budget budget_;
    exec::Budget::Ticket staging_;
    std::shared_ptr<Registry> registry_;
    std::mutex queue_mutex_;
    std::condition_variable ready_;
    struct Job { const Write* owner; std::function<void()> run; };
    std::deque<Job> urgent_, background_;
    std::vector<std::thread> workers_;
    std::atomic<std::size_t> reads_{0};
    bool stopping_ = false;
};
}
