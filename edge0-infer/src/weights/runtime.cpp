#include "weights/storage.h"
#include "ggml-backend.h"
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>

namespace edge0::weights {
bool supports_direct_read() { return platform::supports_direct_read(); }
namespace {
Error conflict(const std::string& text) { return {ErrorCategory::conflict, text}; }
Error invalid(const std::string& text) { return {ErrorCategory::invalid_argument, text}; }
}
struct Registry {
    std::mutex mutex;
    std::condition_variable changed;
    std::map<std::uint64_t, std::weak_ptr<Storage>> storages;
    std::map<std::string, std::pair<std::uint64_t, std::size_t>> ids;
    std::map<std::pair<std::string,ReadMode>,std::shared_ptr<platform::ReadFile>> files;

};
class Usage final : public exec::Use {
public:
    Usage(std::shared_ptr<Storage> storage, std::size_t region, ggml_tensor* tensor = nullptr,
        std::shared_ptr<exec::Buffer> allocation = {}) : storage_(std::move(storage)), region_(region),
        tensor_(tensor ? tensor : storage_->regions_[region].tensor),
        allocation_(allocation ? std::move(allocation) : storage_->allocation_) {}
    ~Usage() override { if (active_) release(); }
    exec::Status release() override {
        std::lock_guard<std::mutex> lock(storage_->registry_->mutex);
        if (!active_) return conflict("Use already released");
        --storage_->regions_[region_].uses; active_ = false; storage_->registry_->changed.notify_all(); return {};
    }
    bool valid() const override {
        std::lock_guard<std::mutex> lock(storage_->registry_->mutex);
        return active_ && storage_->allocation_ && storage_->regions_[region_].state == RegionState::published;
    }
    bool matches(ggml_tensor* tensor, ggml_backend_t backend) const override {
        std::lock_guard<std::mutex> lock(storage_->registry_->mutex);
        return active_ && storage_->allocation_ && tensor_ == tensor && allocation_->backend->handle() == backend;
    }
    std::shared_ptr<void> keep_alive() const override { return allocation_; }
    std::string identity() const override { return std::to_string(storage_->identity_) + "/" + std::to_string(region_); }
private:
    std::shared_ptr<Storage> storage_;
    std::size_t region_;
    ggml_tensor* tensor_;
    std::shared_ptr<exec::Buffer> allocation_;
    bool active_ = true;
};
Storage::~Storage() {
    std::lock_guard<std::mutex> lock(registry_->mutex);
    for (const auto& region : regions_) if (!region.id.empty()) registry_->ids.erase(region.id);
    registry_->storages.erase(identity_);
}
ggml_tensor* Storage::tensor(std::size_t region) const {
    std::lock_guard<std::mutex> lock(registry_->mutex);
    return allocation_ && region < regions_.size() ? regions_[region].tensor : nullptr;
}
RegionState Storage::state(std::size_t region) const {
    std::lock_guard<std::mutex> lock(registry_->mutex);
    return regions_.at(region).state;
}
Weights::Weights(exec::Budget budget, std::size_t threads) : budget_(std::move(budget)), registry_(std::make_shared<Registry>()) {
    for (std::size_t i = 0; i < std::max(std::size_t{1}, threads); ++i) workers_.emplace_back([this] {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                ready_.wait(lock, [this] { return stopping_ || !urgent_.empty() || !background_.empty(); });
                if (stopping_ && urgent_.empty() && background_.empty()) return;
                auto& queue=urgent_.empty()?background_:urgent_;
                job = std::move(queue.front().run); queue.pop_front();
            }
            job();
        }
    });
}
Weights::~Weights() {
    { std::lock_guard<std::mutex> lock(queue_mutex_); stopping_ = true; }
    ready_.notify_all(); for (auto& worker : workers_) worker.join();
}
Result<std::shared_ptr<Storage>> Weights::create_storage(const std::shared_ptr<exec::Backend>& backend,
    const std::vector<Array>& arrays, exec::Purpose purpose, Protection protection) {
    return create_storage_on(backend, arrays, protection,
        backend ? ggml_backend_get_default_buffer_type(backend->handle()) : nullptr, purpose);
}
Result<std::shared_ptr<Storage>> Weights::create_storage_on(const std::shared_ptr<exec::Backend>& backend,
    const std::vector<Array>& arrays, Protection protection, ggml_backend_buffer_type_t type, exec::Purpose purpose) {
    std::vector<exec::Layout> layouts; std::size_t count = 0;
    for (const auto& array : arrays) {
        if (array.regions == 0 || array.regions > 1000000 || array.layout.dimensions.empty() || array.layout.dimensions.size() > 3)
            return invalid("Invalid weight array");
        auto layout = array.layout; layout.dimensions.push_back(static_cast<std::int64_t>(array.regions));
        layouts.push_back(layout); count += array.regions;
    }
    auto allocation = exec::allocate_buffer_on(budget_, backend, layouts, count, type, purpose);
    if (!allocation) return allocation.error();
    auto storage = std::make_shared<Storage>(); storage->registry_ = registry_; storage->allocation_ = allocation.value(); storage->protection_ = protection;
    static std::atomic<std::uint64_t> next_identity{1};
    storage->identity_ = next_identity.fetch_add(1);
    { std::lock_guard<std::mutex> lock(registry_->mutex); registry_->storages[storage->identity_] = storage; }
    for (std::size_t a = 0; a < arrays.size(); ++a) {
        auto* array = allocation.value()->tensors[a];
        storage->arrays_.push_back({storage->regions_.size(), arrays[a].regions});
        auto dimensions = std::array<std::int64_t, 4>{1, 1, 1, 1};
        std::copy(arrays[a].layout.dimensions.begin(), arrays[a].layout.dimensions.end(), dimensions.begin());
        const auto stride = array->nb[arrays[a].layout.dimensions.size()];
        for (std::size_t r = 0; r < arrays[a].regions; ++r) {
            auto* tensor = ggml_view_4d(allocation.value()->context, array, dimensions[0], dimensions[1], dimensions[2], dimensions[3],
                array->nb[1], array->nb[2], array->nb[3], r * stride);
            if (ggml_backend_view_init(tensor) != GGML_STATUS_SUCCESS)
                return Error{ErrorCategory::backend_error, "Weight view initialization failed"};
            storage->regions_.push_back({tensor, RegionState::empty, {}, 0, 0, {}, {}});
        }
    }
    return storage;
}
Result<std::vector<LoadItem>> Weights::plan_resident_load(const std::shared_ptr<Storage>& storage, const std::vector<LoadItem>& items) const {
    if (!storage || storage->registry_ != registry_ || items.empty()) return invalid("Invalid load target or empty load");
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_ || storage->closing_) return conflict("Storage released or closing");
    std::set<std::size_t> regions; std::set<std::string> ids;
    for (const auto& item : items) {
        if (item.region >= storage->regions_.size() || item.id.empty() || !regions.insert(item.region).second)
            return invalid("Duplicate or invalid load item");
        if (!ids.insert(item.id).second) return conflict("Duplicate weight ID in load group: " + item.id);
        auto* tensor = storage->regions_[item.region].tensor;
        if (item.source.layout.type != tensor->type || item.source.bytes != ggml_nbytes(tensor) ||
            item.source.layout.dimensions.empty() || item.source.layout.dimensions.size() > 4)
            return invalid("Weight source representation or size mismatch");
        for (std::size_t i = 0; i < 4; ++i)
            if ((i < item.source.layout.dimensions.size() ? item.source.layout.dimensions[i] : 1) != tensor->ne[i])
                return invalid("Weight source shape mismatch");
    }
    return items;
}
Result<std::shared_ptr<Write>> Weights::begin_write(const std::shared_ptr<Storage>& storage, const std::vector<LoadItem>& items) {
    auto planned = plan_resident_load(storage, items); if (!planned) return planned.error();
    auto write = std::make_shared<Write>(); write->items_ = items;
    std::map<std::pair<std::string,ReadMode>,std::size_t> selected_files;
    for (const auto& item : items) {
        const auto key=std::make_pair(item.source.path,item.source.read_mode);
        auto found=selected_files.find(key);
        std::size_t index;
        if(found==selected_files.end()) {
            std::shared_ptr<platform::ReadFile> file;
            {
                std::lock_guard<std::mutex> lock(registry_->mutex);
                auto cached=registry_->files.find(key);
                if(cached==registry_->files.end()) {
                    auto opened=platform::ReadFile::open(key.first,key.second);if(!opened) return opened.error();
                    cached=registry_->files.emplace(key,opened.value()).first;
                }
                file=cached->second;
            }
            auto identity=file->identity();if(!identity) return identity.error();
            index=write->files_.size();selected_files.emplace(key,index);
            write->files_.push_back(file);write->identities_.push_back(identity.value());write->file_keys_.push_back(key);
        } else index=found->second;
        const auto& identity=write->identities_[index];
        if(item.source.offset>identity.size || item.source.bytes>identity.size-item.source.offset)
            return invalid("Weight source slice out of bounds");
        write->item_files_.push_back(index);
    }
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_ || storage->closing_) return conflict("Storage released or closing");
    std::set<std::size_t> selected; for (const auto& item : items) selected.insert(item.region);
    if (storage->protection_ == Protection::group)
        for (const auto& region : storage->regions_) if (region.uses || region.state == RegionState::writing || region.state == RegionState::isolated)
            return conflict("Storage group is in use, writing or isolated");
    for (const auto& item : items) {
        const auto& region = storage->regions_[item.region];
        if (region.uses || region.state == RegionState::writing || region.state == RegionState::isolated)
            return conflict("Region is protected: " + std::to_string(item.region));
        auto found = registry_->ids.find(item.id);
        if (found != registry_->ids.end() && !(found->second.first == storage->identity_ && selected.count(found->second.second)))
            return conflict("Weight ID held by storage " + std::to_string(found->second.first) + " region " + std::to_string(found->second.second));
    }
    write->storage_ = storage;
    write->ranges_.resize(items.size()); write->touched_.resize(items.size());
    for (const auto& item : items) {
        auto& region = storage->regions_[item.region]; write->previous_.push_back(region);
        if (!region.id.empty()) registry_->ids.erase(region.id);
    }
    for (const auto& item : items) {
        auto& region = storage->regions_[item.region]; region.id = item.id; region.state = RegionState::writing;
        region.metadata = item.metadata; region.pending = write;
        registry_->ids[item.id] = {storage->identity_, item.region};
    }
    return write;
}
void Write::rollback(bool isolate_all) {
    if (!storage_ || finished_) return;
    for (auto& future : futures_) if (future.valid()) future.wait();
    for (auto& future : writers_) future.wait();
    std::lock_guard<std::mutex> lock(storage_->registry_->mutex);
    for (const auto& item : items_) storage_->registry_->ids.erase(item.id);
    for (std::size_t i = 0; i < items_.size(); ++i) {
        auto& region = storage_->regions_[items_[i].region];
        const bool touched = isolate_all || touched_[i];
        if (!touched && (previous_[i].id.empty() || !storage_->registry_->ids.count(previous_[i].id))) {
            region = previous_[i];
            if (!region.id.empty()) storage_->registry_->ids[region.id] = {storage_->identity_, items_[i].region};
        } else {
            region.id.clear(); region.state = touched ? RegionState::isolated : RegionState::empty;
        }
    }
    finished_ = true;
    if (!error_) error_ = conflict("Write abandoned");
    storage_->registry_->changed.notify_all();
}
Write::~Write() { rollback(); }
std::size_t Weights::staging_bytes() const {
    constexpr std::size_t per_thread = 4 * 1024 * 1024;
    return workers_.size() * per_thread;
}
exec::Status Weights::ensure_staging() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (staging_) return {};
    // One 4 MiB slab per write thread, held until this registry is destroyed.
    auto ticket = budget_.reserve(staging_bytes(), exec::Purpose::staging, "host");
    if (!ticket) return ticket.error();
    staging_ = std::move(ticket.value());
    return {};
}
exec::Status Weights::fill_regions(const std::shared_ptr<Write>& write, Priority priority) {
    if (!write || !write->storage_ || write->storage_->registry_ != registry_) return invalid("Wrong write");
    std::lock_guard<std::mutex> operation(write->operation_);
    if (write->started_ || write->finished_) return conflict("Invalid or already started write");
    { std::lock_guard<std::mutex> lock(registry_->mutex);
      if (write->storage_->closing_) return conflict("Storage is closing"); }
    auto allocation = write->storage_->allocation_;
    const bool staged = !direct(allocation->backend->handle(), allocation->buffer);
    // Model loading reserves staging before any read; this only covers direct callers of the
    // weights facility, and still fails before reading when the budget cannot hold it.
    if (staged) if (auto error = ensure_staging()) return error;
    write->started_ = true;
    std::fill(write->touched_.begin(),write->touched_.end(),true);
    struct Part { std::size_t item, file; std::uint64_t offset; std::size_t tensor_offset, bytes; ggml_tensor* tensor; bool first; };
    static constexpr std::size_t maximum_bytes=4*1024*1024, maximum_segments=64;
    std::vector<Part> parts;
    for(std::size_t i=0;i<write->items_.size();++i) {
        const auto& item=write->items_[i];
        auto* tensor=write->storage_->regions_[item.region].tensor;
        for(std::size_t offset=0;offset<item.source.bytes;) {
            const auto bytes=std::min(maximum_bytes,item.source.bytes-offset);
            parts.push_back({i,write->item_files_[i],item.source.offset+offset,offset,bytes,tensor,offset==0});
            offset+=bytes;
        }
    }
    std::stable_sort(parts.begin(),parts.end(),[](const Part& a,const Part& b) { return std::tie(a.file,a.offset)<std::tie(b.file,b.offset); });
    for(std::size_t begin=0;begin<parts.size();) {
        auto end=begin+1;std::size_t bytes=parts[begin].bytes;
        while(end<parts.size() && end-begin<maximum_segments && parts[end].file==parts[begin].file &&
              parts[end].offset==parts[end-1].offset+parts[end-1].bytes && parts[end].bytes<=maximum_bytes-bytes) {
            bytes+=parts[end].bytes;++end;
        }
        const auto file=write->files_[parts[begin].file];const auto identity=write->identities_[parts[begin].file];
        const auto offset=parts[begin].offset;
        std::vector<Part> slice(parts.begin()+static_cast<std::ptrdiff_t>(begin), parts.begin()+static_cast<std::ptrdiff_t>(end));
        std::vector<std::size_t> regions;
        for (const auto& part : slice) if (part.first) regions.push_back(write->items_[part.item].region);
        const auto hook=before_read;
        // Retain allocation and source until completion, without retaining the transaction.
        auto task=std::make_shared<std::packaged_task<exec::Status()>>([this,file,identity,offset,slice,regions,hook,allocation,staged] {
            (void)allocation;
            for(auto region:regions) { if(hook) hook(region);++reads_; }
            if (!staged) {
                std::vector<platform::ReadTarget> targets;
                for (const auto& part : slice)
                    targets.push_back({static_cast<char*>(part.tensor->data)+part.tensor_offset, part.bytes});
                return file->read_vector(offset, targets, identity);
            }
            thread_local std::vector<unsigned char> scratch;
            if (scratch.size() < maximum_bytes) scratch.resize(maximum_bytes);
            std::vector<platform::ReadTarget> targets;
            std::size_t cursor = 0;
            for (const auto& part : slice) {
                targets.push_back({reinterpret_cast<char*>(scratch.data()+cursor), part.bytes});
                cursor += part.bytes;
            }
            if (auto error = file->read_vector(offset, targets, identity)) return error;
            cursor = 0;
            for (const auto& part : slice) {
                ggml_backend_tensor_set(part.tensor, scratch.data()+cursor, part.tensor_offset, part.bytes);
                cursor += part.bytes;
            }
            return exec::Status{};
        });
        write->futures_.push_back(task->get_future());
        { std::lock_guard<std::mutex> lock(queue_mutex_);(priority==Priority::urgent?urgent_:background_).push_back({write.get(),[task] { (*task)(); }}); }
        ready_.notify_one();begin=end;
    }
    return {};
}
exec::Status Weights::promote(const std::shared_ptr<Write>& write) {
    if(!write || !write->storage_ || write->storage_->registry_!=registry_) return invalid("Wrong write");
    { std::lock_guard<std::mutex> lock(queue_mutex_);
    for(auto it=background_.begin();it!=background_.end();) {
        if(it->owner==write.get()) { urgent_.push_back(std::move(*it)); it=background_.erase(it); }
        else ++it;
    }
    }
    ready_.notify_all();
    if(after_promote) after_promote();
    return {};
}
bool Weights::ready(const std::shared_ptr<Write>& write) const {
    if(!write) return true;
    std::lock_guard<std::mutex> operation(write->operation_);
    const auto ready=[](const auto& future) {
        return !future.valid() || future.wait_for(std::chrono::seconds(0))==std::future_status::ready;
    };
    return std::all_of(write->futures_.begin(),write->futures_.end(),ready) &&
        std::all_of(write->writers_.begin(),write->writers_.end(),ready);
}
Result<std::vector<Binding>> Weights::complete(const std::shared_ptr<Write>& write) {
    if(auto error=promote(write)) return *error;
    if (!write || !write->storage_ || write->storage_->registry_ != registry_) return invalid("Wrong write");
    std::lock_guard<std::mutex> operation(write->operation_);
    if (!write->started_ || write->finished_) return conflict("No unfinished write");
    exec::Status error;
    for (auto& future : write->futures_) {
        try { auto result = future.get(); if (result && !error) error = result; }
        catch (const std::exception& exception) { if (!error) error = Error{ErrorCategory::backend_error, exception.what()}; }
        catch (...) { if (!error) error = Error{ErrorCategory::backend_error, "Writer threw"}; }
    }
    for (auto& future : write->writers_) {
        auto result = future.get(); if (result && !error) error = result;
    }
    if (!write->writers_.empty()) for (std::size_t i = 0; i < write->items_.size(); ++i) {
        auto ranges = write->ranges_[i]; std::sort(ranges.begin(), ranges.end());
        std::size_t end = 0;
        for (auto range : ranges) { if (range.first != end) error = invalid("Incomplete writer ranges"); end = range.second; }
        if (end != write->items_[i].source.bytes) error = invalid("Incomplete writer coverage");
    }
    for (std::size_t i = 0; i < write->files_.size(); ++i) {
        auto current=write->files_[i]->identity();
        auto path=platform::ReadFile::open(write->file_keys_[i].first,write->file_keys_[i].second);
        auto path_identity=path?path.value()->identity():Result<platform::FileIdentity>(path.error());
        if(!current || !(current.value()==write->identities_[i]) || !path_identity || !(path_identity.value()==write->identities_[i]))
            if(!error) error=Error{ErrorCategory::backend_error,"Source file identity changed"};
    }
    if (error) { write->error_ = error; write->rollback(true); return *error; }
    std::lock_guard<std::mutex> lock(registry_->mutex);
    std::vector<Binding> bindings;
    for (const auto& item : write->items_) {
        auto& region = write->storage_->regions_[item.region];
        region.state = RegionState::published; ++region.generation;
        bindings.push_back({write->storage_->identity_, item.region, item.id, region.generation});
    }
    write->finished_ = true; write->bindings_ = bindings;
    registry_->changed.notify_all();
    return bindings;
}
Result<View> Weights::acquire_views(const std::shared_ptr<Storage>& storage, const Binding& binding) {
    return acquire_views(storage, binding, {});
}
Result<View> Weights::acquire_views(const std::shared_ptr<Storage>& storage,
    const Binding& binding, const std::shared_ptr<exec::Backend>& backend) {
    if (!storage || storage->registry_ != registry_) return invalid("Wrong registry");
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_ || storage->closing_ || binding.storage != storage->identity_ || binding.region >= storage->regions_.size())
        return conflict("Wrong storage identity");
    auto& region = storage->regions_[binding.region];
    if (region.state != RegionState::published || region.id != binding.id || region.generation != binding.generation)
        return conflict("Stale or unpublished binding");
    if (storage->protection_ == Protection::group)
        for (const auto& other : storage->regions_) if (other.state == RegionState::writing || other.state == RegionState::isolated)
            return conflict("Storage group is writing or isolated");
    auto allocation = storage->allocation_;
    auto* tensor = region.tensor;
    if (backend) {
        const auto found = storage->consumers_.find(backend.get());
        if (found != storage->consumers_.end()) {
            allocation = found->second.allocation; tensor = found->second.regions[binding.region];
        } else if (backend != allocation->backend) return invalid("Weight consumer is outside storage scope");
    }
    ++region.uses;
    return View{tensor, std::make_shared<Usage>(storage, binding.region, tensor, allocation)};
}
exec::Status Weights::release_use(const std::shared_ptr<exec::Use>& use) {
    return use ? use->release() : exec::Status{invalid("Missing use")};
}
exec::Status Weights::release_storage(const std::shared_ptr<Storage>& storage) {
    if (!storage || storage->registry_ != registry_) return invalid("Wrong registry");
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_) return conflict("Storage already released");
    for (const auto& region : storage->regions_) if (region.uses || region.state == RegionState::writing)
        return conflict("Storage is in use or writing");
    for (auto& region : storage->regions_) {
        if (!region.id.empty()) registry_->ids.erase(region.id);
        region.id.clear(); region.tensor = nullptr;
    }
    storage->consumers_.clear();
    storage->allocation_.reset(); registry_->changed.notify_all(); return {};
}
}

namespace edge0::weights {
Result<RegionInfo> Weights::query_state(const std::shared_ptr<Storage>& storage, std::size_t index) const {
    if (!storage || storage->registry_ != registry_) return invalid("Wrong registry");
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_ || index >= storage->regions_.size()) return invalid("Invalid region");
    const auto& r = storage->regions_[index];
    return RegionInfo{{storage->identity_, index, r.id, r.generation}, r.state, r.uses, r.metadata, r.pending.lock()};
}
Result<RegionInfo> Weights::query_state(const std::string& id) const {
    std::lock_guard<std::mutex> lock(registry_->mutex);
    auto found = registry_->ids.find(id);
    if (found == registry_->ids.end()) return conflict("Weight ID is absent: " + id);
    auto storage = registry_->storages.at(found->second.first).lock();
    if (!storage) return conflict("Storage expired");
    const auto index = found->second.second;
    const auto& r = storage->regions_[index];
    return RegionInfo{{storage->identity_, index, r.id, r.generation}, r.state, r.uses, r.metadata, r.pending.lock()};
}
Result<std::vector<Binding>> Weights::await_write(const std::shared_ptr<Write>& write) {
    if (!write || !write->storage_ || write->storage_->registry_ != registry_) return invalid("Wrong write");
    if(auto error=promote(write)) return *error;
    std::unique_lock<std::mutex> lock(registry_->mutex);
    registry_->changed.wait(lock, [&] { return write->finished_; });
    if (write->error_) return *write->error_;
    return write->bindings_;
}
exec::Status Weights::abandon(const std::shared_ptr<Write>& write) {
    if (!write || !write->storage_ || write->storage_->registry_ != registry_) return invalid("Wrong write");
    std::lock_guard<std::mutex> operation(write->operation_);
    write->rollback(); return {};
}
Result<std::shared_ptr<Writer>> Weights::writable_regions(const std::shared_ptr<Write>& write,
    std::size_t region, std::size_t offset, std::size_t bytes) {
    if (!write || !write->storage_ || write->storage_->registry_ != registry_) return invalid("Wrong write");
    std::lock_guard<std::mutex> operation(write->operation_);
    if (write->finished_ || !write->futures_.empty()) return conflict("Write already filled or finished");
    { std::lock_guard<std::mutex> lock(registry_->mutex);
      if (write->storage_->closing_) return conflict("Storage is closing"); }
    auto found = std::find_if(write->items_.begin(), write->items_.end(), [region](const LoadItem& item) { return item.region == region; });
    if (found == write->items_.end() || !bytes || offset > found->source.bytes || bytes > found->source.bytes-offset)
        return invalid("Writer range out of bounds");
    const auto allocation = write->storage_->allocation_;
    if (!direct(allocation->backend->handle(), allocation->buffer)) return Error{ErrorCategory::unsupported, "Storage cannot be directly filled"};
    auto& ranges = write->ranges_[std::size_t(found-write->items_.begin())];
    for (auto r : ranges) if (offset < r.second && offset+bytes > r.first) return conflict("Overlapping writer ranges");
    auto writer = std::make_shared<Writer>();
    writer->storage_ = write->storage_; writer->region_ = region; writer->offset_ = offset; writer->size_ = bytes;
    write->writers_.push_back(writer->completion_.get_future().share());
    ranges.emplace_back(offset, offset+bytes); write->started_ = true;
    write->touched_[std::size_t(found-write->items_.begin())]=true;
    return writer;
}
Writer::~Writer() { if (!finished_) finish(conflict("Writer abandoned")); }
exec::Status Writer::write(std::size_t offset, const void* data, std::size_t bytes) {
    if (finished_) return conflict("Writer finished");
    if (!data || offset > size_ || bytes > size_-offset) return invalid("Writer range out of bounds");
    std::memcpy(static_cast<char*>(storage_->regions_[region_].tensor->data)+offset_+offset, data, bytes);
    filled_.emplace_back(offset,offset+bytes); return {};
}
exec::Status Writer::finish(exec::Status error) {
    if (finished_) return conflict("Writer finished");
    std::sort(filled_.begin(),filled_.end()); std::size_t end=0;
    for (auto range : filled_) { if (range.first > end && !error) error=invalid("Incomplete writer contents"); end=std::max(end,range.second); }
    if (end != size_ && !error) error=invalid("Incomplete writer contents");
    finished_=true; completion_.set_value(error); return {};
}
Result<View> Weights::acquire_views(const std::shared_ptr<Storage>& storage, std::size_t array,
    const std::vector<Binding>& bindings) {
    return acquire_views(storage, {}, array, bindings);
}
Result<View> Weights::acquire_views(const std::shared_ptr<Storage>& storage,
    const std::shared_ptr<exec::Backend>& backend, std::size_t array, const std::vector<Binding>& bindings) {
    if (!storage || storage->registry_ != registry_) return invalid("Wrong registry");
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_ || storage->closing_ || storage->protection_ != Protection::group ||
        array >= storage->arrays_.size() || bindings.empty()) return invalid("Invalid group view");
    for (const auto& r : storage->regions_) if (r.state == RegionState::writing || r.state == RegionState::isolated)
        return conflict("Group is writing or isolated");
    const auto range=storage->arrays_[array];
    for (const auto& b : bindings) {
        if (b.storage != storage->identity_ || b.region < range.first || b.region-range.first >= range.second)
            return invalid("Binding outside array");
        const auto& r=storage->regions_[b.region];
        if (r.state != RegionState::published || r.id != b.id || r.generation != b.generation)
            return conflict("Stale or unpublished binding");
    }
    const auto region=bindings.front().region;
    auto allocation = storage->allocation_;
    if (backend) {
        const auto found = storage->consumers_.find(backend.get());
        if (found != storage->consumers_.end()) allocation = found->second.allocation;
        else if (backend != allocation->backend) return invalid("Weight consumer is outside storage scope");
    }
    auto* tensor=allocation->tensors[array];
    ++storage->regions_[region].uses;
    return View{tensor,std::make_shared<Usage>(storage,region,tensor,allocation)};
}
exec::Status Weights::evict(const std::shared_ptr<Storage>& storage, const std::vector<std::size_t>& regions) {
    if (!storage || storage->registry_ != registry_) return invalid("Wrong registry");
    std::lock_guard<std::mutex> lock(registry_->mutex);
    if (!storage->allocation_ || storage->closing_) return conflict("Storage closed");
    if (storage->protection_ == Protection::group)
        for (const auto& r : storage->regions_) if (r.uses || r.state == RegionState::writing || r.state == RegionState::isolated)
            return conflict("Group protected");
    for (auto i : regions) {
        if (i >= storage->regions_.size()) return invalid("Invalid region");
        const auto& r=storage->regions_[i];
        if (r.uses || r.state != RegionState::published) return conflict("Region not evictable");
    }
    for (auto i : regions) {
        auto& r=storage->regions_[i]; registry_->ids.erase(r.id);
        r.id.clear(); r.metadata.reset(); r.pending.reset(); r.state=RegionState::empty;
    }
    return {};
}
exec::Status Weights::close(const std::shared_ptr<Storage>& storage) {
    if (!storage || storage->registry_ != registry_) return invalid("Wrong registry");
    std::vector<std::shared_ptr<Write>> pending;
    {
        std::lock_guard<std::mutex> lock(registry_->mutex);
        if (!storage->allocation_) return {};
        storage->closing_=true;
        for (const auto& r : storage->regions_) if (auto w=r.pending.lock()) pending.push_back(w);
    }
    for (const auto& w : pending) abandon(w);
    {
        std::unique_lock<std::mutex> lock(registry_->mutex);
        registry_->changed.wait(lock,[&] {
            return std::all_of(storage->regions_.begin(),storage->regions_.end(),[](const Storage::Region& r) {
                return !r.uses && r.state != RegionState::writing;
            });
        });
    }
    return release_storage(storage);
}
}

namespace edge0::weights {
Result<std::vector<std::shared_ptr<Storage>>> Weights::create_storage(
    const std::vector<std::shared_ptr<exec::Backend>>& consumers, const std::vector<Array>& arrays, exec::Purpose purpose,
    Protection protection, const platform::ConsumptionQuery& query) {
    if (consumers.empty()) return invalid("Missing weight consumers");
    std::vector<ggml_backend_t> handles;
    for (const auto& consumer : consumers) {
        if (!consumer) return invalid("Missing weight consumer");
        handles.push_back(consumer->handle());
    }
    if (consumers.size() == 1) {
        auto storage = create_storage(consumers.front(), arrays, purpose, protection);
        if (!storage) return storage.error();
        return std::vector<std::shared_ptr<Storage>>{storage.value()};
    }
    const auto home = platform::choose_storage_home(handles, query);
    std::vector<std::shared_ptr<Storage>> result;
    for (std::size_t i = 0; i < (home ? 1 : consumers.size()); ++i) {
        auto storage = create_storage_on(consumers[i], arrays, protection,
            home ? home : ggml_backend_get_default_buffer_type(handles[i]), purpose);
        if (!storage) return storage.error();
        result.push_back(storage.value());
    }
    if (!home) return result;
    auto& storage = result.front();
    for (const auto& consumer : consumers) {
        const auto mode = query ? query(home, consumer->handle()) : platform::consumption(home, consumer->handle());
        auto allocation = exec::consume_buffer(budget_, storage->allocation_, consumer, mode, storage->regions_.size());
        if (!allocation) return allocation.error();
        Storage::ConsumerView view{allocation.value(), {}};
        if (allocation.value() == storage->allocation_) {
            for (const auto& region : storage->regions_) view.regions.push_back(region.tensor);
        } else {
            for (std::size_t a = 0; a < arrays.size(); ++a) {
                auto* array = allocation.value()->tensors[a];
                const auto range = storage->arrays_[a];
                for (std::size_t r = 0; r < range.second; ++r) {
                    auto* original = storage->regions_[range.first + r].tensor;
                    const auto offset = static_cast<unsigned char*>(original->data) -
                        static_cast<unsigned char*>(storage->allocation_->tensors[a]->data);
                    auto* tensor = ggml_view_4d(allocation.value()->context, array,
                        original->ne[0], original->ne[1], original->ne[2], original->ne[3],
                        original->nb[1], original->nb[2], original->nb[3], static_cast<std::size_t>(offset));
                    if (ggml_backend_view_init(tensor) != GGML_STATUS_SUCCESS)
                        return Error{ErrorCategory::backend_error, "Consumer weight view initialization failed"};
                    view.regions.push_back(tensor);
                }
            }
        }
        storage->consumers_.emplace(consumer.get(), std::move(view));
    }
    return result;
}
}
