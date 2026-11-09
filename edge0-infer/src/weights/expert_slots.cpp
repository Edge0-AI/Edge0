#include "weights/expert_slots.h"
#include <algorithm>
#include <limits>
namespace edge0::weights {
namespace {
Error conflict(const char* message) { return {ErrorCategory::conflict,message}; }
std::string id(std::uint32_t layer,std::int32_t expert,std::size_t projection) {
    static const char* names[]{"gate","up","down"};
    return "slots/"+std::to_string(layer)+"/"+std::to_string(expert)+"/"+names[projection];
}
}
Result<std::shared_ptr<ExpertPool>> ExpertPool::create_pools(Weights& weights,
    const std::shared_ptr<exec::Backend>& backend,const PoolDescription& description) {
    if (!description.slots || description.slots>description.experts.size())
        return Error{ErrorCategory::invalid_argument,"Invalid pool capacity"};
    std::vector<Array> arrays;
    for (const auto& layout:description.projections) arrays.push_back({layout,description.slots});
    auto storage=weights.create_storage(backend,arrays, edge0::exec::Purpose::experts,Protection::group);
    if (!storage) return storage.error();
    auto pool=std::make_shared<ExpertPool>(); pool->weights_=&weights;
    pool->storage_=storage.value(); pool->description_=description; pool->last_used_.resize(description.slots);
    return pool;
}
Result<std::shared_ptr<Preparation>> ExpertPool::prepare(const std::vector<std::int32_t>& requested, Priority priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !pending_.expired()) return conflict("Pool closed or preparation pending");
    const auto count=description_.slots;
    std::vector<RegionInfo> current;
    for (std::size_t i=0;i<count*3;++i) {
        auto region=weights_->query_state(storage_,i); if(!region) return region.error();
        if (region.value().uses || region.value().state==RegionState::writing || region.value().state==RegionState::isolated)
            return conflict("Pool is read, writing or isolated");
        current.push_back(region.value());
    }
    std::set<std::int32_t> experts(requested.begin(),requested.end());
    if (experts.empty() || experts.size()>count) return Error{ErrorCategory::invalid_argument,"Expert demand exceeds pool capacity"};
    for (auto expert:experts) if (expert<0 || std::size_t(expert)>=description_.experts.size())
        return Error{ErrorCategory::invalid_argument,"Expert outside source range"};
    auto preparation=std::make_shared<Preparation>(); preparation->pool_=shared_from_this();
    auto& result=preparation->result_; result.activity.demand=experts.size();
    std::set<std::size_t> protected_slots;
    // Resolve every hit before selecting any replacement.
    for (auto expert:experts) for (std::size_t slot=0;slot<count;++slot) {
        if (current[slot].state!=RegionState::published || current[slot].binding.id!=id(description_.layer,expert,0)) continue;
        bool hit=true;
        for (std::size_t p=1;p<3;++p) hit=hit && current[p*count+slot].state==RegionState::published &&
            current[p*count+slot].binding.id==id(description_.layer,expert,p);
        if (!hit) return conflict("Inconsistent projection mapping");
        result.mapping[expert]=slot; protected_slots.insert(slot); ++result.activity.hits; break;
    }
    std::vector<LoadItem> items;
    for (auto expert:experts) {
        if(result.mapping.count(expert)) continue;
        std::size_t selected=count;
        for(std::size_t slot=0;slot<count;++slot) {
            if(protected_slots.count(slot)) continue;
            if(current[slot].state==RegionState::empty) { selected=slot; break; }
            if(selected==count || last_used_[slot]<last_used_[selected]) selected=slot;
        }
        if(selected==count) return conflict("No unprotected slot");
        protected_slots.insert(selected); result.mapping[expert]=selected; ++result.activity.loads; result.loaded_experts.push_back(expert);
        for(std::size_t p=0;p<3;++p) {
            const auto& source=description_.experts[std::size_t(expert)][p];
            items.push_back({p*count+selected,id(description_.layer,expert,p),source});
            result.activity.bytes+=source.bytes;
        }
    }
    if(!items.empty()) {
        auto write=weights_->begin_write(storage_,items); if(!write) return write.error();
        preparation->write_=write.value();
        if(auto error=weights_->fill_regions(write.value(),priority)) { weights_->abandon(write.value()); return *error; }
    }
    pending_=preparation;
    return preparation;
}
bool ExpertPool::ready(const std::shared_ptr<Preparation>& preparation) const {
    return preparation && weights_->ready(preparation->write_);
}
Result<Prepared> ExpertPool::await_ready(const std::shared_ptr<Preparation>& preparation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!preparation || preparation->pool_.get()!=this || preparation->done_ || pending_.lock()!=preparation) return conflict("Invalid preparation");
    if(preparation->write_) {
        auto completed=weights_->complete(preparation->write_);
        if(!completed) { preparation->done_=true; pending_.reset(); return completed.error(); }
    }
    auto& result=preparation->result_;
    for(auto entry:result.mapping) {
        last_used_[entry.second]=++clock_;
        for(std::size_t p=0;p<3;++p) {
            auto region=weights_->query_state(storage_,p*description_.slots+entry.second);
            if(!region) return region.error();
            result.bindings[p].push_back(region.value().binding);
        }
    }
    activity_+=result.activity; preparation->done_=true; pending_.reset(); return result;
}
exec::Status ExpertPool::abandon(const std::shared_ptr<Preparation>& preparation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!preparation || preparation->pool_.get()!=this) return conflict("Invalid preparation");
    if(preparation->done_) return {};
    if(pending_.lock()!=preparation) return conflict("Preparation is not current");
    if(preparation->write_) weights_->abandon(preparation->write_);
    preparation->done_=true; pending_.reset(); return {};
}
Preparation::~Preparation() {
    // Write destruction joins actual writers and rolls back unpublished content.
    write_.reset();
}
exec::Status ExpertPool::close() {
    std::shared_ptr<Preparation> pending;
    { std::lock_guard<std::mutex> lock(mutex_); closed_=true; pending=pending_.lock(); }
    if(pending) abandon(pending);
    return weights_->close(storage_);
}
Activity ExpertPool::activity() const { std::lock_guard<std::mutex> lock(mutex_); return activity_; }
}
