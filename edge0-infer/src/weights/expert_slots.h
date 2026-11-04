#pragma once
#include "weights/storage.h"
#include "edge0/weight_activity.h"
#include <array>
namespace edge0::weights {
using Activity = ExpertActivity;
struct PoolDescription {
    std::uint32_t layer = 0;
    std::size_t slots = 0;
    std::array<exec::Layout,3> projections;
    std::vector<std::array<Source,3>> experts;
};
struct Prepared {
    std::map<std::int32_t,std::size_t> mapping;
    std::array<std::vector<Binding>,3> bindings;
    Activity activity;
    std::vector<std::int32_t> loaded_experts;
};
class Preparation;
class ExpertPool : public std::enable_shared_from_this<ExpertPool> {
public:
    static Result<std::shared_ptr<ExpertPool>> create_pools(Weights& weights,
        const std::shared_ptr<exec::Backend>& backend, const PoolDescription& description);
    Result<std::shared_ptr<Preparation>> prepare(const std::vector<std::int32_t>& experts, Priority priority = Priority::urgent);
    bool ready(const std::shared_ptr<Preparation>& preparation) const;
    Result<Prepared> await_ready(const std::shared_ptr<Preparation>& preparation);
    exec::Status abandon(const std::shared_ptr<Preparation>& preparation);
    exec::Status close();
    std::shared_ptr<Storage> storage() const { return storage_; }
    Activity activity() const;
private:
    friend class Preparation;
    Weights* weights_ = nullptr;
    PoolDescription description_;
    std::shared_ptr<Storage> storage_;
    mutable std::mutex mutex_;
    std::weak_ptr<Preparation> pending_;
    std::vector<std::uint64_t> last_used_;
    std::uint64_t clock_ = 0;
    Activity activity_;
    bool closed_ = false;
};
class Preparation {
public:
    ~Preparation();
private:
    friend class ExpertPool;
    std::shared_ptr<ExpertPool> pool_;
    std::shared_ptr<Write> write_;
    Prepared result_;
    bool done_ = false;
};
}
