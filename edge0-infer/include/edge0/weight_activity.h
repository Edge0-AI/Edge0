#pragma once
#include <cstddef>
namespace edge0 {
struct ExpertActivity {
    std::size_t demand = 0, hits = 0, loads = 0, bytes = 0;
    ExpertActivity& operator+=(const ExpertActivity& other) {
        demand+=other.demand; hits+=other.hits; loads+=other.loads; bytes+=other.bytes; return *this;
    }
};
struct PrefetchActivity {
    std::size_t initiated = 0, late = 0, bytes = 0, tail_bytes = 0, discarded = 0;
    PrefetchActivity& operator+=(const PrefetchActivity& o) {
        initiated+=o.initiated; late+=o.late;
        bytes+=o.bytes; tail_bytes+=o.tail_bytes; discarded+=o.discarded; return *this;
    }
};
struct WeightActivity {
    ExpertActivity decode, prefill;
    bool applicable = false;
    PrefetchActivity prefetch = {};
    WeightActivity& operator+=(const WeightActivity& other) {
        prefetch+=other.prefetch; decode+=other.decode; prefill+=other.prefill; applicable=applicable||other.applicable; return *this;
    }
};
}
