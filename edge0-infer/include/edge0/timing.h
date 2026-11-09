#pragma once
#include <cstddef>
namespace edge0 {
struct DecodeTiming {
    double prepare_seconds = 0, wait_seconds = 0, submit_seconds = 0, route_seconds = 0;
    double binding_seconds = 0;
    double total_seconds = 0;
    std::size_t segments = 0;
};
struct TimingDistribution {
    double median_seconds = 0, p95_seconds = 0, maximum_seconds = 0;
    std::size_t samples = 0;
};
}
