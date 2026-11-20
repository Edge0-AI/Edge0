#include "app/sampling.h"
#include "tests/support/check.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <new>
namespace allocation_probe {
bool active=false;
std::size_t maximum=0,large=0;
constexpr std::size_t threshold=32768;
// Keep the deallocator out of line so GCC does not diagnose an inlined free
// against the matching replacement operator new used by this test.
#if defined(__GNUC__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
void release(void* p) noexcept {std::free(p);}
}
void* operator new(std::size_t bytes) {
    if (allocation_probe::active) {
        allocation_probe::maximum=std::max(allocation_probe::maximum,bytes);
        if (bytes>=allocation_probe::threshold) ++allocation_probe::large;
    }
    if (auto* p=std::malloc(bytes?bytes:1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {allocation_probe::release(p);}
void operator delete(void* p,std::size_t) noexcept {allocation_probe::release(p);}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete[](void* p) noexcept {allocation_probe::release(p);}
void operator delete[](void* p,std::size_t) noexcept {allocation_probe::release(p);}

using namespace edge0;
int main() {
    {
        std::vector<float> row(262144);
        std::vector<std::int32_t> expected(row.size());
        for (std::size_t i=0;i<row.size();++i) {row[i]=float(i%97);expected[i]=static_cast<std::int32_t>(i);}
        std::partial_sort(expected.begin(),expected.begin()+20,expected.end(),[&](auto a,auto b) {
            return row[a]>row[b] || (row[a]==row[b] && a<b);
        });
        for (std::size_t count:{0,1,20}) {
            allocation_probe::maximum=0;allocation_probe::active=true;
            const auto result=app::raw_logprob(row,12345,count);
            allocation_probe::active=false;
            std::cout << "count=" << count << " max_allocation=" << allocation_probe::maximum << "\n";
            CHECK(allocation_probe::maximum<4096);
            CHECK(result.top.size()==count && std::isfinite(result.logprob));
            for (std::size_t i=0;i<count;++i) CHECK(result.top[i].token==expected[i]);
        }
    }
    {
        GenerationConfiguration c;c.temperature=0;c.logit_bias={{0,4}};c.repetition_penalty=2;
        c.presence_penalty=1;c.frequency_penalty=.5;
        // (8+4)/2 - 1 - .5*3 = 3.5, while ID 1 remains 4.
        CHECK(app::sample_logits({8,4},c,{0},[]{return 0.;},{{0,3}}).value()==1);
        c={};c.presence_penalty=2;
        CHECK(app::sample_logits({2,1},c,{0},[]{return 0.;},{}).value()==0);
        CHECK(app::sample_logits({2,1},c,{0},[]{return 0.;},{{0,1}}).value()==1);
        c.presence_penalty=0;c.frequency_penalty=.5;
        CHECK(app::sample_logits({2,1},c,{},[]{return 0.;},{{0,1}}).value()==0);
        CHECK(app::sample_logits({2,1},c,{},[]{return 0.;},{{0,3}}).value()==1);
        c.frequency_penalty=0;c.logit_bias={{0,-100},{1,100}};
        CHECK(app::sample_logits({2,1},c,{},[]{return 0.;}).value()==1);
        const auto p=app::raw_logprob({2,1,1},1,3);
        const auto denominator=std::log(std::exp(2.)+2*std::exp(1.));
        CHECK(std::abs(p.logprob-(1-denominator))<1e-12);
        CHECK(p.top[0].token==0 && p.top[1].token==1 && p.top[2].token==2);
        CHECK(app::raw_logprob({2,1},0,0).top.empty());
        c={};c.repetition_penalty=std::numeric_limits<double>::min();
        CHECK(!app::sample_logits({std::numeric_limits<float>::max(),0},c,{0},[]{return 0.;}));
    }
    GenerationConfiguration c; c.temperature=1; c.top_k=0; c.top_p=1;
    std::size_t draws=0;
    auto sample=[&](std::vector<float> z, double u, app::SamplingHistory h={}) {
        auto result=app::sample_logits(z,c,h,[&] {++draws;return u;});
        CHECK(result); return result.value();
    };
    // Threshold-based Demo top-k retains ID 2 and selects it at u=.9; strict k selects ID 1.
    c.top_k=2; CHECK(sample({1,1,1},.9)==1);
    // Demo count(cumsum<=p) chooses a score threshold retaining all four ties: ID 3 at u=.9;
    // the shortest prefix reaching p=.5 contains only IDs 0 and 1, selecting ID 1.
    c.top_k=0;c.top_p=.5;CHECK(sample({0,0,0,0},.9)==1);
    c.top_p=.8;CHECK(sample({1,0,-1},.9)==1); // Demo count-before-crossing retains only ID 0.
    c.top_p=1;c.top_k=9;CHECK(sample({0,0,0},.9)==2);
    c.top_k=0;CHECK(sample({0,0,0},.9)==2);
    c.temperature=1;CHECK(sample({0,2},.8)==1);
    c.temperature=2;CHECK(sample({0,2},.8)==0);
    c.temperature=0;c.repetition_penalty=2;
    CHECK(sample({4,3},.99,{0})==1);
    CHECK(sample({-1,-1.5f},.99,{0})==1);
    CHECK(sample({0,-1},.99,{0})==0);
    CHECK(sample({8,3},.99,{0,0,0})==0); // Applying twice would select ID 1.
    const auto before=draws;
    CHECK(sample({2,1},.99,{0})==0);
    CHECK(draws==before); // Demo generic sampling at u=.99 consumes one draw and selects ID 1; ours selects ID 0 without drawing.
    c.temperature=1;c.repetition_penalty=1;
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
        auto result=app::sample_logits({0,bad},c,{},[&] {++draws;return .5;});
        CHECK(!result && result.error().category==ErrorCategory::invalid_argument);
    }
    CHECK(draws==before);
    c.temperature=1e-30;
    CHECK(sample({1,2,3},.99)==2);
    CHECK(sample({3,3,2},.75)==1); // Positive temperature still samples tied maxima.
    c.temperature=1;c.top_p=.01;CHECK(sample({0,0,0},.999)==0);
    app::SamplingRandom random(std::mt19937_64::default_seed);
    double value=0;for(int i=0;i<10000;++i)value=random.uniform();
    CHECK(value==static_cast<double>(9981545732273789042ULL>>11)*0x1p-53);
    CHECK(random.draws()==10000);
    std::cout<<"Sampling oracle, three Demo counterexamples, finite logits and RNG known value passed\n";
}
