#include "strata/core/conversation_cache.hpp"
#include <cstdlib>
#include <cstdio>
#include <new>
static long fail_after=-1;
void* operator new(std::size_t n) {
    if(fail_after==0) throw std::bad_alloc();
    if(fail_after>0) --fail_after;
    if(void* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
using strata::core::ConversationCheckpoint;
bool equal(const ConversationCheckpoint& a,const ConversationCheckpoint& b) {
    if(a.ids!=b.ids || a.imgs!=b.imgs || a.gdn!=b.gdn || a.ple!=b.ple || a.tails!=b.tails || a.dead!=b.dead || a.block_pos!=b.block_pos || a.used!=b.used || a.stage_parts.size()!=b.stage_parts.size()) return false;
    for(size_t i=0;i<a.stage_parts.size();++i) if(!equal(a.stage_parts[i],b.stage_parts[i])) return false;
    return true;
}
int main() {
    // #752: every stage count a layer split runs (1-3), with one checkpoint that has no stage parts (it stays in `rest`)
    long total_faults=0;
    for(size_t stages=1;stages<=3;++stages) {
        int faults=0;
        bool finished=false;
        for(int n=0;n<256 && !finished;++n) {
            std::vector<ConversationCheckpoint> checks(4);
            for(size_t i=0;i<checks.size();++i) {
                auto& c=checks[i]; c.ids={1,2,(int)i+3};c.gdn={1,2,3};c.ple={4};c.used=i;
                if(i==2) continue;   // incomplete: no stage parts
                c.stage_parts.resize(stages);
                for(size_t k=0;k<stages;++k) { c.stage_parts[k].gdn={5,6,static_cast<uint8_t>(7+k)}; c.stage_parts[k].tails={static_cast<uint8_t>(k)}; }
            }
            const auto backup=checks;
            fail_after=n;
            try {
                auto split=strata::core::conversation_checkpoints_split(std::move(checks),stages);
                fail_after=0; // Merge-back after a successful split must not allocate.
                if(!strata::core::conversation_checkpoints_merge(std::move(split),checks)) return 2;
                fail_after=-1;
                if(checks.size()!=backup.size()) return 6;
                finished=true;   // the split went through: the merge gives the running state back, in order
                for(size_t i=0;i<checks.size();++i) {
                    const auto& x=checks[i]; const auto& y=backup[i];
                    // (each stage part's identity is its parent's by design, so only the state is compared)
                    if(x.ids!=y.ids || x.gdn!=y.gdn || x.ple!=y.ple || x.used!=y.used || x.stage_parts.size()!=y.stage_parts.size()) return 7;
                    for(size_t k=0;k<x.stage_parts.size();++k) if(x.stage_parts[k].gdn!=y.stage_parts[k].gdn || x.stage_parts[k].tails!=y.stage_parts[k].tails) return 8;
                }
            } catch(const std::bad_alloc&) {
                fail_after=-1;++faults;
                if(checks.size()!=backup.size()) return 4;
                for(size_t i=0;i<checks.size();++i) if(!equal(checks[i],backup[i])) {
                    std::fprintf(stderr,"stages %zu: allocation failure %d partially moved checkpoint %zu\n",stages,n,i);return 1;
                }
            }
        }
        if(!finished || faults==0) return 5;
        std::printf("stages %zu: allocation-failure cases preserved: %d\n",stages,faults);
        total_faults+=faults;
    }
    return total_faults>0?0:3;
}
