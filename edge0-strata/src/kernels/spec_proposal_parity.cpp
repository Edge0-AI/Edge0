// Explicit mu/NULL verifier against a CPU oracle; CUDA/HIP, no model or tracing.
// Equal finite logits at three known IDs, all other logits -inf, give p=1/3
// at each ID independently of an exp implementation. A top-k1 fixture gives
// exact p=delta. Thus no test-only inspection API or SamplerTraceRow is needed.
// These finite per-token checks do not claim arbitrary-logit bitwise equality.
#include "strata/core/spec_prob.hpp"
#include "strata/core/spec_proposal.hpp"
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int V=8192, MaxT=3, A=7, B=4097, C=8190;
constexpr int Entries=strata::core::kSpecQEntries, Stride=strata::core::kSpecQStride;
constexpr int Error=strata::core::proposal::kProposalSamplerErrorToken;
using Mass=std::array<float,3>;
constexpr std::array<int,3> IDs{{A,B,C}};
struct Result {int token;bool accepted;};
struct Case {
    const char* name;int T,nq;std::vector<int32_t> drafts;
    std::vector<Mass> mu,raw;int discarded=B;
    bool error=false,legacy_equal=false;
};
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
void cu(cudaError_t code,const char* message) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string(message)+": "+cudaGetErrorString(code));
}
float uniform(uint64_t seed,uint64_t counter) {
    uint32_t c0=(uint32_t)counter,c1=(uint32_t)(counter>>32),c2=(uint32_t)seed,c3=(uint32_t)(seed>>32);
    for(int i=0;i<10;++i) {
        const uint64_t x=0x9E3779B9u*(uint64_t)c0,y=0xBB67AE85u*(uint64_t)c2;
        const uint32_t n0=(uint32_t)(y>>32)^c1^(uint32_t)i,n2=(uint32_t)(x>>32)^c3;
        c0=n0;c1=(uint32_t)y;c2=n2;c3=(uint32_t)x;
    }
    return (float)(c0>>8)*(1.0f/16777216.0f);
}
double mass(const Mass& q,int id) {
    for(int i=0;i<3;++i)if(IDs[i]==id)return q[i];
    return 0;
}
int choose(const std::array<double,3>& weights,float u,bool normalize) {
    const double sum=weights[0]+weights[1]+weights[2];require(sum>0,"empty reachable law");
    double cum=0;int last=-1;
    for(int i=0;i<3;++i) {
        if(normalize && weights[i]<=0)continue;
        last=IDs[i];cum+=normalize?weights[i]/sum:weights[i];
        if((double)u<cum)return IDs[i];
    }
    require(last>=0,"empty support");return last;
}
Result reference(const Case& c,int row,uint64_t seed,const Mass* alternate=nullptr) {
    const uint64_t pos=365+(uint64_t)row;
    const std::array<double,3> p=c.error?std::array<double,3>{{1,0,0}}:std::array<double,3>{{1.0/3,1.0/3,1.0/3}};
    const bool last=row==c.T-1,null=last && c.nq==c.T;
    if(last && !null)return {choose(p,uniform(seed,pos),false),false};
    const int d=null?-1:c.drafts[row];const bool point=!null && row>=c.nq;
    const Mass q=alternate?*alternate:(row<c.nq?c.mu[row]:Mass{{0,0,0}});
    double pd=0;for(int i=0;i<3;++i)if(IDs[i]==d)pd=p[i];
    if(!null) {
        const double qd=point?1:mass(q,d);
        if(!(qd>0) || !std::isfinite(qd))return {Error,false};
        if(pd>0 && (double)uniform(seed,pos|strata::core::kSpecCtrAccept)*qd<pd)return {d,true};
    }
    std::array<double,3> weights{};
    for(int i=0;i<3;++i)weights[i]=point?(IDs[i]==d?0:p[i]):std::max(0.0,p[i]-mass(q,IDs[i]));
    if(weights[0]+weights[1]+weights[2]<=0)return {Error,false};
    return {choose(weights,uniform(seed,pos|strata::core::kSpecCtrResid),true),false};
}
struct Device {
    float* logits=nullptr;int* out=nullptr;int32_t* draft=nullptr;int32_t* mu=nullptr;
    Device() {
        cu(cudaMalloc((void**)&logits,MaxT*V*sizeof(float)),"logits");
        cu(cudaMalloc((void**)&out,MaxT*sizeof(int)),"out");
        cu(cudaMalloc((void**)&draft,(MaxT-1)*sizeof(int32_t)),"draft");
        cu(cudaMalloc((void**)&mu,MaxT*Stride*sizeof(int32_t)),"mu");
    }
    ~Device(){cudaDeviceSynchronize();cudaFree(logits);cudaFree(out);cudaFree(draft);cudaFree(mu);}
};
std::vector<Case> cases() {
    const Mass raw{{.625f,.25f,.125f}},m{{.625f,0,0}},root{{.625f,.25f,0}};
    return {
        {"T1 NULL",1,1,{}, {m},{raw}},
        {"zero mu NULL",1,1,{}, {Mass{{0,0,0}}},{Mass{{.375f,.375f,.25f}}}},
        {"offer then NULL",2,2,{A},{m,m},{raw,raw}},
        {"two offers then NULL",3,3,{A,A},{m,m,m},{raw,raw,raw}},
        {"plain budget",1,0,{}, {},{}},
        {"offer reject and bonus",2,1,{A},{m},{raw}},
        {"mu then delta then bonus",3,1,{A,B},{m},{raw}},
        {"point then bonus",2,0,{A},{},{}},
        {"UNION low-q rescue",2,1,{B},{root},{raw}},
        {"UNION other draw rejection",2,1,{A},{root},{raw}},
        {"UNION NULL",1,1,{}, {root},{raw},C},
        {"inclusive threshold NULL",1,1,{}, {Mass{{.5f,0,0}}},{Mass{{.5f,.25f,.25f}}}},
        {"zero residual sentinel",1,1,{}, {Mass{{1,0,0}}},{Mass{{1,0,0}}},-1,true},
        {"legacy full-q parity",3,2,{A,B},{raw,raw},{raw,raw},-1,false,true},
        {"missing offered mu sentinel",2,1,{B},{m},{raw}}
    };
}
int run() {
    Device d;long long mismatches=0,rows=0,accepts=0,rejects=0,null_same=0,plain_witness=0,fullq_witness=0,union_witness=0;
    const uint64_t seeds[]={0,1,2,3,7,11,17,29,41,67,101,777,2026101600ULL,2026101700ULL,0x123456789abcdef0ULL,3728620368707436339ULL};
    for(const Case& c:cases()) {
        require((int)c.drafts.size()==c.T-1 && (int)c.mu.size()==c.nq,"bad fixture");
        std::vector<float> logits(c.T*V,-std::numeric_limits<float>::infinity());
        for(int t=0;t<c.T;++t){logits[t*V+A]=0;if(!c.error){logits[t*V+B]=0;logits[t*V+C]=0;}}
        std::vector<int32_t> packed(c.T*Stride,-1);const int permutation[]={2,0,1};
        for(int t=0;t<c.nq;++t)for(int i=0;i<Entries;++i) {
            const float q=i<3?c.mu[t][permutation[i]]:0;
            if(i<3)packed[t*Stride+i]=IDs[permutation[i]];
            std::memcpy(&packed[t*Stride+Entries+i],&q,sizeof(q));
        }
        cu(cudaMemcpy(d.mu,packed.data(),packed.size()*sizeof(int32_t),cudaMemcpyHostToDevice),"copy mu");
        if(!c.drafts.empty())cu(cudaMemcpy(d.draft,c.drafts.data(),c.drafts.size()*sizeof(int32_t),cudaMemcpyHostToDevice),"copy drafts");
        for(uint64_t seed:seeds) {
            strata::kernels::SamplerParams p;p.top_k=c.error?1:3;p.top_p=1;p.min_p=0;p.temperature=1;p.seed=seed;p.counter=365;
            const int32_t* drafts=c.drafts.empty()?nullptr:d.draft;
            cu(cudaMemcpy(d.logits,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice),"copy logits");
            require(strata::kernels::sample_tokens_proposal(d.logits,c.T,V,nullptr,0,p,drafts,d.mu,c.nq,d.out,nullptr),"proposal Split refused");
            cu(cudaDeviceSynchronize(),"proposal sync");std::vector<int> out(c.T);
            cu(cudaMemcpy(out.data(),d.out,c.T*sizeof(int),cudaMemcpyDeviceToHost),"copy out");
            for(int t=0;t<c.T;++t) {
                const auto ref=reference(c,t,seed);++rows;
                if(out[t]!=ref.token){++mismatches;std::fprintf(stderr,"%s seed=%llu row=%d: got=%d expected=%d\n",c.name,(unsigned long long)seed,t,out[t],ref.token);}
                const bool null=t==c.T-1 && c.nq==c.T;
                if(t<c.T-1 && ref.token!=Error)(ref.accepted?accepts:rejects)++;
                if(null && !c.error) {
                    if(out[t]==c.discarded)++null_same;
                    const std::array<double,3> target{{1.0/3,1.0/3,1.0/3}};
                    if(ref.token!=choose(target,uniform(seed,365+t),false))++plain_witness;
                    if(ref.token!=reference(c,t,seed,&c.raw[t]).token)++fullq_witness;
                    if(std::strcmp(c.name,"UNION NULL")==0) {
                        const Mass old{{.625f,0,0}};
                        if(ref.token!=reference(c,t,seed,&old).token)++union_witness;
                    }
                    // NULL is terminal even when residual==discarded D. T1 has
                    // zero offered inputs; no dtok slot exists for its raw D.
                    if(c.T==1 && out[t]==c.discarded)require(c.drafts.empty(),"NULL candidate leaked into offered inputs");
                }
            }
            if(c.legacy_equal) {
                cu(cudaMemcpy(d.logits,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice),"legacy logits");
                require(strata::kernels::sample_tokens_spec(d.logits,c.T,V,nullptr,0,p,drafts,d.mu,c.nq,d.out,nullptr),"legacy Split refused");
                cu(cudaDeviceSynchronize(),"legacy sync");std::vector<int> old(c.T);
                cu(cudaMemcpy(old.data(),d.out,c.T*sizeof(int),cudaMemcpyDeviceToHost),"legacy output");
                require(old==out,"explicit proposal call changed legacy full-q result");
            }
        }
    }
    require(accepts>0 && rejects>0 && null_same>0,"missing accept/reject/NULL-equality coverage");
    require(plain_witness>0 && fullq_witness>0 && union_witness>0,"fixed seeds did not discriminate incorrect laws");
    std::printf("spec_proposal_parity: %lld rows, %lld mismatches; accepts=%lld rejects=%lld NULL==rawD=%lld; ordinary/fullq/omitted-rescue witnesses=%lld/%lld/%lld\n",rows,mismatches,accepts,rejects,null_same,plain_witness,fullq_witness,union_witness);
    return mismatches?1:0;
}
} // namespace
int main() {
    try{return run();}catch(const std::exception& e){std::fprintf(stderr,"spec_proposal_parity: %s\n",e.what());return 1;}
}
