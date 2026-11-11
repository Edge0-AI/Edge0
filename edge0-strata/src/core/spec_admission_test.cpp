// Enumerate proposal and acceptance branches exactly; no Monte Carlo or GPU.
#include "strata/core/spec_prob.hpp"
#include "strata/core/gated_proposal.hpp"
#include "strata/spec/draft_policy.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace strata::core;
using Dist = std::array<double, 2>;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
Dist judged(const Dist& p, const std::array<float,2>& q, int d) {
    const int32_t ids[2] = {0,1};
    const SpecP P{ids,p.data(),2}; const SpecQ Q{ids,q.data(),2};
    const double a=std::min(1.0,p[d]/q[d]); Dist result{};
    if (a>0) result[spec_verify_row(P,d,Q,(float)(a/2),0.5f).token]+=a;
    if (a<1) result[spec_verify_row(P,d,Q,(float)((1+a)/2),0.5f).token]+=1-a;
    return result;
}
Dist legacy_raw_q_plain_p(const Dist& p, const std::array<float,2>& q, bool own_draw, float threshold) {
    Dist result{};
    for (int d=0;d<2;++d) {
        const float gate=own_draw?q[d]:std::max(q[0],q[1]);
        const auto row=gate>=threshold?judged(p,q,d):p;
        for (int i=0;i<2;++i) result[i]+=q[d]*row[i];
    }
    return result;
}
bool equal(const Dist& a,const Dist& b) {
    return std::abs(a[0]-b[0])<1e-12 && std::abs(a[1]-b[1])<1e-12;
}
// Enumerate the repaired first-row mixture using the ACTUAL packet builder.
// No candidate-conditioned renormalization: every draw receives the same root mu.
Dist proposal_contract(const Dist& p,const std::array<float,2>& q,float threshold,
                       strata::core::gated::RootChoice root={}) {
    std::array<int32_t,strata::core::gated::kStride> raw;
    raw.fill(-1);raw[0]=0;raw[1]=1;
    std::memcpy(raw.data()+strata::core::gated::kEntries,q.data(),sizeof(q));
    Dist result{};
    for (int d=0;d<2;++d) {
        const int32_t draft=d;const float score=q[d];strata::core::gated::Packet packet;
        const bool ok=strata::core::gated::make_packet(packet,raw.data(),&draft,&score,1,1,2,2,threshold,false,root);
        check(ok,"valid gated packet");if(!ok)continue;
        Dist mu{strata::core::gated::mass_at(packet.mu.data(),0),strata::core::gated::mass_at(packet.mu.data(),1)};
        double acceptance=0;
        if (!packet.null_tail) {
            check(mu[d]>0,"offer has positive mu");
            acceptance=std::min(1.0,p[d]/mu[d]);
            result[d]+=q[d]*acceptance;
        }
        if (acceptance<1) {
            Dist residual{std::max(0.0,p[0]-mu[0]),std::max(0.0,p[1]-mu[1])};
            const double sum=residual[0]+residual[1];check(sum>0,"reachable failure has residual");
            if(sum>0)for(int y=0;y<2;++y)result[y]+=q[d]*(1-acceptance)*residual[y]/sum;
        }
    }
    return result;
}
int main(int argc,char** argv) {
    const bool expect_pick=!(argc>1 && std::strcmp(argv[1],"top")==0);
    check(spec_gate_pick()==expect_pick,"pick is default; only explicit lowercase top selects Top compatibility");
    const Dist p{0.5,0.5};const std::array<float,2> q{0.875f,0.125f};
    check(equal(legacy_raw_q_plain_p(p,q,true,0.5f),Dist{0.5625,0.4375}),
          "LEGACY negative control: own-draw gate with full-q verifier/plain-p failure is biased");
    check(equal(proposal_contract(p,q,0.5f),p),"new pick mu/NULL mixture preserves p");
    check(equal(spec_gate_pick()?proposal_contract(p,q,0.5f):legacy_raw_q_plain_p(p,q,false,0.5f),p),
          "configured repaired pick or Top compatibility preserves p");
    int top_cases=0,proposal_cases=0;
    for(int a=1;a<16;++a)for(int b=1;b<16;++b) {
        const Dist P{a/16.0,1-a/16.0};const std::array<float,2> Q{b/16.0f,1-b/16.0f};
        for(float threshold:{0.0f,0.5f,0.9f,1.1f}) {
            check(equal(legacy_raw_q_plain_p(P,Q,false,threshold),P),"Top compatibility distribution law");++top_cases;
            check(equal(proposal_contract(P,Q,threshold),P),"H-only proposal mixture");++proposal_cases;
            for(int rescue:{0,1}) {
                check(equal(proposal_contract(P,Q,threshold,{true,rescue,2}),P),"fixed root UNION on every source branch");++proposal_cases;
            }
        }
    }
    // A planner must not learn from draws it may select, including discarded position-keyed lookahead.
    strata::spec::DraftPolicy live(6);
    for(int i=0;i<5;++i) { live.observe(false,4,3,24,1.0);live.observe(false,6,5,24,1.5); }
    const auto request_plan=live;
    auto short_chain=live,long_chain=live;
    short_chain.observe(false,4,3,24,0.2);long_chain.observe(false,4,3,24,8.0);
    const bool short_lookup=short_chain.choose(4,5,24).lookup,long_lookup=long_chain.choose(4,5,24).lookup;
    check(!short_lookup && long_lookup,"draw-dependent chain time can change the actual policy choice");
    Dist conditional{},planned{};
    for(int d=0;d<2;++d) {
        const auto selected=(d==0?long_lookup:short_lookup)?p:judged(p,q,d);
        const auto fixed=request_plan.choose(4,5,24).lookup?p:judged(p,q,d);
        for(int i=0;i<2;++i) { conditional[i]+=q[d]*selected[i];planned[i]+=q[d]*fixed[i]; }
    }
    check(equal(conditional,Dist{0.4375,0.5625}),"LEGACY negative: D-conditioned source without common first measure changes proposal law");
    check(equal(planned,p),"request plan preserves p despite subsequent timing observations");
    std::printf("spec_admission_test: %d Top compatibility cases, %d actual mu-packet cases, legacy negatives; %d failures\n",top_cases,proposal_cases,failures);
    return failures?1:0;
}
