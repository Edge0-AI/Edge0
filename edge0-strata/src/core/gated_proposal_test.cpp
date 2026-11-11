// Actual gated proposal packet contract, CPU-only; hand-specified expectations.
// Includes immutable q bits, root rescue on all branches, NULL versus budget, and invalid packets.
// Does not validate a GPU execution or main's final window copy/commit loop.
#include "strata/core/gated_proposal.hpp"
#include "strata/core/spec_proposal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
namespace ap = strata::core::gated;
namespace ac = strata::core::proposal;
constexpr int A=7, B=4097, C=8190, V=8192;
constexpr float theta=.5f;
using Row=std::array<int32_t,ap::kStride>;
using Rows=std::array<int32_t,ap::kRows*ap::kStride>;
using Floats=std::array<float,ap::kRows>;
using Drafts=std::array<int32_t,ap::kRows>;
int checks=0;
void need(bool ok,const char* msg) {++checks;if(!ok)throw std::runtime_error(msg);}
uint32_t bits(float f) {uint32_t n;std::memcpy(&n,&f,4);return n;}
Row row(float a=.625f,float b=.25f,float c=.125f) {
    Row r; r.fill(-1);
    // IDs/padding deliberately unlike target order. Padding must survive masking.
    for(int i=0;i<ap::kEntries;++i)r[ap::kEntries+i]=(int32_t)(0x3e800000u+(uint32_t)i);
    const int ids[]={C,A,B};const float probs[]={c,a,b};
    for(int i=0;i<3;++i){r[i]=ids[i];std::memcpy(r.data()+ap::kEntries+i,&probs[i],4);}
    return r;
}
void put(Rows& rows,int n,const Row& r) {std::memcpy(rows.data()+n*ap::kStride,r.data(),sizeof(r));}
void expect_mask(const int32_t* actual,const Row& original,float a,float b,float c) {
    Row expected=original;const float p[]={c,a,b};
    for(int i=0;i<3;++i)std::memcpy(expected.data()+ap::kEntries+i,&p[i],4);
    need(std::memcmp(actual,expected.data(),sizeof(expected))==0,"mask differs in retained bits / IDs / padding / +0");
}
struct Expected {int T,nq,offers,attempts,failed;bool suffix,nulltail;};
void packet(const ap::Packet& p,Expected x) {
    need(!p.error,"unexpected packet error");
    need(p.T==x.T && p.n_q==x.nq && p.offers==x.offers && p.attempts==x.attempts && p.failed_index==x.failed && p.from_sfx==x.suffix && p.null_tail==x.nulltail,"packet fields differ from hand-written expectation");
    need(p.T==p.offers+1,"one terminal output after offered prefix");
    need(ac::rowkind(p.T-1,p.T,p.n_q)==(p.null_tail?ac::RowKind::NullTail:ac::RowKind::PlainBonus),"AA packet / Hubble terminal rowkind mismatch");
}
struct Fixture {
    Rows raw{};Drafts d{};Floats scores{};
    Fixture(){raw.fill(-1);d.fill(-777);scores.fill(std::numeric_limits<float>::quiet_NaN());}
    void set(int j,const Row& r,int id,float score){put(raw,j,r);d[j]=id;scores[j]=score;}
    bool call(ap::Packet& out,int generated,int budget,const ap::RootChoice& root={},bool initial=false,int capacity=6) const {
        const Rows saved=raw;const Drafts ds=d;const Floats ss=scores;
        const bool ok=ap::make_packet(out,raw.data(),d.data(),scores.data(),generated,budget,capacity,V,theta,initial,root);
        need(std::memcmp(raw.data(),saved.data(),sizeof raw)==0,"raw q mutated");
        need(std::memcmp(d.data(),ds.data(),sizeof d)==0,"raw drafts mutated");
        need(std::memcmp(scores.data(),ss.data(),sizeof scores)==0,"own-score mutated");
        need(ok==(out.error==nullptr),"return/error inconsistent");
        return ok;
    }
};
void positive(const char* name,const Fixture& f,int n,int budget,ap::RootChoice root,Expected x) {
    ap::Packet p;need(f.call(p,n,budget,root),name);packet(p,x);
    std::printf("{\"case\":\"%s\",\"passed\":true}\n",name);
}
void tests() {
    const Row raw=row();Row masked{};const Row saved=raw;
    need(ap::mask_row(raw.data(),masked.data(),theta,-1,V),"mask H");
    expect_mask(masked.data(),raw,.625f,0,0); // exact GPU T1_NULL mu
    need(ap::mask_row(raw.data(),masked.data(),theta,B,V),"mask UNION");
    expect_mask(masked.data(),raw,.625f,.25f,0); // GPU UNION mu, sum=.875 !=1
    need(std::memcmp(raw.data(),saved.data(),sizeof raw)==0,"mask modified raw");
    const Row edge=row(.5f,std::nextafter(.5f,0.0f),0x1p-25f);
    need(ap::mask_row(edge.data(),masked.data(),theta,-1,V),"mask inclusive edge");
    expect_mask(masked.data(),edge,.5f,0,0);
    const Row above=row(std::nextafter(.5f,1.0f),.25f,.25f-0x1p-24f);
    need(ap::mask_row(above.data(),masked.data(),theta,-1,V),"mask above edge");
    expect_mask(masked.data(),above,std::nextafter(.5f,1.0f),0,0);
    const Row none=row(.375f,.375f,.25f);
    need(ap::mask_row(none.data(),masked.data(),theta,-1,V),"mask zero measure");
    expect_mask(masked.data(),none,0,0,0);
    const Row negzero=row(1.0f,-0.0f,0.0f);
    need(ap::mask_row(negzero.data(),masked.data(),0.0f,-1,V),"theta zero mask");
    expect_mask(masked.data(),negzero,1.0f,-0.0f,0.0f);
    need(bits(ap::mass_at(masked.data(),B))==bits(-0.0f),"retained negative zero bits changed");

    ap::Packet p;
    need(ap::make_packet(p,nullptr,nullptr,nullptr,0,3,6,V,theta,true,{}),"initial no buffers");
    packet(p,{1,0,0,0,-1,false,false});
    need(ap::make_packet(p,nullptr,nullptr,nullptr,0,0,6,V,theta,false,{}),"zero producer budget");
    packet(p,{1,0,0,0,-1,false,false});

    Fixture firstnull;firstnull.set(0,raw,B,.25f);
    positive("root_NULL",firstnull,1,3,{}, {1,1,0,1,0,false,true});
    Fixture allzero;allzero.set(0,none,B,.375f);
    positive("all_zero_mu_NULL",allzero,1,3,{}, {1,1,0,1,0,false,true});
    Fixture one;one.set(0,raw,A,.625f);
    positive("actual_max_steps_1_is_bonus",one,1,1,{}, {2,1,1,1,-1,false,false});
    need(!one.call(p,1,3),"incomplete generated1/budget3 must not become bonus");
    Fixture later;later.set(0,raw,A,.625f);later.set(1,raw,C,.125f);
    positive("second_attempt_NULL",later,2,3,{}, {2,2,1,2,1,false,true});
    need(later.call(p,2,3),"NULL packet");
    need(p.offers==1 && p.failed_index==1 && p.T-1==1,"failed D excluded from offered window input span");
    // Packet contains no token array. These counts exclude failed D from the
    // intended copy span; actual main window construction remains integration work.
    expect_mask(p.mu.data()+ap::kStride,raw,.625f,0,0);
    Fixture full;full.set(0,raw,A,.625f);full.set(1,raw,A,.625f);full.set(2,raw,A,.625f);
    positive("actual_max_steps_3_bonus",full,3,3,{}, {4,3,3,3,-1,false,false});
    need(!full.call(p,3,4),"generated3/budget4 must not guess exhausted budget");

    const ap::RootChoice E{true,B,6};
    positive("UNION_low_D_suffix",firstnull,1,3,E, {6,1,5,1,-1,true,false});
    need(firstnull.call(p,1,3,E),"UNION rescue");expect_mask(p.mu.data(),raw,.625f,.25f,0);
    const Row union_root_snapshot=[](){Row r=row();Row out;need(ap::mask_row(r.data(),out.data(),theta,B,V),"snapshot union");return out;}();
    Fixture high;high.set(0,raw,A,.625f);
    need(high.call(p,1,1,E),"UNION D!=s offer");packet(p,{2,1,1,1,-1,false,false});
    need(std::memcmp(p.mu.data(),union_root_snapshot.data(),sizeof(Row))==0,"UNION must exist on MTP branch");
    Fixture otherlow;otherlow.set(0,raw,C,.125f);
    need(otherlow.call(p,1,3,E),"UNION D!=s NULL");packet(p,{1,1,0,1,0,false,true});
    need(std::memcmp(p.mu.data(),union_root_snapshot.data(),sizeof(Row))==0,"UNION must exist on NULL branch");
    // Later row's B is NOT rescued by the root set.
    Fixture next;next.set(0,raw,A,.625f);next.set(1,raw,B,.25f);
    need(next.call(p,2,3,E),"later H only");packet(p,{2,2,1,2,1,false,true});
    expect_mask(p.mu.data(),raw,.625f,.25f,0);
    expect_mask(p.mu.data()+ap::kStride,raw,.625f,0,0);
    // Eligibility false despite populated rescue fields must mean plain H.
    positive("ineligible_rescue_ignored",firstnull,1,3,{false,B,6}, {1,1,0,1,0,false,true});

    // Frozen source choice must not inspect invalid discarded later materialization.
    // A future row's poison may be present, but only root is consumed if suffix wins.
    Fixture eager=firstnull;
    positive("suffix_ignores_later_D_score_q",eager,3,3,E, {6,1,5,1,-1,true,false});
    Fixture stopped=firstnull;
    positive("NULL_ignores_later_materialization",stopped,3,3,{}, {1,1,0,1,0,false,true});
    Fixture edgeoffer;edgeoffer.set(0,edge,A,.5f);
    positive("inclusive_D_at_theta",edgeoffer,1,1,{}, {2,1,1,1,-1,false,false});
    Fixture edgenull;edgenull.set(0,edge,B,std::nextafter(.5f,0.0f));
    positive("D_just_below_theta_NULL",edgenull,1,3,{}, {1,1,0,1,0,false,true});

    Fixture bad=one;bad.scores[0]=std::nextafter(.625f,1.0f);
    need(!bad.call(p,1,1),"own-score mismatch must fail");
    bad=one;bad.d[0]=123;
    need(!bad.call(p,1,1),"D outside q support must fail");
    bad=one;bad.raw[1]=bad.raw[0];
    need(!bad.call(p,1,1),"duplicate support must fail");
    bad=one;float nan=std::numeric_limits<float>::quiet_NaN();std::memcpy(bad.raw.data()+ap::kEntries+1,&nan,4);
    need(!bad.call(p,1,1),"nonfinite q must fail");
    need(!one.call(p,1,0),"n_generated exceeds actual budget");
    need(!one.call(p,1,1,{},true),"initial stale attempts");
    need(!one.call(p,1,1,{true,V,6}),"out-of-vocab rescue");
    need(!one.call(p,1,1,{true,B,7}),"suffix width exceeds capacity");
    need(!one.call(p,1,6),"budget cannot fill terminal slot");
    need(!ap::make_packet(p,nullptr,nullptr,nullptr,0,1,6,V,theta,false,{}),"missing expected attempt");
    need(!ap::make_packet(p,nullptr,nullptr,nullptr,1,1,6,V,theta,false,{}),"null buffers");
    need(!ap::mask_row(raw.data(),masked.data(),nan,-1,V),"NaN theta");
    need(!ap::mask_row(raw.data(),masked.data(),-.1f,-1,V),"negative theta");
    // Concrete producer/context boundaries: budget is not inferred from count.
    // The failed row may be the LAST physically generated row at actual budget.
    Fixture last_attempt;last_attempt.set(0,raw,A,.625f);last_attempt.set(1,raw,B,.25f);
    need(last_attempt.call(p,2,2,{},false,3),"last producer attempt failed at exact budget");
    packet(p,{2,2,1,2,1,false,true});
    expect_mask(p.mu.data()+ap::kStride,raw,.625f,0,0);
    // Zero remaining proposal budget has NO attempt, regardless of stale bytes.
    Fixture poison;
    need(poison.call(p,0,0,{},false,1),"capacity1 zero producer budget");
    packet(p,{1,0,0,0,-1,false,false});
    need(std::all_of(p.mu.begin(),p.mu.end(),[](int32_t word){return word==0;}),"zero-budget packet retained old mu");
    need(!last_attempt.call(p,1,0,{},false,1),"actual generated row cannot be relabeled zero budget");
    // If a caller has already materialized a failed row at the final target
    // capacity, it must preserve NULL or explicitly reject this packet. Passing
    // min(generated,room-1) here would erase evidence and evade this contract.
    Fixture edge_failed;edge_failed.set(0,raw,B,.25f);
    const bool edge_ok=edge_failed.call(p,1,1,{},false,1);
    if(edge_ok)packet(p,{1,1,0,1,0,false,true});
    else need(p.error!=nullptr,"capacity failure must be explicit, never bonus");
    // A subsequent successful call must clear every stale error/count/mode field.
    need(one.call(p,1,1),"packet reset after errors");packet(p,{2,1,1,1,-1,false,false});
}
} // namespace
int main() {
    try {tests();std::printf("{\"summary\":\"gated_proposal_test\",\"checks\":%d,\"passed\":true,\"GPU_executed\":false,\"actual_main_window_copy_tested\":false}\n",checks);return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"gated_proposal_test FAILED after %d checks: %s\n",checks,e.what());return 1;}
}
