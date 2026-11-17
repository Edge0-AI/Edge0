#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "../../include/strata/kernels/bf16_bits.hpp"
#include <hip/hip_version.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {
#define HIP_CHECK(call) do { const hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e)); std::exit(2); } } while (0)
#define BLAS_CHECK(call) do { const hipblasStatus_t e = (call); if (e != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLAS %s:%d: %s: status=%d\n", __FILE__, __LINE__, #call, (int)e); std::exit(2); } } while (0)
#define LT_CHECK(call) do { const hipblasStatus_t e = (call); if (e != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLASLt %s:%d: %s: status=%d\n", __FILE__, __LINE__, #call, (int)e); std::exit(2); } } while (0)

struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};
struct Events {
    hipEvent_t a = nullptr, b = nullptr;
    Events() { HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b)); }
    ~Events() { if (a) (void)hipEventDestroy(a); if (b) (void)hipEventDestroy(b); }
};

struct Shape { int t, n, k, ldy; bool bf16; };
struct Options {
    size_t workspace = 32U * 1024U * 1024U;
    std::vector<Shape> shapes;
    std::vector<int> tokens{4096, 8192};
    std::string shapes_file, tuning_out;
    bool explicit_shapes = false;
};
struct Err { double rel_l2, max_abs; bool finite; size_t padding_writes; };
struct Best {
    Shape s;
    int heuristic_index, solution_id;
    size_t required_workspace, algo_workspace;
    float mean_ms;
    Err error;
    std::string solution, kernel, config;
};

constexpr int WARMUPS = 2;
constexpr int REPS = 3;
constexpr int MAX_ALGOS = 16;
constexpr double REL_L2_TOL = 1e-4;
constexpr double MAX_ABS_TOL = 1e-2;
constexpr float PADDING_CANARY = 123456.25f;

const char *dtype(bool bf16) { return bf16 ? "bf16" : "f16"; }
const char *status_name(hipblasStatus_t s) {
    switch (s) {
        case HIPBLAS_STATUS_SUCCESS: return "success";
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "not_supported";
        case HIPBLAS_STATUS_INVALID_VALUE: return "invalid_value";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "arch_mismatch";
        default: return "other";
    }
}
std::string json(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (unsigned)c << std::dec;
        else o << (char)c;
    }
    o << '"'; return o.str();
}
std::string config_hex(const hipblasLtMatmulAlgo_t &algo) {
    std::ostringstream o; o << std::hex << std::setfill('0');
    for (uint8_t b : algo.data) o << std::setw(2) << (unsigned)b;
    return o.str();
}
uint16_t input_bits(float x, bool bf16) {
    if (bf16) return strata::kernels::bf16_from_f32(x);
    const __half half = __float2half_rn(x);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}
void fill(std::vector<uint16_t> &v, uint32_t seed, bool bf16) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : v) x = input_bits(dist(rng), bf16);
}
float time_call(hipStream_t stream, const std::function<void()> &f) {
    Events ev;
    HIP_CHECK(hipEventRecord(ev.a, stream));
    f();
    HIP_CHECK(hipEventRecord(ev.b, stream));
    HIP_CHECK(hipEventSynchronize(ev.b));
    HIP_CHECK(hipDeviceSynchronize());
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, ev.a, ev.b));
    return ms;
}
Err compare(const std::vector<float> &ref, const std::vector<float> &got, int n, int t, int ld) {
    long double d2 = 0, r2 = 0; double max_abs = 0; bool finite = true;
    for (int col = 0; col < t; ++col) for (int row = 0; row < n; ++row) {
        const size_t i = (size_t)col * ld + row;
        const double r = ref[i], g = got[i];
        if (!std::isfinite(r) || !std::isfinite(g)) { finite = false; continue; }
        const double d = g - r;
        if (!std::isfinite(d)) { finite = false; continue; }
        d2 += (long double)d * d; r2 += (long double)r * r;
        max_abs = std::max(max_abs, std::abs(d));
    }
    if (!finite) return {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(), false, 0};
    return {std::sqrt((double)(d2 / std::max(r2, 1e-300L))), max_abs, true, 0};
}
size_t padding_writes(const std::vector<float> &got, int n, int t, int ld) {
    size_t writes = 0;
    for (int col = 0; col < t; ++col) for (int row = n; row < ld; ++row)
        if (got[(size_t)col * ld + row] != PADDING_CANARY) ++writes;
    return writes;
}
bool accuracy_ok(const Err &e) {
    return e.finite && e.padding_writes == 0 &&
           e.rel_l2 <= REL_L2_TOL && e.max_abs <= MAX_ABS_TOL;
}
std::vector<std::string> csv(const std::string &s) {
    std::vector<std::string> out; size_t b = 0;
    for (;;) {
        const size_t p = s.find(',', b);
        out.push_back(s.substr(b, p == std::string::npos ? p : p - b));
        if (p == std::string::npos) return out;
        b = p + 1;
    }
}
int positive(const std::string &s, const char *label) {
    char *e = nullptr; const long v = std::strtol(s.c_str(), &e, 10);
    if (s.empty() || !e || *e || v < 1 || v > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "invalid %s: %s\n", label, s.c_str()); std::exit(2);
    }
    return (int)v;
}
void add(Options &o, bool bf16, int t, int n, int k, int ldy) {
    if (ldy < n) { std::fprintf(stderr, "ldy must be >= N\n"); std::exit(2); }
    o.shapes.push_back({t,n,k,ldy,bf16});
}
void add_shape(Options &o, const std::string &v) {
    const auto f = csv(v);
    if (f.size() != 3) { std::fprintf(stderr, "--shape expects T,N,K\n"); std::exit(2); }
    int t=positive(f[0],"T"), n=positive(f[1],"N"), k=positive(f[2],"K");
    add(o,false,t,n,k,n); add(o,true,t,n,k,n); o.explicit_shapes=true;
}
void add_case(Options &o, const std::string &v) {
    const auto f = csv(v);
    if (f.size()!=5 || (f[0]!="f16" && f[0]!="bf16")) {
        std::fprintf(stderr, "--case expects dtype,T,N,K,ldy\n"); std::exit(2);
    }
    add(o,f[0]=="bf16",positive(f[1],"T"),positive(f[2],"N"),positive(f[3],"K"),positive(f[4],"ldy"));
    o.explicit_shapes=true;
}
std::vector<int> parse_tokens(const std::string &s) {
    std::vector<int> out;
    for (const auto &x : csv(s)) out.push_back(positive(x,"token bucket"));
    return out;
}
void load_shapes(Options &o) {
    std::ifstream f(o.shapes_file);
    if (!f) { std::fprintf(stderr, "cannot open shapes file: %s\n",o.shapes_file.c_str()); std::exit(2); }
    const std::string text((std::istreambuf_iterator<char>(f)),{});
    const std::regex obj(R"(\{[^{}]*\})"), dtype_re(R"re("dtype"\s*:\s*"(f16|bf16)")re"),
        n_re(R"("N"\s*:\s*([0-9]+))"), k_re(R"("K"\s*:\s*([0-9]+))"),
        ld_re(R"("ldy"\s*:\s*([0-9]+))");
    int count=0;
    for (auto i=std::sregex_iterator(text.begin(),text.end(),obj); i!=std::sregex_iterator(); ++i) {
        const std::string item=i->str(); std::smatch d,n,k,ld;
        if (!std::regex_search(item,d,dtype_re) || !std::regex_search(item,n,n_re) ||
            !std::regex_search(item,k,k_re) || !std::regex_search(item,ld,ld_re)) continue;
        for (int t:o.tokens) add(o,d[1]=="bf16",t,positive(n[1],"N"),positive(k[1],"K"),positive(ld[1],"ldy"));
        ++count;
    }
    if (!count) { std::fprintf(stderr,"no dtype,N,K,ldy entries in %s\n",o.shapes_file.c_str()); std::exit(2); }
    o.explicit_shapes=true;
}
void usage(const char *p) {
    std::printf("Usage: %s [--workspace-mib N] [--shape T,N,K]... [--case dtype,T,N,K,ldy]...\n",p);
    std::printf("       [--shapes-file PATH [--tokens T1,T2,...]] [--tuning-out PATH]\n");
    std::printf("Defaults: original three shapes, f16+bf16, T=8192, workspace=32 MiB.\n");
}
Options options(int argc,char **argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto next=[&]() -> std::string { if(i+1>=argc){std::fprintf(stderr,"missing option value\n");std::exit(2);} return argv[++i]; };
        if(a=="-h"||a=="--help"){usage(argv[0]);std::exit(0);}
        else if(a=="--workspace-mib"){
            std::string v=next(); char *e=nullptr; unsigned long long x=std::strtoull(v.c_str(),&e,10);
            if(v.empty()||!e||*e||x>std::numeric_limits<size_t>::max()/(1024ULL*1024ULL)){std::fprintf(stderr,"bad workspace MiB\n");std::exit(2);}
            o.workspace=(size_t)x*1024U*1024U;
        } else if(a=="--shape") add_shape(o,next());
        else if(a=="--case") add_case(o,next());
        else if(a=="--shapes-file"){o.shapes_file=next();o.explicit_shapes=true;}
        else if(a=="--tokens") o.tokens=parse_tokens(next());
        else if(a=="--tuning-out") o.tuning_out=next();
        else {std::fprintf(stderr,"unknown option: %s\n",a.c_str());usage(argv[0]);std::exit(2);}
    }
    if(!o.shapes_file.empty()) load_shapes(o);
    if(!o.explicit_shapes) {
        struct DefaultShape { int n,k,ldy; bool bf16; };
        const DefaultShape defs[]={{10240,2560,10240,false},{320,10240,320,false},{2560,320,2560,false},
                                {10240,2560,10240,true},{320,10240,320,true},{2560,320,2560,true}};
        for(const auto &s:defs) add(o,s.bf16,8192,s.n,s.k,s.ldy);
    }
    if(o.shapes.empty()){std::fprintf(stderr,"no shapes requested\n");std::exit(2);}
    return o;
}
std::string arch_name(const char *s) {
    std::string a=s?s:"unknown"; size_t p=a.find(':'); if(p!=std::string::npos)a.resize(p); return a;
}
void emit_candidate(const Shape &s,size_t workspace,int h,int id,size_t req,size_t algows,float ms,const Err &error,
                    const std::string &arch,int version,const std::string &sol,const std::string &kernel,const std::string &config) {
    std::cout<<"candidate_json={\"device_arch\":"<<json(arch)<<",\"hipblaslt_version\":"<<version
      <<",\"dtype\":"<<json(dtype(s.bf16))<<",\"T\":"<<s.t<<",\"N\":"<<s.n<<",\"K\":"<<s.k<<",\"ldy\":"<<s.ldy
      <<",\"workspace_limit_bytes\":"<<workspace<<",\"heuristic_index\":"<<h<<",\"solution_id\":"<<id
      <<",\"required_workspace_bytes\":"<<req<<",\"algo_max_workspace_bytes\":"<<algows
      <<",\"finite\":"<<(error.finite?"true":"false")<<",\"padding_writes\":"<<error.padding_writes
      <<",\"relative_l2\":"<<std::setprecision(12)<<error.rel_l2<<",\"max_abs\":"<<error.max_abs
      <<",\"relative_l2_tolerance\":"<<REL_L2_TOL<<",\"max_abs_tolerance\":"<<MAX_ABS_TOL
      <<",\"mean_ms\":"<<std::setprecision(9)<<ms<<",\"solution_name\":"<<json(sol)
      <<",\"kernel_name\":"<<json(kernel)<<",\"algo_config_hex\":"<<json(config)<<"}\n";
}
void emit_best(const Best &b,size_t workspace,const std::string &arch,int version,int hipver) {
    const auto&s=b.s;
    std::cout<<"best_json={\"device_arch\":"<<json(arch)<<",\"hipblaslt_version\":"<<version
      <<",\"hip_runtime_version\":"<<hipver<<",\"dtype\":"<<json(dtype(s.bf16))
      <<",\"T\":"<<s.t<<",\"N\":"<<s.n<<",\"K\":"<<s.k<<",\"ldy\":"<<s.ldy
      <<",\"workspace_limit_bytes\":"<<workspace<<",\"heuristic_index\":"<<b.heuristic_index
      <<",\"solution_id\":"<<b.solution_id<<",\"required_workspace_bytes\":"<<b.required_workspace
      <<",\"algo_max_workspace_bytes\":"<<b.algo_workspace<<",\"mean_ms\":"<<std::setprecision(9)<<b.mean_ms
      <<",\"finite\":"<<(b.error.finite?"true":"false")<<",\"padding_writes\":"<<b.error.padding_writes
      <<",\"relative_l2\":"<<std::setprecision(12)<<b.error.rel_l2<<",\"max_abs\":"<<b.error.max_abs
      <<",\"relative_l2_tolerance\":"<<REL_L2_TOL<<",\"max_abs_tolerance\":"<<MAX_ABS_TOL
      <<",\"solution_name\":"<<json(b.solution)<<",\"kernel_name\":"<<json(b.kernel)
      <<",\"algo_config_hex\":"<<json(b.config)<<"}\n";
}

void run_case(hipblasHandle_t blas,hipblasLtHandle_t lt,const Shape&s,int case_id,size_t ws,hipStream_t stream,
              const std::string&arch,int version,int hipver,std::vector<Best>&bests) {
    const size_t na=(size_t)s.k*s.n, nb=(size_t)s.k*s.t, out_elems=(size_t)s.ldy*s.t;
    const hipDataType it=s.bf16?HIP_R_16BF:HIP_R_16F;
    const hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    const int m=s.n,n=s.t,k=s.k,lda=s.k,ldb=s.k,ldc=s.ldy;
    const float alpha=1.f,beta=0.f;
    std::vector<uint16_t> ha(na),hb(nb);
    uint32_t seed=0x5A17C3U+(uint32_t)case_id*0x10001U+(s.bf16?0xB16U:0xF16U);
    fill(ha,seed,s.bf16);fill(hb,seed^0x9E3779B9U,s.bf16);
    Buffer da(na*sizeof(uint16_t)),db(nb*sizeof(uint16_t)),dc(out_elems*sizeof(float)),dy(out_elems*sizeof(float)),dws(ws);
    const std::vector<float> canary(out_elems,PADDING_CANARY);
    HIP_CHECK(hipMemcpyAsync(dc.p,canary.data(),out_elems*sizeof(float),hipMemcpyHostToDevice,stream));
    HIP_CHECK(hipMemcpyAsync(da.p,ha.data(),na*sizeof(uint16_t),hipMemcpyHostToDevice,stream));
    HIP_CHECK(hipMemcpyAsync(db.p,hb.data(),nb*sizeof(uint16_t),hipMemcpyHostToDevice,stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    hipblasLtMatmulDesc_t op=nullptr;
    hipblasLtMatrixLayout_t ad=nullptr,bd=nullptr,cd=nullptr;
    hipblasLtMatmulPreference_t pref=nullptr;
    LT_CHECK(hipblasLtMatmulDescCreate(&op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&ad,it,s.k,s.n,lda));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&bd,it,s.k,s.t,ldb));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&cd,HIP_R_32F,s.n,s.t,ldc));
    LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    const uint64_t limit=ws;
    LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));

    BLAS_CHECK(hipblasSetStream(blas,stream));
    auto base=[&]{BLAS_CHECK(hipblasGemmEx(blas,ta,tb,m,n,k,&alpha,da.p,it,lda,db.p,it,ldb,&beta,dc.p,HIP_R_32F,ldc,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));};
    for(int i=0;i<WARMUPS;++i){base();HIP_CHECK(hipDeviceSynchronize());}
    float base_ms[REPS]{};
    for(int i=0;i<REPS;++i)base_ms[i]=time_call(stream,base);
    std::vector<float> ref(out_elems);
    HIP_CHECK(hipMemcpyAsync(ref.data(),dc.p,out_elems*sizeof(float),hipMemcpyDeviceToHost,stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const size_t baseline_padding=padding_writes(ref,s.n,s.t,s.ldy);
    std::printf("baseline dtype=%s T=%d N=%d K=%d ldy=%d padding_writes=%zu\n",dtype(s.bf16),s.t,s.n,s.k,s.ldy,baseline_padding);
    float base_mean=(base_ms[0]+base_ms[1]+base_ms[2])/REPS;

    hipblasLtMatmulHeuristicResult_t hs[MAX_ALGOS]{};
    int count=0;
    const hipblasStatus_t query=hipblasLtMatmulAlgoGetHeuristic(lt,op,ad,bd,cd,cd,pref,MAX_ALGOS,hs,&count);
    std::printf("case dtype=%s T=%d N=%d K=%d ldy=%d workspace_limit_bytes=%zu baseline=hipblasGemmEx rep1_ms=%.4f rep2_ms=%.4f rep3_ms=%.4f mean_ms=%.4f\n",
      dtype(s.bf16),s.t,s.n,s.k,s.ldy,ws,base_ms[0],base_ms[1],base_ms[2],base_mean);
    if(query!=HIPBLAS_STATUS_SUCCESS||count==0){
        std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d status=%s(%d) heuristic_count=%d\n",
          dtype(s.bf16),s.t,s.n,s.k,s.ldy,status_name(query),(int)query,count);
    } else {
        int best_h=-1,best_id=-1; float best_ms=std::numeric_limits<float>::infinity();
        size_t best_req=0,best_maxws=0; std::string best_sol,best_kernel,best_config;
        Err best_error{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),false,0};
        for(int h=0;h<count;++h){
            if(hs[h].state!=HIPBLAS_STATUS_SUCCESS||hs[h].workspaceSize>ws){
                std::printf("lt_skip dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d status=%s(%d) required_workspace_bytes=%zu\n",
                  dtype(s.bf16),s.t,s.n,s.k,s.ldy,h,status_name(hs[h].state),(int)hs[h].state,hs[h].workspaceSize);
                continue;
            }
            auto algo=hs[h].algo;
            const int id=hipblaslt_ext::getIndexFromAlgo(algo);
            const std::string sol=hipblaslt_ext::getSolutionNameFromAlgo(lt,algo);
            const std::string kernel=hipblaslt_ext::getKernelNameFromAlgo(lt,algo);
            const std::string config=config_hex(algo);
            auto run=[&]{LT_CHECK(hipblasLtMatmul(lt,op,&alpha,da.p,ad,db.p,bd,&beta,dy.p,cd,dy.p,cd,&algo,dws.p,ws,stream));};

            HIP_CHECK(hipMemcpyAsync(dy.p,canary.data(),out_elems*sizeof(float),hipMemcpyHostToDevice,stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            const hipblasStatus_t check_status=hipblasLtMatmul(lt,op,&alpha,da.p,ad,db.p,bd,&beta,dy.p,cd,dy.p,cd,&algo,dws.p,ws,stream);
            if(check_status!=HIPBLAS_STATUS_SUCCESS){
                std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d status=%s(%d) phase=accuracy_check\n",
                  dtype(s.bf16),s.t,s.n,s.k,s.ldy,h,id,status_name(check_status),(int)check_status);continue;
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            std::vector<float> checked_y(out_elems);
            HIP_CHECK(hipMemcpyAsync(checked_y.data(),dy.p,out_elems*sizeof(float),hipMemcpyDeviceToHost,stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            Err error=compare(ref,checked_y,s.n,s.t,s.ldy);
            error.padding_writes=padding_writes(checked_y,s.n,s.t,s.ldy);
            if(!accuracy_ok(error)){
                std::printf("lt_reject dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d finite=%s relative_l2=%.12g relative_l2_tolerance=%.3g max_abs=%.12g max_abs_tolerance=%.3g padding_writes=%zu reason=accuracy_gate\n",
                  dtype(s.bf16),s.t,s.n,s.k,s.ldy,h,id,error.finite?"true":"false",
                  error.rel_l2,REL_L2_TOL,error.max_abs,MAX_ABS_TOL,error.padding_writes);continue;
            }

            bool failed=false;
            for(int i=0;i<WARMUPS;++i){
                const hipblasStatus_t st=hipblasLtMatmul(lt,op,&alpha,da.p,ad,db.p,bd,&beta,dy.p,cd,dy.p,cd,&algo,dws.p,ws,stream);
                if(st!=HIPBLAS_STATUS_SUCCESS){
                    std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d status=%s(%d)\n",
                      dtype(s.bf16),s.t,s.n,s.k,s.ldy,h,id,status_name(st),(int)st);failed=true;break;
                }
                HIP_CHECK(hipDeviceSynchronize());
            }
            if(failed)continue;
            float ms[REPS]{};
            for(int i=0;i<REPS;++i)ms[i]=time_call(stream,run);
            const float mean=(ms[0]+ms[1]+ms[2])/REPS;
            std::printf("lt dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d solution=%s kernel=%s algo_config_hex=%s required_workspace_bytes=%zu algo_max_workspace_bytes=%zu finite=%s relative_l2=%.12g max_abs=%.12g padding_writes=%zu rep1_ms=%.4f rep2_ms=%.4f rep3_ms=%.4f mean_ms=%.4f\n",
              dtype(s.bf16),s.t,s.n,s.k,s.ldy,h,id,sol.c_str(),kernel.c_str(),config.c_str(),hs[h].workspaceSize,algo.max_workspace_bytes,
              error.finite?"true":"false",error.rel_l2,error.max_abs,error.padding_writes,ms[0],ms[1],ms[2],mean);
            emit_candidate(s,ws,h,id,hs[h].workspaceSize,algo.max_workspace_bytes,mean,error,arch,version,sol,kernel,config);
            if(mean<best_ms){
                best_h=h;best_id=id;best_ms=mean;best_req=hs[h].workspaceSize;best_maxws=algo.max_workspace_bytes;
                best_sol=sol;best_kernel=kernel;best_config=config;best_error=error;
            }
        }
        if(best_h<0)std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d status=no_valid_heuristics heuristic_count=%d\n",
                                dtype(s.bf16),s.t,s.n,s.k,s.ldy,count);
        else{
            Best b{s,best_h,best_id,best_req,best_maxws,best_ms,best_error,best_sol,best_kernel,best_config};
            bests.push_back(b);
            std::printf("lt_best dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d solution=%s kernel=%s algo_config_hex=%s required_workspace_bytes=%zu algo_max_workspace_bytes=%zu mean_ms=%.4f speedup_vs_blas=%.4fx finite=%s relative_l2=%.9g max_abs=%.9g padding_writes=%zu\n",
              dtype(s.bf16),s.t,s.n,s.k,s.ldy,best_h,best_id,best_sol.c_str(),best_kernel.c_str(),best_config.c_str(),best_req,best_maxws,best_ms,base_mean/best_ms,
              best_error.finite?"true":"false",best_error.rel_l2,best_error.max_abs,best_error.padding_writes);
            emit_best(b,ws,arch,version,hipver);
        }
    }
    (void)hipblasLtMatmulPreferenceDestroy(pref);
    (void)hipblasLtMatrixLayoutDestroy(cd);(void)hipblasLtMatrixLayoutDestroy(bd);(void)hipblasLtMatrixLayoutDestroy(ad);
    (void)hipblasLtMatmulDescDestroy(op);
}
void tuning_file(const std::string &path,const std::string &arch,int version,const std::vector<Best>&rows){
    if(path.empty())return;
    std::ofstream f(path);
    if(!f){std::fprintf(stderr,"cannot write tuning output: %s\n",path.c_str());std::exit(2);}
    f<<"# solution IDs are scoped to this hipBLASLt version and device architecture\n";
    f<<"STRATA_HIPBLASLT_TUNING_V1 "<<arch<<" "<<version<<"\n";
    for(const auto &b:rows)f<<dtype(b.s.bf16)<<" "<<b.s.n<<" "<<b.s.k<<" "<<b.s.ldy<<" "<<b.s.t<<" "<<b.solution_id<<"\n";
}
} // namespace

int main(int argc,char **argv){
    Options o=options(argc,argv);
    hipblasHandle_t blas=nullptr;hipblasLtHandle_t lt=nullptr;
    BLAS_CHECK(hipblasCreate(&blas));LT_CHECK(hipblasLtCreate(&lt));
    int device=0,hipver=0,version=0;hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&device));HIP_CHECK(hipGetDeviceProperties(&prop,device));HIP_CHECK(hipRuntimeGetVersion(&hipver));
    const hipblasStatus_t vs=hipblasLtGetVersion(lt,&version);
    if(vs!=HIPBLAS_STATUS_SUCCESS){std::fprintf(stderr,"hipblasLtGetVersion failed: %d\n",(int)vs);return 2;}
    const std::string arch=arch_name(prop.gcnArchName);
    std::printf("meta device_arch=%s hipblaslt_version=%d hipblaslt_header_version=%d.%d.%d hip_runtime_version=%d\n",
      arch.c_str(),version,HIPBLASLT_VERSION_MAJOR,HIPBLASLT_VERSION_MINOR,HIPBLASLT_VERSION_PATCH,hipver);
    hipStream_t stream=nullptr;HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::vector<Best>bests;
    for(size_t i=0;i<o.shapes.size();++i)run_case(blas,lt,o.shapes[i],(int)i,o.workspace,stream,arch,version,hipver,bests);
    tuning_file(o.tuning_out,arch,version,bests);
    HIP_CHECK(hipStreamSynchronize(stream));HIP_CHECK(hipStreamDestroy(stream));
    LT_CHECK(hipblasLtDestroy(lt));BLAS_CHECK(hipblasDestroy(blas));
    return 0;
}
