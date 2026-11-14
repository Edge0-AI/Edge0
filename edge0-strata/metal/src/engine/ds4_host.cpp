// metal/src/engine/ds4_host.cpp - host side of the ported dsv41 kernels: compile the MSL once, wrap the engine's own buffers,
// dispatch.  First entry point: the Q8 activation quantiser (ds4_gguf.cu:201).
//
// The two runtimes meet on pointers: strata::gpu owns the engine's device memory (a device pointer is its MTLBuffer's
// gpuAddress) and exposes host_view(device_pointer); mtl::Context::wrap() adopts that address without copying and a
// Dispatch names it through Arg::B().  See RIGS/m1002.md for the derivation.
#include "mtl_rt.h"

#include <strata/gpu/runtime.hpp>



#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <strata/gpu/runtime.hpp>
#include <vector>

namespace strata::dsv41::metal_host {
mtl::Context& finish_cx();   // ds4_finish.cpp's Context: every kernel that orders with the chunk path's dispatches uses it
int shared_lib();   // from ds4_finish.cpp: the library the shared Context compiled


namespace {
mtl::Context* g_cx = nullptr;
int g_lib = -1;
bool g_tried = false;

bool ensure(mtl::Context& cx) {
    if (g_cx == &cx && g_lib >= 0) return true;
    if (g_tried && g_lib < 0) return false;
    g_tried = true;
    // the kernel lives in its own file and is self-contained (metal_stdlib + ggml_grids.h)
    g_lib = cx.compile({"../src/engine/msl/ds4_quant.metal"}, "", false);
    if (g_lib < 0) {
        std::fprintf(stderr, "dsv41 metal: compiling ds4_quant.metal failed\n");
        return false;
    }
    g_cx = &cx;
    return true;
}
}  // namespace

// q8(x, ldx, n) -> q (int8, n bytes) and d (half, n/32); ldx is the stride in values between blocks of 32
bool q8_quant(mtl::Context& cx, const float* x, int ldx, int n, void* q, void* d) {
    if (!ensure(cx)) return false;
    const size_t xb = (size_t) ldx * (size_t) (n / 32) * sizeof(float), qb = (size_t) n, db = (size_t) (n / 32) * 2;
    mtl::Buffer bx = cx.wrap(strata::gpu::host_view(x), xb);
    mtl::Buffer bq = cx.wrap(q, qb);
    mtl::Buffer bd = cx.wrap(d, db);
    mtl::Pipeline p;
    try {
        p = cx.pipeline(g_lib, "ds4_q8_quant");
    } catch (...) {
        return false;
    }
    mtl::Dispatch ds;
    ds.p = &p;
    ds.args = {mtl::Arg::B(bx), mtl::Arg::V((unsigned) ldx), mtl::Arg::B(bq), mtl::Arg::B(bd)};
    ds.grid[0] = (uint32_t) (n / 32);
    ds.grid[1] = ds.grid[2] = 1;
    ds.tg[0] = 32;
    ds.tg[1] = ds.tg[2] = 1;
    mtl::require_fit(p, 32, "ds4_q8_quant");
    cx.run({ds});
    return true;
}

}  // namespace strata::dsv41::metal_host

namespace strata::dsv41::metal_host {
namespace {
mtl::Context* g_init_cx = nullptr;
int g_lib_init = -1;

mtl::Context& shared_cx() {
    if (!g_init_cx) {
        g_init_cx = new mtl::Context();
        g_lib_init = g_init_cx->compile({"../src/engine/msl/ds4_init.metal"}, "", false);
        if (g_lib_init < 0) std::fprintf(stderr, "dsv41 metal: compiling ds4_init.metal failed\n");
    }
    return *g_init_cx;
}
}  // namespace

// ds4_kernels.cu:1477 - pre[i] = (i % 8) == 0 ? 1 : 0 for i < T*8
bool init_pre(float* pre, int T, void*) {
    // cpu path (T*8 floats of a fixed pattern): the GPU route had to wrap a whole page to reach 32 bytes, and that aliasing
    // buffer overlapped the pages holding h
    void* host = strata::gpu::host_view(pre);
    if (!host) { std::fprintf(stderr, "dsv41 metal: init_pre has no host view\n"); return false; }
    float* p = (float*) host;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 8; ++i) p[t * 8 + i] = (i == 0) ? 1.0f : 0.0f;
    return true;
}


// ds4_kernels.cu:130 - the BF16 embedding gather, hc copies a token
bool embed(const uint16_t* rows, int T, float* h, int dim, int hc, void*) {
    strata::dsv41::metal_host::shared_cx();
    if (g_lib_init < 0) { std::fprintf(stderr, "dsv41 metal: embed: no library\n"); return false; }
    // host_view() returns a different address from the device pointer on this runtime, so wrapping it wrote into a mapping the
    // engine never reads.  On Apple silicon a device pointer is a valid CPU address for shared storage: wrap that instead.
    // DS4_HOSTVIEW=1 restores the old behaviour.
    const bool use_hv = std::getenv("DS4_HOSTVIEW") != nullptr;
    void* hrows = use_hv ? strata::gpu::host_view((void*) rows) : (void*) rows;
    void* hh = use_hv ? strata::gpu::host_view(h) : (void*) h;
    if (!hrows || !hh) { std::fprintf(stderr, "dsv41 metal: embed has no host view\n"); return false; }
    if (std::getenv("DS4_DBG_HV"))
        std::fprintf(stderr, "DBG hostview: rows %p -> %p   h %p -> %p\n",
                     (void*) rows, hrows, (void*) h, hh);
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        return strata::dsv41::metal_host::shared_cx().wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    mtl::Buffer br = wrap(hrows, (size_t) T * dim * 2);
    mtl::Buffer bh = wrap(hh, (size_t) T * hc * dim * 4);
    if (!br.handle || !bh.handle) { std::fprintf(stderr, "dsv41 metal: embed wrap failed\n"); return false; }
    const size_t dro = (size_t) (((uintptr_t) hrows) - (((uintptr_t) hrows) & ~(uintptr_t) (page - 1)));
    const size_t dho = (size_t) (((uintptr_t) hh) - (((uintptr_t) hh) & ~(uintptr_t) (page - 1)));
    mtl::Pipeline pl;
    try { pl = strata::dsv41::metal_host::shared_cx().pipeline(g_lib_init, "ds4_embed"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(br, dro), mtl::Arg::B(bh, dho), mtl::Arg::V((unsigned) dim), mtl::Arg::V((unsigned) hc)};
    d.grid[0] = (uint32_t) ((dim + 255) / 256);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    strata::dsv41::metal_host::shared_cx().run({d});
    if (std::getenv("DS4_DBG_EMBWRITE")) {
        strata::dsv41::metal_host::shared_cx().flush_runs();
        const float* f0 = (const float*) hh;
        const uint16_t* r16 = (const uint16_t*) hrows;
        std::fprintf(stderr, "DBG embwrite: h=%p rows=%p h_off=%zu rows_off=%zu\n", hh, hrows, dho, dro);
        std::fprintf(stderr, "DBG embwrite: rows[0..3]=%04x %04x %04x %04x  h[0..3]=%g %g %g %g\n",
                     (unsigned) r16[0], (unsigned) r16[1], (unsigned) r16[2], (unsigned) r16[3],
                     f0[0], f0[1], f0[2], f0[3]);
    }
    return true;
}


// ds4_model.cu:2737 - C[t][m] = sum_k A[m][k] * B[t][k], F32 operands, alpha = 1, beta = 0
bool sgemm_tn(const float* A, const float* B, float* C, int m, int nt, int k, int lda, int ldb, int ldc) {
    finish_cx();
    const int lib = shared_lib();
    if (lib < 0) return false;
    void* ha = strata::gpu::host_view((void*) A);
    void* hb = strata::gpu::host_view((void*) B);
    void* hc = strata::gpu::host_view((void*) C);
    if (!ha || !hb || !hc) { std::fprintf(stderr, "dsv41 metal: sgemm_tn has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        return finish_cx().wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer ba = wrap(ha, (size_t) m * lda * 4);
    mtl::Buffer bb = wrap(hb, (size_t) nt * ldb * 4);
    mtl::Buffer bc = wrap(hc, (size_t) nt * ldc * 4);
    if (!ba.handle || !bb.handle || !bc.handle) return false;
    mtl::Pipeline pl;
    try { pl = finish_cx().pipeline(lib, "ds4_sgemm_tn"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(ba, off(ha)), mtl::Arg::B(bb, off(hb)), mtl::Arg::B(bc, off(hc)),
              mtl::Arg::V((unsigned) m), mtl::Arg::V((unsigned) nt), mtl::Arg::V((unsigned) k),
              mtl::Arg::V((unsigned) lda), mtl::Arg::V((unsigned) ldb), mtl::Arg::V((unsigned) ldc)};
    const uint64_t total = (uint64_t) m * (uint64_t) nt;
    d.grid[0] = (uint32_t) ((total + 127) / 128);
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 128; d.tg[1] = d.tg[2] = 1;
    finish_cx().run({d});
    if (std::getenv("DS4_DBG_SG")) {
        finish_cx().flush_runs();
        const float* fa = (const float*) ha; const float* fb = (const float*) hb; const float* fc = (const float*) hc;
        std::fprintf(stderr, "sgemm m %d nt %d k %d lda %d ldb %d ldc %d A %g %g B %g %g C %g %g %g\n", m, nt, k, lda, ldb, ldc, fa[0], fa[1], fb[0], fb[1], fc[0], fc[1], fc[2]);
    }
    return true;
}


// embed through raw device addresses: the struct carries the gpuAddress of rows and h, which the runtime's residency set keeps
// resident, so the kernel writes the engine's own memory instead of a CPU mirror.
struct Ds4EmbedArgsHost {
    uint64_t rows;
    uint64_t h;
    uint32_t dim;
    uint32_t hc;
};

bool embed_addr(const uint16_t* rows, int T, float* h, int dim, int hc) {
    // the engine uploads emb_rows on ITS stream; this kernel runs on mtl::Context queue.  Order the two before reading.
    strata::gpu::stream_synchronize(nullptr);
    if (std::getenv("DS4_DBG_ROWS")) {   // what is the engine actually handing us as the embedding rows?
        const uint16_t* rv = (const uint16_t*) strata::gpu::host_view((void*) rows);
        std::fprintf(stderr, "DBG rows_in: ptr=%p view=%p first=%04x %04x %04x %04x\n",
                     (const void*) rows, (const void*) rv,
                     rv ? (unsigned) rv[0] : 0u, rv ? (unsigned) rv[1] : 0u,
                     rv ? (unsigned) rv[2] : 0u, rv ? (unsigned) rv[3] : 0u);
    }
    strata::dsv41::metal_host::shared_cx();
    const int lib = shared_lib();
    if (lib < 0) return false;
    Ds4EmbedArgsHost a{};
    a.rows = (uint64_t) (uintptr_t) rows;
    a.h = (uint64_t) (uintptr_t) h;
    a.dim = (uint32_t) dim;
    a.hc = (uint32_t) hc;
    mtl::Pipeline pl;
    try { pl = strata::dsv41::metal_host::shared_cx().pipeline(lib, "ds4_embed_j"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::V(a)};
    d.grid[0] = (uint32_t) ((dim + 255) / 256);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    strata::dsv41::metal_host::shared_cx().run({d});
    strata::dsv41::metal_host::shared_cx().flush_runs();
    if (std::getenv("DS4_DBG_CMP")) {
        const float* hv = (const float*) strata::gpu::host_view(h);
        {
            const size_t nv = (size_t) T * hc * dim;
            size_t nf = 0, nz = 0;
            for (size_t i = 0; i < nv; ++i) { const float v = hv[i]; if (!(v == v)) ++nf; else if (v == 0.0f) ++nz; }
            std::fprintf(stderr, "DBG cmp: after embed n=%.0f nonfinite=%zu zeros=%zu\n", (double) nv, nf, nz);
        }
        std::fprintf(stderr, "DBG cmp: view %g %g %g | h=%g h+1=%g\n",
                     hv ? hv[0] : -1.0f, hv ? hv[1] : -1.0f, hv ? hv[2] : -1.0f,
                     hv ? hv[(size_t) dim * hc] : -1.0f, hv ? hv[(size_t) dim * hc + 1] : -1.0f);
    }
    if (std::getenv("DS4_DBG_ARGS")) {
        mtl::Buffer bp = strata::dsv41::metal_host::shared_cx().wrap(
            (void*) (((uintptr_t) strata::gpu::host_view(h)) & ~(uintptr_t) 16383), 16384);
        mtl::Pipeline pp = strata::dsv41::metal_host::shared_cx().pipeline(lib, "ds4_args_probe");
        mtl::Dispatch pd;
        pd.p = &pp;
        pd.args = {mtl::Arg::B(bp, ((uintptr_t) strata::gpu::host_view(h)) & 16383),
                   mtl::Arg::V((unsigned) dim), mtl::Arg::V((unsigned) hc)};
        pd.grid[0] = 1; pd.grid[1] = pd.grid[2] = 1; pd.tg[0] = 32; pd.tg[1] = pd.tg[2] = 1;
        strata::dsv41::metal_host::shared_cx().run({pd});
        strata::dsv41::metal_host::shared_cx().flush_runs();
        float got[3] = {0, 0, 0};
        { const float* hv2 = (const float*) strata::gpu::host_view(h); if (hv2) { got[0] = hv2[0]; got[1] = hv2[1]; } }
        std::fprintf(stderr, "DBG args: kernel sees dim=%g hc=%g\n", got[0], got[1]);
    }
    return true;
}

}  // namespace strata::dsv41::metal_host