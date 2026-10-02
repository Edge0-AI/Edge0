#include "qgemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include "mlx/backend/metal/device.h"

#include "expertmap.hpp"
#include "fwd_common.hpp"

using mlx::core::array;
using mlx::core::Device;

namespace e0n {
namespace {

constexpr const char *kMslKernel = R"metal(
#include <metal_stdlib>
using namespace metal;
inline float bf2f(ushort v) {
  uint bits = uint(v) << 16;
  return *(thread float*)&bits;
}
kernel void e0n_dequant_gemm(
    const device float* x [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device ushort* s [[buffer(2)]],
    const device ushort* b [[buffer(3)]],
    const device uint* idxs [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint& O [[buffer(6)]],
    constant uint& I [[buffer(7)]],
    constant uint& T [[buffer(8)]],
    uint3 g [[threadgroup_position_in_grid]],
    uint3 t [[thread_position_in_threadgroup]]) {
  const uint G = I / 64;
  const uint o = g.x * 32u + t.x;
  const uint tk = g.y;
  const uint e = g.z;
  if (o >= O || tk >= T) return;
  const uint expert = idxs[e];
  const uint wrow = (expert * O + o) * (I / 8);
  const uint srow = (expert * O + o) * G;
  float acc = 0.f;
  for (uint k = 0; k < I; k++) {
    const uint word = w[wrow + (k >> 3)];
    const uint nib = (word >> (4u * (k & 7u))) & 0xFu;
    acc += x[tk * I + k] * (bf2f(s[srow + (k >> 6)]) * float(nib) + bf2f(b[srow + (k >> 6)]));
  }
  y[(e * T + tk) * O + o] = acc;
}
)metal";

inline float bf16f(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

bool read_full_tensor(const ModelFiles &files, const std::string &prefix,
                      const std::string &name, std::vector<uint8_t> &out, std::string &err) {
  const TensorInfo *t = files.find(prefix + "." + name);
  if (!t) { err = "missing tensor " + prefix + "." + name; return false; }
  out.resize(t->end - t->begin);
  int fd = ::open(files.shard_path(t->shard).c_str(), O_RDONLY);
  if (fd < 0) { err = "open failed " + t->shard; return false; }
  size_t done = 0;
  while (done < out.size()) {
    ssize_t got = ::pread(fd, out.data() + done, out.size() - done, t->begin + done);
    if (got <= 0) { ::close(fd); err = "pread failed " + name; return false; }
    done += got;
  }
  ::close(fd);
  return true;
}

std::string fused_prefix(const ModelFiles &files, int layer) {
  const bool qwen = files.find("language_model.model.layers.0.mlp.gate.weight") != nullptr;
  const std::string base =
      qwen ? "language_model.model.layers." + std::to_string(layer)
           : "model.layers." + std::to_string(layer);
  for (const std::string &cand : {base + ".mlp.experts", base + ".mlp.switch_mlp", base + ".mlp"}) {
    if (files.find(cand + ".gate_proj.weight")) return cand;
  }
  return "";
}

bool msl_proj(MTL::CommandQueue *q, MTL::ComputePipelineState *pso, MTL::Buffer *xbuf,
              MTL::Buffer *w, MTL::Buffer *s, MTL::Buffer *b, MTL::Buffer *idx, MTL::Buffer *y,
              int ES, int O, int I, std::string &err) {
  uint32_t uO = O, uI = I, uT = 1;
  MTL::CommandBuffer *cb = q->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(pso);
  enc->setBuffer(xbuf, 0, 0);
  enc->setBuffer(w, 0, 1);
  enc->setBuffer(s, 0, 2);
  enc->setBuffer(b, 0, 3);
  enc->setBuffer(idx, 0, 4);
  enc->setBuffer(y, 0, 5);
  enc->setBytes(&uO, sizeof(uO), 6);
  enc->setBytes(&uI, sizeof(uI), 7);
  enc->setBytes(&uT, sizeof(uT), 8);
  enc->dispatchThreads(MTL::Size::Make((O + 31) / 32 * 32, 1, ES), MTL::Size::Make(32, 1, 1));
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (cb->error() != nullptr) {
    err = "MSL dispatch error";
    return false;
  }
  return true;
}

}  // namespace

bool dims_of(const ModelFiles &files, int layer, ExpertMapDims &out, std::string &err) {
  const std::string prefix = fused_prefix(files, layer);
  if (prefix.empty()) { err = "layer has no fused prefix"; return false; }
  const TensorInfo *g = files.find(prefix + ".gate_proj.weight");
  const TensorInfo *d = files.find(prefix + ".down_proj.weight");
  if (!g || !d) { err = "missing gate/down weights"; return false; }
  int64_t E = 0;
  if (!files.config_int("num_experts", E)) { err = "missing num_experts"; return false; }
  auto rows_of = [&](const TensorInfo *t) {
    return t->shape.size() == 3 ? t->shape[1] : t->shape[0] / E;
  };
  out.E = E;
  out.hid = static_cast<int>(g->shape.back() * 8);
  out.ffn = static_cast<int>(rows_of(g));
  return true;
}

struct QuantLayer::Impl {
  GemmPath path = GemmPath::kMlxGatherQmm;
  std::vector<array> w, s, b;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> keepers;
  MTL::Device *mtl = nullptr;
  MTL::CommandQueue *q = nullptr;
  MTL::ComputePipelineState *pso = nullptr;
  MTL::Buffer *bw[3] = {}, *bs[3] = {}, *bb[3] = {};
  ~Impl() {
    for (int i = 0; i < 3; i++) {
      if (bw[i]) bw[i]->release();
      if (bs[i]) bs[i]->release();
      if (bb[i]) bb[i]->release();
    }
  }
};

QuantLayer::QuantLayer(std::unique_ptr<Impl> i) : impl_(std::move(i)) {}
QuantLayer::~QuantLayer() = default;

std::unique_ptr<QuantLayer> QuantLayer::load(const ModelFiles &files, const ExpertMapDims &dims,
                                             int layer, GemmPath path, std::string &err) {
  const std::string prefix = fused_prefix(files, layer);
  if (prefix.empty()) { err = "layer has no fused prefix"; return nullptr; }

  auto l = std::unique_ptr<QuantLayer>(new QuantLayer(std::unique_ptr<Impl>(new Impl())));
  l->e_ = static_cast<int>(dims.E);
  l->hid_ = dims.hid;
  l->ffn_ = dims.ffn;
  auto &im = *l->impl_;
  im.path = path;

  const int E = static_cast<int>(dims.E), HID = dims.hid, FFN = dims.ffn;
  const char *parts[3] = {"gate_proj", "up_proj", "down_proj"};
  const int in_dim[3] = {HID, HID, FFN};
  const int out_rows[3] = {FFN, FFN, HID};
  std::vector<std::vector<uint8_t>> wb(3), sb(3), bb(3);
  for (int p = 0; p < 3; p++) {
    if (!read_full_tensor(files, prefix, std::string(parts[p]) + ".weight", wb[p], err) ||
        !read_full_tensor(files, prefix, std::string(parts[p]) + ".scales", sb[p], err) ||
        !read_full_tensor(files, prefix, std::string(parts[p]) + ".biases", bb[p], err))
      return nullptr;
  }
  if (path == GemmPath::kMlxGatherQmm) {
    for (int p = 0; p < 3; p++) {
      auto kw = std::make_shared<std::vector<uint8_t>>(std::move(wb[p]));
      auto ks = std::make_shared<std::vector<uint8_t>>(std::move(sb[p]));
      auto kb = std::make_shared<std::vector<uint8_t>>(std::move(bb[p]));
      const int Gi = in_dim[p] / 64;
      im.w.push_back(array(kw->data(), {E, out_rows[p], in_dim[p] / 8}, mlx::core::uint32,
                           [kw](void *) {}));
      im.s.push_back(array(ks->data(), {E, out_rows[p], Gi}, mlx::core::bfloat16, [ks](void *) {}));
      im.b.push_back(array(kb->data(), {E, out_rows[p], Gi}, mlx::core::bfloat16, [kb](void *) {}));
      im.keepers.insert(im.keepers.end(), {kw, ks, kb});
    }
  } else {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    auto &mdev = mlx::core::metal::device(Device::gpu);
    im.mtl = mdev.mtl_device();
    im.q = mdev.get_queue(mlx::core::default_stream(Device::gpu));
    MTL::Library *lib = mdev.get_library("e0n_dequant_gemm", [] { return std::string(kMslKernel); });
    im.pso = lib ? mdev.get_kernel("e0n_dequant_gemm", lib) : nullptr;
    if (!im.pso) { pool->release(); err = "MSL kernel compile failed"; return nullptr; }
    auto nbuf = [&](const void *d, size_t bytes) {
      return im.mtl->newBuffer(d, bytes, MTL::ResourceStorageModeShared);
    };
    for (int p = 0; p < 3; p++) {
      im.bw[p] = nbuf(wb[p].data(), wb[p].size());
      im.bs[p] = nbuf(sb[p].data(), sb[p].size());
      im.bb[p] = nbuf(bb[p].data(), bb[p].size());
    }
    pool->release();
  }
  return l;
}

bool QuantLayer::moe_proxy(std::vector<float> &x,
                           const std::vector<std::vector<uint32_t>> &layer_indices,
                           std::string &err) {
  if (layer_indices.empty()) return true;
  const int HID = hid_, FFN = ffn_;
  auto &im = *impl_;
  if (im.path == GemmPath::kMlxGatherQmm) {
    auto xg = array(x.begin(), {1, HID});
    for (const auto &indices : layer_indices) {
      const int ES = (int)indices.size();
      auto ia = array(indices.begin(), {ES});
      auto a = mlx::core::gather_qmm(xg, im.w[0], im.s[0], im.b[0], std::nullopt, ia, true, 64,
                                     4, "affine");
      auto b = mlx::core::gather_qmm(xg, im.w[1], im.s[1], im.b[1], std::nullopt, ia, true, 64,
                                     4, "affine");
      auto ga = mlx::core::sum(a, {0}, false);
      auto h = (ga * mlx::core::sigmoid(ga)) * mlx::core::sum(b, {0}, false);
      auto d = mlx::core::gather_qmm(h, im.w[2], im.s[2], im.b[2], std::nullopt, ia, true, 64,
                                     4, "affine");
      xg = xg + mlx::core::sum(d, {0}, false);
    }
    auto cpu = mlx::core::copy(xg, mlx::core::default_stream(Device::cpu));
    mlx::core::eval(cpu);
    const float *p = cpu.data<float>();
    x.assign(p, p + HID);
    return true;
  }
  const int MAX_ES = (int)layer_indices[0].size();
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Buffer *bidx = im.mtl->newBuffer(MAX_ES * 4, MTL::ResourceStorageModeShared);
  MTL::Buffer *bx = im.mtl->newBuffer((size_t)HID * 4, MTL::ResourceStorageModeShared);
  MTL::Buffer *bg = im.mtl->newBuffer((size_t)MAX_ES * FFN * 4, MTL::ResourceStorageModeShared);
  MTL::Buffer *bu = im.mtl->newBuffer((size_t)MAX_ES * FFN * 4, MTL::ResourceStorageModeShared);
  MTL::Buffer *bd = im.mtl->newBuffer((size_t)MAX_ES * HID * 4, MTL::ResourceStorageModeShared);
  bool ok = true;
  std::vector<float> h(FFN);
  for (const auto &indices : layer_indices) {
    const int ES = (int)indices.size();
    ::memcpy(bidx->contents(), indices.data(), ES * 4);
    ::memcpy(bx->contents(), x.data(), HID * 4);
    ok = msl_proj(im.q, im.pso, bx, im.bw[0], im.bs[0], im.bb[0], bidx, bg, ES, FFN, HID, err) &&
         msl_proj(im.q, im.pso, bx, im.bw[1], im.bs[1], im.bb[1], bidx, bu, ES, FFN, HID, err);
    if (!ok) break;
    const float *pg = (const float *)bg->contents();
    const float *pu = (const float *)bu->contents();
    for (int o = 0; o < FFN; o++) {
      double gs = 0, us = 0;
      for (int e = 0; e < ES; e++) {
        float gv = pg[(size_t)e * FFN + o];
        gs += (double)gv / (1.0 + std::exp(-(double)gv));  // silu
        us += pu[(size_t)e * FFN + o];
      }
      h[o] = (float)(gs * us);
    }
    ::memcpy(bx->contents(), h.data(), FFN * 4);
    ok = msl_proj(im.q, im.pso, bx, im.bw[2], im.bs[2], im.bb[2], bidx, bd, ES, HID, FFN, err);
    if (!ok) break;
    const float *pd = (const float *)bd->contents();
    for (int k = 0; k < HID; k++) {
      double acc = 0;
      for (int e = 0; e < ES; e++) acc += pd[(size_t)e * HID + k];
      x[k] += (float)acc;
    }
  }
  bidx->release(); bx->release(); bg->release(); bu->release(); bd->release();
  pool->release();
  return ok;
}

std::vector<double> fp64_qmm_ref(const float *x, int T, int I, const uint32_t *W,
                                 const uint16_t *S, const uint16_t *B, int O, int G,
                                 const std::vector<int64_t> &sel) {
  std::vector<double> ref;
  ref.reserve(sel.size() * T * O);
  for (int64_t e : sel) {
    const uint32_t *we = W + static_cast<size_t>(e) * O * (I / 8);
    const uint16_t *se = S + static_cast<size_t>(e) * O * G;
    const uint16_t *be = B + static_cast<size_t>(e) * O * G;
    for (int t = 0; t < T; t++) {
      for (int o = 0; o < O; o++) {
        double acc = 0;
        for (int k = 0; k < I; k++) {
          const uint32_t word = we[static_cast<size_t>(o) * (I / 8) + k / 8];
          const int nib = (word >> (4 * (k % 8))) & 0xF;
          acc += (double)x[(size_t)t * I + k] *
                 ((double)bf16f(se[o * G + k / 64]) * nib + (double)bf16f(be[o * G + k / 64]));
        }
        ref.push_back(acc);
      }
    }
  }
  return ref;
}

bool gather_qmm_mlx(const uint32_t *W, const uint16_t *S, const uint16_t *B, int E_all, int O,
                    int I, int G, const float *x, int T, const std::vector<int64_t> &sel,
                    std::vector<float> &y, std::string &err) {
  {
    auto &cx = exec_counters();
    cx.qmm_calls.fetch_add(1, std::memory_order_relaxed);
    cx.qmm_experts.fetch_add(sel.size(), std::memory_order_relaxed);
    cx.qmm_weight_bytes.fetch_add(sel.size() * ((size_t)O * (I / 8) * 4 + 2 * (size_t)O * G * 2),
                                  std::memory_order_relaxed);
  }
  std::vector<uint32_t> idx(sel.begin(), sel.end());
  auto wa = array((void *)W, {E_all, O, I / 8}, mlx::core::uint32, [](void *) {});
  auto sa = array((void *)S, {E_all, O, G}, mlx::core::bfloat16, [](void *) {});
  auto ba = array((void *)B, {E_all, O, G}, mlx::core::bfloat16, [](void *) {});
  auto xa = array(x, {T, I});
  auto ia = array(idx.begin(), {static_cast<int>(idx.size())});
  try {
    auto ya = mlx::core::gather_qmm(xa, wa, sa, ba, std::nullopt, ia, true, 64, 4, "affine");
    auto ycpu = mlx::core::copy(ya, mlx::core::default_stream(Device::cpu));
    exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
    mlx::core::eval(ycpu);
    const float *p = ycpu.data<float>();
    y.assign(p, p + sel.size() * T * O);
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }
  return true;
}

mlx::core::array gather_qmm_graph(const uint32_t *W, const uint16_t *S, const uint16_t *B,
                                   int E_all, int O, int I, int G, const mlx::core::array &xa,
                                   int T, const std::vector<int64_t> &sel) {
  auto &cx = exec_counters();
  cx.qmm_calls.fetch_add(1, std::memory_order_relaxed);
  cx.qmm_experts.fetch_add(sel.size(), std::memory_order_relaxed);
  cx.qmm_weight_bytes.fetch_add(sel.size() * ((size_t)O * (I / 8) * 4 + 2 * (size_t)O * G * 2),
                                std::memory_order_relaxed);
  std::vector<uint32_t> idx(sel.begin(), sel.end());
  auto wa = array((void *)W, {E_all, O, I / 8}, mlx::core::uint32, [](void *) {});
  auto sa = array((void *)S, {E_all, O, G}, mlx::core::bfloat16, [](void *) {});
  auto ba = array((void *)B, {E_all, O, G}, mlx::core::bfloat16, [](void *) {});
  auto ia = array(idx.begin(), {static_cast<int>(idx.size())});
  return mlx::core::gather_qmm(xa, wa, sa, ba, std::nullopt, ia, true, 64, 4, "affine");
}

double max_rel(const std::vector<float> &got, const std::vector<double> &ref) {  double norm = 0, mr = 0;
  for (double v : ref) norm = std::max(norm, std::fabs(v));
  for (size_t i = 0; i < ref.size() && i < got.size(); i++)
    mr = std::max(mr, std::fabs((double)got[i] - ref[i]) / (norm + 1e-12));
  return mr;
}

}  // namespace e0n
