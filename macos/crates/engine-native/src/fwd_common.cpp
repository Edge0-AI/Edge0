#include "fwd_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dense.hpp"
#include "hotstack.hpp"  // Entry (9-region expert blob) lives here, not in expertmap.hpp

namespace e0n {
namespace mx = mlx::core;

mlx::core::Shape mkshape(std::initializer_list<int64_t> dims) {
  mlx::core::Shape s;
  for (auto d : dims) s.push_back(static_cast<int32_t>(d));
  return s;
}

bool QuantLin::load(const ModelFiles &files, const std::string &weight_name, std::string &err) {
  auto dot = weight_name.rfind('.');
  const std::string mod = dot == std::string::npos ? weight_name : weight_name.substr(0, dot);
  if (!files.read_tensor(weight_name, wb, err) || !files.read_tensor(mod + ".scales", sb, err))
    return false;
  std::string b_err;
  if (!files.read_tensor(mod + ".biases", bb, b_err)) bb.clear();
  const TensorInfo *tw = files.find(weight_name), *ts = files.find(mod + ".scales");
  if (tw->dtype != "U32" || ts->dtype != "BF16") {
    err = "QuantLin expected U32 weight + BF16 scales: " + weight_name;
    return false;
  }
  if (!files.config_quant_for(weight_name, bits, group)) {
    err = "QuantLin missing quantization config: " + weight_name;
    return false;
  }
  const int64_t per_word = 32 / bits;
  int64_t r = 1;
  for (size_t i = 0; i + 1 < tw->shape.size(); i++) r *= tw->shape[i];
  rows = r;
  cols = tw->shape.back() * per_word;
  if (ts->shape.back() * group != cols) {
    err = "QuantLin scale group count mismatch: " + weight_name;
    return false;
  }
  mlx::core::array wa(wb.data(), mkshape({rows, tw->shape.back()}), mlx::core::uint32,
                      [](void *) {});
  mlx::core::array sa(sb.data(), mkshape({rows, ts->shape.back()}), mlx::core::bfloat16,
                      [](void *) {});
  w = wa;
  s = sa;
  if (!bb.empty())
    b = mlx::core::array(bb.data(), mkshape({rows, ts->shape.back()}), mlx::core::bfloat16,
                         [](void *) {});
  loaded = true;
  return true;
}

mlx::core::array QuantLin::apply(const mlx::core::array &x) const {
  return mlx::core::quantized_matmul(x, w, s, biases(), /*transpose=*/true, group, bits,
                                     "affine");
}

bool RawLin::load(const ModelFiles &files, const std::string &name, std::string &err) {
  const TensorInfo *t = files.find(name);
  if (!t) { err = "missing tensor: " + name; return false; }
  if (t->dtype != "BF16" && t->dtype != "F16" && t->dtype != "F32") {
    err = "RawLin accepts only BF16/F16/F32: " + name;
    return false;
  }
  if (!files.read_tensor(name, wb, err)) return false;
  int64_t r = 1;
  for (size_t i = 0; i + 1 < t->shape.size(); i++) r *= t->shape[i];
  if (t->shape.size() == 1) r = 1;
  rows = r;
  cols = t->shape.back();
  auto dt = t->dtype == "BF16"   ? mlx::core::bfloat16
            : t->dtype == "F16"  ? mlx::core::float16
                                 : mlx::core::float32;
  w = mlx::core::array(wb.data(), mkshape(t->shape), dt, [](void *) {});
  return true;
}

mlx::core::array RawLin::apply(const mlx::core::array &x) const {
  return mlx::core::matmul(x, mlx::core::transpose(w));
}

// Upstream nn.silu is x * sigmoid(x); x/(1+exp(-x)) is algebraically equal but
// rounds differently (one division vs reciprocal+mul).
mlx::core::array e0_silu(const mlx::core::array &x) {
  return x * mlx::core::sigmoid(x);
}

mlx::core::array e0_swiglu_precise(mlx::core::Dtype out_dt, const mlx::core::array &gate,
                                   const mlx::core::array &up) {
  auto g = e0_silu(mx::astype(gate, mx::float32));
  auto u = mx::astype(up, mx::float32);
  return mx::astype(g * u, out_dt);
}

mlx::core::array e0_rms_norm(const mlx::core::array &x,
                             const std::optional<mlx::core::array> &weight, double eps) {
  return mx::fast::rms_norm(x, weight, static_cast<float>(eps));
}

ExecCounters &exec_counters() {
  static ExecCounters c;
  return c;
}

uint64_t TimeSink::monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
bool TimingCounters::enabled() {
  static const bool on = [] {
    const char *e = getenv("EDGE0_TIMING");
    return e && !strcmp(e, "1");
  }();
  return on;
}
TimingCounters &timing_counters() {
  static TimingCounters c;
  return c;
}

ExecMode exec_mode() {
  static const ExecMode m = [] {
    const char *e = getenv("EDGE0_EXEC");
    if (!e || !*e) return ExecMode::Batched;
    if (!strcmp(e, "batched")) return ExecMode::Batched;
    if (!strcmp(e, "legacy")) return ExecMode::Legacy;
    std::fprintf(stderr, "edge0: unknown EDGE0_EXEC=%s, using legacy (batched|legacy)\n", e);
    return ExecMode::Legacy;
  }();
  return m;
}

bool prefill_ondemand() {
  static const bool on = [] {
    const char *e = getenv("EDGE0_PREFILL");
    if (!e || !*e) return true;
    if (!strcmp(e, "ondemand")) return true;
    if (!strcmp(e, "production")) return false;
    std::fprintf(stderr, "edge0: unknown EDGE0_PREFILL=%s, using ondemand (ondemand|production)\n", e);
    return true;
  }();
  return on;
}

bool to_host32(const mx::array &a, std::vector<float> &out, std::string &err) {
  E0_TIME(timing_counters().host_sync_ns);
  exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
  try {
    auto cpu = mx::copy(mx::astype(a, mx::float32), mx::default_stream(mx::Device::cpu));
    mx::eval(cpu);
    const float *p = cpu.data<float>();
    out.assign(p, p + cpu.size());
  } catch (const std::exception &ex) {
    err = std::string("to_host32: ") + ex.what();
    return false;
  }
  return true;
}

mx::array wrap_f32(const std::vector<float> &v, mx::Shape shape) {
  exec_counters().host_wraps.fetch_add(1, std::memory_order_relaxed);
  return mx::array(v.begin(), std::move(shape), mx::float32);
}

bool expert_ptrs(const Entry &e, ExpertPtrs &p, std::string &err) {
  if (e.regions.size() != 9) { err = "expert region count != 9"; return false; }
  for (size_t i = 0; i < 9; i++) {
    const uint8_t *base = e.blob.data() + e.blob_offsets[i];
    const std::string &nm = e.regions[i].name;
    const uint32_t **wp = nullptr;
    const uint16_t **sp = nullptr, **bp = nullptr;
    if (nm == "gate_proj.weight") wp = &p.gw;
    else if (nm == "up_proj.weight") wp = &p.uw;
    else if (nm == "down_proj.weight") wp = &p.dw;
    else if (nm == "gate_proj.scales") sp = &p.gs;
    else if (nm == "gate_proj.biases") bp = &p.gb;
    else if (nm == "up_proj.scales") sp = &p.us;
    else if (nm == "up_proj.biases") bp = &p.ub;
    else if (nm == "down_proj.scales") sp = &p.ds;
    else if (nm == "down_proj.biases") bp = &p.db;
    else { err = "unknown region name " + nm; return false; }
    if (wp) *wp = reinterpret_cast<const uint32_t *>(base);
    if (sp) *sp = reinterpret_cast<const uint16_t *>(base);
    if (bp) *bp = reinterpret_cast<const uint16_t *>(base);
  }
  return true;
}

bool embed_lookup(const QuantLin &emb, const std::vector<int32_t> &ids, int64_t H, int64_t V,
                  std::vector<float> &out, std::string &err) {
  const int64_t groups = H / 64, packed = H / 8;
  if (emb.rows != V) { err = "embed rows != vocab"; return false; }
  out.assign(ids.size() * static_cast<size_t>(H), 0.f);
  for (size_t i = 0; i < ids.size(); i++) {
    int64_t id = ids[i];
    if (id < 0 || id >= V) { err = "token id out of range " + std::to_string(id); return false; }
    const uint32_t *wp = reinterpret_cast<const uint32_t *>(emb.wb.data()) + id * packed;
    const uint16_t *sp = reinterpret_cast<const uint16_t *>(emb.sb.data()) + id * groups;
    const uint16_t *bp = reinterpret_cast<const uint16_t *>(emb.bb.data()) + id * groups;
    std::vector<float> row;
    dequant_row(wp, sp, bp, packed, 4, 64, row);
    std::memcpy(out.data() + i * H, row.data(), H * 4);
  }
  return true;
}

}  // namespace e0n
