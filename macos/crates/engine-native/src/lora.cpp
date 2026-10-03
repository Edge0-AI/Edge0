#include "lora.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

namespace e0n {
namespace lora {
namespace mx = mlx::core;
using json = nlohmann::json;

namespace {

void jsonl_capture(const std::string &path, const std::string &key, int layer,
                   const mx::array &base, const mx::array &delta) {
  try {
    mx::array be = mx::astype(base, mx::float32), de = mx::astype(delta, mx::float32);
    mx::eval(be, de);
    const size_t n = be.size();
    if (de.size() != n) return;
    auto side = [](const float *p, size_t n) {
      double ss = 0;
      for (size_t i = 0; i < n; i++) ss += double(p[i]) * p[i];
      json j;
      json pref = json::array();
      for (size_t i = 0; i < 64 && i < n; i++) pref.push_back(p[i]);
      j["prefix"] = pref;
      j["summary"] = {{"l2", std::sqrt(ss)}};
      return j;
    };
    json rec;
    rec["key"] = key;
    rec["layer"] = layer;
    rec["step"] = -1;
    rec["n"] = static_cast<int64_t>(n);
    rec["base"] = side(be.data<float>(), n);
    rec["delta"] = side(de.data<float>(), n);
    std::ofstream f(path, std::ios::app);
    f << rec.dump() << "\n";
  } catch (const std::exception &) {
  }
}

bool parse_meta_field(const json &m, const char *key, int64_t &out) {
  if (!m.contains(key)) return false;
  const json &v = m[key];
  if (v.is_number_integer()) { out = v.get<int64_t>(); return true; }
  if (v.is_string()) {
    try { out = std::stoll(v.get<std::string>()); return true; } catch (...) { return false; }
  }
  return false;
}

int slot_of_35(int layer, const std::string &tail, std::string &err) {
  const bool full = (layer + 1) % 4 == 0;
  auto need = [&](bool cond) {
    if (cond) return true;
    err = "lora layer family mismatch: layer=" + std::to_string(layer) + " tail=" + tail;
    return false;
  };
  if (tail == "linear_attn.in_proj_a") return need(!full) ? S35_IN_A : -1;
  if (tail == "linear_attn.in_proj_b") return need(!full) ? S35_IN_B : -1;
  if (tail == "linear_attn.in_proj_qkv") return need(!full) ? S35_IN_QKV : -1;
  if (tail == "linear_attn.in_proj_z") return need(!full) ? S35_IN_Z : -1;
  if (tail == "linear_attn.out_proj") return need(!full) ? S35_OUT_PROJ : -1;
  if (tail == "self_attn.q_proj") return need(full) ? S35_Q : -1;
  if (tail == "self_attn.k_proj") return need(full) ? S35_K : -1;
  if (tail == "self_attn.v_proj") return need(full) ? S35_V : -1;
  if (tail == "self_attn.o_proj") return need(full) ? S35_O : -1;
  if (tail == "mlp.shared_expert.gate_proj") return S35_SH_GATE;
  if (tail == "mlp.shared_expert.up_proj") return S35_SH_UP;
  if (tail == "mlp.shared_expert.down_proj") return S35_SH_DOWN;
  err = "lora unknown target: " + tail;
  return -1;
}

int slot_of_8b(int layer, const std::string &tail, std::string &err) {
  const bool mla = (layer + 1) % 4 == 0;
  auto need = [&](bool cond) {
    if (cond) return true;
    err = "lora layer family mismatch: layer=" + std::to_string(layer) + " tail=" + tail;
    return false;
  };
  if (tail == "attention.q_proj") return need(!mla) ? S8B_Q : -1;
  if (tail == "attention.k_proj") return need(!mla) ? S8B_K : -1;
  if (tail == "attention.v_proj") return need(!mla) ? S8B_V : -1;
  if (tail == "attention.o_proj") return need(!mla) ? S8B_O : -1;
  if (tail == "attention.b_proj") return need(!mla) ? S8B_B : -1;
  if (tail == "attention.f_proj") return need(!mla) ? S8B_F : -1;
  if (tail == "attention.g_proj") return mla ? S8B_M_G : S8B_G;
  if (tail == "attention.q_b_proj") return need(mla) ? S8B_M_QB : -1;
  if (tail == "attention.kv_b_proj") return need(mla) ? S8B_M_KVB : -1;
  if (tail == "attention.dense") return need(mla) ? S8B_M_DENSE : -1;
  if (tail == "mlp.gate_proj") return need(layer == 0) ? S8B_D_GATE : -1;
  if (tail == "mlp.up_proj") return need(layer == 0) ? S8B_D_UP : -1;
  if (tail == "mlp.down_proj") return need(layer == 0) ? S8B_D_DOWN : -1;
  err = "lora unknown target: " + tail;
  return -1;
}

}  // namespace

bool Bundle::load_raw(std::map<std::string, std::pair<std::string, std::vector<uint8_t>>> tensors,
                      const std::string &meta_json, bool is_8b, int num_layers,
                      std::string &err) {
  try {
    json m = json::parse(meta_json);
    int64_t r = 0, a = 0, k = 0;
    if (!parse_meta_field(m, "r", r) || !parse_meta_field(m, "alpha", a) ||
        !parse_meta_field(m, "K", k) || r <= 0 || a <= 0) {
      err = "lora metadata missing r/alpha/K or not positive";
      return false;
    }
    meta_.r = r;
    meta_.alpha = a;
    meta_.k = k;
    meta_.scale = static_cast<float>(a) / static_cast<float>(r);
    meta_.ok = true;
  } catch (const std::exception &ex) {
    err = std::string("lora metadata parse: ") + ex.what();
    return false;
  }

  struct Half {
    std::pair<std::string, std::vector<uint8_t>> A, B;
  };
  std::map<std::string, Half> halves;
  for (auto &kv : tensors) {
    const std::string &key = kv.first;
    const size_t sfx = 7;  // ".lora_A"
    bool is_a = key.size() > sfx && key.compare(key.size() - sfx, sfx, ".lora_A") == 0;
    bool is_b = key.size() > sfx && key.compare(key.size() - sfx, sfx, ".lora_B") == 0;
    if (!is_a && !is_b) {
      err = "lora unexpected key (not .lora_A/.lora_B): " + key;
      return false;
    }
    std::string base = key.substr(0, key.size() - sfx);
    if (is_a) halves[base].A = kv.second;
    else halves[base].B = kv.second;
  }

  const std::string pfx35 = "language_model.model.layers.";
  const std::string pfx8 = "model.layers.";
  ns_ = is_8b ? static_cast<int>(S8B_NSLOT) : static_cast<int>(S35_NSLOT);
  slots_.assign(num_layers, std::vector<int>(ns_, -1));

  for (auto &h : halves) {
    const std::string &base = h.first;
    const bool has_a = !h.second.A.second.empty();
    const bool has_b = !h.second.B.second.empty();
    if (!has_a || !has_b) {
      err = "lora pairing incomplete: " + base;
      return false;
    }
    const size_t n_a = h.second.A.second.size(), n_b = h.second.B.second.size();
    std::string dta = h.second.A.first, dtb = h.second.B.first;
    auto tag = [](std::string &d) {
      auto pos = d.find('|');
      std::string shape;
      if (pos != std::string::npos) {
        shape = d.substr(pos + 1);
        d = d.substr(0, pos);
      }
      return shape;
    };
    std::string sha = tag(dta), shb = tag(dtb);
    if (dta != "F16" || dtb != "F16") {
      err = "lora expected F16 domain: " + base;
      return false;
    }
    auto parse_dims = [](const std::string &s, std::vector<int64_t> &out) {
      // "r,in" / "out,r"
      size_t c = s.find(',');
      if (c == std::string::npos) return false;
      try {
        out.push_back(std::stoll(s.substr(0, c)));
        out.push_back(std::stoll(s.substr(c + 1)));
      } catch (...) { return false; }
      return true;
    };
    std::vector<int64_t> da, db;
    if (!parse_dims(sha, da) || !parse_dims(shb, db) || da.size() != 2 || db.size() != 2) {
      err = "lora shape markers missing: " + base;
      return false;
    }
    const int64_t r = da[0], in = da[1], out = db[0];
    if (db[1] != r || r != meta_.r || in <= 0 || out <= 0) {
      err = "lora shape mismatch (r/in/out): " + base + " A=[" + std::to_string(r) + "," +
            std::to_string(in) + "] B=[" + std::to_string(out) + "," + std::to_string(db[1]) + "]";
      return false;
    }
    if (n_a != static_cast<size_t>(2 * r * in) || n_b != static_cast<size_t>(2 * out * r)) {
      err = "lora byte size does not match shape: " + base;
      return false;
    }
    std::string prefix = is_8b ? pfx8 : pfx35;
    if (base.compare(0, prefix.size(), prefix) != 0) {
      std::string other = is_8b ? pfx35 : pfx8;
      if (base.compare(0, other.size(), other) == 0) {
        err = "lora tier prefix mismatch: " + base;
        return false;
      }
      err = "lora unknown prefix: " + base;
      return false;
    }
    std::string rest = base.substr(prefix.size());
    size_t dot = rest.find('.');
    int layer = -1;
    if (dot == std::string::npos) {
      err = "lora key shape invalid: " + base;
      return false;
    }
    try {
      layer = std::stoi(rest.substr(0, dot));
    } catch (...) {
      err = "lora layer is not a number: " + base;
      return false;
    }
    if (layer < 0 || layer >= num_layers) {
      err = "lora layer out of range: " + base;
      return false;
    }
    std::string tail = rest.substr(dot + 1);
    std::string serr;
    int slot = is_8b ? slot_of_8b(layer, tail, serr) : slot_of_35(layer, tail, serr);
    if (slot < 0) {
      err = serr.empty() ? "lora target parse failed: " + base : serr;
      return false;
    }
    if (slots_[layer][slot] >= 0) {
      err = "lora slot already bound: " + base;
      return false;
    }
    auto p = std::make_unique<Pair>();
    p->layer = layer;
    p->slot = slot;
    p->r = r;
    p->in = in;
    p->out = out;
    p->ab = std::move(h.second.A.second);
    p->bb = std::move(h.second.B.second);
    p->key = base;
    slots_[layer][slot] = static_cast<int>(pairs_.size());
    bytes_ += static_cast<int64_t>(p->ab.size() + p->bb.size());
    pairs_.push_back(std::move(p));
  }

  for (auto &p : pairs_) {
    p->A = mx::array(p->ab.data(), mx::Shape{static_cast<int32_t>(p->r), static_cast<int32_t>(p->in)},
                     mx::float16, [](void *) {});
    p->B = mx::array(p->bb.data(), mx::Shape{static_cast<int32_t>(p->out), static_cast<int32_t>(p->r)},
                     mx::float16, [](void *) {});
  }
  for (auto &p : pairs_) keys_sorted_.push_back(p->key);
  std::sort(keys_sorted_.begin(), keys_sorted_.end());
  hit_.assign(pairs_.size(), 0);
  return true;
}

bool Bundle::load_file(const std::string &path, bool is_8b, int num_layers, std::string &err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "lora file unreadable: " + path; return false; }
  uint64_t hl = 0;
  f.read(reinterpret_cast<char *>(&hl), 8);
  if (!f || hl > (64u << 20)) { err = "lora header length invalid: " + path; return false; }
  std::vector<char> hs(hl, '\0');
  f.read(hs.data(), static_cast<std::streamsize>(hl));
  if (!f) { err = "lora header truncated: " + path; return false; }
  json hdr;
  try {
    hdr = json::parse(std::string(hs.begin(), hs.end()));
  } catch (const std::exception &ex) {
    err = std::string("lora header parse: ") + ex.what();
    return false;
  }
  std::string meta = "{}";
  if (hdr.contains("__metadata__")) {
    const json &mv = hdr["__metadata__"];
    if (mv.is_string()) meta = mv.get<std::string>();
    else if (mv.is_object()) {
      if (mv.size() == 1 && mv.contains("__metadata__") && mv["__metadata__"].is_string())
        meta = mv["__metadata__"].get<std::string>();
      else
        meta = mv.dump();
    }
  }
  std::map<std::string, std::pair<std::string, std::vector<uint8_t>>> tensors;
  const uint64_t base_off = 8 + hl;
  for (auto &el : hdr.items()) {
    if (el.key() == "__metadata__") continue;
    const json &v = el.value();
    if (!v.contains("dtype") || !v.contains("data_offsets")) continue;
    std::string dt = v["dtype"].get<std::string>();
    std::vector<int64_t> shape;
    for (auto &d : v["shape"]) shape.push_back(d.get<int64_t>());
    std::string tag = dt;
    if (shape.size() == 2) tag += "|" + std::to_string(shape[0]) + "," + std::to_string(shape[1]);
    uint64_t b = v["data_offsets"][0].get<uint64_t>() + base_off;
    uint64_t e = v["data_offsets"][1].get<uint64_t>() + base_off;
    std::vector<uint8_t> buf(e - b);
    f.seekg(static_cast<std::streamoff>(b));
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!f) { err = "lora tensor truncated: " + el.key(); return false; }
    tensors[el.key()] = {tag, std::move(buf)};
  }
  if (!load_raw(std::move(tensors), meta, is_8b, num_layers, err)) return false;
  if (const char *sc = std::getenv("E0_LORA_SCAFFOLD")) cap_path_ = sc;
  return true;
}

bool Bundle::has_pair(int layer, int slot) const { return find_pair(layer, slot) != nullptr; }

int Bundle::n_hits() const {
  int n = 0;
  for (char c : hit_) n += c != 0;
  return n;
}

bool Bundle::pair_raw(int layer, int slot, std::vector<uint8_t> &a, std::vector<uint8_t> &b,
                      int64_t &r, int64_t &in, int64_t &out) const {
  const Pair *p = find_pair(layer, slot);
  if (!p) return false;
  a = p->ab;
  b = p->bb;
  r = p->r;
  in = p->in;
  out = p->out;
  return true;
}

mlx::core::array Bundle::dressed(const QuantLin &lin, const mlx::core::array &x, int layer,
                                  int slot) const {
  mx::array y = lin.apply(x);
  const Pair *pp = find_pair(layer, slot);
  if (!pp) return y;
  const Pair &p = *pp;
  hit_[static_cast<size_t>(slots_[layer][slot])] = 1;
  if (p.in != lin.cols || p.out != lin.rows) {
    throw std::runtime_error("lora wiring rank mismatch: " + p.key + " file=[in=" + std::to_string(p.in) +
                             ",out=" + std::to_string(p.out) + "] lin=[" + std::to_string(lin.cols) +
                             "," + std::to_string(lin.rows) + "]");
  }
  mx::array xf = mx::astype(x, mx::float16);
  mx::array d = mx::matmul(mx::matmul(xf, mx::transpose(p.A)), mx::transpose(p.B));
  d = mx::astype(meta_.scale * d, y.dtype());
  mx::array out = y + d;
  if (!cap_path_.empty() && cap_done_.insert(p.key).second)
    jsonl_capture(cap_path_, p.key, p.layer, y, d);
  return out;
}

bool Bundle::dressed_variant(const QuantLin &lin, const mlx::core::array &x, int layer, int slot,
                             int which, mx::array &out, std::string &err) const {
  const Pair *p = find_pair(layer, slot);
  if (!p) { err = "dressed_variant: no pairing"; return false; }
  try {
    if (which == 1) {
      mx::array xf = mx::astype(x, mx::float16);
      mx::array d = mx::matmul(mx::matmul(xf, mx::transpose(p->A)), mx::transpose(p->B));
      out = mx::astype(meta_.scale * d, lin.apply(x).dtype());
      return true;
    }
    if (which == 0) {
      float wrong = static_cast<float>(meta_.alpha);
      mx::array xf = mx::astype(x, mx::float16);
      mx::array d = mx::matmul(mx::matmul(xf, mx::transpose(p->A)), mx::transpose(p->B));
      out = lin.apply(x) + mx::astype(wrong * d, lin.apply(x).dtype());
      return true;
    }
    if (which == 2) {
      mx::array xf = mx::astype(x, mx::float32);
      mx::array a32 = mx::astype(p->A, mx::float32), b32 = mx::astype(p->B, mx::float32);
      mx::array d = mx::matmul(mx::matmul(xf, mx::transpose(a32)), mx::transpose(b32));
      out = lin.apply(x) + mx::astype(meta_.scale * d, lin.apply(x).dtype());
      return true;
    }
  } catch (const std::exception &ex) {
    err = std::string("dressed_variant: ") + ex.what();
    return false;
  }
  err = "dressed_variant: unknown which";
  return false;
}

bool Bundle::delta_bits(int layer, int slot, const mx::array &x, std::vector<uint16_t> &bits,
                        std::string &err) const {
  const Pair *p = find_pair(layer, slot);
  if (!p) { err = "delta_bits: no pairing"; return false; }
  try {
    mx::array xf = mx::astype(x, mx::float16);
    mx::array d = mx::matmul(mx::matmul(xf, mx::transpose(p->A)), mx::transpose(p->B));
    d = mx::astype(meta_.scale * d, mx::float16);
    mx::eval(d);
    const uint16_t *hp = d.data<uint16_t>();
    bits.assign(hp, hp + d.size());
  } catch (const std::exception &ex) {
    err = std::string("delta_bits: ") + ex.what();
    return false;
  }
  return true;
}

}  // namespace lora
}  // namespace e0n
