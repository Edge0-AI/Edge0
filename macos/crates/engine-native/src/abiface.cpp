#include "engine-abi.h"

#include <algorithm>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <dirent.h>

#include "memstat.hpp"
#include "fwd_common.hpp"
#include "safio.hpp"
#include "expertmap.hpp"
#include "hotstack.hpp"
#include "expertio.hpp"
#include "net35.hpp"
#include "net8b.hpp"
#include "genstream.hpp"
#include "prerouter.hpp"
#include "prefixcache.hpp"
#include "sampling.hpp"

#include <nlohmann/json.hpp>

extern "C" {
typedef struct ETok ETok;
void *etok_create(const char *model_dir);
void etok_destroy(void *t);
int etok_last_error(char *buf, size_t buf_len, size_t *written);
int etok_tokenize(void *t, const char *text, int32_t *ids_out, size_t ids_cap,
                  size_t *written);
int etok_decode(void *t, const int32_t *ids, size_t n_ids, char *buf, size_t buf_len,
                size_t *written);
int etok_render(void *t, const char *ctx_json, char *buf, size_t buf_len, size_t *written);
}

namespace {

std::string esc(const std::string &s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o;
}

E0Status json_out(const std::string &json, char *buf, size_t buf_len, size_t *written) {
  if (written) *written = json.size();
  if (buf == nullptr) return E0_OK;
  if (buf_len < json.size()) return E0_ERR_BUFFER;
  std::memcpy(buf, json.data(), json.size());
  return E0_OK;
}

bool is_dir(const char *p) {
  struct stat st;
  return p && stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

}  // namespace

std::string etok_err_json() {
  size_t w = 0;
  etok_last_error(nullptr, 0, &w);
  std::string msg(w, '\0');
  if (w) etok_last_error(msg.data(), w, &w);
  return std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") + esc(msg) + "\"}";
}

namespace e0n { namespace gen { class StreamBase; } }

struct E0Engine {
  std::string tier;
  std::string model_dir;
  std::string config_json;
  std::mutex mu;
  std::string last_error;
  std::shared_ptr<e0n::ModelFiles> files;
  std::shared_ptr<e0n::ExpertMap> experts;
  std::unique_ptr<e0n::PrefixCache> prefix_cache;
  std::string model_identity;
  std::unique_ptr<e0n::Net35> net35;
  std::unique_ptr<e0n::State35> st35;
  std::unique_ptr<e0n::Net8b> net8b;
  std::unique_ptr<e0n::State8b> st8b;
  std::vector<e0n::RouteRec> last_recs;
  bool has_recs = false;
  std::vector<int32_t> eos_ids;
  e0n::gen::StreamBase *active = nullptr;
};

struct CachedState35 {
  e0n::State35 state;
  std::vector<float> logits;
};

struct CachedState8 {
  e0n::State8b state;
  std::vector<float> logits;
};

namespace {

mlx::core::array snapshot_array(const mlx::core::array &a) {
  auto out = mlx::core::copy(a);
  mlx::core::eval(out);
  return out;
}

template <typename T>
std::vector<mlx::core::array> snapshot_arrays(const std::vector<mlx::core::array> &src) {
  std::vector<mlx::core::array> out;
  out.reserve(src.size());
  for (const auto &a : src) out.push_back(snapshot_array(a));
  return out;
}

std::shared_ptr<CachedState35> capture_state35(const e0n::State35 &src,
                                                const std::vector<float> &logits) {
  auto out = std::make_shared<CachedState35>();
  out->state.pr = src.pr;
  out->state.conv = snapshot_arrays<float>(src.conv);
  out->state.recur = snapshot_arrays<float>(src.recur);
  out->state.kc = snapshot_arrays<float>(src.kc);
  out->state.vc = snapshot_arrays<float>(src.vc);
  out->state.is_linear = src.is_linear;
  out->state.position = src.position;
  out->state.started = src.started;
  out->logits = logits;
  return out;
}

std::shared_ptr<CachedState8> capture_state8(const e0n::State8b &src,
                                               const std::vector<float> &logits) {
  auto out = std::make_shared<CachedState8>();
  out->state.pr = src.pr;
  out->state.cq = snapshot_arrays<float>(src.cq);
  out->state.ck = snapshot_arrays<float>(src.ck);
  out->state.cv = snapshot_arrays<float>(src.cv);
  out->state.ssm = snapshot_arrays<float>(src.ssm);
  out->state.kc = snapshot_arrays<float>(src.kc);
  out->state.vc = snapshot_arrays<float>(src.vc);
  out->state.is_mla = src.is_mla;
  out->state.position = src.position;
  out->state.started = src.started;
  out->logits = logits;
  return out;
}

void restore_state35(e0n::State35 &dst, const e0n::State35 &src) {
  dst.pr = src.pr;
  dst.conv = snapshot_arrays<float>(src.conv);
  dst.recur = snapshot_arrays<float>(src.recur);
  dst.kc = snapshot_arrays<float>(src.kc);
  dst.vc = snapshot_arrays<float>(src.vc);
  dst.is_linear = src.is_linear;
  dst.position = src.position;
  dst.started = src.started;
}

void restore_state8(e0n::State8b &dst, const e0n::State8b &src) {
  dst.pr = src.pr;
  dst.cq = snapshot_arrays<float>(src.cq);
  dst.ck = snapshot_arrays<float>(src.ck);
  dst.cv = snapshot_arrays<float>(src.cv);
  dst.ssm = snapshot_arrays<float>(src.ssm);
  dst.kc = snapshot_arrays<float>(src.kc);
  dst.vc = snapshot_arrays<float>(src.vc);
  dst.is_mla = src.is_mla;
  dst.position = src.position;
  dst.started = src.started;
}

uint64_t state_bytes(const e0n::State35 &s) {
  uint64_t n = sizeof(s);
  for (const auto *v : {&s.conv, &s.recur, &s.kc, &s.vc})
    for (const auto &a : *v) n += a.nbytes();
  for (const auto &v : s.pr.cap_hidden) n += v.size() * sizeof(float);
  for (const auto &v : s.pr.cap_exec) n += v.size() * sizeof(int64_t);
  for (const auto &v : s.pr.prev_exec) n += v.size() * sizeof(int64_t);
  return n;
}

uint64_t state_bytes(const e0n::State8b &s) {
  uint64_t n = sizeof(s);
  for (const auto *v : {&s.cq, &s.ck, &s.cv, &s.ssm, &s.kc, &s.vc})
    for (const auto &a : *v) n += a.nbytes();
  for (const auto &v : s.pr.cap_hidden) n += v.size() * sizeof(float);
  for (const auto &v : s.pr.cap_exec) n += v.size() * sizeof(int64_t);
  for (const auto &v : s.pr.prev_exec) n += v.size() * sizeof(int64_t);
  return n;
}

uint64_t snapshot_bytes(const CachedState35 &s) {
  return state_bytes(s.state) + s.logits.size() * sizeof(float);
}

uint64_t snapshot_bytes(const CachedState8 &s) {
  return state_bytes(s.state) + s.logits.size() * sizeof(float);
}

}  // namespace

struct E0Tokenizer {
  std::unique_ptr<void, void (*)(void *)> raw;
  explicit E0Tokenizer(void *h)
      : raw(h, [](void *p) { if (p) etok_destroy(p); }) {}
};

static std::string g_create_error;
static std::mutex g_tok_err_mu;

#include <unordered_map>

static std::mutex g_pool_mu;
static std::unordered_map<E0Engine *, std::unique_ptr<E0Engine>> g_engines;
static std::unordered_map<E0Tokenizer *, std::unique_ptr<E0Tokenizer>> g_toks;
static std::unordered_map<E0Stream *, std::unique_ptr<e0n::gen::StreamBase>> g_streams;

#include "genstream.hpp"

template <class MapT, class PtrT>
static PtrT pool_new(MapT &m, PtrT p) {
  std::lock_guard<std::mutex> g(g_pool_mu);
  using V = typename MapT::mapped_type::element_type;
  m.emplace(p, std::unique_ptr<V>(static_cast<V *>((void *)p)));
  return p;
}
template <class MapT, class PtrT>
static void pool_free(MapT &m, PtrT p) {
  std::lock_guard<std::mutex> g(g_pool_mu);
  m.erase(p);
}

namespace {
uint64_t logits_hash(const std::vector<float> &v) {
  uint64_t h = 1469598103934665603ull;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(float); i++) { h ^= p[i]; h *= 1099511628211ull; }
  return h;
}
}  // namespace

template <class NetT, class StateT>
E0Status test_forward_run(std::unique_ptr<NetT> &net, std::unique_ptr<StateT> &st, E0Engine *h,
                          const std::vector<int32_t> &tok, uint32_t steps, char *buf,
                          size_t buf_len, size_t *written) {
  std::string err;
  std::vector<float> lg;
  std::vector<e0n::RouteRec> recs;
  if (!net->prefill(tok, *st, lg, recs, err)) {
    h->last_error = "{\"error\":\"E-MODEL-INVALID\",\"message\":\"" + esc(err) + "\"}";
    return E0_ERR_INTERNAL;
  }
  h->last_recs = recs;
  h->has_recs = true;
  auto argmax = [&](const std::vector<float> &v) {
    size_t b = 0;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[b]) b = i;
    return static_cast<int32_t>(b);
  };
  std::string json = "{\"prefill_logits_finite\":" +
                     std::string(std::all_of(lg.begin(), lg.end(), [](float x) { return std::isfinite(x); }) ? "true" : "false") +
                     ",\"lh\":" + std::to_string(logits_hash(lg)) +
                     ",\"argmax\":" + std::to_string(argmax(lg)) + ",\"steps\":[";
  int32_t cur = argmax(lg);
  for (uint32_t s = 0; s < steps; s++) {
    if (!net->step(cur, *st, lg, recs, err)) {
      h->last_error = "{\"error\":\"E-MODEL-INVALID\",\"message\":\"" + esc(err) + "\"}";
      return E0_ERR_INTERNAL;
    }
    h->last_recs = recs;
    h->has_recs = true;
    bool fin = std::all_of(lg.begin(), lg.end(), [](float x) { return std::isfinite(x); });
    cur = argmax(lg);
    json += (s ? ",{" : "{") + std::string("\"finite\":") + (fin ? "true" : "false") +
            ",\"lh\":" + std::to_string(logits_hash(lg)) +
            ",\"argmax\":" + std::to_string(cur) + "}";
  }
  json += "]}";
  return json_out(json, buf, buf_len, written);
}

static void sweep_terminal(const char *what) {
  (void)what;
  std::lock_guard<std::mutex> pg(g_pool_mu);
  for (auto it = g_streams.begin(); it != g_streams.end();) {
    if (it->second && it->second->terminal()) it = g_streams.erase(it);
    else ++it;
  }
}

extern "C" {

#ifndef E0N_VERSION
#define E0N_VERSION "0.1.0"
#endif

uint32_t e0_abi_version(void) { return E0_ABI_VERSION; }
const char *e0_engine_version(void) { return E0N_VERSION; }

E0Engine *e0_engine_create(const char *tier_id, const char *model_dir,
                           const char *config_json) {
  auto note = [](const std::string &e) { g_create_error = e; };
  if (!tier_id || !*tier_id) {
    note("{\"error\":\"E-MODEL-INVALID\",\"message\":\"tier_id is empty\"}");
    return nullptr;
  }
  auto files = std::make_shared<e0n::ModelFiles>();
  std::string err;
  if (!files->open(model_dir ? model_dir : "", err)) {
    note(std::string("{\"error\":\"E-MODEL-PATH\",\"message\":\"") + esc(err) + "\"}");
    return nullptr;
  }
  auto experts = std::make_shared<e0n::ExpertMap>();
  if (!e0n::ExpertMap::build(*files, *experts, err)) {
    note(std::string("{\"error\":\"E-MODEL-LAYOUT\",\"message\":\"") + esc(err) + "\"}");
    return nullptr;
  }
  uint64_t budget = 2ull << 30;
  int io_threads = 3;
  try {
    auto cj = nlohmann::json::parse(config_json ? config_json : "{}", nullptr, false);
    if (!cj.is_discarded()) {
      if (cj.contains("cache_budget_bytes") && cj["cache_budget_bytes"].is_number_unsigned())
        budget = cj["cache_budget_bytes"].get<uint64_t>();
      if (cj.contains("io_threads") && cj["io_threads"].is_number_integer())
        io_threads = std::max(1, cj["io_threads"].get<int>());
    }
  } catch (...) {}
  auto *h = new E0Engine();
  h->tier = tier_id;
  h->model_dir = model_dir ? model_dir : "";
  h->config_json = config_json ? config_json : "{}";
  h->files = std::move(files);
  h->experts = std::move(experts);
  e0n::PrefixCacheConfig pcc;
  pcc.enabled = true;
  std::string identity = std::string(tier_id) + "|" + (model_dir ? model_dir : "|") +
                         "|abi=" + std::to_string(E0_ABI_VERSION);
  try {
    auto cj = nlohmann::json::parse(h->config_json, nullptr, false);
    if (!cj.is_discarded()) {
      if (cj.contains("model_identity")) identity = cj["model_identity"].dump();
      if (cj.contains("prefix_cache") && cj["prefix_cache"].is_object()) {
        const auto &pc = cj["prefix_cache"];
        if (pc.contains("enabled") && pc["enabled"].is_boolean())
          pcc.enabled = pc["enabled"].get<bool>();
        if (pc.contains("budget_bytes") && pc["budget_bytes"].is_number_unsigned())
          pcc.budget_bytes = pc["budget_bytes"].get<uint64_t>();
        if (pc.contains("min_prefix_tokens") && pc["min_prefix_tokens"].is_number_unsigned())
          pcc.min_prefix_tokens = std::max<uint32_t>(1, pc["min_prefix_tokens"].get<uint32_t>());
      }
    }
  } catch (...) {
    identity = std::string(tier_id) + "|" + (model_dir ? model_dir : "|") +
               "|abi=" + std::to_string(E0_ABI_VERSION);
  }
  h->model_identity = std::move(identity);
  h->prefix_cache = std::make_unique<e0n::PrefixCache>(pcc);
  try {
    auto cj = nlohmann::json::parse(h->config_json, nullptr, false);
    if (!cj.is_discarded() && cj.contains("forward") && cj["forward"].is_string()) {
      const std::string fwd = cj["forward"].get<std::string>();
      if (fwd == "35b" || fwd == "8b") {
        const std::string want_tier = "edge0-" + fwd;
        if (h->tier != want_tier) {
          pool_free(g_engines, h);
          note("{\"error\":\"E-MODEL-INVALID\",\"message\":\"forward=" + fwd +
               " expected tier=" + want_tier + "\"}");
          return nullptr;
        }
        std::shared_ptr<const e0n::lora::Bundle> lo;
        if (cj.contains("lora") && cj["lora"].is_boolean() && cj["lora"].get<bool>()) {
          e0n::TierRouting treg;
          if (!e0n::tier_routing(h->tier, treg)) {
            pool_free(g_engines, h);
            note("{\"error\":\"E-MODEL-INVALID\",\"message\":\"tier missing from registry\"}");
            return nullptr;
          }
          auto b = std::make_shared<e0n::lora::Bundle>();
          std::string lerr;
          if (!b->load_file(h->model_dir + "/" + treg.lora_file, fwd == "8b",
                            treg.num_layers, lerr)) {
            pool_free(g_engines, h);
            const bool missing = lerr.rfind("lora file unreadable", 0) == 0;
            note(std::string("{\"error\":\"") + (missing ? "E-MODEL-INVALID" : "E-MODEL-LAYOUT") +
                 "\",\"message\":\"lora: " + esc(lerr) + "\"}");
            return nullptr;
          }
          lo = b;
        }
        std::string ferr;
        if (fwd == "35b") {
          h->net35 = e0n::Net35::create(h->files, h->experts, ferr, lo);
          if (!h->net35) {
            pool_free(g_engines, h);
            note(std::string("{\"error\":\"E-MODEL-LAYOUT\",\"message\":\"forward ctx: ") +
                 esc(ferr) + "\"}");
            return nullptr;
          }
          h->st35 = std::make_unique<e0n::State35>();
          h->net35->init_state(*h->st35);
        } else {
          h->net8b = e0n::Net8b::create(h->files, h->experts, ferr, lo);
          if (!h->net8b) {
            pool_free(g_engines, h);
            note(std::string("{\"error\":\"E-MODEL-LAYOUT\",\"message\":\"forward ctx: ") +
                 esc(ferr) + "\"}");
            return nullptr;
          }
          h->st8b = std::make_unique<e0n::State8b>();
          h->net8b->init_state(*h->st8b);
        }
        std::ifstream cf(h->model_dir + "/config.json");
        if (cf) {
          auto cj2 = nlohmann::json::parse(cf, nullptr, false);
          if (!cj2.is_discarded() && cj2.contains("eos_token_id")) {
            const auto &x = cj2["eos_token_id"];
            if (x.is_number_integer()) {
              h->eos_ids.push_back(x.get<int32_t>());
            } else if (x.is_array()) {
              for (const auto &e : x)
                if (e.is_number_integer()) h->eos_ids.push_back(e.get<int32_t>());
            }
          }
        }
      }
    }
  } catch (const std::exception &e) {
    note(std::string("{\"error\":\"E-INTERNAL\",\"message\":\"create: ") +
         esc(e.what()) + "\"}");
    delete h;
    return nullptr;
  }
  return pool_new(g_engines, h);
}

E0Stream *e0_generate_begin(E0Engine *h, const int32_t *prompt_tokens, uint32_t n_tokens,
                            const char *params_json) {
  if (!h) return nullptr;
  {
    std::lock_guard<std::mutex> pg(g_pool_mu);
    if (!g_engines.count(h)) return nullptr;
  }
  sweep_terminal("begin");
  std::lock_guard<std::mutex> g(h->mu);
  e0n::sampling::TierDefaults td = e0n::sampling::tier_defaults(h->tier);
  auto p = e0n::sampling::parse_params(params_json ? params_json : "{}", td, h->eos_ids);
  bool transport_only = false;
  if (p.teacher_tokens.empty()) {
    auto pv0 = nlohmann::json::parse(params_json ? params_json : "{}", nullptr, false);
    if (!pv0.is_discarded() && pv0.contains("replay_tokens") && pv0["replay_tokens"].is_array()) {
      for (const auto &it : pv0["replay_tokens"])
        if (it.is_number_integer()) p.teacher_tokens.push_back(it.get<int32_t>());
      transport_only = !p.teacher_tokens.empty();
    }
  }
  if (params_json) {
    auto pv = nlohmann::json::parse(params_json, nullptr, false);
    if (!pv.is_discarded() && pv.contains("stop") && pv["stop"].is_array()) {
      for (const auto &it : pv["stop"]) {
        if (!it.is_string() || it.get_ref<const std::string &>().empty()) continue;
        void *tk = etok_create(h->model_dir.c_str());
        if (!tk) {
          h->last_error = etok_err_json();
          return nullptr;
        }
        std::vector<int32_t> ids;
        size_t w = 0;
        const std::string &str = it.get_ref<const std::string &>();
        if (etok_tokenize(tk, str.c_str(), nullptr, 0, &w) == 0) {
          ids.resize(w);
          if (w) etok_tokenize(tk, str.c_str(), ids.data(), w, &w);
        }
        etok_destroy(tk);
        p.stop_seqs.push_back(std::move(ids));
      }
    }
  }
  std::vector<int32_t> prompt(prompt_tokens, prompt_tokens + n_tokens);
  const size_t prompt_len = prompt.size();
  e0n::gen::StreamBase *ns;
#ifdef E0N_TEST_SCAFFOLD
  if (const char *sc = getenv("E0N_SCRIPT")) {
    std::vector<std::vector<float>> script;
    const int n = atoi(sc);
    for (int i = 0; i < n; i++) {
      std::vector<float> lg(8, -10.f);
      lg[i % 8] = 5.f;
      script.push_back(std::move(lg));
    }
    ns = new e0n::gen::ScriptedStream(std::move(prompt), std::move(p), std::move(script));
    ns->set_owner(h);
    if (h->active) h->active->seal();
    h->active = ns;
    return pool_new(g_streams, (E0Stream *)ns);
  }
#endif
  if (transport_only && (h->net35 || h->net8b)) {
    ns = new e0n::gen::StreamBase(std::move(prompt), std::move(p));
    ns->set_transport();
    ns->set_owner(h);
    if (h->active) h->active->seal();
    h->active = ns;
    return pool_new(g_streams, (E0Stream *)ns);
  }
  if (!h->net35 && !h->net8b) {
    const bool empty_turn = p.teacher_tokens.empty() && prompt.empty();
    ns = new e0n::gen::StreamBase(std::move(prompt), std::move(p));
    ns->set_transport();
    ns->set_owner(h);
    if (empty_turn) ns->set_empty_done();
    if (h->active) h->active->seal();
    h->active = ns;
    return pool_new(g_streams, (E0Stream *)ns);
  }
  std::vector<int32_t> prefill_prompt = prompt;
  size_t cached_tokens = 0;
  if (h->net35) {
    auto hit = h->prefix_cache->lookup(h->model_identity, prompt);
    if (hit.snapshot) {
      try {
        auto snap = std::static_pointer_cast<CachedState35>(hit.snapshot);
        restore_state35(*h->st35, snap->state);
        cached_tokens = hit.matched_tokens;
        prefill_prompt.assign(prompt.begin() + static_cast<std::ptrdiff_t>(cached_tokens), prompt.end());
      } catch (const std::exception &e) {
        h->prefix_cache->clear();
        h->last_error = std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") +
                        esc(e.what()) + "\"}";
        return nullptr;
      }
    } else {
      h->net35->reset_state(*h->st35);
    }
    ns = new e0n::gen::Stream<e0n::Net35, e0n::State35>(
        h->net35.get(), h->st35.get(),
        [net = h->net35.get()](const e0n::Step &plan) { net->prefetch(plan); }, h,
        std::move(prompt), std::move(p));
    ns->set_prefill_prompt(std::move(prefill_prompt));
    if (cached_tokens == prompt_len && hit.snapshot) {
      auto snap = std::static_pointer_cast<CachedState35>(hit.snapshot);
      ns->set_cached_logits(snap->logits);
    }
    ns->set_on_prefill([h](const std::vector<int32_t> &full, const std::vector<float> &logits) {
      try {
        auto snap = capture_state35(*h->st35, logits);
        h->prefix_cache->put(h->model_identity, full, snap, snapshot_bytes(*snap));
      } catch (const std::exception &e) {
        h->last_error = std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") +
                        esc(e.what()) + "\"}";
      }
    });
    ns->set_on_complete([h](const std::vector<int32_t> &full, const std::vector<float> &logits) {
      try {
        auto snap = capture_state35(*h->st35, logits);
        h->prefix_cache->put(h->model_identity, full, snap, snapshot_bytes(*snap));
      } catch (const std::exception &e) {
        h->last_error = std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") +
                        esc(e.what()) + "\"}";
      }
    });
  } else {
    auto hit = h->prefix_cache->lookup(h->model_identity, prompt);
    if (hit.snapshot) {
      try {
        auto snap = std::static_pointer_cast<CachedState8>(hit.snapshot);
        restore_state8(*h->st8b, snap->state);
        cached_tokens = hit.matched_tokens;
        prefill_prompt.assign(prompt.begin() + static_cast<std::ptrdiff_t>(cached_tokens), prompt.end());
      } catch (const std::exception &e) {
        h->prefix_cache->clear();
        h->last_error = std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") +
                        esc(e.what()) + "\"}";
        return nullptr;
      }
    } else {
      h->net8b->reset_state(*h->st8b);
    }
    ns = new e0n::gen::Stream<e0n::Net8b, e0n::State8b>(
        h->net8b.get(), h->st8b.get(),
        [net = h->net8b.get()](const e0n::Step &plan) { net->prefetch(plan); }, h,
        std::move(prompt), std::move(p));
    ns->set_prefill_prompt(std::move(prefill_prompt));
    if (cached_tokens == prompt_len && hit.snapshot) {
      auto snap = std::static_pointer_cast<CachedState8>(hit.snapshot);
      ns->set_cached_logits(snap->logits);
    }
    ns->set_on_prefill([h](const std::vector<int32_t> &full, const std::vector<float> &logits) {
      try {
        auto snap = capture_state8(*h->st8b, logits);
        h->prefix_cache->put(h->model_identity, full, snap, snapshot_bytes(*snap));
      } catch (const std::exception &e) {
        h->last_error = std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") +
                        esc(e.what()) + "\"}";
      }
    });
    ns->set_on_complete([h](const std::vector<int32_t> &full, const std::vector<float> &logits) {
      try {
        auto snap = capture_state8(*h->st8b, logits);
        h->prefix_cache->put(h->model_identity, full, snap, snapshot_bytes(*snap));
      } catch (const std::exception &e) {
        h->last_error = std::string("{\"error\":\"E-INTERNAL\",\"message\":\"") +
                        esc(e.what()) + "\"}";
      }
    });
  }
  h->prefix_cache->set_last_prefill_tokens(prompt_len - cached_tokens);
  if (h->active) h->active->seal();
  h->active = ns;
  return pool_new(g_streams, (E0Stream *)ns);
}

namespace {
e0n::gen::StreamBase *checked_stream(E0Engine *h, E0Stream *stream) {
  if (!h || !stream) return nullptr;
  std::lock_guard<std::mutex> pg(g_pool_mu);
  if (!g_engines.count(h)) return nullptr;
  auto it = g_streams.find(stream);
  if (it == g_streams.end() || !it->second || !it->second->owner_of(h)) return nullptr;
  return it->second.get();
}
}  // namespace

E0Status e0_next(E0Engine *h, E0Stream *stream, E0TokenOut *out) {
  auto *st = checked_stream(h, stream);
  if (!st) return E0_ERR_INVALID;
  std::lock_guard<std::mutex> g(st->mu);
  if (out) {
    if (out->size < sizeof(E0TokenOut)) return E0_ERR_INVALID;
    out->token = 0;
    out->flags = 0;
  }
  int32_t tok = 0;
  const bool got = st->next_token(tok);
  if (got) {
    if (out) out->token = tok;
    return E0_NOTIFIED;
  }
  const bool errored = st->errored();
  const std::string em = st->err_msg();
  {
    std::lock_guard<std::mutex> pg(g_pool_mu);
    if (st->owner()) {
      st->owner()->last_recs = st->last_recs();
      st->owner()->has_recs = true;
      if (errored) {
        st->owner()->last_error =
            "{\"error\":\"E-INTERNAL\",\"message\":" + esc(em) + "}";
        if (st->owner()->active == st) st->owner()->active = nullptr;
        g_streams.erase(stream);
      }
    }
  }
  if (errored) return E0_ERR_INTERNAL;
  if (out) out->flags = E0_TOKEN_FINISHED;
  return E0_NOTIFIED;
}

E0Status e0_cancel(E0Engine *h, E0Stream *stream) {
  if (!h) return E0_ERR_INVALID;
  auto *st = checked_stream(h, stream);
  if (!st) return E0_OK;
  std::lock_guard<std::mutex> g(st->mu);
  st->cancel();
  return E0_OK;
}

E0Status e0_stats(E0Engine *h, char *buf, size_t buf_len, size_t *written) {
  if (!h) return E0_ERR_INVALID;
  e0n::MemSample m = e0n::mem_sample();
  e0n::ExpertIo::Metrics c{};
  if (h->net35) c = h->net35->io_metrics();
  else if (h->net8b) c = h->net8b->io_metrics();
  std::string route = "{\"route_source_by_layer\":{";
  {
    std::lock_guard<std::mutex> g(h->mu);
    if (h->has_recs) {
      for (size_t i = 0; i < h->last_recs.size(); i++) {
        if (i) route += ",";
        route += "\"" + std::to_string(h->last_recs[i].layer) + "\":\"" +
                 (h->last_recs[i].predicted ? "predicted" : "routed") + "\"";
      }
    }
  }
  route += "}";
  auto cx = e0n::exec_counters().snapshot();
  std::string diag = ",\"diag\":{\"qmm_calls\":" + std::to_string(cx.qmm_calls) +
                     ",\"qmm_experts\":" + std::to_string(cx.qmm_experts) +
                     ",\"qmm_weight_bytes\":" + std::to_string(cx.qmm_weight_bytes) +
                     ",\"host_syncs\":" + std::to_string(cx.host_syncs) +
                     ",\"host_wraps\":" + std::to_string(cx.host_wraps) +
                     ",\"leases\":" + std::to_string(cx.leases) + "}";
  const auto pc = h->prefix_cache ? h->prefix_cache->metrics() : e0n::PrefixCacheMetrics{};
  std::string prefix = ",\"prefix_cache\":{\"enabled\":" +
                       std::string(pc.enabled ? "true" : "false") +
                       ",\"budget_bytes\":" + std::to_string(pc.budget_bytes) +
                       ",\"min_prefix_tokens\":" + std::to_string(pc.min_prefix_tokens) +
                       ",\"hits\":" + std::to_string(pc.hits) +
                       ",\"misses\":" + std::to_string(pc.misses) +
                       ",\"cached_tokens\":" + std::to_string(pc.cached_tokens) +
                       ",\"entries\":" + std::to_string(pc.entries) +
                       ",\"bytes\":" + std::to_string(pc.bytes) +
                       ",\"evictions\":" + std::to_string(pc.evictions) +
                       ",\"last_cached_tokens\":" + std::to_string(pc.last_cached_tokens) +
                       ",\"last_prefill_tokens\":" + std::to_string(pc.last_prefill_tokens) +
                       ",\"last_reason\":\"" + esc(pc.last_reason) + "\"}";
  std::string json =
      route + diag + prefix + ",\"loaded_experts\":" + std::to_string(c.loads) +
      ",\"cache_hits\":" + std::to_string(c.hits) +
      ",\"cache_misses\":" + std::to_string(c.misses) +
      ",\"cache_evictions\":" + std::to_string(c.evictions) +
      ",\"memory\":{\"resident_peak_bytes\":" + std::to_string(m.resident_peak_bytes) +
      ",\"footprint_bytes\":" + std::to_string(m.footprint_bytes) +
      ",\"expert_cache_heap_bytes\":" + std::to_string(c.heap_bytes) +
      ",\"file_backed_bytes\":0},"
      "\"tier\":\"" + esc(h->tier) + "\"}";
  return json_out(json, buf, buf_len, written);
}

E0Status e0_turn_reset(E0Engine *h) {
  if (!h) return E0_ERR_INVALID;
  std::lock_guard<std::mutex> g(h->mu);
  if (h->active) {
    h->active->seal();
    h->active = nullptr;
  }
  sweep_terminal("turn_reset");
  if (h->net35 && h->st35) {
    h->net35->reset_state(*h->st35);
    h->last_recs.clear();
    h->has_recs = false;
  }
  if (h->net8b && h->st8b) {
    h->net8b->reset_state(*h->st8b);
    h->last_recs.clear();
    h->has_recs = false;
  }
  h->last_error.clear();
  return E0_OK;
}

E0Status e0_last_error(E0Engine *h, char *buf, size_t buf_len, size_t *written) {
  std::string msg;
  if (h) {
    std::lock_guard<std::mutex> g(h->mu);
    msg = h->last_error.empty()
              ? std::string("{\"error\":\"E-INTERNAL\",\"message\":\"no error recorded\"}")
              : h->last_error;
  } else {
    std::lock_guard<std::mutex> g(g_tok_err_mu);
    msg = g_create_error.empty()
              ? std::string("{\"error\":\"E-INTERNAL\",\"message\":\"no error recorded\"}")
              : g_create_error;
  }
  return json_out(msg, buf, buf_len, written);
}

void e0_engine_destroy(E0Engine *h) {
  if (!h) return;
  {
    std::lock_guard<std::mutex> pg(g_pool_mu);
    for (auto it = g_streams.begin(); it != g_streams.end();) {
      if (it->second && it->second->owner_of(h))
        it = g_streams.erase(it);
      else
        ++it;
    }
  }
  pool_free(g_engines, h);
}

extern "C++" {
namespace {

void tok_err(const std::string &msg) {
  std::lock_guard<std::mutex> g(g_tok_err_mu);
  g_create_error = msg;
}

template <class F>
bool etok_out(F f, std::string &out, const char *fail_what) {
  size_t w = 0;
  if (f(nullptr, 0, &w) != 0) {
    tok_err(etok_err_json());
    return false;
  }
  out.assign(w + 1, '\0');
  if (w && f(out.data(), w, &w) != 0) {
    tok_err(etok_err_json());
    return false;
  }
  out.resize(w);
  return true;
}

bool etok_ids(void *raw, const std::string &text, std::vector<int32_t> &ids) {
  size_t n = 0;
  if (etok_tokenize(raw, text.c_str(), nullptr, 0, &n) != 0) {
    tok_err(etok_err_json());
    return false;
  }
  ids.assign(n, 0);
  if (n && etok_tokenize(raw, text.c_str(), ids.data(), n, &n) != 0) {
    tok_err(etok_err_json());
    return false;
  }
  return true;
}

}  // namespace
}  // extern "C++"

E0Tokenizer *e0_tok_create(const char *model_dir) {
  if (!is_dir(model_dir)) {
    tok_err(std::string("{\"error\":\"E-MODEL-PATH\",\"message\":\"directory does not exist: ") +
            esc(model_dir ? model_dir : "") + "\"}");
    return nullptr;
  }
  void *raw = etok_create(model_dir);
  if (!raw) {
    tok_err(etok_err_json());
    return nullptr;
  }
  return pool_new(g_toks, new E0Tokenizer(raw));
}

E0Status e0_tok_tokenize(E0Tokenizer *t, const char *text_json, char *buf, size_t buf_len,
                         size_t *written) {
  if (!t) return E0_ERR_INVALID;
  nlohmann::json v = nlohmann::json::parse(text_json ? text_json : "", nullptr, false);
  if (v.is_discarded() || !v.is_object()) {
    tok_err("{\"error\":\"E-INVALID-ARG\",\"message\":\"invalid text_json\"}");
    return E0_ERR_INTERNAL;
  }
  std::vector<std::vector<int32_t>> batches;
  if (v.contains("text") && v["text"].is_string()) {
    std::vector<int32_t> ids;
    if (!etok_ids(t->raw.get(), v["text"].get<std::string>(), ids)) return E0_ERR_INTERNAL;
    batches.push_back(std::move(ids));
  } else if (v.contains("texts") && v["texts"].is_array()) {
    for (const auto &item : v["texts"]) {
      if (!item.is_string()) {
        tok_err("{\"error\":\"E-INVALID-ARG\",\"message\":\"texts contains a non-string element\"}");
        return E0_ERR_INTERNAL;
      }
      std::vector<int32_t> ids;
      if (!etok_ids(t->raw.get(), item.get<std::string>(), ids)) return E0_ERR_INTERNAL;
      batches.push_back(std::move(ids));
    }
  } else {
    tok_err("{\"error\":\"E-INVALID-ARG\",\"message\":\"need text or texts\"}");
    return E0_ERR_INTERNAL;
  }
  nlohmann::json out = {{"tokens", batches}};
  return json_out(out.dump(), buf, buf_len, written);
}

E0Status e0_tok_apply_template(E0Tokenizer *t, const char *messages_json, char *buf,
                               size_t buf_len, size_t *written) {
  if (!t) return E0_ERR_INVALID;
  nlohmann::json v = nlohmann::json::parse(messages_json ? messages_json : "", nullptr, false);
  if (v.is_discarded()) {
    tok_err("{\"error\":\"E-INVALID-ARG\",\"message\":\"invalid messages_json\"}");
    return E0_ERR_INTERNAL;
  }
  nlohmann::json ctx = nlohmann::json::object();
  bool agp = true;
  if (v.is_array()) {
    ctx["messages"] = v;
  } else {
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (it.key() == "add_generation_prompt" && it.value().is_boolean())
        agp = it.value().get<bool>();
      else
        ctx[it.key()] = it.value();
    }
    if (!ctx.contains("messages")) ctx["messages"] = nlohmann::json::array();
  }
  ctx["add_generation_prompt"] = agp;
  std::string cj = ctx.dump();
  void *raw = t->raw.get();
  std::string prompt;
  if (!etok_out([&](char *b, size_t l, size_t *w) {
        return etok_render(raw, cj.c_str(), b, l, w);
      },
      prompt, "render"))
    return E0_ERR_INTERNAL;
  std::vector<int32_t> ids;
  if (!etok_ids(raw, prompt, ids)) return E0_ERR_INTERNAL;
  nlohmann::json out = {{"prompt", prompt}, {"token_count", ids.size()}};
  return json_out(out.dump(), buf, buf_len, written);
}

E0Status e0_tok_decode(E0Tokenizer *t, const char *tokens_json, char *buf, size_t buf_len,
                       size_t *written) {
  if (!t) return E0_ERR_INVALID;
  nlohmann::json v = nlohmann::json::parse(tokens_json ? tokens_json : "", nullptr, false);
  if (v.is_discarded() || !v.is_object() || !v.contains("tokens") || !v["tokens"].is_array()) {
    tok_err("{\"error\":\"E-INVALID-ARG\",\"message\":\"need {\\\"tokens\\\":[...]} \"}");
    return E0_ERR_INTERNAL;
  }
  std::vector<int32_t> ids;
  for (const auto &x : v["tokens"]) {
    if (!x.is_number_integer()) {
      tok_err("{\"error\":\"E-INVALID-ARG\",\"message\":\"tokens contains a non-integer\"}");
      return E0_ERR_INTERNAL;
    }
    ids.push_back(x.get<int32_t>());
  }
  void *raw = t->raw.get();
  std::string text;
  if (!etok_out([&](char *b, size_t l, size_t *w) {
        return etok_decode(raw, ids.data(), ids.size(), b, l, w);
      },
      text, "decode"))
    return E0_ERR_INTERNAL;
  nlohmann::json out = {{"text", text}};
  return json_out(out.dump(), buf, buf_len, written);
}

void e0_tok_destroy(E0Tokenizer *t) { pool_free(g_toks, t); }

E0Status e0_test_forward(E0Engine *h, const int32_t *tokens, uint32_t n_tokens,
                         uint32_t steps, char *buf, size_t buf_len, size_t *written) {
  if (!h) return E0_ERR_INVALID;
  std::lock_guard<std::mutex> g(h->mu);
  if ((!h->net35 || !h->st35) && (!h->net8b || !h->st8b)) {
    h->last_error = "{\"error\":\"E-INVALID-ARG\",\"message\":\"forward context not enabled (config {\"forward\":\"35b\"|\"8b\"})\"}";
    return E0_ERR_INVALID;
  }
  std::vector<int32_t> tok(tokens, tokens + n_tokens);
  if (h->net35) return test_forward_run(h->net35, h->st35, h, tok, steps, buf, buf_len, written);
  return test_forward_run(h->net8b, h->st8b, h, tok, steps, buf, buf_len, written);
}

}  // extern "C"
