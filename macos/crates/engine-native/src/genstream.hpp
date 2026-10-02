// Generation stream: ABI wrapper around net prefill/step plus sampling.
#pragma once
#include <cstdint>
#include <mutex>
#include <functional>
#include <string>
#include <vector>

#include <algorithm>

#include "net35.hpp"
#include "prerouter.hpp"
#include "sampling.hpp"
#include "provider.hpp"

struct E0Engine;

namespace e0n { namespace gen {

class StreamBase {
 public:
  StreamBase(const StreamBase &) = delete;
  StreamBase &operator=(const StreamBase &) = delete;
  virtual ~StreamBase() = default;
  std::mutex mu;  // stream-level; the engine handle may die first so the lock is not on E0Engine
  bool next_token(int32_t &tok_out) {
    if (finished_) return false;
    if (cancelled_) { finished_ = true; return false; }
    if (generated_ >= p_.max_new_tokens) { finish_natural(); return false; }
    std::string err;
    std::vector<float> lg;
    std::vector<RouteRec> recs;
    if (exhausted()) { finished_ = true; return false; }
    if (transport_) {
      if (generated_ >= p_.teacher_tokens.size() || generated_ >= p_.max_new_tokens) {
        finish_natural();
        return false;
      }
      tok_out = p_.teacher_tokens[generated_++];
      hist_.push_back(tok_out);
      emitted_ = true;
      return true;
    }
    if (!started_) {
      if (prefill_prompt_.empty() && cached_logits_set_) {
        lg = cached_logits_;
      } else {
        if (!prefill(prefill_prompt_, lg, recs, err)) { note_err(err); return false; }
        if (on_prefill_) on_prefill_(prompt_, lg);
      }
      last_logits_ = lg;
      last_recs_ = recs;
      cur_ = p_.teacher_tokens.empty() ? host_argmax(lg) : p_.teacher_tokens[0];
      started_ = true;
    } else {
      if (warm_) {
        Step plan = live_next_step(plan_state_());
        if (!plan.experts.empty()) warm_(plan);
      }
      if (!step(cur_, lg, recs, err)) { note_err(err); return false; }
      last_logits_ = lg;
      last_recs_ = recs;
    }
    int32_t t;
    if (!p_.teacher_tokens.empty()) {
      if (generated_ >= p_.teacher_tokens.size()) { finish_natural(); return false; }
      t = p_.teacher_tokens[generated_];
    } else {
      t = sampling::sample_step(lg, p_, !emitted_, hist_, rng_);
    }
    emitted_ = true;
    // EOS / stop-id: terminate without emitting that token.
    for (int32_t e : p_.eos_ids)
      if (t == e) { finish_natural(); return false; }
    for (int32_t e : p_.stop_token_ids)
      if (t == e) { finish_natural(); return false; }
    hist_.push_back(t);
    // String stop: subsequence ending at this token, wholly in the generated region,
    // rolls back the matched span. hist_ = prompt + emitted (incl. this t);
    // generated-region start gp = hist_.size() - generated_ - 1.
    {
      const size_t gp = hist_.size() - generated_ - 1;
      for (const auto &seq : p_.stop_seqs) {
        if (seq.empty() || seq.size() > generated_ + 1) continue;
        const size_t base = hist_.size() - seq.size();
        if (base < gp) continue;
        bool hit = true;
        for (size_t i = 0; i < seq.size(); i++)
          if (hist_[base + i] != seq[i]) { hit = false; break; }
        if (hit) {
          generated_ -= (seq.size() - 1);
          hist_.resize(base);
          finish_natural();
          return false;
        }
      }
    }
    generated_++;
    cur_ = t;
    tok_out = t;
    return true;
  }
  void cancel() { cancelled_ = true; }
  void set_transport() { transport_ = true; }
  void set_empty_done() { finished_ = true; }
  void set_prefill_prompt(std::vector<int32_t> ids) { prefill_prompt_ = std::move(ids); }
  void set_on_prefill(std::function<void(const std::vector<int32_t> &)> cb) {
    on_prefill_ = [cb = std::move(cb)](const std::vector<int32_t> &ids,
                                       const std::vector<float> &) { cb(ids); };
  }
  void set_on_prefill(std::function<void(const std::vector<int32_t> &,
                                         const std::vector<float> &)> cb) {
    on_prefill_ = std::move(cb);
  }
  void set_cached_logits(std::vector<float> logits) {
    cached_logits_ = std::move(logits);
    cached_logits_set_ = true;
  }
  void set_on_complete(std::function<void(const std::vector<int32_t> &,
                                          const std::vector<float> &)> cb) {
    on_complete_ = std::move(cb);
  }
  void set_on_complete(std::function<void(const std::vector<int32_t> &)> cb) {
    on_complete_ = [cb = std::move(cb)](const std::vector<int32_t> &ids,
                                        const std::vector<float> &) { cb(ids); };
  }
  void set_owner(E0Engine *h) { owner_ = h; }
  bool finished() const { return finished_; }
  bool errored() const { return err_; }
  const std::string &err_msg() const { return err_msg_; }
  void seal() { finished_ = true; }
  const std::vector<RouteRec> &last_recs() const { return last_recs_; }
  bool owner_of(E0Engine *h) const { return owner_ == h; }
  E0Engine *owner() { return owner_; }
  bool started() const { return started_; }
  size_t generated() const { return generated_; }
  bool terminal() const { return finished_ && !err_; }
  const std::vector<int32_t> &history() const { return hist_; }
  StreamBase(std::vector<int32_t> prompt, sampling::Params p)
      : prompt_(std::move(prompt)), prefill_prompt_(prompt_), p_(std::move(p)), hist_(prompt_) {}

 protected:
  virtual bool prefill(const std::vector<int32_t> &, std::vector<float> &,
                       std::vector<RouteRec> &, std::string &err) {
    err = "stream without forward context";
    return false;
  }
  virtual bool step(int32_t, std::vector<float> &, std::vector<RouteRec> &,
                    std::string &err) {
    err = "stream without forward context";
    return false;
  }
  virtual const PrerouterState &plan_state_() const { static PrerouterState k; return k; }
  virtual bool exhausted() const { return false; }
  void note_err(std::string m) { err_ = true; err_msg_ = std::move(m); finished_ = true; }
  void finish_natural() {
    if (!finished_) {
      finished_ = true;
      if (!completion_called_ && on_complete_) {
        completion_called_ = true;
        on_complete_(hist_, last_logits_);
      }
    }
  }
  static int32_t host_argmax(const std::vector<float> &v) {
    size_t b = 0;
    for (size_t i = 1; i < v.size(); i++)
      if (v[i] > v[b]) b = i;  // ties take the smaller id (strict >)
    return (int32_t)b;
  }

  sampling::Params p_;
  std::vector<int32_t> prompt_;
  std::vector<int32_t> prefill_prompt_;
  std::vector<int32_t> hist_;
  size_t generated_ = 0;
  bool emitted_ = false;
  bool cancelled_ = false;
  bool finished_ = false;
  bool err_ = false;
  std::string err_msg_;
  sampling::Rng rng_{};
  std::function<void(const Step &)> warm_;
  std::function<void(const std::vector<int32_t> &, const std::vector<float> &)> on_prefill_;
  std::function<void(const std::vector<int32_t> &, const std::vector<float> &)> on_complete_;
  std::vector<float> cached_logits_;
  std::vector<float> last_logits_;
  bool cached_logits_set_ = false;
  bool completion_called_ = false;
  E0Engine *owner_ = nullptr;  // destroy scans the pool and drops streams of a dead engine
  std::vector<RouteRec> last_recs_;

 private:
  bool started_ = false;
  bool transport_ = false;
  int32_t cur_ = 0;
};

template <class NetT, class StateT>
class Stream : public StreamBase {
 public:
  Stream(NetT *net, StateT *st, std::function<void(const Step &)> warm, E0Engine *owner,
         std::vector<int32_t> prompt, sampling::Params p)
      : StreamBase(std::move(prompt), std::move(p)), net_(net), st_(st) {
    this->warm_ = std::move(warm);
    this->owner_ = owner;
  }

 protected:
  bool prefill(const std::vector<int32_t> &ids, std::vector<float> &lg,
               std::vector<RouteRec> &recs, std::string &err) override {
    return net_->prefill(ids, *st_, lg, recs, err);
  }
  bool step(int32_t tok, std::vector<float> &lg, std::vector<RouteRec> &recs,
            std::string &err) override {
    return net_->step(tok, *st_, lg, recs, err);
  }
  const PrerouterState &plan_state_() const override { return st_->pr; }

 private:
  NetT *net_;
  StateT *st_;
};

class ScriptedStream : public StreamBase {
 public:
  ScriptedStream(std::vector<int32_t> prompt, sampling::Params p,
                 std::vector<std::vector<float>> script)
      : StreamBase(std::move(prompt), std::move(p)), script_(std::move(script)) {}

 protected:
  bool prefill(const std::vector<int32_t> &, std::vector<float> &lg,
               std::vector<RouteRec> &recs, std::string &) override {
    recs.clear();
    lg = script_[0];
    return true;
  }
  bool step(int32_t, std::vector<float> &lg, std::vector<RouteRec> &recs,
            std::string &) override {
    recs.clear();
    idx_++;
    lg = script_[idx_];
    return true;
  }
  bool exhausted() const override { return idx_ + 1 >= script_.size(); }

 private:
  std::vector<std::vector<float>> script_;
  size_t idx_ = 0;
};

}}  // namespace e0n::gen
