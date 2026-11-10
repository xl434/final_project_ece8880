#include "gpt2_cpu.h"

#include <cstdio>
#include <cstdlib>

#include "../common/utils.h"
#include "ops.h"

namespace gpt2 {

const char* const kOpNames[kNumOps] = {
    "embed", "ln1", "qkv_matmul", "kv_write", "attention", "attproj_matmul", "residual1",
    "ln2", "fc_matmul", "gelu", "fcproj_matmul", "residual2", "ln_f", "lm_head", "argmax"};

State::State()
    : kcache((size_t)kL * kMaxT * kC), vcache((size_t)kL * kMaxT * kC),
      x((size_t)kMaxT * kC), ln((size_t)kMaxT * kC), qkv((size_t)kMaxT * kC3),
      atty((size_t)kMaxT * kC), attproj((size_t)kMaxT * kC), res1((size_t)kMaxT * kC),
      ln2((size_t)kMaxT * kC), fch((size_t)kMaxT * kC4), fch_gelu((size_t)kMaxT * kC4),
      fcproj((size_t)kMaxT * kC), lnf(kC), logits(kVp) {}

namespace {

// Times one operator call when a profiler is attached.
struct Scope {
  Profiler* p;
  Op op;
  Timer t;
  Scope(Profiler* p_, Op op_) : p(p_), op(op_) {}
  ~Scope() {
    if (p) {
      p->ms[op] += t.ms();
      p->calls[op] += 1;
    }
  }
};

void record(Trace* tr, const std::string& name, const float* last_row, int n) {
  if (tr) tr->t[name].assign(last_row, last_row + n);
}

std::string lname(int l, const char* s) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "l%02d_%s", l, s);
  return buf;
}

}  // namespace

int forward(const Weights& w, State& st, const int* tokens, int T, Profiler* prof, Trace* tr) {
  if (T <= 0 || st.pos + T > kMaxT) {
    std::fprintf(stderr, "forward: context overflow (pos %d + T %d > %d)\n", st.pos, T, kMaxT);
    std::exit(EXIT_FAILURE);
  }
  const int pos0 = st.pos;
  const long lastC = (long)(T - 1) * kC;
  float* x = st.x.data();  // residual stream (T, C)
  {
    Scope s(prof, kOpEmbed);
    encoder_forward(x, tokens, T, pos0, w.wte, w.wpe, kC);
  }
  record(tr, "embed", x + lastC, kC);

  for (int l = 0; l < kL; l++) {
    { Scope s(prof, kOpLn1); layernorm_forward(st.ln.data(), x, w.l_ln1w(l), w.l_ln1b(l), T, kC); }
    { Scope s(prof, kOpQkv); matmul_forward(st.qkv.data(), st.ln.data(), w.l_qkvw(l), w.l_qkvb(l), T, kC, kC3); }
    { Scope s(prof, kOpKvWrite); kv_cache_write(st.k(l), st.v(l), st.qkv.data(), T, pos0, kC); }
    { Scope s(prof, kOpAttn); attention_forward(st.atty.data(), st.qkv.data(), st.k(l), st.v(l), T, pos0, kC, kNH); }
    { Scope s(prof, kOpAttProj); matmul_forward(st.attproj.data(), st.atty.data(), w.l_attprojw(l), w.l_attprojb(l), T, kC, kC); }
    { Scope s(prof, kOpRes1); residual_forward(st.res1.data(), x, st.attproj.data(), T * kC); }
    { Scope s(prof, kOpLn2); layernorm_forward(st.ln2.data(), st.res1.data(), w.l_ln2w(l), w.l_ln2b(l), T, kC); }
    { Scope s(prof, kOpFc); matmul_forward(st.fch.data(), st.ln2.data(), w.l_fcw(l), w.l_fcb(l), T, kC, kC4); }
    { Scope s(prof, kOpGelu); gelu_forward(st.fch_gelu.data(), st.fch.data(), T * kC4); }
    { Scope s(prof, kOpFcProj); matmul_forward(st.fcproj.data(), st.fch_gelu.data(), w.l_fcprojw(l), w.l_fcprojb(l), T, kC4, kC); }
    { Scope s(prof, kOpRes2); residual_forward(x, st.res1.data(), st.fcproj.data(), T * kC); }
    if (tr) {
      record(tr, lname(l, "ln1"), st.ln.data() + lastC, kC);
      record(tr, lname(l, "qkv"), st.qkv.data() + (long)(T - 1) * kC3, kC3);
      record(tr, lname(l, "atty"), st.atty.data() + lastC, kC);
      record(tr, lname(l, "attproj"), st.attproj.data() + lastC, kC);
      record(tr, lname(l, "res1"), st.res1.data() + lastC, kC);
      record(tr, lname(l, "ln2"), st.ln2.data() + lastC, kC);
      record(tr, lname(l, "fch"), st.fch.data() + (long)(T - 1) * kC4, kC4);
      record(tr, lname(l, "fch_gelu"), st.fch_gelu.data() + (long)(T - 1) * kC4, kC4);
      record(tr, lname(l, "fcproj"), st.fcproj.data() + lastC, kC);
      record(tr, lname(l, "res2"), x + lastC, kC);
    }
  }
  // Inference only needs the last token's logits.
  { Scope s(prof, kOpLnf); layernorm_forward(st.lnf.data(), x + lastC, w.lnfw, w.lnfb, 1, kC); }
  record(tr, "lnf", st.lnf.data(), kC);
  { Scope s(prof, kOpLmHead); lm_head_forward(st.logits.data(), st.lnf.data(), w.wte, kC, kV); }
  int next;
  { Scope s(prof, kOpArgmax); next = argmax(st.logits.data(), kV); }
  st.pos += T;
  return next;
}

std::vector<int> generate(const Weights& w, State& st, const std::vector<int>& prompt, int n_new,
                          Profiler* decode_prof, double* prefill_ms,
                          std::vector<double>* decode_ms) {
  std::vector<int> out;
  Timer t;
  int next = forward(w, st, prompt.data(), (int)prompt.size());
  if (prefill_ms) *prefill_ms = t.ms();
  out.push_back(next);
  for (int i = 1; i < n_new; i++) {
    t.reset();
    next = forward(w, st, &next, 1, decode_prof);
    if (decode_ms) decode_ms->push_back(t.ms());
    out.push_back(next);
  }
  return out;
}

}  // namespace gpt2
