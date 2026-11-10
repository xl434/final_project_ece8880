// GPT-2 CPU inference with a KV cache: prefill (T tokens) and decode (T = 1)
// use the same forward function; only the last token's logits are computed.
#ifndef GPT2_CPU_GPT2_CPU_H_
#define GPT2_CPU_GPT2_CPU_H_

#include <map>
#include <string>
#include <vector>

#include "../common/config.h"
#include "../common/weights.h"

namespace gpt2 {

enum Op {
  kOpEmbed, kOpLn1, kOpQkv, kOpKvWrite, kOpAttn, kOpAttProj, kOpRes1,
  kOpLn2, kOpFc, kOpGelu, kOpFcProj, kOpRes2, kOpLnf, kOpLmHead, kOpArgmax, kNumOps
};
extern const char* const kOpNames[kNumOps];

// Accumulated wall time per operator (milliseconds).
struct Profiler {
  double ms[kNumOps] = {0};
  long calls[kNumOps] = {0};
  void clear() { *this = Profiler(); }
};

// Copies of the last token's activations, keyed like the golden files
// ("embed", "l03_ln1", ..., "lnf").
struct Trace {
  std::map<std::string, std::vector<float>> t;
};

struct State {
  std::vector<float> kcache, vcache;   // (L, maxT, C)
  std::vector<float> x, ln, qkv, atty, attproj, res1, ln2, fch, fch_gelu, fcproj;  // (maxT, *)
  std::vector<float> lnf, logits;      // last token only: (C), (Vp)
  int pos = 0;                          // number of tokens already in the cache
  State();
  float* k(int l) { return kcache.data() + (long)l * kMaxT * kC; }
  float* v(int l) { return vcache.data() + (long)l * kMaxT * kC; }
};

// Runs T new tokens at positions st.pos .. st.pos+T-1, appends them to the KV
// cache, writes the logits of the last token to st.logits and returns argmax.
int forward(const Weights& w, State& st, const int* tokens, int T, Profiler* prof = nullptr,
            Trace* trace = nullptr);

// Greedy generation: prefill `prompt`, then n_new - 1 decode steps.
// Returns the n_new generated tokens. prefill_ms / decode_ms are optional.
std::vector<int> generate(const Weights& w, State& st, const std::vector<int>& prompt, int n_new,
                          Profiler* decode_prof = nullptr, double* prefill_ms = nullptr,
                          std::vector<double>* decode_ms = nullptr);

}  // namespace gpt2

#endif
