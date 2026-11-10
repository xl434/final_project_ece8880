// FP32 GPT-2 weights loaded from llm.c's gpt2_124M.bin (format version 3).
// Layout (all row-major, as in llm.c train_gpt2.c):
//   wte (Vp, C)  wpe (maxT, C)
//   ln1w/ln1b (L, C)  qkvw (L, 3C, C)  qkvb (L, 3C)  attprojw (L, C, C)  attprojb (L, C)
//   ln2w/ln2b (L, C)  fcw (L, 4C, C)   fcb (L, 4C)   fcprojw (L, C, 4C)  fcprojb (L, C)
//   lnfw/lnfb (C)
// Linear weights are (out, in): y[o] = sum_i W[o * in + i] * x[i] + b[o].
#ifndef GPT2_COMMON_WEIGHTS_H_
#define GPT2_COMMON_WEIGHTS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "config.h"

namespace gpt2 {

struct Weights {
  std::vector<float> mem;  // all parameters, contiguous
  size_t num_parameters = 0;
  float* wte = nullptr;
  float* wpe = nullptr;
  float* ln1w = nullptr;
  float* ln1b = nullptr;
  float* qkvw = nullptr;
  float* qkvb = nullptr;
  float* attprojw = nullptr;
  float* attprojb = nullptr;
  float* ln2w = nullptr;
  float* ln2b = nullptr;
  float* fcw = nullptr;
  float* fcb = nullptr;
  float* fcprojw = nullptr;
  float* fcprojb = nullptr;
  float* lnfw = nullptr;
  float* lnfb = nullptr;

  // Per-layer views.
  const float* l_ln1w(int l) const { return ln1w + (size_t)l * kC; }
  const float* l_ln1b(int l) const { return ln1b + (size_t)l * kC; }
  const float* l_qkvw(int l) const { return qkvw + (size_t)l * kC3 * kC; }
  const float* l_qkvb(int l) const { return qkvb + (size_t)l * kC3; }
  const float* l_attprojw(int l) const { return attprojw + (size_t)l * kC * kC; }
  const float* l_attprojb(int l) const { return attprojb + (size_t)l * kC; }
  const float* l_ln2w(int l) const { return ln2w + (size_t)l * kC; }
  const float* l_ln2b(int l) const { return ln2b + (size_t)l * kC; }
  const float* l_fcw(int l) const { return fcw + (size_t)l * kC4 * kC; }
  const float* l_fcb(int l) const { return fcb + (size_t)l * kC4; }
  const float* l_fcprojw(int l) const { return fcprojw + (size_t)l * kC * kC4; }
  const float* l_fcprojb(int l) const { return fcprojb + (size_t)l * kC; }
};

// Loads the checkpoint; exits with a message on any format/size mismatch.
void load_gpt2_weights(Weights& w, const std::string& path);

// Default paths, overridable with the GPT2_MODEL_DIR environment variable.
std::string default_model_dir();

}  // namespace gpt2

#endif
