// FPGA host model for GPT-2 decode: owns the (host copies of the) device
// buffers, packs weights for the GEMV / LM-head kernels once, and runs one
// token at a time through the HLS kernels with tapa::invoke. Any operator
// class can be forced onto the CPU (src/cpu operators) for fallback/debug.
//
// Bitstream "" = TAPA software simulation; ".xo" = TAPA fast cosim; ".xclbin"
// = board (all kernels must be linked into that one xclbin, see
// link_config.ini). With tapa::invoke every call copies its mmap arguments
// to/from the device; the persistent-buffer flow for the board is described
// by hbm_plan() and is a next step (see PROGRESS_REPORT.md).
#ifndef GPT2_HOST_FPGA_MODEL_H_
#define GPT2_HOST_FPGA_MODEL_H_

#include <array>
#include <string>
#include <vector>

#include "../../kernels/common/host_pack.h"
#include "../common/config.h"
#include "../common/weights.h"

namespace gpt2 {

enum OpClass { kLayerNorm, kGemv, kAttention, kGelu, kResidual, kLmHead, kNumOpClasses };
extern const char* const kOpClassNames[kNumOpClasses];

struct HostOptions {
  std::string bitstream;                        // "" = software simulation
  std::array<bool, kNumOpClasses> on_cpu{};     // force CPU fallback per class
  bool fuse_residual = true;                    // residual add inside the GEMV
};

// Per-token activations of one block (for tests).
struct BlockActs {
  std::vector<float> ln1, qkv, atty, res1, ln2, fch, fch_gelu, res2;
};

class FpgaModel {
 public:
  FpgaModel(const Weights& w, const HostOptions& opt);

  // One decode step: token at position pos (= tokens already cached).
  // Returns the argmax next token.
  int step(int token, int pos);
  // One transformer block on x (C floats) at position pos; updates the KV
  // cache of `layer` and returns the block output in acts->res2.
  void block(int layer, const std::vector<float>& x, int pos, BlockActs* acts);

  float* kcache_layer(int l) { return kcache_.data() + (size_t)l * kMaxT * kC; }
  float* vcache_layer(int l) { return vcache_.data() + (size_t)l * kMaxT * kC; }
  double last_logit() const { return last_logit_; }
  long kernel_calls[kNumOpClasses] = {0};
  double kernel_ms[kNumOpClasses] = {0};

  // Prints the HBM buffer placement used by link_config.ini.
  static void print_hbm_plan();

 private:
  void layernorm(const float* x, const float* g, const float* b, float* y);
  void gemv(const std::vector<aligned_vector<float>>& wp, const float* W, const float* b,
            const float* x, const float* res, float* y, int C, int OC);
  void attention(const float* qkv, int layer, int pos, float* y);
  void gelu(const float* x, float* y, int n);
  void residual(const float* a, const float* b, float* y, int n);
  int lm_head(const float* x);

  const Weights& w_;
  HostOptions opt_;
  // packed GEMV weights [layer][port], packed LM head [port]
  std::vector<std::vector<aligned_vector<float>>> qkvw_, attprojw_, fcw_, fcprojw_;
  std::vector<aligned_vector<float>> lmw_;
  aligned_vector<float> kcache_, vcache_;  // (L, maxT, C) each
  double last_logit_ = 0;
};

}  // namespace gpt2

#endif
