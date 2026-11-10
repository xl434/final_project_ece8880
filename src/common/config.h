// GPT-2 (124M) model dimensions. These match the header of llm.c's
// gpt2_124M.bin; load_gpt2_weights() checks the file against them.
#ifndef GPT2_COMMON_CONFIG_H_
#define GPT2_COMMON_CONFIG_H_

namespace gpt2 {

constexpr int kMaxT = 1024;   // maximum context length
constexpr int kV = 50257;     // vocabulary size
constexpr int kVp = 50304;    // vocabulary padded to a multiple of 128 (llm.c)
constexpr int kL = 12;        // transformer blocks
constexpr int kNH = 12;       // attention heads
constexpr int kC = 768;       // channels (d_model)
constexpr int kHS = kC / kNH; // head size (64)
constexpr int kC3 = 3 * kC;   // fused QKV width
constexpr int kC4 = 4 * kC;   // MLP hidden width
constexpr int kEotToken = 50256;
constexpr float kLnEps = 1e-5f;

}  // namespace gpt2

#endif
