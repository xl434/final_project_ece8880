// CPU reference operators for GPT-2 inference (batch 1, FP32).
// Adapted from llm.c train_gpt2.c forward functions (B removed, backward
// buffers removed, attention rewritten to use a KV cache). Each function is
// the functional reference for the HLS kernel of the same name.
//
// Shapes: T = number of new tokens in this call, C = channels, OC = output
// channels. Activations are row-major (T, C). Linear weights are (OC, C).
// KV cache of one layer: kcache/vcache (maxT, C); head h owns columns
// [h*HS, (h+1)*HS) of every row (same layout as the K/V part of qkv).
#ifndef GPT2_CPU_OPS_H_
#define GPT2_CPU_OPS_H_

namespace gpt2 {

// out[t] = wte[tokens[t]] + wpe[pos0 + t]
void encoder_forward(float* out, const int* tokens, int T, int pos0, const float* wte,
                     const float* wpe, int C);

// out[t] = (x[t] - mean) / sqrt(var + 1e-5) * w + b   (biased variance)
void layernorm_forward(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C);

// out[t][o] = bias[o] + sum_i inp[t][i] * weight[o][i]   (bias may be null)
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias, int T,
                    int C, int OC);

// Append K,V of the T new tokens (taken from qkv (T, 3C)) at rows pos0..pos0+T-1.
void kv_cache_write(float* kcache, float* vcache, const float* qkv, int T, int pos0, int C);

// Causal multi-head attention of the T new tokens (queries from qkv (T, 3C))
// against cache rows 0..pos0+t (inclusive). The cache must already contain the
// new tokens (kv_cache_write). out (T, C), heads concatenated.
void attention_forward(float* out, const float* qkv, const float* kcache, const float* vcache,
                       int T, int pos0, int C, int NH);

// tanh-approximation GELU ("gelu_new" in HF GPT-2).
void gelu_forward(float* out, const float* inp, int N);

// out = a + b
void residual_forward(float* out, const float* a, const float* b, int N);

// logits[v] = sum_i x[i] * wte[v][i] for v < V (weight tying; no bias), x (C)
void lm_head_forward(float* logits, const float* x, const float* wte, int C, int V);

}  // namespace gpt2

#endif
