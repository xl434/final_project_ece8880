#include "ops.h"

#include <cmath>

namespace gpt2 {

void encoder_forward(float* out, const int* tokens, int T, int pos0, const float* wte,
                     const float* wpe, int C) {
  for (int t = 0; t < T; t++) {
    float* out_t = out + (long)t * C;
    const float* wte_ix = wte + (long)tokens[t] * C;
    const float* wpe_t = wpe + (long)(pos0 + t) * C;
    for (int i = 0; i < C; i++) out_t[i] = wte_ix[i] + wpe_t[i];
  }
}

void layernorm_forward(float* out, const float* inp, const float* weight, const float* bias,
                       int T, int C) {
  const float eps = 1e-5f;
  for (int t = 0; t < T; t++) {
    const float* x = inp + (long)t * C;
    float m = 0.0f;
    for (int i = 0; i < C; i++) m += x[i];
    m = m / C;
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
      float xshift = x[i] - m;
      v += xshift * xshift;
    }
    v = v / C;
    float s = 1.0f / sqrtf(v + eps);
    float* out_t = out + (long)t * C;
    for (int i = 0; i < C; i++) out_t[i] = (s * (x[i] - m)) * weight[i] + bias[i];
  }
}

// Dot product with 16 independent partial sums. Without -ffast-math the
// compiler may not reassociate a single running sum, so the explicit lanes are
// what lets -O3 vectorize this loop (the FPGA GEMV uses the same idea).
static inline float dot(const float* a, const float* b, int n) {
  float acc[16] = {0};
  int i = 0;
  for (; i + 16 <= n; i += 16)
    for (int k = 0; k < 16; k++) acc[k] += a[i + k] * b[i + k];
  float s = 0.0f;
  for (int k = 0; k < 16; k++) s += acc[k];
  for (; i < n; i++) s += a[i] * b[i];
  return s;
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias, int T,
                    int C, int OC) {
  if (T == 1) {
    // decode: GEMV, parallel over output channels
#pragma omp parallel for schedule(static)
    for (int o = 0; o < OC; o++)
      out[o] = (bias ? bias[o] : 0.0f) + dot(inp, weight + (long)o * C, C);
    return;
  }
  // prefill: llm.c's register tiling, LOOP_UNROLL tokens share each weight load
  const int LOOP_UNROLL = 8;
  const int Tmain = T / LOOP_UNROLL * LOOP_UNROLL;
#pragma omp parallel for schedule(static)
  for (int obt = 0; obt < Tmain; obt += LOOP_UNROLL) {
    for (int o = 0; o < OC; o++) {
      float result[LOOP_UNROLL];
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) result[ibt] = bias ? bias[o] : 0.0f;
      for (int i = 0; i < C; i++) {
        float w = weight[i + (long)o * C];
        for (int ibt = 0; ibt < LOOP_UNROLL; ibt++)
          result[ibt] += inp[(long)(obt + ibt) * C + i] * w;
      }
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) out[(long)(obt + ibt) * OC + o] = result[ibt];
    }
  }
  // remaining tokens: plain dot products
#pragma omp parallel for schedule(static)
  for (int o = 0; o < OC; o++)
    for (int t = Tmain; t < T; t++)
      out[(long)t * OC + o] =
          (bias ? bias[o] : 0.0f) + dot(inp + (long)t * C, weight + (long)o * C, C);
}

void kv_cache_write(float* kcache, float* vcache, const float* qkv, int T, int pos0, int C) {
  for (int t = 0; t < T; t++) {
    const float* row = qkv + (long)t * 3 * C;
    float* k = kcache + (long)(pos0 + t) * C;
    float* v = vcache + (long)(pos0 + t) * C;
    for (int i = 0; i < C; i++) {
      k[i] = row[C + i];
      v[i] = row[2 * C + i];
    }
  }
}

void attention_forward(float* out, const float* qkv, const float* kcache, const float* vcache,
                       int T, int pos0, int C, int NH) {
  const int hs = C / NH;
  const float scale = 1.0f / sqrtf((float)hs);
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int t = 0; t < T; t++) {
    for (int h = 0; h < NH; h++) {
      const int npos = pos0 + t + 1;  // causal: rows 0 .. pos0 + t
      const float* q = qkv + (long)t * 3 * C + h * hs;
      float* att = new float[npos];
      // pass 1: scores and max
      float maxval = -INFINITY;
      for (int t2 = 0; t2 < npos; t2++) {
        const float* k = kcache + (long)t2 * C + h * hs;
        float val = 0.0f;
        for (int i = 0; i < hs; i++) val += q[i] * k[i];
        val *= scale;
        if (val > maxval) maxval = val;
        att[t2] = val;
      }
      // pass 2: exp and sum
      float expsum = 0.0f;
      for (int t2 = 0; t2 < npos; t2++) {
        float e = expf(att[t2] - maxval);
        expsum += e;
        att[t2] = e;
      }
      const float inv = 1.0f / expsum;
      // pass 3: weighted sum of values
      float* o = out + (long)t * C + h * hs;
      for (int i = 0; i < hs; i++) o[i] = 0.0f;
      for (int t2 = 0; t2 < npos; t2++) {
        const float a = att[t2] * inv;
        const float* v = vcache + (long)t2 * C + h * hs;
        for (int i = 0; i < hs; i++) o[i] += a * v[i];
      }
      delete[] att;
    }
  }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, const float* inp, int N) {
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
}

void residual_forward(float* out, const float* a, const float* b, int N) {
  for (int i = 0; i < N; i++) out[i] = a[i] + b[i];
}

void lm_head_forward(float* logits, const float* x, const float* wte, int C, int V) {
  matmul_forward(logits, x, wte, nullptr, 1, C, V);
}

}  // namespace gpt2
