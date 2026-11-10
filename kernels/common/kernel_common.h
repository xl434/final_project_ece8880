// Types and constants shared by the GPT-2 HLS kernels (TAPA / Vitis HLS).
#ifndef GPT2_KERNEL_COMMON_H_
#define GPT2_KERNEL_COMMON_H_

#include <hls_vector.h>
#include <tapa.h>

// One 512-bit AXI word = 16 FP32 values.
using float_v16 = hls::vector<float, 16>;
constexpr int kVec = 16;

// GPT-2 124M dimensions (mirror src/common/config.h; kernels cannot include
// host-only code).
constexpr int kKC = 768;              // channels
constexpr int kKC4 = 4 * kKC;         // MLP hidden width (largest GEMV input)
constexpr int kKNH = 12;              // heads
constexpr int kKHS = kKC / kKNH;      // head size 64
constexpr int kKMaxT = 1024;          // max context
constexpr int kKL = 12;               // layers
constexpr int kKV = 50257;            // vocabulary
constexpr int kKVp = 50304;           // padded vocabulary (rows of wte)
constexpr int kCWords = kKC / kVec;   // 48 words per C-vector
constexpr int kMaxCWords = kKC4 / kVec;  // 192 words per 4C-vector
constexpr int kHeadWords = kKHS / kVec;  // 4 words per head

// Sum of 16 products as a balanced tree (fixed order; no reassociation flag
// needed and the same order in csim and RTL).
inline float tree_sum16(const float p[16]) {
#pragma HLS INLINE
  float s8[8], s4[4], s2[2];
#pragma HLS ARRAY_PARTITION variable=s8 type=complete
#pragma HLS ARRAY_PARTITION variable=s4 type=complete
#pragma HLS ARRAY_PARTITION variable=s2 type=complete
  for (int i = 0; i < 8; ++i) s8[i] = p[2 * i] + p[2 * i + 1];
  for (int i = 0; i < 4; ++i) s4[i] = s8[2 * i] + s8[2 * i + 1];
  for (int i = 0; i < 2; ++i) s2[i] = s4[2 * i] + s4[2 * i + 1];
  return s2[0] + s2[1];
}

inline float dot16(const float_v16& a, const float_v16& b) {
#pragma HLS INLINE
  float p[16];
#pragma HLS ARRAY_PARTITION variable=p type=complete
  for (int k = 0; k < 16; ++k) p[k] = a[k] * b[k];
  return tree_sum16(p);
}

#endif
