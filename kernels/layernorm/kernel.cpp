#include "kernel.h"

#include <cmath>

void LnReadX(tapa::mmap<float_v16> x, tapa::ostream<float_v16>& x_s, int rows, int C) {
  const int n = rows * (C / kVec);
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kCWords avg=kCWords
    x_s.write(x[i]);
  }
}

// Sum of kLnAcc partial sums (small fixed tree).
static float fold_acc(const float a[kLnAcc]) {
#pragma HLS INLINE
  return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}

void LnCore(tapa::istream<float_v16>& x_s, tapa::mmap<float_v16> gamma,
            tapa::mmap<float_v16> beta, tapa::ostream<float_v16>& y_s, int rows, int C) {
  const int cw = C / kVec;
  float_v16 g[kCWords], b[kCWords], buf[kCWords];
  for (int i = 0; i < cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kCWords avg=kCWords
    g[i] = gamma[i];
    b[i] = beta[i];
  }
  const float inv_c = 1.0f / C;

  for (int row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
    float acc[kLnAcc];
#pragma HLS ARRAY_PARTITION variable=acc type=complete
    for (int k = 0; k < kLnAcc; ++k) acc[k] = 0.0f;
    // pass 1: mean
    for (int i = 0; i < cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kCWords avg=kCWords
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
      const float_v16 v = x_s.read();
      buf[i] = v;
      float p[16];
#pragma HLS ARRAY_PARTITION variable=p type=complete
      for (int k = 0; k < 16; ++k) p[k] = v[k];
      acc[i % kLnAcc] += tree_sum16(p);
    }
    const float mean = fold_acc(acc) * inv_c;
    // pass 2: biased variance
    for (int k = 0; k < kLnAcc; ++k) acc[k] = 0.0f;
    for (int i = 0; i < cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kCWords avg=kCWords
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
      const float_v16 v = buf[i];
      float p[16];
#pragma HLS ARRAY_PARTITION variable=p type=complete
      for (int k = 0; k < 16; ++k) {
        const float d = v[k] - mean;
        p[k] = d * d;
      }
      acc[i % kLnAcc] += tree_sum16(p);
    }
    const float rstd = 1.0f / std::sqrt(fold_acc(acc) * inv_c + 1e-5f);
    // pass 3: normalize, scale, shift
    for (int i = 0; i < cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kCWords avg=kCWords
      const float_v16 v = buf[i], gi = g[i], bi = b[i];
      float_v16 o;
      for (int k = 0; k < 16; ++k) o[k] = (rstd * (v[k] - mean)) * gi[k] + bi[k];
      y_s.write(o);
    }
  }
}

void LnWrite(tapa::istream<float_v16>& y_s, tapa::mmap<float_v16> y, int rows, int C) {
  const int n = rows * (C / kVec);
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kCWords avg=kCWords
    y[i] = y_s.read();
  }
}

void LayerNormKernel(tapa::mmap<float_v16> x, tapa::mmap<float_v16> gamma,
                     tapa::mmap<float_v16> beta, tapa::mmap<float_v16> y, int rows, int C) {
  tapa::stream<float_v16, 2 * kCWords> x_s("x_s");
  tapa::stream<float_v16, 4> y_s("y_s");
  tapa::task()
      .invoke(LnReadX, x, x_s, rows, C)
      .invoke(LnCore, x_s, gamma, beta, y_s, rows, C)
      .invoke(LnWrite, y_s, y, rows, C);
}
