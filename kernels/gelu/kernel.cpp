#include "kernel.h"

#include <cmath>

constexpr int kTc = kMaxCWords;  // one 3072-wide MLP hidden vector

void GeluRead(tapa::mmap<float_v16> x, tapa::ostream<float_v16>& x_s, int n_words) {
  for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTc max=kTc avg=kTc
    x_s.write(x[i]);
  }
}

void GeluCore(tapa::istream<float_v16>& x_s, tapa::ostream<float_v16>& y_s, int n_words) {
  const float kS = 0.7978845608028654f;  // sqrt(2/pi)
  for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTc max=kTc avg=kTc
    const float_v16 v = x_s.read();
    float_v16 o;
    for (int k = 0; k < 16; ++k) {
      const float x = v[k];
      const float u = kS * (x + 0.044715f * x * x * x);
      o[k] = x / (1.0f + std::exp(-2.0f * u));
    }
    y_s.write(o);
  }
}

void GeluWrite(tapa::istream<float_v16>& y_s, tapa::mmap<float_v16> y, int n_words) {
  for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTc max=kTc avg=kTc
    y[i] = y_s.read();
  }
}

void GeluKernel(tapa::mmap<float_v16> x, tapa::mmap<float_v16> y, int n_words) {
  tapa::stream<float_v16, 4> x_s("x_s");
  tapa::stream<float_v16, 4> y_s("y_s");
  tapa::task()
      .invoke(GeluRead, x, x_s, n_words)
      .invoke(GeluCore, x_s, y_s, n_words)
      .invoke(GeluWrite, y_s, y, n_words);
}
