#include "kernel.h"

constexpr int kTc = kCWords;  // one 768-wide residual vector

void ResRead(tapa::mmap<float_v16> m, tapa::ostream<float_v16>& s, int n_words) {
  for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTc max=kTc avg=kTc
    s.write(m[i]);
  }
}

void ResAdd(tapa::istream<float_v16>& a_s, tapa::istream<float_v16>& b_s,
            tapa::ostream<float_v16>& y_s, int n_words) {
  for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTc max=kTc avg=kTc
    y_s.write(a_s.read() + b_s.read());
  }
}

void ResWrite(tapa::istream<float_v16>& y_s, tapa::mmap<float_v16> y, int n_words) {
  for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTc max=kTc avg=kTc
    y[i] = y_s.read();
  }
}

void ResidualKernel(tapa::mmap<float_v16> a, tapa::mmap<float_v16> b, tapa::mmap<float_v16> y,
                    int n_words) {
  tapa::stream<float_v16, 4> a_s("a_s");
  tapa::stream<float_v16, 4> b_s("b_s");
  tapa::stream<float_v16, 4> y_s("y_s");
  tapa::task()
      .invoke(ResRead, a, a_s, n_words)
      .invoke(ResRead, b, b_s, n_words)
      .invoke(ResAdd, a_s, b_s, y_s, n_words)
      .invoke(ResWrite, y_s, y, n_words);
}
