#include "kernel.h"

// Representative trip counts for the HLS latency report: the MLP fc layer
// (C = 768, OC = 3072). Range: C in [768, 3072], OC in [768, 3072].
constexpr int kTcCw = kKC / kVec;                                  // 48
constexpr int kTcCwMax = kMaxCWords;                               // 192
constexpr int kTcIters = (kKC4 / kGemvRowBlock) * kTcCw * kGemvRows;  // fc layer

// Read x once and broadcast it to every PE.
void GemvReadX(tapa::mmap<float_v16> x, tapa::ostreams<float_v16, kGemvPorts>& x_s, int C) {
  const int cw = C / kVec;
  for (int i = 0; i < cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcCw max=kTcCwMax avg=kTcCw
    const float_v16 v = x[i];
    for (int p = 0; p < kGemvPorts; ++p) x_s[p].write(v);
  }
}

// Stream this port's packed weights, one 512-bit word per cycle.
void GemvReadW(tapa::mmap<float_v16> w, tapa::ostream<float_v16>& w_s, int C, int OC) {
  const int n = OC / kGemvPorts * (C / kVec);
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcIters max=kTcIters avg=kTcIters
    w_s.write(w[i]);
  }
}

// One PE: 16 MACs per cycle. Loop order: row group g, word w, row r.
void GemvPE(tapa::istream<float_v16>& x_s, tapa::istream<float_v16>& w_s,
            tapa::ostream<float>& y_s, int C, int OC) {
  float_v16 xbuf[kMaxCWords];
  const int cw = C / kVec;
  for (int i = 0; i < cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcCw max=kTcCwMax avg=kTcCw
    xbuf[i] = x_s.read();
  }

  // acc[r] is read and written once per cycle, and the same r comes back only
  // every kGemvRows cycles, which is longer than the fadd latency.
  float acc[kGemvRows];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
  const int n = OC / kGemvRowBlock * cw * kGemvRows;
  int w = 0, r = 0;
  for (int it = 0; it < n; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcIters max=kTcIters avg=kTcIters
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
    const float t = dot16(w_s.read(), xbuf[w]);
    const float s = (w == 0 ? 0.0f : acc[r]) + t;
    acc[r] = s;
    if (w == cw - 1) y_s.write(s);
    if (r == kGemvRows - 1) {
      r = 0;
      w = (w == cw - 1) ? 0 : w + 1;
    } else {
      ++r;
    }
  }
}

// Collect PE outputs in row order and add bias (+ residual).
void GemvWriteY(tapa::istreams<float, kGemvPorts>& y_s, tapa::mmap<float> bias,
                tapa::mmap<float> residual, tapa::mmap<float> y, int OC, int use_residual) {
  int p = 0, r = 0;
  for (int row = 0; row < OC; ++row) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kKC max=kKC4 avg=kKC4
    float v = 0.0f;
    for (int c = 0; c < kGemvPorts; ++c)
      if (c == p) v = y_s[c].read();
    float o = v + bias[row];
    if (use_residual) o += residual[row];
    y[row] = o;
    if (r == kGemvRows - 1) {
      r = 0;
      p = (p == kGemvPorts - 1) ? 0 : p + 1;
    } else {
      ++r;
    }
  }
}

void GemvKernel(tapa::mmap<float_v16> x, tapa::mmaps<float_v16, kGemvPorts> w,
                tapa::mmap<float> bias, tapa::mmap<float> residual, tapa::mmap<float> y,
                int C, int OC, int use_residual) {
  tapa::streams<float_v16, kGemvPorts, 4> x_s("x_s");
  tapa::streams<float_v16, kGemvPorts, 32> w_s("w_s");
  tapa::streams<float, kGemvPorts, 2 * kGemvRows> y_s("y_s");

  tapa::task()
      .invoke(GemvReadX, x, x_s, C)
      .invoke<tapa::join, kGemvPorts>(GemvReadW, w, w_s, C, OC)
      .invoke<tapa::join, kGemvPorts>(GemvPE, x_s, w_s, y_s, C, OC)
      .invoke(GemvWriteY, y_s, bias, residual, y, OC, use_residual);
}
