#include "kernel.h"

#include <cmath>

constexpr int kGroups = kKVp / kLmRowBlock;          // row groups per PE (393)
constexpr int kWordsPerPort = kKVp / kLmPorts * kCWords;
constexpr int kIters = kGroups * kCWords * kLmRows;  // = kWordsPerPort

void LmReadX(tapa::mmap<float_v16> x, tapa::ostreams<float_v16, kLmPorts>& x_s) {
  for (int i = 0; i < kCWords; ++i) {
#pragma HLS PIPELINE II=1
    const float_v16 v = x[i];
    for (int p = 0; p < kLmPorts; ++p) x_s[p].write(v);
  }
}

void LmReadW(tapa::mmap<float_v16> w, tapa::ostream<float_v16>& w_s) {
  for (int i = 0; i < kWordsPerPort; ++i) {
#pragma HLS PIPELINE II=1
    w_s.write(w[i]);
  }
}

static bool better(float v, int i, float bv, int bi) {
#pragma HLS INLINE
  return v > bv || (v == bv && i < bi);
}

// PE `port` (from tapa::seq()) owns row blocks b = g * kLmPorts + port.
void LmPE(int port, tapa::istream<float_v16>& x_s, tapa::istream<float_v16>& w_s,
          tapa::ostream<float>& val_s, tapa::ostream<int>& idx_s) {
  float_v16 xbuf[kCWords];
  for (int i = 0; i < kCWords; ++i) {
#pragma HLS PIPELINE II=1
    xbuf[i] = x_s.read();
  }
  float acc[kLmRows], bv[kLmRows];
  int bi[kLmRows];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=bv type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=bi type=ram_2p impl=lutram
  for (int r = 0; r < kLmRows; ++r) {
#pragma HLS PIPELINE II=1
    bv[r] = -INFINITY;
    bi[r] = kKVp;
  }
  int w = 0, r = 0, row_base = port * kLmRows;
  for (int it = 0; it < kIters; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
#pragma HLS DEPENDENCE variable=bv type=inter dependent=false
#pragma HLS DEPENDENCE variable=bi type=inter dependent=false
    const float t = dot16(w_s.read(), xbuf[w]);
    const float s = (w == 0 ? 0.0f : acc[r]) + t;
    acc[r] = s;
    // a lane sees its rows in increasing order, so '>' keeps the lowest index
    const int row = row_base + r;
    if (w == kCWords - 1 && row < kKV && s > bv[r]) {
      bv[r] = s;
      bi[r] = row;
    }
    if (r == kLmRows - 1) {
      r = 0;
      if (w == kCWords - 1) {
        w = 0;
        row_base += kLmRowBlock;
      } else {
        ++w;
      }
    } else {
      ++r;
    }
  }
  float best_v = bv[0];
  int best_i = bi[0];
  for (int k = 1; k < kLmRows; ++k) {
    if (better(bv[k], bi[k], best_v, best_i)) {
      best_v = bv[k];
      best_i = bi[k];
    }
  }
  val_s.write(best_v);
  idx_s.write(best_i);
}

void LmMerge(tapa::istreams<float, kLmPorts>& val_s, tapa::istreams<int, kLmPorts>& idx_s,
             tapa::mmap<int> out_idx, tapa::mmap<float> out_val) {
  float best_v = -INFINITY;
  int best_i = kKVp;
  for (int p = 0; p < kLmPorts; ++p) {
    const float v = val_s[p].read();
    const int i = idx_s[p].read();
    if (better(v, i, best_v, best_i)) {
      best_v = v;
      best_i = i;
    }
  }
  out_idx[0] = best_i;
  out_val[0] = best_v;
}

void LmHeadKernel(tapa::mmap<float_v16> x, tapa::mmaps<float_v16, kLmPorts> w,
                  tapa::mmap<int> out_idx, tapa::mmap<float> out_val) {
  tapa::streams<float_v16, kLmPorts, 4> x_s("x_s");
  tapa::streams<float_v16, kLmPorts, 32> w_s("w_s");
  tapa::streams<float, kLmPorts, 2> val_s("val_s");
  tapa::streams<int, kLmPorts, 2> idx_s("idx_s");
  tapa::task()
      .invoke(LmReadX, x, x_s)
      .invoke<tapa::join, kLmPorts>(LmReadW, w, w_s)
      .invoke<tapa::join, kLmPorts>(LmPE, tapa::seq(), x_s, w_s, val_s, idx_s)
      .invoke(LmMerge, val_s, idx_s, out_idx, out_val);
}
