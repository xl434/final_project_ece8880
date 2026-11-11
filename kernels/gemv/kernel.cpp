#include "kernel.h"

#ifndef TAPA_TARGET_DEVICE_
#include <vector>
#endif

// Representative trip counts for the HLS latency report: the MLP fc layer
// (C = 768, OC = 3072) for one decode token (T = 1). Range: C in [768, 3072],
// OC in [768, 3072], T in [1, 1024].
constexpr int kTcCw = kKC / kVec;                                     // 48
constexpr int kTcCwMax = kMaxCWords;                                  // 192
constexpr int kTcIters = (kKC4 / kGemvRowBlock) * kTcCw * kGemvRows;  // fc layer, T = 1
constexpr int kTcItersMax = kTcIters * 4 * kGemvTb;                   // fc-proj, full chunk
constexpr int kTcChunks = (kKMaxT + kGemvChunk - 1) / kGemvChunk;

static int min_int(int a, int b) { return a < b ? a : b; }

// Read X (T rows, in order) once and broadcast it to every PE.
void GemvReadX(tapa::mmap<float_v16> x, tapa::ostreams<float_v16, kGemvPorts>& x_s, int T,
               int C) {
  const int n = T * (C / kVec);
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcCw max=kTcCwMax*kKMaxT avg=kTcCw
    const float_v16 v = x[i];
    for (int p = 0; p < kGemvPorts; ++p) x_s[p].write(v);
  }
}

// Stream this port's packed weights once per token chunk.
void GemvReadW(tapa::mmap<float_v16> w, tapa::ostream<float_v16>& w_s, int T, int C, int OC) {
  const int n = OC / kGemvPorts * (C / kVec);
  const int total = (T + kGemvChunk - 1) / kGemvChunk * n;
  int i = 0;
  for (int k = 0; k < total; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcIters max=kTcIters*kTcChunks avg=kTcIters
    w_s.write(w[i]);
    i = (i == n - 1) ? 0 : i + 1;
  }
}

// One PE. For each token chunk: load the chunk's X rows, then for every
// weight word (order: row group g, word w, row r) multiply it with all tokens
// of the chunk, kGemvTok tokens per cycle. Emits, for each row of the PE and
// each token group, the finished sums of kGemvTok tokens.
void GemvPE(tapa::istream<float_v16>& x_s, tapa::istream<float_v16>& w_s,
            tapa::ostream<gemv_tok_t>& y_s, int T, int C, int OC) {
  // xbuf[p][tb][w] = word w of token tb * kGemvTok + p of the chunk (URAM).
  // In software simulation a task runs on a small coroutine stack, so this
  // large buffer is allocated on the heap there instead.
#ifdef TAPA_TARGET_DEVICE_
  float_v16 xbuf[kGemvTok][kGemvTb][kMaxCWords];
#pragma HLS ARRAY_PARTITION variable=xbuf dim=1 type=complete
#pragma HLS BIND_STORAGE variable=xbuf type=ram_2p impl=uram
#else
  std::vector<float_v16> xstore((size_t)kGemvTok * kGemvTb * kMaxCWords);
  auto xbuf = reinterpret_cast<float_v16(*)[kGemvTb][kMaxCWords]>(xstore.data());
#endif
  // acc[p][r][tb]: running sum of row r for token tb * kGemvTok + p
  float acc[kGemvTok][kGemvRows][kGemvTb];
#pragma HLS ARRAY_PARTITION variable=acc dim=1 type=complete
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  const int cw = C / kVec;
  const int groups = OC / kGemvRowBlock;
  for (int t0 = 0; t0 < T; t0 += kGemvChunk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=kTcChunks avg=1
    const int tc = min_int(kGemvChunk, T - t0);
    const int ntb = (tc + kGemvTok - 1) / kGemvTok;

    // load the chunk's X rows (unused token lanes of the last group stay stale;
    // their results are never written)
    int lt = 0, lw = 0;
    for (int i = 0; i < tc * cw; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcCw max=kTcCwMax*kGemvChunk avg=kTcCw
      const float_v16 v = x_s.read();
      for (int p = 0; p < kGemvTok; ++p)
        if (p == lt % kGemvTok) xbuf[p][lt / kGemvTok][lw] = v;
      if (lw == cw - 1) {
        lw = 0;
        ++lt;
      } else {
        ++lw;
      }
    }

    // acc[p][r][tb] comes back every kGemvRows * ntb >= kGemvRows cycles,
    // longer than the fadd latency.
    const int n = groups * cw * kGemvRows * ntb;
    int w = 0, r = 0, tb = 0;
    float_v16 wv;
    for (int it = 0; it < n; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kTcIters max=kTcItersMax avg=kTcIters
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
      if (tb == 0) wv = w_s.read();
      gemv_tok_t o;
      for (int p = 0; p < kGemvTok; ++p) {
        const float t = dot16(wv, xbuf[p][tb][w]);
        const float s = (w == 0 ? 0.0f : acc[p][r][tb]) + t;
        acc[p][r][tb] = s;
        o[p] = s;
      }
      if (w == cw - 1) y_s.write(o);
      if (tb == ntb - 1) {
        tb = 0;
        if (r == kGemvRows - 1) {
          r = 0;
          w = (w == cw - 1) ? 0 : w + 1;
        } else {
          ++r;
        }
      } else {
        ++tb;
      }
    }
  }
}

// Collect PE results row block by row block (block b = word b of a Y row),
// assemble one 512-bit word per token, add bias (+ residual) and write it.
void GemvWriteY(tapa::istreams<gemv_tok_t, kGemvPorts>& y_s, tapa::mmap<float_v16> bias,
                tapa::mmap<float_v16> residual, tapa::mmap<float_v16> y, int T, int OC,
                int use_residual) {
  // ybuf[p][r][tb]: row r of the block for token tb * kGemvTok + p
  float ybuf[kGemvTok][kGemvRows][kGemvTb];
#pragma HLS ARRAY_PARTITION variable=ybuf dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=ybuf dim=2 type=complete
  const int ocw = OC / kVec;
  const int groups = OC / kGemvRowBlock;
  for (int t0 = 0; t0 < T; t0 += kGemvChunk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=kTcChunks avg=1
    const int tc = min_int(kGemvChunk, T - t0);
    const int ntb = (tc + kGemvTok - 1) / kGemvTok;
    for (int blk = 0; blk < groups * kGemvPorts; ++blk) {
#pragma HLS LOOP_TRIPCOUNT min=kKC/kVec max=kKC4/kVec avg=kKC4/kVec
      const int pe = blk % kGemvPorts;
      // the PE emits row r, token group tb in order (r outer, tb inner)
      int r = 0, tb = 0;
      for (int i = 0; i < kGemvRows * ntb; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kGemvRows max=kGemvRows*kGemvTb avg=kGemvRows
        gemv_tok_t v;
        for (int c = 0; c < kGemvPorts; ++c)
          if (c == pe) v = y_s[c].read();
        for (int p = 0; p < kGemvTok; ++p)
          for (int k = 0; k < kGemvRows; ++k)
            if (k == r) ybuf[p][k][tb] = v[p];
        if (tb == ntb - 1) {
          tb = 0;
          ++r;
        } else {
          ++tb;
        }
      }
      const float_v16 bw = bias[blk];
      for (int j = 0; j < tc; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=kGemvChunk avg=1
        const int p = j % kGemvTok, g = j / kGemvTok;
        float_v16 o;
        for (int k = 0; k < kGemvRows; ++k) {
          float v = 0.0f;
          for (int q = 0; q < kGemvTok; ++q)
            if (q == p) v = ybuf[q][k][g];
          o[k] = v;
        }
        o += bw;
        const int addr = (t0 + j) * ocw + blk;
        if (use_residual) o += residual[addr];
        y[addr] = o;
      }
    }
  }
}

void GemvKernel(tapa::mmap<float_v16> x, tapa::mmaps<float_v16, kGemvPorts> w,
                tapa::mmap<float_v16> bias, tapa::mmap<float_v16> residual,
                tapa::mmap<float_v16> y, int T, int C, int OC, int use_residual) {
  tapa::streams<float_v16, kGemvPorts, 4> x_s("x_s");
  tapa::streams<float_v16, kGemvPorts, 32> w_s("w_s");
  tapa::streams<gemv_tok_t, kGemvPorts, 2 * kGemvRows * kGemvTb> y_s("y_s");

  tapa::task()
      .invoke(GemvReadX, x, x_s, T, C)
      .invoke<tapa::join, kGemvPorts>(GemvReadW, w, w_s, T, C, OC)
      .invoke<tapa::join, kGemvPorts>(GemvPE, x_s, w_s, y_s, T, C, OC)
      .invoke(GemvWriteY, y_s, bias, residual, y, T, OC, use_residual);
}
