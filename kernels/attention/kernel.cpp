#include "kernel.h"

#include <cmath>

// Representative trip count for the latency report: full context.
constexpr int kTcPos = kKMaxT - 1;
constexpr int kTcRows = kTcPos * kCWords;

// Split the new token's qkv: q goes to the heads, k and v to AttnKv.
void AttnReadQkv(tapa::mmap<float_v16> qkv, tapa::ostreams<float_v16, kKNH>& q_s,
                 tapa::ostream<float_v16>& knew_s, tapa::ostream<float_v16>& vnew_s) {
  for (int w = 0; w < 3 * kCWords; ++w) {
#pragma HLS PIPELINE II=1
    const float_v16 v = qkv[w];
    if (w < kCWords) {
      for (int h = 0; h < kKNH; ++h)
        if (h == w / kHeadWords) q_s[h].write(v);
    } else if (w < 2 * kCWords) {
      knew_s.write(v);
    } else {
      vnew_s.write(v);
    }
  }
}

// Append the new K/V row, then stream rows 0..pos to the heads.
void AttnKv(tapa::mmap<float_v16> kcache, tapa::mmap<float_v16> vcache,
            tapa::istream<float_v16>& knew_s, tapa::istream<float_v16>& vnew_s,
            tapa::ostreams<float_v16, kKNH>& k_s, tapa::ostreams<float_v16, kKNH>& v_s,
            int layer, int pos) {
  float_v16 kn[kCWords], vn[kCWords];
  const int layer_base = layer * kKMaxT * kCWords;
  const int new_base = layer_base + pos * kCWords;
  for (int w = 0; w < kCWords; ++w) {
#pragma HLS PIPELINE II=1
    kn[w] = knew_s.read();
    kcache[new_base + w] = kn[w];
  }
  for (int w = 0; w < kCWords; ++w) {
#pragma HLS PIPELINE II=1
    vn[w] = vnew_s.read();
    vcache[new_base + w] = vn[w];
  }
  // cached rows 0..pos-1: one K word and one V word per cycle
  int w = 0;
  for (int i = 0; i < pos * kCWords; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=kTcRows avg=kTcRows
    const float_v16 k = kcache[layer_base + i];
    const float_v16 v = vcache[layer_base + i];
    for (int h = 0; h < kKNH; ++h)
      if (h == w / kHeadWords) {
        k_s[h].write(k);
        v_s[h].write(v);
      }
    w = (w == kCWords - 1) ? 0 : w + 1;
  }
  // row pos: the new token, forwarded from on-chip buffers
  for (int w2 = 0; w2 < kCWords; ++w2) {
#pragma HLS PIPELINE II=1
    for (int h = 0; h < kKNH; ++h)
      if (h == w2 / kHeadWords) {
        k_s[h].write(kn[w2]);
        v_s[h].write(vn[w2]);
      }
  }
}

// One head: online softmax over rows 0..pos.
//   m' = max(m, s); l' = l e^(m-m') + e^(s-m'); acc' = acc e^(m-m') + e^(s-m') v
void AttnHead(tapa::istream<float_v16>& q_s, tapa::istream<float_v16>& k_s,
              tapa::istream<float_v16>& v_s, tapa::ostream<float_v16>& o_s, int pos) {
  float_v16 q[kHeadWords];
  for (int w = 0; w < kHeadWords; ++w) {
#pragma HLS PIPELINE II=1
    q[w] = q_s.read() * 0.125f;  // 1/sqrt(64): exact power of two
  }
  float m = -INFINITY, l = 0.0f;
  float_v16 acc[kHeadWords];
#pragma HLS ARRAY_PARTITION variable=acc type=complete
  for (int w = 0; w < kHeadWords; ++w) acc[w] = 0.0f;

  for (int t = 0; t <= pos; ++t) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=kKMaxT avg=kKMaxT
    float_v16 kv[kHeadWords], vv[kHeadWords];
    float d[kHeadWords];
#pragma HLS ARRAY_PARTITION variable=kv type=complete
#pragma HLS ARRAY_PARTITION variable=vv type=complete
#pragma HLS ARRAY_PARTITION variable=d type=complete
    for (int w = 0; w < kHeadWords; ++w) {
      kv[w] = k_s.read();
      vv[w] = v_s.read();
      d[w] = dot16(q[w], kv[w]);
    }
    const float s = (d[0] + d[1]) + (d[2] + d[3]);
    const float m_new = s > m ? s : m;
    const float alpha = std::exp(m - m_new);
    const float p = std::exp(s - m_new);
    l = l * alpha + p;
    for (int w = 0; w < kHeadWords; ++w) acc[w] = acc[w] * alpha + vv[w] * p;
    m = m_new;
  }
  const float inv = 1.0f / l;
  for (int w = 0; w < kHeadWords; ++w) {
#pragma HLS PIPELINE II=1
    o_s.write(acc[w] * inv);
  }
}

void AttnWrite(tapa::istreams<float_v16, kKNH>& o_s, tapa::mmap<float_v16> attn_out) {
  for (int w = 0; w < kCWords; ++w) {
#pragma HLS PIPELINE II=1
    float_v16 v;
    for (int h = 0; h < kKNH; ++h)
      if (h == w / kHeadWords) v = o_s[h].read();
    attn_out[w] = v;
  }
}

void AttentionKernel(tapa::mmap<float_v16> qkv, tapa::mmap<float_v16> kcache,
                     tapa::mmap<float_v16> vcache, tapa::mmap<float_v16> attn_out, int layer,
                     int pos) {
  tapa::streams<float_v16, kKNH, kHeadWords> q_s("q_s");
  tapa::stream<float_v16, kCWords> knew_s("knew_s");
  tapa::stream<float_v16, kCWords> vnew_s("vnew_s");
  tapa::streams<float_v16, kKNH, 2 * kHeadWords> k_s("k_s");
  tapa::streams<float_v16, kKNH, 2 * kHeadWords> v_s("v_s");
  tapa::streams<float_v16, kKNH, kHeadWords> o_s("o_s");
  tapa::task()
      .invoke(AttnReadQkv, qkv, q_s, knew_s, vnew_s)
      .invoke(AttnKv, kcache, vcache, knew_s, vnew_s, k_s, v_s, layer, pos)
      .invoke<tapa::join, kKNH>(AttnHead, q_s, k_s, v_s, o_s, pos)
      .invoke(AttnWrite, o_s, attn_out);
}
