#include "kernel.h"

#include <cmath>

// Representative trip counts for the latency report: one decode token at the
// full context (cur_pos = 1023).
constexpr int kTcRows = kKMaxT;
constexpr int kTcBlocks = (kKMaxT + kAttnQ - 1) / kAttnQ;

static int min_int(int a, int b) { return a < b ? a : b; }

// Copies word w (16 floats) into slot w % 4 of head w / 4's vector.
static void put_word(attn_vec_t vec[kKNH], int w, const float_v16& v) {
#pragma HLS INLINE
  for (int h = 0; h < kKNH; ++h)
    for (int k = 0; k < kHeadWords; ++k)
      if (h * kHeadWords + k == w)
        for (int i = 0; i < kVec; ++i) vec[h][k * kVec + i] = v[i];
}

// For each query block: send the block's queries to the heads, then stream
// every K/V row the block needs (rows 0 .. cur_pos + last query).
void AttnKv(tapa::mmap<float_v16> qkv, tapa::mmap<float_v16> kcache,
            tapa::mmap<float_v16> vcache, tapa::ostreams<attn_vec_t, kKNH>& q_s,
            tapa::ostreams<attn_vec_t, kKNH>& k_s, tapa::ostreams<attn_vec_t, kKNH>& v_s,
            int layer, int cur_pos, int T) {
  const int lb = layer * kKMaxT * kCWords;  // first word of this layer's cache
  const int rw = 3 * kCWords;               // words per qkv row
  attn_vec_t qv[kKNH], kv[kKNH], vv[kKNH];
#pragma HLS ARRAY_PARTITION variable=qv type=complete
#pragma HLS ARRAY_PARTITION variable=kv type=complete
#pragma HLS ARRAY_PARTITION variable=vv type=complete
  for (int i0 = 0; i0 < T; i0 += kAttnQ) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=kTcBlocks avg=1
    const int qn = min_int(kAttnQ, T - i0);
    // queries of the block: word w of query qi goes to head w / 4
    int qi = 0, w = 0;
    for (int i = 0; i < qn * kCWords; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kAttnQ*kCWords avg=kCWords
      put_word(qv, w, qkv[(i0 + qi) * rw + w]);
      if (w % kHeadWords == kHeadWords - 1) q_s[w / kHeadWords].write(qv[w / kHeadWords]);
      if (w == kCWords - 1) {
        w = 0;
        ++qi;
      } else {
        ++w;
      }
    }
    // cached rows 0 .. cur_pos-1: one K word and one V word per cycle
    int j = 0;
    w = 0;
    for (int i = 0; i < cur_pos * kCWords; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=kTcRows*kCWords avg=kTcRows*kCWords
      put_word(kv, w, kcache[lb + i]);
      put_word(vv, w, vcache[lb + i]);
      if (w % kHeadWords == kHeadWords - 1) {
        k_s[w / kHeadWords].write(kv[w / kHeadWords]);
        v_s[w / kHeadWords].write(vv[w / kHeadWords]);
      }
      w = (w == kCWords - 1) ? 0 : w + 1;
    }
    // new rows cur_pos .. cur_pos + i0 + qn - 1, read from qkv (K words, then
    // V words of each row); the rows of this block are also written to the cache
    int t = 0, c = 0;  // c = word in [k | v] of the row (0 .. 2*48-1)
    for (int i = 0; i < (i0 + qn) * 2 * kCWords; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2*kCWords max=kKMaxT*2*kCWords avg=2*kCWords
      const float_v16 word = qkv[t * rw + kCWords + c];
      const bool is_k = c < kCWords;
      const int ww = is_k ? c : c - kCWords;
      const int addr = lb + (cur_pos + t) * kCWords + ww;
      const bool own = t >= i0;  // row belongs to this block: write it once
      if (is_k) {
        put_word(kv, ww, word);
        if (own) kcache[addr] = word;
      } else {
        put_word(vv, ww, word);
        if (own) vcache[addr] = word;
        if (ww % kHeadWords == kHeadWords - 1) {
          k_s[ww / kHeadWords].write(kv[ww / kHeadWords]);
          v_s[ww / kHeadWords].write(vv[ww / kHeadWords]);
        }
      }
      if (c == 2 * kCWords - 1) {
        c = 0;
        ++t;
      } else {
        ++c;
      }
    }
  }
}

static float dot64(const attn_vec_t& a, const attn_vec_t& b) {
#pragma HLS INLINE
  float p[kVec];
#pragma HLS ARRAY_PARTITION variable=p type=complete
  for (int k = 0; k < kHeadWords; ++k) {
    float q[kVec];
#pragma HLS ARRAY_PARTITION variable=q type=complete
    for (int i = 0; i < kVec; ++i) q[i] = a[k * kVec + i] * b[k * kVec + i];
    p[k] = tree_sum16(q);
  }
  return (p[0] + p[1]) + (p[2] + p[3]);
}

// One head: online softmax for up to kAttnQ queries of a block. Per row, the
// head loops over the kAttnQ query slots (one every kAttnII cycles); query qi of the
// block only uses rows 0 .. cur_pos + i0 + qi (causal mask).
//   m' = max(m, s); l' = l e^(m-m') + e^(s-m'); acc' = acc e^(m-m') + e^(s-m') v
void AttnHead(tapa::istream<attn_vec_t>& q_s, tapa::istream<attn_vec_t>& k_s,
              tapa::istream<attn_vec_t>& v_s, tapa::ostream<attn_vec_t>& o_s, int cur_pos,
              int T) {
  float q[kAttnQ][kKHS], acc[kAttnQ][kKHS], m[kAttnQ], l[kAttnQ];
#pragma HLS ARRAY_PARTITION variable=q dim=2 type=complete
#pragma HLS ARRAY_PARTITION variable=acc dim=2 type=complete
  for (int i0 = 0; i0 < T; i0 += kAttnQ) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=kTcBlocks avg=1
    const int qn = min_int(kAttnQ, T - i0);
    for (int qi = 0; qi < qn; ++qi) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=kAttnQ avg=1
      const attn_vec_t v = q_s.read();
      for (int d = 0; d < kKHS; ++d) {
        q[qi][d] = v[d] * 0.125f;  // 1/sqrt(64): exact power of two
        acc[qi][d] = 0.0f;
      }
      m[qi] = -INFINITY;
      l[qi] = 0.0f;
    }
    // A query slot is updated every kAttnQ iterations = kAttnQ * kAttnII cycles;
    // the update chain (max, exp, multiply-add) must be shorter than that.
    const int nrows = cur_pos + i0 + qn;
    int j = 0, qi = 0;
    attn_vec_t kr, vr;
    for (int it = 0; it < nrows * kAttnQ; ++it) {
#pragma HLS PIPELINE II=kAttnII
#pragma HLS LOOP_TRIPCOUNT min=kAttnQ max=kTcRows*kAttnQ avg=kTcRows*kAttnQ
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW dependent=true distance=kAttnQ
#pragma HLS DEPENDENCE variable=m type=inter direction=RAW dependent=true distance=kAttnQ
#pragma HLS DEPENDENCE variable=l type=inter direction=RAW dependent=true distance=kAttnQ
      if (qi == 0) {
        kr = k_s.read();
        vr = v_s.read();
      }
      if (qi < qn && j <= cur_pos + i0 + qi) {
        attn_vec_t qq;
        for (int d = 0; d < kKHS; ++d) qq[d] = q[qi][d];
        const float s = dot64(qq, kr);
        const float m_old = m[qi];
        const float m_new = s > m_old ? s : m_old;
        const float alpha = std::exp(m_old - m_new);
        const float p = std::exp(s - m_new);
        l[qi] = l[qi] * alpha + p;
        for (int d = 0; d < kKHS; ++d) acc[qi][d] = acc[qi][d] * alpha + vr[d] * p;
        m[qi] = m_new;
      }
      if (qi == kAttnQ - 1) {
        qi = 0;
        ++j;
      } else {
        ++qi;
      }
    }
    for (int qo = 0; qo < qn; ++qo) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=kAttnQ avg=1
      const float inv = 1.0f / l[qo];
      attn_vec_t o;
      for (int d = 0; d < kKHS; ++d) o[d] = acc[qo][d] * inv;
      o_s.write(o);
    }
  }
}

// Collect the heads' outputs, one token row (48 words) at a time.
void AttnWrite(tapa::istreams<attn_vec_t, kKNH>& o_s, tapa::mmap<float_v16> attn_out, int T) {
  attn_vec_t o;
  int h = 0, k = 0;
  for (int i = 0; i < T * kCWords; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kCWords max=kKMaxT*kCWords avg=kCWords
    if (k == 0)
      for (int c = 0; c < kKNH; ++c)
        if (c == h) o = o_s[c].read();
    float_v16 word;
    for (int x = 0; x < kVec; ++x) word[x] = o[k * kVec + x];
    attn_out[i] = word;
    if (k == kHeadWords - 1) {
      k = 0;
      h = (h == kKNH - 1) ? 0 : h + 1;
    } else {
      ++k;
    }
  }
}

void AttentionKernel(tapa::mmap<float_v16> qkv, tapa::mmap<float_v16> kcache,
                     tapa::mmap<float_v16> vcache, tapa::mmap<float_v16> attn_out, int layer,
                     int cur_pos, int T) {
  tapa::streams<attn_vec_t, kKNH, 2> q_s("q_s");
  tapa::streams<attn_vec_t, kKNH, 4> k_s("k_s");
  tapa::streams<attn_vec_t, kKNH, 4> v_s("v_s");
  tapa::streams<attn_vec_t, kKNH, 2> o_s("o_s");
  tapa::task()
      .invoke(AttnKv, qkv, kcache, vcache, q_s, k_s, v_s, layer, cur_pos, T)
      .invoke<tapa::join, kKNH>(AttnHead, q_s, k_s, v_s, o_s, cur_pos, T)
      .invoke(AttnWrite, o_s, attn_out, T);
}
