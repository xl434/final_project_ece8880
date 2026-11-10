// Decode attention with a KV cache (one new token, one layer).
//
// Inputs: qkv = [q | k | v] of the new token (3C floats, output of the QKV
// GEMV); kcache / vcache = (L, maxT, C) caches; `layer`; `pos` = position of
// the new token (= number of tokens already cached).
// The kernel appends k, v at row `pos` of this layer and computes, for every
// head h, softmax(q_h K_h^T / sqrt(64)) V_h over rows 0..pos with an online
// (single-pass) softmax. Output: C floats, heads concatenated.
//
// TAPA dataflow:
//   AttnReadQkv --q_s[h]--------------------------------> AttnHead[h] --o_s[h]--> AttnWrite
//              \--knew_s, vnew_s--> AttnKv --k_s[h], v_s[h]--^
// AttnKv streams one 512-bit K word and one V word per cycle (row t = 48
// words; word w belongs to head w / 4), so all 12 heads work in parallel and
// each head receives its 64-float slice of every row. Cache rows < pos come
// from HBM; row pos is forwarded on chip (and written back to HBM), so there
// is no read-after-write through memory.
#ifndef GPT2_KERNEL_ATTENTION_H_
#define GPT2_KERNEL_ATTENTION_H_

#include "../common/kernel_common.h"

void AttentionKernel(tapa::mmap<float_v16> qkv, tapa::mmap<float_v16> kcache,
                     tapa::mmap<float_v16> vcache, tapa::mmap<float_v16> attn_out, int layer,
                     int pos);

#endif
