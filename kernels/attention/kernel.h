// Attention with a KV cache, for decode (T = 1) and prefill (T new tokens).
//
// Inputs: qkv = [q | k | v] rows of the T new tokens (T x 3C floats, output of
// the QKV GEMV); kcache / vcache = (L, maxT, C) caches; `layer`; `cur_pos` =
// position of the first new token (= number of tokens already cached).
// The kernel appends the T new k, v rows at rows cur_pos .. cur_pos+T-1 of
// this layer and computes, for every new token t and head h,
// softmax(q_h K_h^T / sqrt(64)) V_h over rows 0..cur_pos+t (causal), with an
// online (single-pass) softmax. Output: T x C floats, heads concatenated.
//
// TAPA dataflow:
//   AttnKv --q_s[h], k_s[h], v_s[h]--> AttnHead[h] --o_s[h]--> AttnWrite
// Queries are processed in blocks of up to kAttnQ. For each block AttnKv
// streams every K/V row the block needs once (old rows from the cache, one K
// and one V word per cycle; new rows from qkv, which also writes them to the
// cache). Each head receives its 64-float slice of every row and keeps the
// softmax state of all queries of the block; it updates one query every
// kAttnII cycles, so a row costs kAttnQ * kAttnII = 48 cycles, the same as a
// cached row takes to arrive. Decode is a block with one query.
// kAttnII trades DSPs for prefill speed: II = 12 (4 queries per block) costs
// about 205 DSPs per head; II = 4 (12 queries) about 346; II = 1 about 850,
// which for 12 heads exceeds the chip.
#ifndef GPT2_KERNEL_ATTENTION_H_
#define GPT2_KERNEL_ATTENTION_H_

#include "../common/kernel_common.h"

#ifndef ATTN_II
#define ATTN_II 12
#endif
constexpr int kAttnII = ATTN_II;          // cycles per query update
constexpr int kAttnQ = kCWords / kAttnII;  // queries per block: kAttnQ * kAttnII = 48 cycles per row
static_assert(kCWords % kAttnII == 0, "ATTN_II must divide 48");
using attn_vec_t = hls::vector<float, kKHS>;  // one head's 64-float slice of a row

// Not `pos`: Vitis HLS renames a port named pos to pos_r, which breaks TAPA's
// RTL wiring.
void AttentionKernel(tapa::mmap<float_v16> qkv, tapa::mmap<float_v16> kcache,
                     tapa::mmap<float_v16> vcache, tapa::mmap<float_v16> attn_out, int layer,
                     int cur_pos, int T);

#endif
