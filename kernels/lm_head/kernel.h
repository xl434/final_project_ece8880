// LM head fused with argmax: next = argmax_{v < V} (wte[v] . x), x = ln_f
// output (C = 768). Only the token ID and its logit leave the chip, so the
// 50257 logits are never written to memory.
//
// Same weight-streaming structure as the GEMV (kLmPorts PEs, 16 MACs per
// cycle each, kLmRows interleaved rows). wte (Vp = 50304 rows, rows >= V are
// zero padding) must be packed with pack_gemv_weights(..., kLmPorts, kLmRows).
// Each PE keeps a running best per interleaved row lane, reduces the lanes at
// the end, and LmMerge picks the global maximum. Ties resolve to the lowest
// token ID (same as src/cpu argmax / torch.argmax). Padding rows are masked.
#ifndef GPT2_KERNEL_LM_HEAD_H_
#define GPT2_KERNEL_LM_HEAD_H_

#include "../common/kernel_common.h"

#ifndef LM_PORTS
#define LM_PORTS 8
#endif
constexpr int kLmPorts = LM_PORTS;
constexpr int kLmRows = 16;
constexpr int kLmRowBlock = kLmPorts * kLmRows;
static_assert(kKVp % kLmRowBlock == 0, "padded vocabulary must fill whole row blocks");

void LmHeadKernel(tapa::mmap<float_v16> x, tapa::mmaps<float_v16, kLmPorts> w,
                  tapa::mmap<int> out_idx, tapa::mmap<float> out_val);

#endif
