// Residual add y = a + b, 16 lanes per cycle.
// TAPA dataflow: ResReadA, ResReadB -> ResAdd -> ResWrite.
// (The GEMV kernel can also fuse this add into its output stage; this
// standalone kernel covers the unfused path.)
#ifndef GPT2_KERNEL_RESIDUAL_H_
#define GPT2_KERNEL_RESIDUAL_H_

#include "../common/kernel_common.h"

void ResidualKernel(tapa::mmap<float_v16> a, tapa::mmap<float_v16> b, tapa::mmap<float_v16> y,
                    int n_words);

#endif
