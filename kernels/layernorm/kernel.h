// LayerNorm over rows of C channels: y = (x - mean) * rstd * gamma + beta,
// rstd = 1 / sqrt(var + 1e-5), biased variance (as in GPT-2 / llm.c).
//
// TAPA dataflow: LnReadX -> LnCore -> LnWrite. LnCore buffers one row
// (C/16 words) and makes three II=1 passes over it: sum, squared deviation,
// normalize. Each pass folds the 16 lanes with an adder tree and accumulates
// into kLnAcc interleaved partial sums so the FP add latency is hidden.
#ifndef GPT2_KERNEL_LAYERNORM_H_
#define GPT2_KERNEL_LAYERNORM_H_

#include "../common/kernel_common.h"

constexpr int kLnAcc = 8;  // interleaved partial sums (> fadd latency)

// x, y: rows x C floats; gamma, beta: C floats. C must be a multiple of 16
// and <= 768.
void LayerNormKernel(tapa::mmap<float_v16> x, tapa::mmap<float_v16> gamma,
                     tapa::mmap<float_v16> beta, tapa::mmap<float_v16> y, int rows, int C);

#endif
