// GELU (tanh approximation, "gelu_new"), 16 lanes per cycle.
//   gelu(x) = 0.5 x (1 + tanh(u)),  u = sqrt(2/pi) (x + 0.044715 x^3)
// computed through the exact identity 0.5 (1 + tanh(u)) = 1 / (1 + exp(-2u)),
// i.e. gelu(x) = x / (1 + exp(-2u)). This needs one exp and one divide per
// element and, unlike 1 + tanh(u), has no cancellation for negative x.
// TAPA dataflow: GeluRead -> GeluCore -> GeluWrite.
#ifndef GPT2_KERNEL_GELU_H_
#define GPT2_KERNEL_GELU_H_

#include "../common/kernel_common.h"

// x, y: n_words * 16 floats.
void GeluKernel(tapa::mmap<float_v16> x, tapa::mmap<float_v16> y, int n_words);

#endif
