// Decode GEMV: y = W x + b (+ residual), W is (OC, C) FP32.
//
// Architecture (TAPA dataflow):
//   x mmap --GemvReadX--> x_s[p] (broadcast)
//   w[p] mmap --GemvReadW[p]--> w_s[p] --GemvPE[p]--> y_s[p] --GemvWriteY--> y mmap
//                                                    bias, residual mmaps --^
// Each of the kGemvPorts PEs owns one HBM port and consumes one 512-bit word
// (16 weights) per cycle: 16 FP32 MACs/cycle/port. Rows are processed in
// groups of kGemvRows interleaved rows so that consecutive iterations update
// different accumulators; the FP add latency (< kGemvRows cycles) is hidden
// and the loop runs at II=1 without reassociation.
//
// Weights must be pre-packed on the host (kernels/common/host_pack.h,
// pack_gemv_weights): port p holds row blocks b = g*kGemvPorts + p, each
// stored as [word w][row r][16 floats].
#ifndef GPT2_KERNEL_GEMV_H_
#define GPT2_KERNEL_GEMV_H_

#include "../common/kernel_common.h"

#ifndef GEMV_PORTS
#define GEMV_PORTS 8
#endif
constexpr int kGemvPorts = GEMV_PORTS;  // HBM ports (= PEs) for the weights
constexpr int kGemvRows = 16;           // interleaved rows per PE (> fadd latency)
constexpr int kGemvRowBlock = kGemvPorts * kGemvRows;  // OC must be a multiple

void GemvKernel(tapa::mmap<float_v16> x, tapa::mmaps<float_v16, kGemvPorts> w,
                tapa::mmap<float> bias, tapa::mmap<float> residual, tapa::mmap<float> y,
                int C, int OC, int use_residual);

#endif
