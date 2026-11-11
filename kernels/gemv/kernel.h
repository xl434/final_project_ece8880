// GEMV / GEMM for decode and prefill: Y = X W^T + b (+ residual).
// X is (T, C) row-major (T tokens), W is (OC, C) FP32, Y is (T, OC).
// Decode is T = 1; prefill runs all T prompt tokens in one call.
//
// Architecture (TAPA dataflow):
//   x mmap --GemvReadX--> x_s[p] (broadcast)
//   w[p] mmap --GemvReadW[p]--> w_s[p] --GemvPE[p]--> y_s[p] --GemvWriteY--> y mmap
//                                                    bias, residual mmaps --^
// Tokens are processed in chunks of up to kGemvChunk. For each chunk every PE
// keeps the chunk's X rows on chip (URAM) and the weights are streamed once,
// so each 512-bit weight word is reused for every token of the chunk. Per
// cycle a PE multiplies one weight word with kGemvTok tokens (kGemvTok x 16
// MACs); a word stays in the PE for ceil(tc / kGemvTok) cycles. Decode
// (T = 1) still reads one weight word per cycle per port.
//
// Rows are processed in groups of kGemvRows interleaved rows (a row block),
// so an accumulator is reused only every kGemvRows x ceil(tc / kGemvTok)
// cycles, longer than the FP add latency: the loop runs at II = 1.
//
// Weights must be pre-packed on the host (kernels/common/host_pack.h,
// pack_gemv_weights): port p holds row blocks b = g*kGemvPorts + p, each
// stored as [word w][row r][16 floats]. Row block b is word b of a Y row.
#ifndef GPT2_KERNEL_GEMV_H_
#define GPT2_KERNEL_GEMV_H_

#include "../common/kernel_common.h"

#ifndef GEMV_PORTS
#define GEMV_PORTS 8
#endif
#ifndef GEMV_TOKENS
#define GEMV_TOKENS 4
#endif
#ifndef GEMV_CHUNK
#define GEMV_CHUNK 64
#endif
constexpr int kGemvPorts = GEMV_PORTS;  // HBM ports (= PEs) for the weights
constexpr int kGemvRows = 16;           // interleaved rows per PE (> fadd latency)
constexpr int kGemvRowBlock = kGemvPorts * kGemvRows;  // OC must be a multiple
constexpr int kGemvTok = GEMV_TOKENS;   // tokens multiplied per weight word per cycle
constexpr int kGemvChunk = GEMV_CHUNK;  // tokens whose X rows are held on chip
constexpr int kGemvTb = kGemvChunk / kGemvTok;  // token groups per chunk
static_assert(kGemvChunk % kGemvTok == 0, "chunk must be a multiple of the token lanes");

// One row's partial results for kGemvTok tokens, from a PE to the writer.
using gemv_tok_t = hls::vector<float, kGemvTok>;

void GemvKernel(tapa::mmap<float_v16> x, tapa::mmaps<float_v16, kGemvPorts> w,
                tapa::mmap<float_v16> bias, tapa::mmap<float_v16> residual,
                tapa::mmap<float_v16> y, int T, int C, int OC, int use_residual);

#endif
