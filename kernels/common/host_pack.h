// Host-side helpers shared by kernel testbenches, tests/ and src/host:
// aligned buffers and the weight packing expected by the GEMV / LM-head
// kernels. Host code only (not compiled by HLS).
#ifndef GPT2_KERNEL_HOST_PACK_H_
#define GPT2_KERNEL_HOST_PACK_H_

#include <tapa.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

// Packs W (OC, C) row-major for a GEMV with `ports` ports and `rows`
// interleaved rows per PE. Row block b = rows [b*rows, (b+1)*rows) goes to
// port b % ports; inside a block the order is [word w][row r][16 lanes].
// Rows >= OC_valid are zero (used to pad the vocabulary to a full block).
inline std::vector<aligned_vector<float>> pack_gemv_weights(const float* W, int OC, int C,
                                                            int ports, int rows,
                                                            int OC_valid = -1) {
  if (OC_valid < 0) OC_valid = OC;
  if (OC % (ports * rows) != 0 || C % 16 != 0) {
    std::fprintf(stderr, "pack_gemv_weights: OC=%d must be a multiple of %d, C=%d of 16\n", OC,
                 ports * rows, C);
    std::exit(EXIT_FAILURE);
  }
  const int cw = C / 16;
  std::vector<aligned_vector<float>> out(ports, aligned_vector<float>((size_t)OC / ports * C));
  std::vector<size_t> pos(ports, 0);
  for (int b = 0; b < OC / rows; ++b) {
    auto& dst = out[b % ports];
    size_t& o = pos[b % ports];
    for (int w = 0; w < cw; ++w)
      for (int r = 0; r < rows; ++r) {
        const int row = b * rows + r;
        for (int k = 0; k < 16; ++k)
          dst[o++] = row < OC_valid ? W[(size_t)row * C + w * 16 + k] : 0.0f;
      }
  }
  return out;
}

#endif
