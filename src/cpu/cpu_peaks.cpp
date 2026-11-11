// Measures the two CPU ceilings used in the analysis (roofline):
//   * memory bandwidth: read-only streaming over a 1 GiB array (far larger than
//     the 32 MB L3 slice), the access pattern of GPT-2 decode reading weights;
//   * FP32 compute: independent AVX2 FMA chains on registers only (no memory).
// Each test runs with 1 thread and with all threads; best of 5 repetitions.
//   cpu_peaks [--threads 1,8]   -> results/cpu_peaks.csv
#include <immintrin.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "../common/utils.h"

namespace {

double bandwidth_gbs(const float* a, size_t n, int threads) {
  double best = 0;
  for (int rep = 0; rep < 5; ++rep) {
    gpt2::Timer t;
    double total = 0;
#pragma omp parallel num_threads(threads) reduction(+ : total)
    {
      __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i += 32) {
        s0 = _mm256_add_ps(s0, _mm256_load_ps(a + i));
        s1 = _mm256_add_ps(s1, _mm256_load_ps(a + i + 8));
        s2 = _mm256_add_ps(s2, _mm256_load_ps(a + i + 16));
        s3 = _mm256_add_ps(s3, _mm256_load_ps(a + i + 24));
      }
      float o[8];
      _mm256_storeu_ps(o, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
      for (float v : o) total += v;
    }
    const double s = t.ms() / 1e3;
    if (total == 12345.0) std::printf(" ");  // keep the sum alive
    best = std::max(best, n * sizeof(float) / s / 1e9);
  }
  return best;
}

double compute_gflops(int threads) {
  const long iters = 50'000'000;
  double best = 0;
  for (int rep = 0; rep < 5; ++rep) {
    gpt2::Timer t;
    float sink = 0;
#pragma omp parallel num_threads(threads) reduction(+ : sink)
    {
      // 10 independent chains hide the FMA latency (4 cycles x 2 pipes)
      __m256 a[10];
      const __m256 m = _mm256_set1_ps(0.999999f), c = _mm256_set1_ps(1e-7f);
      for (int k = 0; k < 10; ++k) a[k] = _mm256_set1_ps(1.0f + k);
      for (long i = 0; i < iters; ++i)
        for (int k = 0; k < 10; ++k) a[k] = _mm256_fmadd_ps(a[k], m, c);
      float o[8];
      __m256 s = a[0];
      for (int k = 1; k < 10; ++k) s = _mm256_add_ps(s, a[k]);
      _mm256_storeu_ps(o, s);
      sink += o[0];
    }
    const double s = t.ms() / 1e3;
    if (sink == 12345.0f) std::printf(" ");
    best = std::max(best, (double)iters * 10 * 8 * 2 * threads / s / 1e9);
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string threads_arg = "1," + std::to_string(omp_get_max_threads());
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--threads") threads_arg = argv[i + 1];

  const size_t n = (size_t)1 << 28;  // 2^28 floats = 1 GiB
  float* a = static_cast<float*>(std::aligned_alloc(64, n * sizeof(float)));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) a[i] = 1.0f;  // first touch by the reading threads

  const std::string path = std::string(GPT2_DEFAULT_RESULTS_DIR) + "/cpu_peaks.csv";
  FILE* f = std::fopen(path.c_str(), "w");
  std::fprintf(f, "threads,mem_bandwidth_gbs,fp32_gflops\n");
  std::stringstream ss(threads_arg);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const int t = std::atoi(item.c_str());
    const double bw = bandwidth_gbs(a, n, t), gf = compute_gflops(t);
    std::printf("threads %d: memory read bandwidth %.1f GB/s, FP32 FMA peak %.1f GFLOP/s\n", t, bw,
                gf);
    std::fprintf(f, "%d,%.2f,%.2f\n", t, bw, gf);
  }
  std::fclose(f);
  std::printf("wrote %s\n", path.c_str());
  std::free(a);
  return 0;
}
