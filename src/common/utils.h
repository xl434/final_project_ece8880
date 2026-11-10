// Shared helpers: argmax, timers, binary file I/O, tolerance-based comparison.
#ifndef GPT2_COMMON_UTILS_H_
#define GPT2_COMMON_UTILS_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace gpt2 {

// First index of the maximum (ties -> lowest index, like torch.argmax / numpy).
inline int argmax(const float* x, int n) {
  int best = 0;
  float bv = x[0];
  for (int i = 1; i < n; ++i)
    if (x[i] > bv) {
      bv = x[i];
      best = i;
    }
  return best;
}

class Timer {
 public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}
  void reset() { t0_ = std::chrono::steady_clock::now(); }
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

template <typename T>
std::vector<T> read_binary(const std::string& path, long expected_count = -1) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "read_binary: cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<T> v(bytes / sizeof(T));
  if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) {
    std::fprintf(stderr, "read_binary: short read %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::fclose(f);
  if (expected_count >= 0 && (long)v.size() != expected_count) {
    std::fprintf(stderr, "read_binary: %s has %zu elements, expected %ld\n", path.c_str(),
                 v.size(), expected_count);
    std::exit(EXIT_FAILURE);
  }
  return v;
}

template <typename T>
void write_binary(const std::string& path, const T* data, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(data, sizeof(T), n, f) != n) {
    std::fprintf(stderr, "write_binary: cannot write %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::fclose(f);
}

// Tolerance: element i passes if |got - ref| <= atol + rtol * |ref|, with
// atol = atol_rel * max|ref| (scale-aware absolute floor for near-zero refs).
struct Tolerance {
  double rtol = 1e-4;
  double atol_rel = 1e-4;
};

struct ErrStats {
  size_t n = 0;
  double max_abs = 0;      // max |got - ref|
  double max_rel = 0;      // max |got - ref| / |ref| over |ref| >= 1e-2 * max|ref|
  double norm_rel = 0;     // ||got - ref||_2 / ||ref||_2
  double ref_max_abs = 0;  // max |ref|
  size_t worst = 0;        // index of max |got - ref|
  size_t violations = 0;   // elements outside the tolerance
  bool pass = true;
};

inline ErrStats compare(const float* got, const float* ref, size_t n, Tolerance tol = {}) {
  ErrStats s;
  s.n = n;
  for (size_t i = 0; i < n; ++i) s.ref_max_abs = std::max(s.ref_max_abs, (double)std::fabs(ref[i]));
  const double atol = tol.atol_rel * s.ref_max_abs;
  const double rel_floor = 1e-2 * s.ref_max_abs;
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double r = ref[i];
    double d = std::fabs((double)got[i] - r);
    if (!(d == d)) d = INFINITY;  // NaN counts as an error
    if (d > s.max_abs) {
      s.max_abs = d;
      s.worst = i;
    }
    if (std::fabs(r) >= rel_floor && std::fabs(r) > 0)
      s.max_rel = std::max(s.max_rel, d / std::fabs(r));
    num += d * d;
    den += r * r;
    if (!(d <= atol + tol.rtol * std::fabs(r))) ++s.violations;
  }
  s.norm_rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  s.pass = s.violations == 0;
  return s;
}

inline void print_err(const char* name, const ErrStats& s) {
  std::printf("  %-22s n=%-7zu max_abs=%.3e max_rel=%.3e norm_rel=%.3e |ref|max=%.3e  %s\n",
              name, s.n, s.max_abs, s.max_rel, s.norm_rel, s.ref_max_abs,
              s.pass ? "PASS" : "FAIL");
}

}  // namespace gpt2

#endif
