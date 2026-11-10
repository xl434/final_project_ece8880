// Testbench helpers: result bookkeeping and the bitstream flag convention of
// the course labs ("" = software simulation, .xo = TAPA fast cosim,
// .hw_emu.xclbin = Vitis hardware emulation, .xclbin = board).
#ifndef GPT2_KERNEL_TB_UTIL_H_
#define GPT2_KERNEL_TB_UTIL_H_

#include <gflags/gflags.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../src/common/utils.h"
#include "host_pack.h"

DECLARE_string(bitstream);

struct TbReport {
  const char* kernel;
  int n_pass = 0, n_fail = 0;
  explicit TbReport(const char* k) : kernel(k) {}

  // Kernel output vs the src/cpu reference (the pass/fail criterion).
  bool check(const std::string& name, const float* got, const float* ref, size_t n,
             gpt2::Tolerance tol = {}) {
    gpt2::ErrStats s = gpt2::compare(got, ref, n, tol);
    gpt2::print_err(name.c_str(), s);
    if (!s.pass)
      std::printf("    worst element %zu: got %.8g ref %.8g (%zu violations)\n", s.worst,
                  got[s.worst], ref[s.worst], s.violations);
    (s.pass ? n_pass : n_fail)++;
    return s.pass;
  }
  // Informational comparison with the Hugging Face golden data (not counted).
  void info(const std::string& name, const float* got, const std::vector<float>& golden) {
    gpt2::ErrStats s = gpt2::compare(got, golden.data(), golden.size(), {1e-4, 1e-4});
    std::printf("  [vs HF] %-14s max_abs=%.3e max_rel=%.3e norm_rel=%.3e\n", name.c_str(),
                s.max_abs, s.max_rel, s.norm_rel);
  }
  void expect(const std::string& name, bool ok) {
    std::printf("  %-22s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    (ok ? n_pass : n_fail)++;
  }
  int finish() const {
    std::printf("%s: %d passed, %d failed -> %s\n", kernel, n_pass, n_fail,
                n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
  }
};

inline const char* run_mode() {
  const std::string& b = FLAGS_bitstream;
  if (b.empty()) return "software simulation";
  if (b.size() > 3 && b.compare(b.size() - 3, 3, ".xo") == 0) return "TAPA fast cosim (RTL)";
  if (b.find(".hw_emu.xclbin") != std::string::npos) return "Vitis hardware emulation";
  return "FPGA board";
}

inline std::vector<float> random_vec(size_t n, float lo, float hi, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

template <typename T>
aligned_vector<T> to_aligned(const std::vector<T>& v, size_t pad_to = 0) {
  aligned_vector<T> a(v.begin(), v.end());
  if (a.size() < pad_to) a.resize(pad_to, T());
  return a;
}

#endif
