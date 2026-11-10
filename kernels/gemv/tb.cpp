// Unit testbench for GemvKernel: real GPT-2 weights and activations from
// data/ (first decode step) plus random edge cases. Reference: src/cpu
// matmul_forward (+ residual_forward).
#include <cstdio>
#include <vector>

#include "../../data/golden_loader.h"
#include "../../src/common/weights.h"
#include "../../src/cpu/ops.h"
#include "../common/tb_util.h"
#include "kernel.h"

DEFINE_string(bitstream, "", "path to bitstream (.xo / .xclbin); empty = software simulation");

using namespace gpt2;

// Runs the kernel for y = W x + b (+ res) and checks it against src/cpu.
static void run_case(TbReport& rep, const std::string& name, const float* W, const float* b,
                     const std::vector<float>& x, const std::vector<float>* res, int C, int OC,
                     std::vector<float>& y_out) {
  auto wp = pack_gemv_weights(W, OC, C, kGemvPorts, kGemvRows);
  auto xa = to_aligned(x);
  aligned_vector<float> ba(b, b + OC);
  aligned_vector<float> ra = res ? to_aligned(*res) : aligned_vector<float>(16, 0.0f);
  aligned_vector<float> y(OC);
  double us = tapa::invoke(GemvKernel, FLAGS_bitstream,
                           tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
                           tapa::read_only_mmaps<float, kGemvPorts>(wp).reinterpret<float_v16>(),
                           tapa::read_only_mmap<float>(ba), tapa::read_only_mmap<float>(ra),
                           tapa::write_only_mmap<float>(y), C, OC, res ? 1 : 0) * 1e-3;
  std::vector<float> ref(OC);
  matmul_forward(ref.data(), x.data(), W, b, 1, C, OC);
  if (res) residual_forward(ref.data(), ref.data(), res->data(), OC);
  std::printf("%s: C=%d OC=%d residual=%d (%.0f us)\n", name.c_str(), C, OC, res ? 1 : 0, us);
  rep.check(name, y.data(), ref.data(), OC);
  y_out.assign(y.begin(), y.end());
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::printf("GemvKernel: %d ports x %d rows, mode: %s\n", kGemvPorts, kGemvRows, run_mode());
  TbReport rep("gemv");
  Weights w;
  load_gpt2_weights(w, default_model_dir() + "/gpt2_124M.bin");
  std::vector<float> y;

  // 1) layer 0 QKV: x = ln1 output of the first decode step
  run_case(rep, "l00_qkv", w.l_qkvw(0), w.l_qkvb(0), golden::act(0, "ln1"), nullptr, kC, kC3, y);
  rep.info("l00_qkv", y.data(), golden::act(0, "qkv"));

  // 2) layer 5 attention projection fused with residual 1
  {
    auto res = golden::act(4, "res2");  // block input = previous block output
    run_case(rep, "l05_attproj+res", w.l_attprojw(5), w.l_attprojb(5), golden::act(5, "atty"),
             &res, kC, kC, y);
    rep.info("l05_res1", y.data(), golden::act(5, "res1"));
  }
  // 3) layer 11 MLP fc (768 -> 3072)
  run_case(rep, "l11_fc", w.l_fcw(11), w.l_fcb(11), golden::act(11, "ln2"), nullptr, kC, kC4, y);
  rep.info("l11_fch", y.data(), golden::act(11, "fch"));

  // 4) layer 11 MLP projection (3072 -> 768) fused with residual 2
  {
    auto res = golden::act(11, "res1");
    run_case(rep, "l11_fcproj+res", w.l_fcprojw(11), w.l_fcprojb(11), golden::act(11, "fch_gelu"),
             &res, kC4, kC, y);
    rep.info("l11_res2", y.data(), golden::act(11, "res2"));
  }

  // 5) edge cases with random data: smallest legal shape (one word, one row
  //    block), and a long reduction with mixed magnitudes.
  {
    const int C = 16, OC = kGemvRowBlock;
    auto W = random_vec((size_t)OC * C, -1, 1, 1), b = random_vec(OC, -1, 1, 2);
    run_case(rep, "rand_min_shape", W.data(), b.data(), random_vec(C, -1, 1, 3), nullptr, C, OC, y);
  }
  {
    const int C = kC4, OC = 2 * kGemvRowBlock;
    auto W = random_vec((size_t)OC * C, -1, 1, 4), b = random_vec(OC, -100, 100, 5);
    auto x = random_vec(C, -50, 50, 6), res = random_vec(OC, -1000, 1000, 7);
    run_case(rep, "rand_long_mixed+res", W.data(), b.data(), x, &res, C, OC, y);
  }
  return rep.finish();
}
