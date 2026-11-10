// Unit testbench for LmHeadKernel (LM head + argmax). Reference: src/cpu
// lm_head_forward + argmax.
//   * real: ln_f output of the first decode step with the real wte. GPT-2's
//     logits are negative (about -108 here), so the zero padding rows
//     (logit 0) would win if they were not masked: this case checks masking.
//   * tie: two identical rows (ids 100 and 40000) share the maximum -> 100.
//   * last row: the maximum is the last valid token (50256).
#include "../../data/golden_loader.h"
#include "../../src/common/config.h"
#include "../../src/common/weights.h"
#include "../../src/cpu/ops.h"
#include "../common/tb_util.h"
#include "kernel.h"

DEFINE_string(bitstream, "", "path to bitstream (.xo / .xclbin); empty = software simulation");

using namespace gpt2;

static void run_case(TbReport& rep, const std::string& name, const float* wte,
                     const std::vector<float>& x, int expect_id = -1) {
  auto wp = pack_gemv_weights(wte, kVp, kC, kLmPorts, kLmRows, kV);
  auto xa = to_aligned(x);
  aligned_vector<int> idx(16, -1);
  aligned_vector<float> val(16, 0.0f);
  double us = tapa::invoke(LmHeadKernel, FLAGS_bitstream,
                           tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
                           tapa::read_only_mmaps<float, kLmPorts>(wp).reinterpret<float_v16>(),
                           tapa::write_only_mmap<int>(idx), tapa::write_only_mmap<float>(val)) *
              1e-3;
  std::vector<float> logits(kV);
  lm_head_forward(logits.data(), x.data(), wte, kC, kV);
  const int ref = argmax(logits.data(), kV);
  std::printf("%s: kernel id %d logit %.6f | cpu id %d logit %.6f (%.0f us)\n", name.c_str(),
              idx[0], val[0], ref, logits[ref], us);
  rep.expect(name + " argmax", idx[0] == ref && (expect_id < 0 || ref == expect_id));
  rep.check(name + " max logit", val.data(), &logits[ref], 1);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::printf("LmHeadKernel: %d ports x %d rows, mode: %s\n", kLmPorts, kLmRows, run_mode());
  TbReport rep("lm_head");
  Weights w;
  load_gpt2_weights(w, default_model_dir() + "/gpt2_124M.bin");

  // real: expected token = 2nd generated token of the golden run
  const int golden_next = golden::i32("output_ids.i32")[1];
  run_case(rep, "real_decode0", w.wte, golden::act(-1, "lnf"), golden_next);

  // synthetic tables (rows >= V stay zero, as in the real checkpoint)
  std::vector<float> syn((size_t)kVp * kC, 0.0f);
  auto r = random_vec((size_t)kV * kC, -0.1f, 0.1f, 41);
  std::copy(r.begin(), r.end(), syn.begin());
  auto u = random_vec(kC, -1, 1, 42);
  std::vector<float> x(kC);
  for (int i = 0; i < kC; ++i) x[i] = 4.0f * u[i];
  for (int i = 0; i < kC; ++i) syn[(size_t)100 * kC + i] = syn[(size_t)40000 * kC + i] = u[i];
  run_case(rep, "tie_100_40000", syn.data(), x, 100);

  for (int i = 0; i < kC; ++i) syn[(size_t)100 * kC + i] = syn[(size_t)40000 * kC + i] = 0.0f;
  for (int i = 0; i < kC; ++i) syn[(size_t)(kV - 1) * kC + i] = u[i];
  run_case(rep, "max_at_last_row", syn.data(), x, kV - 1);
  return rep.finish();
}
