// Unit testbench for GeluKernel. Reference: src/cpu gelu_forward (tanhf).
// Inputs: real pre-activation MLP vectors (fch) of layers 0, 5 and 11 from
// the first decode step, plus a dense sweep over [-20, 20] covering the
// saturated tails, and exact zeros.
#include "../../data/golden_loader.h"
#include "../../src/cpu/ops.h"
#include "../common/tb_util.h"
#include "kernel.h"

DEFINE_string(bitstream, "", "path to bitstream (.xo / .xclbin); empty = software simulation");

using namespace gpt2;

static void run_case(TbReport& rep, const std::string& name, const std::vector<float>& x,
                     std::vector<float>& y_out) {
  const int n = (int)x.size();
  auto xa = to_aligned(x);
  aligned_vector<float> y(n);
  tapa::invoke(GeluKernel, FLAGS_bitstream,
               tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(y).reinterpret<float_v16>(), n / 16);
  std::vector<float> ref(n);
  gelu_forward(ref.data(), x.data(), n);
  rep.check(name, y.data(), ref.data(), n);
  y_out.assign(y.begin(), y.end());
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::printf("GeluKernel, mode: %s\n", run_mode());
  TbReport rep("gelu");
  std::vector<float> y;
  for (int l : {0, 5, 11}) {
    char name[32];
    std::snprintf(name, sizeof name, "l%02d_gelu", l);
    run_case(rep, name, golden::act(l, "fch"), y);
    rep.info(name, y.data(), golden::act(l, "fch_gelu"));
  }
  // sweep [-20, 20) in 4096 steps + zeros
  std::vector<float> sweep(4096 + 16, 0.0f);
  for (int i = 0; i < 4096; ++i) sweep[i] = -20.0f + 40.0f * i / 4096;
  run_case(rep, "sweep_-20_20", sweep, y);
  return rep.finish();
}
