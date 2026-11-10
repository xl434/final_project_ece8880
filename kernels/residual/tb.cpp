// Unit testbench for ResidualKernel. Reference: src/cpu residual_forward.
// Inputs: real block input + attention projection (-> res1) and res1 + MLP
// projection (-> res2) from the first decode step, plus random data.
#include "../../data/golden_loader.h"
#include "../../src/cpu/ops.h"
#include "../common/tb_util.h"
#include "kernel.h"

DEFINE_string(bitstream, "", "path to bitstream (.xo / .xclbin); empty = software simulation");

using namespace gpt2;

static void run_case(TbReport& rep, const std::string& name, const std::vector<float>& a,
                     const std::vector<float>& b, std::vector<float>& y_out) {
  const int n = (int)a.size();
  auto aa = to_aligned(a), ba = to_aligned(b);
  aligned_vector<float> y(n);
  tapa::invoke(ResidualKernel, FLAGS_bitstream,
               tapa::read_only_mmap<float>(aa).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ba).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(y).reinterpret<float_v16>(), n / 16);
  std::vector<float> ref(n);
  residual_forward(ref.data(), a.data(), b.data(), n);
  // FP32 addition is exactly reproducible: require bit-exact results.
  rep.check(name, y.data(), ref.data(), n, {0.0, 0.0});
  y_out.assign(y.begin(), y.end());
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::printf("ResidualKernel, mode: %s\n", run_mode());
  TbReport rep("residual");
  std::vector<float> y;
  run_case(rep, "l03_res1", golden::act(2, "res2"), golden::act(3, "attproj"), y);
  rep.info("l03_res1", y.data(), golden::act(3, "res1"));
  run_case(rep, "l11_res2", golden::act(11, "res1"), golden::act(11, "fcproj"), y);
  rep.info("l11_res2", y.data(), golden::act(11, "res2"));
  run_case(rep, "rand_4x768", random_vec(4 * 768, -1e3, 1e3, 21), random_vec(4 * 768, -1, 1, 22), y);
  return rep.finish();
}
