// Unit testbench for LayerNormKernel. Reference: src/cpu layernorm_forward.
// Inputs: real residual-stream vectors of the first decode step (ln1 of
// layers 0 and 11, ln2 of layer 6, final ln_f, whose input has |x| ~ 200),
// plus random multi-row and offset/large-magnitude cases.
#include "../../data/golden_loader.h"
#include "../../src/common/weights.h"
#include "../../src/cpu/ops.h"
#include "../common/tb_util.h"
#include "kernel.h"

#include <cmath>

DEFINE_string(bitstream, "", "path to bitstream (.xo / .xclbin); empty = software simulation");

using namespace gpt2;

// FP64 LayerNorm, used to judge ill-conditioned inputs where the FP32
// reference itself is inaccurate.
static std::vector<float> layernorm_fp64(const std::vector<float>& x, const float* g,
                                         const float* b, int rows) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* xr = x.data() + (size_t)r * kC;
    double m = 0, v = 0;
    for (int i = 0; i < kC; ++i) m += xr[i];
    m /= kC;
    for (int i = 0; i < kC; ++i) v += (xr[i] - m) * (xr[i] - m);
    const double s = 1.0 / std::sqrt(v / kC + 1e-5);
    for (int i = 0; i < kC; ++i) y[(size_t)r * kC + i] = (float)((xr[i] - m) * s * g[i] + b[i]);
  }
  return y;
}

static void run_case(TbReport& rep, const std::string& name, const std::vector<float>& x,
                     const float* g, const float* b, int rows, std::vector<float>& y_out,
                     bool fp64_ref = false) {
  auto xa = to_aligned(x);
  aligned_vector<float> ga(g, g + kC), ba(b, b + kC), y((size_t)rows * kC);
  tapa::invoke(LayerNormKernel, FLAGS_bitstream,
               tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ga).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ba).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(y).reinterpret<float_v16>(), rows, kC);
  std::vector<float> ref((size_t)rows * kC);
  layernorm_forward(ref.data(), x.data(), g, b, rows, kC);
  if (fp64_ref) {
    // Large common offset: FP32 rounding of the mean dominates, so both the
    // kernel and src/cpu are judged against an FP64 LayerNorm instead.
    auto r64 = layernorm_fp64(x, g, b, rows);
    rep.check(name + " vs fp64", y.data(), r64.data(), r64.size());
    auto c = gpt2::compare(ref.data(), r64.data(), r64.size());
    auto k = gpt2::compare(y.data(), ref.data(), ref.size());
    std::printf("  (info) src/cpu vs fp64: max_abs=%.3e; kernel vs src/cpu: max_abs=%.3e\n",
                c.max_abs, k.max_abs);
  } else {
    rep.check(name, y.data(), ref.data(), ref.size());
  }
  y_out.assign(y.begin(), y.end());
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::printf("LayerNormKernel, mode: %s\n", run_mode());
  TbReport rep("layernorm");
  Weights w;
  load_gpt2_weights(w, default_model_dir() + "/gpt2_124M.bin");
  std::vector<float> y;

  run_case(rep, "l00_ln1", golden::act(-1, "embed"), w.l_ln1w(0), w.l_ln1b(0), 1, y);
  rep.info("l00_ln1", y.data(), golden::act(0, "ln1"));
  run_case(rep, "l06_ln2", golden::act(6, "res1"), w.l_ln2w(6), w.l_ln2b(6), 1, y);
  rep.info("l06_ln2", y.data(), golden::act(6, "ln2"));
  run_case(rep, "l11_ln1", golden::act(10, "res2"), w.l_ln1w(11), w.l_ln1b(11), 1, y);
  rep.info("l11_ln1", y.data(), golden::act(11, "ln1"));
  run_case(rep, "ln_f", golden::act(11, "res2"), w.lnfw, w.lnfb, 1, y);
  rep.info("ln_f", y.data(), golden::act(-1, "lnf"));

  // random: 4 rows at once (prefill-style), and a large common offset that
  // stresses the two-pass variance.
  run_case(rep, "rand_4rows", random_vec(4 * kC, -3, 3, 11), w.l_ln1w(3), w.l_ln1b(3), 4, y);
  {
    auto x = random_vec(kC, -1, 1, 12);
    for (auto& v : x) v += 1000.0f;
    run_case(rep, "rand_offset1000", x, w.l_ln2w(3), w.l_ln2b(3), 1, y, true);
  }
  return rep.finish();
}
