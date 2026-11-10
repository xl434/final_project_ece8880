// Integration test: one transformer block through the HLS kernels (TAPA
// software simulation, via src/host/FpgaModel::block) vs the src/cpu
// operator chain on the same input, for the first decode step.
// Input of block l = golden block input (embed for l = 0, else res2 of l-1);
// KV cache rows 0..pos-1 = golden (Hugging Face) K/V of the prompt.
// Every intermediate is checked against src/cpu (tolerance 1e-4) and also
// reported against the Hugging Face activations.
#include <gflags/gflags.h>

#include <cstdio>
#include <cstring>

#include "../data/golden_loader.h"
#include "../kernels/common/tb_util.h"
#include "../src/cpu/ops.h"
#include "../src/host/fpga_model.h"

DEFINE_string(bitstream, "", "\"\" = software simulation");
DEFINE_string(layers, "0,11", "layers to test");
DEFINE_bool(unfused, false, "use the standalone residual kernel instead of GEMV fusion");

using namespace gpt2;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TbReport rep(FLAGS_unfused ? "block_test (unfused residual)" : "block_test");
  Weights w;
  load_gpt2_weights(w, default_model_dir() + "/gpt2_124M.bin");
  HostOptions opt;
  opt.bitstream = FLAGS_bitstream;
  opt.fuse_residual = !FLAGS_unfused;
  FpgaModel model(w, opt);
  const int pos = (int)golden::decode0_info("position");

  std::stringstream ss(FLAGS_layers);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const int l = std::atoi(item.c_str());
    std::printf("block %d, position %d, %s:\n", l, pos, run_mode());
    const auto x = l == 0 ? golden::act(-1, "embed") : golden::act(l - 1, "res2");
    const auto gk = golden::act(l, "k"), gv = golden::act(l, "v");
    std::memcpy(model.kcache_layer(l), gk.data(), (size_t)pos * kC * 4);
    std::memcpy(model.vcache_layer(l), gv.data(), (size_t)pos * kC * 4);

    // CPU reference chain on its own copy of the cache
    std::vector<float> kc((size_t)kMaxT * kC, 0.0f), vc((size_t)kMaxT * kC, 0.0f);
    std::memcpy(kc.data(), gk.data(), (size_t)pos * kC * 4);
    std::memcpy(vc.data(), gv.data(), (size_t)pos * kC * 4);
    BlockActs r;
    r.ln1.resize(kC); r.qkv.resize(kC3); r.atty.resize(kC); r.res1.resize(kC);
    r.ln2.resize(kC); r.fch.resize(kC4); r.fch_gelu.resize(kC4); r.res2.resize(kC);
    std::vector<float> tmp(kC);
    layernorm_forward(r.ln1.data(), x.data(), w.l_ln1w(l), w.l_ln1b(l), 1, kC);
    matmul_forward(r.qkv.data(), r.ln1.data(), w.l_qkvw(l), w.l_qkvb(l), 1, kC, kC3);
    kv_cache_write(kc.data(), vc.data(), r.qkv.data(), 1, pos, kC);
    attention_forward(r.atty.data(), r.qkv.data(), kc.data(), vc.data(), 1, pos, kC, kNH);
    matmul_forward(tmp.data(), r.atty.data(), w.l_attprojw(l), w.l_attprojb(l), 1, kC, kC);
    residual_forward(r.res1.data(), x.data(), tmp.data(), kC);
    layernorm_forward(r.ln2.data(), r.res1.data(), w.l_ln2w(l), w.l_ln2b(l), 1, kC);
    matmul_forward(r.fch.data(), r.ln2.data(), w.l_fcw(l), w.l_fcb(l), 1, kC, kC4);
    gelu_forward(r.fch_gelu.data(), r.fch.data(), kC4);
    matmul_forward(tmp.data(), r.fch_gelu.data(), w.l_fcprojw(l), w.l_fcprojb(l), 1, kC4, kC);
    residual_forward(r.res2.data(), r.res1.data(), tmp.data(), kC);

    BlockActs a;
    model.block(l, x, pos, &a);
    char p[16];
    std::snprintf(p, sizeof p, "l%02d_", l);
    const std::string pre = p;
    struct { const char* n; std::vector<float> BlockActs::*f; } fields[] = {
        {"ln1", &BlockActs::ln1}, {"qkv", &BlockActs::qkv}, {"atty", &BlockActs::atty},
        {"res1", &BlockActs::res1}, {"ln2", &BlockActs::ln2}, {"fch", &BlockActs::fch},
        {"fch_gelu", &BlockActs::fch_gelu}, {"res2", &BlockActs::res2}};
    for (auto& f : fields) {
      rep.check(pre + f.n, (a.*f.f).data(), (r.*f.f).data(), (r.*f.f).size());
      rep.info(pre + f.n, (a.*f.f).data(), golden::act(l, f.n));
    }
    rep.check(pre + "kcache", model.kcache_layer(l), kc.data(), (size_t)(pos + 1) * kC);
    rep.check(pre + "vcache", model.vcache_layer(l), vc.data(), (size_t)(pos + 1) * kC);
  }
  return rep.finish();
}
