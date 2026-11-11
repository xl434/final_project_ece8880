// Integration test: one transformer block through the HLS kernels (TAPA
// software simulation, via src/host/FpgaModel::block) vs the src/cpu
// operator chain on the same input.
//
// Decode (default): the first decode step (one token at position 5).
//   Input of block l = golden block input (embed for l = 0, else res2 of l-1);
//   KV cache rows 0..pos-1 = golden (Hugging Face) K/V of the prompt. Every
//   intermediate is also reported against the Hugging Face activations.
// Prefill (--prefill_T=N): the first N tokens of the long prompt at positions
//   0..N-1 in one pass. Input of block l = the CPU model's output of block l-1
//   on the same tokens (real activations, not random).
// Every intermediate (all N rows) and the new KV-cache rows are checked
// against src/cpu (tolerance 1e-4).
#include <gflags/gflags.h>

#include <cstdio>
#include <cstring>

#include "../data/golden_loader.h"
#include "../kernels/common/tb_util.h"
#include "../src/cpu/ops.h"
#include "../src/host/fpga_model.h"

DEFINE_string(bitstream, "", "\"\" = software simulation");
DEFINE_string(xo_dir, "", "run every kernel on xo_dir/<k>/<k>.xo (RTL co-simulation), "
                          "e.g. build/kernels");
DEFINE_string(layers, "0,11", "layers to test");
DEFINE_bool(unfused, false, "use the standalone residual kernel instead of GEMV fusion");
DEFINE_int32(prefill_T, 0, "0 = decode step; N > 0 = prefill of N tokens of the long prompt");

using namespace gpt2;

// src/cpu reference of one block for T tokens at positions pos0.. on the
// caches kc, vc (one layer, maxT rows).
static void cpu_block(const Weights& w, int l, const std::vector<float>& x, int T, int pos0,
                      float* kc, float* vc, BlockActs* r) {
  r->ln1.resize((size_t)T * kC); r->qkv.resize((size_t)T * kC3); r->atty.resize((size_t)T * kC);
  r->res1.resize((size_t)T * kC); r->ln2.resize((size_t)T * kC); r->fch.resize((size_t)T * kC4);
  r->fch_gelu.resize((size_t)T * kC4); r->res2.resize((size_t)T * kC);
  std::vector<float> tmp((size_t)T * kC);
  layernorm_forward(r->ln1.data(), x.data(), w.l_ln1w(l), w.l_ln1b(l), T, kC);
  matmul_forward(r->qkv.data(), r->ln1.data(), w.l_qkvw(l), w.l_qkvb(l), T, kC, kC3);
  kv_cache_write(kc, vc, r->qkv.data(), T, pos0, kC);
  attention_forward(r->atty.data(), r->qkv.data(), kc, vc, T, pos0, kC, kNH);
  matmul_forward(tmp.data(), r->atty.data(), w.l_attprojw(l), w.l_attprojb(l), T, kC, kC);
  residual_forward(r->res1.data(), x.data(), tmp.data(), T * kC);
  layernorm_forward(r->ln2.data(), r->res1.data(), w.l_ln2w(l), w.l_ln2b(l), T, kC);
  matmul_forward(r->fch.data(), r->ln2.data(), w.l_fcw(l), w.l_fcb(l), T, kC, kC4);
  gelu_forward(r->fch_gelu.data(), r->fch.data(), T * kC4);
  matmul_forward(tmp.data(), r->fch_gelu.data(), w.l_fcprojw(l), w.l_fcprojb(l), T, kC4, kC);
  residual_forward(r->res2.data(), r->res1.data(), tmp.data(), T * kC);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const bool prefill = FLAGS_prefill_T > 0;
  std::string title = FLAGS_unfused ? "block_test (unfused residual)" : "block_test";
  if (prefill) title += " (prefill T=" + std::to_string(FLAGS_prefill_T) + ")";
  TbReport rep(title.c_str());
  Weights w;
  load_gpt2_weights(w, default_model_dir() + "/gpt2_124M.bin");
  HostOptions opt;
  opt.bitstream = FLAGS_bitstream;
  opt.xo_dir = FLAGS_xo_dir;
  opt.fuse_residual = !FLAGS_unfused;
  FpgaModel model(w, opt);
  const int T = prefill ? FLAGS_prefill_T : 1;
  const int pos = prefill ? 0 : (int)golden::decode0_info("position");

  // prefill: CPU inputs of every block for the first T tokens of the long prompt
  std::vector<std::vector<float>> pre_in;
  if (prefill) {
    const auto ids = golden::i32("long_prompt_ids.i32");
    std::vector<float> x((size_t)T * kC);
    encoder_forward(x.data(), ids.data(), T, 0, w.wte, w.wpe, kC);
    std::vector<float> kc((size_t)kMaxT * kC), vc((size_t)kMaxT * kC);
    for (int l = 0; l < kL; ++l) {
      pre_in.push_back(x);
      BlockActs r;
      cpu_block(w, l, x, T, 0, kc.data(), vc.data(), &r);
      x = r.res2;
    }
  }

  std::stringstream ss(FLAGS_layers);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const int l = std::atoi(item.c_str());
    std::printf("block %d, positions %d..%d, %s:\n", l, pos, pos + T - 1, run_mode());
    std::vector<float> x;
    std::vector<float> kc((size_t)kMaxT * kC, 0.0f), vc((size_t)kMaxT * kC, 0.0f);
    if (prefill) {
      x = pre_in[l];
    } else {
      x = l == 0 ? golden::act(-1, "embed") : golden::act(l - 1, "res2");
      const auto gk = golden::act(l, "k"), gv = golden::act(l, "v");
      std::memcpy(model.kcache_layer(l), gk.data(), (size_t)pos * kC * 4);
      std::memcpy(model.vcache_layer(l), gv.data(), (size_t)pos * kC * 4);
      std::memcpy(kc.data(), gk.data(), (size_t)pos * kC * 4);
      std::memcpy(vc.data(), gv.data(), (size_t)pos * kC * 4);
    }

    BlockActs r;  // CPU reference chain on its own copy of the cache
    cpu_block(w, l, x, T, pos, kc.data(), vc.data(), &r);
    BlockActs a;
    model.block(l, x, pos, &a, T);
    char p[16];
    std::snprintf(p, sizeof p, "l%02d_", l);
    const std::string pre = p;
    struct { const char* n; std::vector<float> BlockActs::*f; } fields[] = {
        {"ln1", &BlockActs::ln1}, {"qkv", &BlockActs::qkv}, {"atty", &BlockActs::atty},
        {"res1", &BlockActs::res1}, {"ln2", &BlockActs::ln2}, {"fch", &BlockActs::fch},
        {"fch_gelu", &BlockActs::fch_gelu}, {"res2", &BlockActs::res2}};
    for (auto& f : fields) {
      rep.check(pre + f.n, (a.*f.f).data(), (r.*f.f).data(), (r.*f.f).size());
      if (!prefill) rep.info(pre + f.n, (a.*f.f).data(), golden::act(l, f.n));
    }
    rep.check(pre + "kcache", model.kcache_layer(l), kc.data(), (size_t)(pos + T) * kC);
    rep.check(pre + "vcache", model.vcache_layer(l), vc.data(), (size_t)(pos + T) * kC);
  }
  return rep.finish();
}
