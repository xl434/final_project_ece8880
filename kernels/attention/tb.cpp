// Unit testbench for AttentionKernel. Reference: src/cpu kv_cache_write +
// attention_forward. Cases:
//   * real data: layers 0, 6, 11 of the first decode step (pos = 5), cache
//     rows 0..4 from the Hugging Face K/V of the prompt;
//   * pos = 0 (only the new token: output must equal v);
//   * full context pos = 1023 with random K/V and large scores (softmax stress),
//     on layer 7 to exercise the layer offset;
//   * prefill (T > 1 new tokens, causal): a short prompt after cached rows,
//     a 100-token prompt from position 0 (three query blocks, the last one
//     partial), and 50 tokens after 200 cached rows.
// Also checks that the new row is written to the cache and nothing else changes.
#include <cstring>

#include "../../data/golden_loader.h"
#include "../../src/common/config.h"
#include "../../src/cpu/ops.h"
#include "../common/tb_util.h"
#include "kernel.h"

DEFINE_string(bitstream, "", "path to bitstream (.xo / .xclbin); empty = software simulation");

using namespace gpt2;

constexpr size_t kLayerElems = (size_t)kMaxT * kC;
constexpr size_t kCacheElems = (size_t)kL * kLayerElems;

static void run_case(TbReport& rep, const std::string& name, const std::vector<float>& qkv,
                     aligned_vector<float>& kc, aligned_vector<float>& vc, int layer, int pos,
                     std::vector<float>& y_out, int T = 1) {
  // reference on copies of the caches
  std::vector<float> kref(kc.begin(), kc.end()), vref(vc.begin(), vc.end());
  float* kl = kref.data() + layer * kLayerElems;
  float* vl = vref.data() + layer * kLayerElems;
  kv_cache_write(kl, vl, qkv.data(), T, pos, kC);
  std::vector<float> ref((size_t)T * kC);
  attention_forward(ref.data(), qkv.data(), kl, vl, T, pos, kC, kNH);

  auto qa = to_aligned(qkv);
  aligned_vector<float> y((size_t)T * kC);
  double us = tapa::invoke(AttentionKernel, FLAGS_bitstream,
                           tapa::read_only_mmap<float>(qa).reinterpret<float_v16>(),
                           tapa::read_write_mmap<float>(kc).reinterpret<float_v16>(),
                           tapa::read_write_mmap<float>(vc).reinterpret<float_v16>(),
                           tapa::write_only_mmap<float>(y).reinterpret<float_v16>(), layer, pos,
                           T) *
              1e-3;
  std::printf("%s: layer %d pos %d T %d (%.0f us)\n", name.c_str(), layer, pos, T, us);
  rep.check(name, y.data(), ref.data(), (size_t)T * kC);
  const bool cache_ok = std::memcmp(kc.data(), kref.data(), kCacheElems * 4) == 0 &&
                        std::memcmp(vc.data(), vref.data(), kCacheElems * 4) == 0;
  rep.expect(name + " cache", cache_ok);
  y_out.assign(y.begin(), y.end());
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::printf("AttentionKernel: %d heads in parallel, mode: %s\n", kNH, run_mode());
  TbReport rep("attention");
  aligned_vector<float> kc(kCacheElems, 0.0f), vc(kCacheElems, 0.0f);
  std::vector<float> y;

  // real data
  const int pos = (int)golden::decode0_info("position");
  for (int l : {0, 6, 11}) {
    auto k = golden::act(l, "k"), v = golden::act(l, "v");  // (pos+1, C)
    std::memcpy(kc.data() + l * kLayerElems, k.data(), (size_t)pos * kC * 4);
    std::memcpy(vc.data() + l * kLayerElems, v.data(), (size_t)pos * kC * 4);
    char name[32];
    std::snprintf(name, sizeof name, "l%02d_attn", l);
    run_case(rep, name, golden::act(l, "qkv"), kc, vc, l, pos, y);
    rep.info(name, y.data(), golden::act(l, "atty"));
  }

  // pos = 0: softmax over a single position, output = v
  run_case(rep, "pos0", random_vec(kC3, -2, 2, 31), kc, vc, 2, 0, y);

  // full context, random cache, scores up to ~|8*8*64/8| -> exp range stress
  {
    const int l = 7, p = kMaxT - 1;
    auto kr = random_vec((size_t)p * kC, -1, 1, 32), vr = random_vec((size_t)p * kC, -3, 3, 33);
    std::memcpy(kc.data() + l * kLayerElems, kr.data(), kr.size() * 4);
    std::memcpy(vc.data() + l * kLayerElems, vr.data(), vr.size() * 4);
    run_case(rep, "rand_pos1023", random_vec(kC3, -8, 8, 34), kc, vc, l, p, y);
  }
  // prefill: T new tokens at positions pos .. pos+T-1
  struct Pre { const char* name; int layer, pos, T; };
  for (const Pre& c : {Pre{"pre_T5_pos3", 4, 3, 5}, Pre{"pre_T100_pos0", 9, 0, 100},
                       Pre{"pre_T50_pos200", 10, 200, 50}}) {
    auto kr = random_vec((size_t)c.pos * kC, -1, 1, 40 + c.layer);
    auto vr = random_vec((size_t)c.pos * kC, -3, 3, 50 + c.layer);
    std::memcpy(kc.data() + c.layer * kLayerElems, kr.data(), kr.size() * 4);
    std::memcpy(vc.data() + c.layer * kLayerElems, vr.data(), vr.size() * 4);
    run_case(rep, c.name, random_vec((size_t)c.T * kC3, -3, 3, 60 + c.layer), kc, vc, c.layer,
             c.pos, y, c.T);
  }
  return rep.finish();
}
