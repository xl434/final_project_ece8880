#include "fpga_model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "../../kernels/attention/kernel.h"
#include "../../kernels/gelu/kernel.h"
#include "../../kernels/gemv/kernel.h"
#include "../../kernels/layernorm/kernel.h"
#include "../../kernels/lm_head/kernel.h"
#include "../../kernels/residual/kernel.h"
#include "../common/utils.h"
#include "../cpu/ops.h"

namespace gpt2 {

const char* const kOpClassNames[kNumOpClasses] = {"layernorm", "gemv", "attention",
                                                  "gelu", "residual", "lm_head"};

namespace {
aligned_vector<float> copy_of(const float* p, size_t n) { return aligned_vector<float>(p, p + n); }

// Times one operator call.
struct Clock {
  double* dst;
  long* cnt;
  Timer t;
  Clock(double* d, long* c) : dst(d), cnt(c) {}
  ~Clock() {
    *dst += t.ms();
    ++*cnt;
  }
};
}  // namespace

FpgaModel::FpgaModel(const Weights& w, const HostOptions& opt)
    : w_(w), opt_(opt), kcache_((size_t)kL * kMaxT * kC, 0.0f),
      vcache_((size_t)kL * kMaxT * kC, 0.0f) {
  Timer t;
  if (!opt_.on_cpu[kGemv]) {
    for (int l = 0; l < kL; ++l) {
      qkvw_.push_back(pack_gemv_weights(w.l_qkvw(l), kC3, kC, kGemvPorts, kGemvRows));
      attprojw_.push_back(pack_gemv_weights(w.l_attprojw(l), kC, kC, kGemvPorts, kGemvRows));
      fcw_.push_back(pack_gemv_weights(w.l_fcw(l), kC4, kC, kGemvPorts, kGemvRows));
      fcprojw_.push_back(pack_gemv_weights(w.l_fcprojw(l), kC, kC4, kGemvPorts, kGemvRows));
    }
  } else {
    qkvw_.resize(kL);
    attprojw_.resize(kL);
    fcw_.resize(kL);
    fcprojw_.resize(kL);
  }
  if (!opt_.on_cpu[kLmHead]) lmw_ = pack_gemv_weights(w.wte, kVp, kC, kLmPorts, kLmRows, kV);
  std::printf("[host] packed weights in %.0f ms (GEMV %d ports x %d rows, LM head %d ports)\n",
              t.ms(), kGemvPorts, kGemvRows, kLmPorts);
}

void FpgaModel::layernorm(const float* x, const float* g, const float* b, float* y) {
  Clock c(&kernel_ms[kLayerNorm], &kernel_calls[kLayerNorm]);
  if (opt_.on_cpu[kLayerNorm]) return layernorm_forward(y, x, g, b, 1, kC);
  auto xa = copy_of(x, kC), ga = copy_of(g, kC), ba = copy_of(b, kC);
  aligned_vector<float> ya(kC);
  tapa::invoke(LayerNormKernel, opt_.bitstream,
               tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ga).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ba).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(ya).reinterpret<float_v16>(), 1, kC);
  std::memcpy(y, ya.data(), kC * 4);
}

void FpgaModel::gemv(const std::vector<aligned_vector<float>>& wp, const float* W, const float* b,
                     const float* x, const float* res, float* y, int C, int OC) {
  Clock c(&kernel_ms[kGemv], &kernel_calls[kGemv]);
  if (opt_.on_cpu[kGemv]) {
    matmul_forward(y, x, W, b, 1, C, OC);
    if (res) residual_forward(y, y, res, OC);
    return;
  }
  auto xa = copy_of(x, C), ba = copy_of(b, OC);
  auto ra = res ? copy_of(res, OC) : aligned_vector<float>(16, 0.0f);
  aligned_vector<float> ya(OC);
  auto& wpm = const_cast<std::vector<aligned_vector<float>>&>(wp);
  tapa::invoke(GemvKernel, opt_.bitstream, tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
               tapa::read_only_mmaps<float, kGemvPorts>(wpm).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ba), tapa::read_only_mmap<float>(ra),
               tapa::write_only_mmap<float>(ya), C, OC, res ? 1 : 0);
  std::memcpy(y, ya.data(), OC * 4);
}

void FpgaModel::attention(const float* qkv, int layer, int pos, float* y) {
  Clock c(&kernel_ms[kAttention], &kernel_calls[kAttention]);
  if (opt_.on_cpu[kAttention]) {
    kv_cache_write(kcache_layer(layer), vcache_layer(layer), qkv, 1, pos, kC);
    return attention_forward(y, qkv, kcache_layer(layer), vcache_layer(layer), 1, pos, kC, kNH);
  }
  auto qa = copy_of(qkv, kC3);
  aligned_vector<float> ya(kC);
  tapa::invoke(AttentionKernel, opt_.bitstream,
               tapa::read_only_mmap<float>(qa).reinterpret<float_v16>(),
               tapa::read_write_mmap<float>(kcache_).reinterpret<float_v16>(),
               tapa::read_write_mmap<float>(vcache_).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(ya).reinterpret<float_v16>(), layer, pos);
  std::memcpy(y, ya.data(), kC * 4);
}

void FpgaModel::gelu(const float* x, float* y, int n) {
  Clock c(&kernel_ms[kGelu], &kernel_calls[kGelu]);
  if (opt_.on_cpu[kGelu]) return gelu_forward(y, x, n);
  auto xa = copy_of(x, n);
  aligned_vector<float> ya(n);
  tapa::invoke(GeluKernel, opt_.bitstream, tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(ya).reinterpret<float_v16>(), n / 16);
  std::memcpy(y, ya.data(), n * 4);
}

void FpgaModel::residual(const float* a, const float* b, float* y, int n) {
  Clock c(&kernel_ms[kResidual], &kernel_calls[kResidual]);
  if (opt_.on_cpu[kResidual]) return residual_forward(y, a, b, n);
  auto aa = copy_of(a, n), ba = copy_of(b, n);
  aligned_vector<float> ya(n);
  tapa::invoke(ResidualKernel, opt_.bitstream,
               tapa::read_only_mmap<float>(aa).reinterpret<float_v16>(),
               tapa::read_only_mmap<float>(ba).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(ya).reinterpret<float_v16>(), n / 16);
  std::memcpy(y, ya.data(), n * 4);
}

int FpgaModel::lm_head(const float* x) {
  Clock c(&kernel_ms[kLmHead], &kernel_calls[kLmHead]);
  if (opt_.on_cpu[kLmHead]) {
    std::vector<float> logits(kV);
    lm_head_forward(logits.data(), x, w_.wte, kC, kV);
    const int id = argmax(logits.data(), kV);
    last_logit_ = logits[id];
    return id;
  }
  auto xa = copy_of(x, kC);
  aligned_vector<float> out(16, -1.0f);  // {token ID, logit}
  tapa::invoke(LmHeadKernel, opt_.bitstream, tapa::read_only_mmap<float>(xa).reinterpret<float_v16>(),
               tapa::read_only_mmaps<float, kLmPorts>(lmw_).reinterpret<float_v16>(),
               tapa::write_only_mmap<float>(out));
  last_logit_ = out[1];
  return (int)out[0];
}

void FpgaModel::block(int l, const std::vector<float>& x, int pos, BlockActs* a) {
  a->ln1.resize(kC);
  a->qkv.resize(kC3);
  a->atty.resize(kC);
  a->res1.resize(kC);
  a->ln2.resize(kC);
  a->fch.resize(kC4);
  a->fch_gelu.resize(kC4);
  a->res2.resize(kC);
  layernorm(x.data(), w_.l_ln1w(l), w_.l_ln1b(l), a->ln1.data());
  gemv(qkvw_[l], w_.l_qkvw(l), w_.l_qkvb(l), a->ln1.data(), nullptr, a->qkv.data(), kC, kC3);
  attention(a->qkv.data(), l, pos, a->atty.data());
  if (opt_.fuse_residual) {
    gemv(attprojw_[l], w_.l_attprojw(l), w_.l_attprojb(l), a->atty.data(), x.data(),
         a->res1.data(), kC, kC);
  } else {
    std::vector<float> proj(kC);
    gemv(attprojw_[l], w_.l_attprojw(l), w_.l_attprojb(l), a->atty.data(), nullptr, proj.data(),
         kC, kC);
    residual(x.data(), proj.data(), a->res1.data(), kC);
  }
  layernorm(a->res1.data(), w_.l_ln2w(l), w_.l_ln2b(l), a->ln2.data());
  gemv(fcw_[l], w_.l_fcw(l), w_.l_fcb(l), a->ln2.data(), nullptr, a->fch.data(), kC, kC4);
  gelu(a->fch.data(), a->fch_gelu.data(), kC4);
  if (opt_.fuse_residual) {
    gemv(fcprojw_[l], w_.l_fcprojw(l), w_.l_fcprojb(l), a->fch_gelu.data(), a->res1.data(),
         a->res2.data(), kC4, kC);
  } else {
    std::vector<float> proj(kC);
    gemv(fcprojw_[l], w_.l_fcprojw(l), w_.l_fcprojb(l), a->fch_gelu.data(), nullptr, proj.data(),
         kC4, kC);
    residual(a->res1.data(), proj.data(), a->res2.data(), kC);
  }
}

int FpgaModel::step(int token, int pos) {
  std::vector<float> x(kC);
  encoder_forward(x.data(), &token, 1, pos, w_.wte, w_.wpe, kC);  // host: table lookup
  BlockActs a;
  for (int l = 0; l < kL; ++l) {
    block(l, x, pos, &a);
    x.swap(a.res2);
  }
  std::vector<float> lnf(kC);
  layernorm(x.data(), w_.lnfw, w_.lnfb, lnf.data());
  return lm_head(lnf.data());
}

void FpgaModel::print_hbm_plan() {
  const double MB = 1e6;
  const double gemv_w = 12.0 * kC * kC * kL * 4;  // qkv + attproj + fc + fcproj, all layers
  const double lm_w = (double)kVp * kC * 4;
  const double kv = (double)kL * kMaxT * kC * 4;
  std::printf("HBM placement plan (FP32, 32 pseudo-channels x 512 MB; see link_config.ini)\n");
  std::printf("  HBM[0..%d]   GEMV weights, port p: row blocks p, p+%d, ... of all 48 matrices: "
              "%.1f MB per channel\n", kGemvPorts - 1, kGemvPorts, gemv_w / kGemvPorts / MB);
  std::printf("  HBM[%d..%d]  LM head (packed wte): %.1f MB per channel\n", kGemvPorts,
              kGemvPorts + kLmPorts - 1, lm_w / kLmPorts / MB);
  std::printf("  HBM[%d]      K cache (L, maxT, C): %.1f MB\n", kGemvPorts + kLmPorts, kv / MB);
  std::printf("  HBM[%d]      V cache (L, maxT, C): %.1f MB\n", kGemvPorts + kLmPorts + 1, kv / MB);
  std::printf("  HBM[%d]      activations, biases, LN params, token output: < 1 MB\n",
              kGemvPorts + kLmPorts + 2);
  std::printf("  every channel holds <= %.1f MB of its 512 MB\n",
              std::max(gemv_w / kGemvPorts, std::max(lm_w / kLmPorts, kv)) / MB);
}

}  // namespace gpt2
