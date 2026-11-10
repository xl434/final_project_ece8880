// FPGA host program for GPT-2 (124M) greedy decoding (built by
// `make -C src/host` into build/bin/host_main).
//
//   host_main [--bitstream=FILE] [--prompt=FILE.i32] [--n=32]
//             [--cpu_ops=layernorm,gemv,attention,gelu,residual,lm_head]
//             [--fuse_residual=true] [--check] [--plan]
//
// Tokenization is done on the host (prompt token IDs from Python, decoding
// with src/common/tokenizer). The prompt is prefilled token by token through
// the decode kernels (KV cache), then n tokens are generated greedily.
// --check compares the generated IDs with data/output_ids.i32 (exact match).
#include <gflags/gflags.h>

#include <cstdio>
#include <sstream>

#include "../../data/golden_loader.h"
#include "../common/tokenizer.h"
#include "../common/utils.h"
#include "fpga_model.h"

DEFINE_string(bitstream, "", "\"\" = software simulation, .xo = TAPA cosim, .xclbin = board");
DEFINE_string(prompt, "", "int32 prompt token-ID file (default data/prompt_ids.i32)");
DEFINE_int32(n, 32, "tokens to generate");
DEFINE_string(cpu_ops, "", "comma-separated operator classes to run on the CPU");
DEFINE_bool(fuse_residual, true, "fuse residual adds into the GEMV output stage");
DEFINE_bool(check, false, "compare generated IDs with data/output_ids.i32");
DEFINE_bool(plan, false, "print the HBM placement plan and exit");

using namespace gpt2;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_plan) {
    FpgaModel::print_hbm_plan();
    return 0;
  }
  HostOptions opt;
  opt.bitstream = FLAGS_bitstream;
  opt.fuse_residual = FLAGS_fuse_residual;
  std::stringstream ss(FLAGS_cpu_ops);
  std::string item;
  while (std::getline(ss, item, ',')) {
    bool found = false;
    for (int i = 0; i < kNumOpClasses; ++i)
      if (item == kOpClassNames[i]) opt.on_cpu[i] = found = true;
    if (!found && !item.empty()) {
      std::fprintf(stderr, "unknown operator class '%s'\n", item.c_str());
      return 2;
    }
  }
  std::printf("[host] mode: %s; ops on CPU:", opt.bitstream.empty() ? "software simulation"
                                                                    : opt.bitstream.c_str());
  for (int i = 0; i < kNumOpClasses; ++i)
    if (opt.on_cpu[i]) std::printf(" %s", kOpClassNames[i]);
  std::printf("%s; residual %s\n", FLAGS_cpu_ops.empty() ? " none" : "",
              opt.fuse_residual ? "fused into GEMV" : "separate kernel");

  Weights w;
  load_gpt2_weights(w, default_model_dir() + "/gpt2_124M.bin");
  Tokenizer tok;
  tok.load(default_model_dir() + "/gpt2_tokenizer.bin");
  const std::vector<int> prompt = FLAGS_prompt.empty() ? golden::i32("prompt_ids.i32")
                                                       : read_binary<int>(FLAGS_prompt);
  if ((int)prompt.size() + FLAGS_n > kMaxT) {
    std::fprintf(stderr, "prompt + n exceeds the %d-token context\n", kMaxT);
    return 2;
  }
  FpgaModel model(w, opt);

  Timer t;
  int pos = 0, next = -1;
  for (int id : prompt) next = model.step(id, pos++);  // prefill through the KV cache
  const double prefill_ms = t.ms();
  std::vector<int> out{next};
  t.reset();
  while ((int)out.size() < FLAGS_n) {
    next = model.step(out.back(), pos++);
    out.push_back(next);
    std::printf("\r[host] generated %zu/%d", out.size(), FLAGS_n);
    std::fflush(stdout);
  }
  const double decode_ms = t.ms();
  std::printf("\noutput ids:");
  for (int id : out) std::printf(" %d", id);
  if (tok.ok()) std::printf("\n---\n%s%s\n---\n", tok.decode(prompt).c_str(), tok.decode(out).c_str());
  std::printf("[host] wall time: prefill %.1f ms (%zu tokens), decode %.1f ms (%d tokens) "
              "-- simulation time, not FPGA performance\n",
              prefill_ms, prompt.size(), decode_ms, FLAGS_n - 1);
  for (int i = 0; i < kNumOpClasses; ++i)
    std::printf("  %-10s %5ld calls %10.1f ms\n", kOpClassNames[i], model.kernel_calls[i],
                model.kernel_ms[i]);

  if (FLAGS_check) {
    const auto ref = golden::i32("output_ids.i32");
    int match = 0;
    const int n = std::min<int>(ref.size(), out.size());
    for (int i = 0; i < n; ++i) match += out[i] == ref[i];
    const bool ok = match == n && FLAGS_prompt.empty();
    std::printf("token IDs vs golden: %d/%d match -> %s\n", match, n, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
  return 0;
}
