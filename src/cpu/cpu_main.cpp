// CPU baseline for GPT-2 (124M) FP32 inference with a KV cache.
//
//   (built by `make -C src/cpu` into build/bin/cpu_main)
//   cpu_main generate [--prompt FILE.i32] [--n 32] [--threads N]
//   cpu_main verify   [--threads N]       compare with Hugging Face golden data
//   cpu_main profile  [--threads 1,8] [--ctx 128,512,1024] [--reps 3]
//
// Paths default to the repository layout (models/, data/, results/) and can be
// overridden with GPT2_MODEL_DIR / GPT2_DATA_DIR / GPT2_RESULTS_DIR.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../../data/golden_loader.h"
#include "../common/config.h"
#include "../common/tokenizer.h"
#include "../common/utils.h"
#include "../common/weights.h"
#include "gpt2_cpu.h"
#include "ops.h"

using namespace gpt2;

#ifndef GPT2_DEFAULT_RESULTS_DIR
#define GPT2_DEFAULT_RESULTS_DIR "results"
#endif
#ifndef GPT2_CXXFLAGS
#define GPT2_CXXFLAGS "unknown"
#endif

namespace {

std::string results_dir() {
  const char* env = std::getenv("GPT2_RESULTS_DIR");
  return env ? env : GPT2_DEFAULT_RESULTS_DIR;
}

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(std::atoi(item.c_str()));
  return v;
}

struct Args {
  std::string mode = "generate";
  std::string prompt;
  int n = 32;
  std::string threads;
  std::string ctx = "128,512,1024";
  int reps = 3;
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.mode = argv[1];
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--prompt") a.prompt = val();
    else if (k == "--n") a.n = std::atoi(val().c_str());
    else if (k == "--threads") a.threads = val();
    else if (k == "--ctx") a.ctx = val();
    else if (k == "--reps") a.reps = std::atoi(val().c_str());
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      std::exit(2);
    }
  }
  return a;
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
  return "unknown";
}

void print_env(FILE* f) {
  std::fprintf(f, "cpu_model: %s\n", cpu_model().c_str());
  std::fprintf(f, "omp_max_threads: %d (online cpus usable by this process: %d)\n",
               omp_get_max_threads(), omp_get_num_procs());
  std::fprintf(f, "compiler: g++ %s\n", __VERSION__);
  std::fprintf(f, "cxxflags: %s\n", GPT2_CXXFLAGS);
}

void load(Weights& w) {
  const std::string path = default_model_dir() + "/gpt2_124M.bin";
  Timer t;
  load_gpt2_weights(w, path);
  std::printf("loaded %s: %zu parameters (%.1f MB FP32) in %.0f ms\n", path.c_str(),
              w.num_parameters, w.num_parameters * 4.0 / 1e6, t.ms());
}

// ---------------------------------------------------------------- generate
int run_generate(const Args& a) {
  if (!a.threads.empty()) omp_set_num_threads(std::atoi(a.threads.c_str()));
  Weights w;
  load(w);
  Tokenizer tok;
  if (!tok.load(default_model_dir() + "/gpt2_tokenizer.bin"))
    std::fprintf(stderr, "warning: tokenizer not loaded; printing IDs only\n");
  std::vector<int> prompt = a.prompt.empty()
                                ? golden::i32("prompt_ids.i32")
                                : read_binary<int>(a.prompt);
  State st;
  double prefill_ms = 0;
  std::vector<double> dms;
  auto out = generate(w, st, prompt, a.n, nullptr, &prefill_ms, &dms);
  std::printf("prompt (%zu tokens): %s\n", prompt.size(), tok.ok() ? tok.decode(prompt).c_str() : "");
  std::printf("output ids:");
  for (int id : out) std::printf(" %d", id);
  std::printf("\n---\n%s%s\n---\n", tok.ok() ? tok.decode(prompt).c_str() : "",
              tok.ok() ? tok.decode(out).c_str() : "");
  double mean = dms.empty() ? 0 : std::accumulate(dms.begin(), dms.end(), 0.0) / dms.size();
  std::printf("threads %d: prefill %.2f ms, decode %.2f ms/token (%.2f tokens/s)\n",
              omp_get_max_threads(), prefill_ms, mean, mean > 0 ? 1000.0 / mean : 0.0);
  return 0;
}

// ---------------------------------------------------------------- verify
int run_verify(const Args& a) {
  if (!a.threads.empty()) omp_set_num_threads(std::atoi(a.threads.c_str()));
  Weights w;
  load(w);
  // HF (PyTorch/MKL) and this code accumulate in different orders; errors
  // grow through 12 layers, so the tolerance is stated per comparison.
  const Tolerance tol{1e-4, 1e-4};
  std::printf("tolerance vs HF: |got-ref| <= %.0e*max|ref| + %.0e*|ref|\n", tol.atol_rel, tol.rtol);

  FILE* csv = std::fopen((results_dir() + "/cpu_verify.csv").c_str(), "w");
  std::fprintf(csv, "tensor,n,max_abs,max_rel,norm_rel,ref_max_abs,pass\n");
  bool all_ok = true;
  auto check = [&](const std::string& name, const float* got, const std::vector<float>& ref) {
    ErrStats s = compare(got, ref.data(), ref.size(), tol);
    print_err(name.c_str(), s);
    std::fprintf(csv, "%s,%zu,%.6e,%.6e,%.6e,%.6e,%d\n", name.c_str(), s.n, s.max_abs, s.max_rel,
                 s.norm_rel, s.ref_max_abs, s.pass);
    all_ok &= s.pass;
  };

  const auto prompt = golden::i32("prompt_ids.i32");
  const auto ref_ids = golden::i32("output_ids.i32");
  const int pos = (int)golden::decode0_info("position");
  State st;
  // prefill
  std::vector<int> out;
  out.push_back(forward(w, st, prompt.data(), (int)prompt.size()));
  // first decode step with activation trace
  Trace tr;
  int tok = out.back();
  if (st.pos != pos) {
    std::fprintf(stderr, "position mismatch %d vs %d\n", st.pos, pos);
    return 1;
  }
  out.push_back(forward(w, st, &tok, 1, nullptr, &tr));
  std::printf("first decode step (input token %d at position %d):\n", tok, pos);
  check("embed", tr.t["embed"].data(), golden::act(-1, "embed"));
  const char* names[] = {"ln1", "qkv", "atty", "attproj", "res1", "ln2", "fch", "fch_gelu",
                         "fcproj", "res2"};
  for (int l = 0; l < kL; l++) {
    for (const char* n : names) {
      char key[32];
      std::snprintf(key, sizeof key, "l%02d_%s", l, n);
      check(key, tr.t[key].data(), golden::act(l, n));
    }
    char kk[32], vk[32];
    std::snprintf(kk, sizeof kk, "l%02d_kcache", l);
    std::snprintf(vk, sizeof vk, "l%02d_vcache", l);
    check(kk, st.k(l), golden::act(l, "k"));  // rows 0..pos
    check(vk, st.v(l), golden::act(l, "v"));
  }
  check("lnf", tr.t["lnf"].data(), golden::act(-1, "lnf"));
  check("logits", st.logits.data(), golden::f32("decode0_logits.f32"));

  // top-5 of that step
  std::vector<int> idx(kV);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int x, int y) { return st.logits[x] > st.logits[y]; });
  std::ifstream t5(golden::data_dir() + "/decode0_top5.txt");
  bool top5_ok = true;
  std::printf("  top5 (cpu vs hf):");
  for (int i = 0; i < 5; i++) {
    int id;
    float v;
    std::string rest;
    t5 >> id >> v;
    std::getline(t5, rest);
    std::printf(" %d/%d", idx[i], id);
    top5_ok &= idx[i] == id;
  }
  std::printf("  %s\n", top5_ok ? "MATCH" : "DIFF");
  all_ok &= top5_ok;

  // remaining decode steps
  while ((int)out.size() < (int)ref_ids.size()) {
    tok = out.back();
    out.push_back(forward(w, st, &tok, 1));
  }
  int mism = 0;
  for (size_t i = 0; i < ref_ids.size(); i++) mism += out[i] != ref_ids[i];
  std::printf("short prompt: %zu/%zu generated token IDs match HF %s\n", ref_ids.size() - mism,
              ref_ids.size(), mism ? "FAIL" : "PASS");
  std::fprintf(csv, "token_ids_short,%zu,%d,,,,%d\n", ref_ids.size(), mism, mism == 0);
  all_ok &= mism == 0;

  // long prompt: context 992 -> 1024
  const auto lp = golden::i32("long_prompt_ids.i32");
  const auto lref = golden::i32("long_output_ids.i32");
  State st2;
  auto lout = generate(w, st2, lp, (int)lref.size());
  int lm = 0;
  for (size_t i = 0; i < lref.size(); i++) lm += lout[i] != lref[i];
  std::printf("long prompt (%zu tokens, context up to %d): %zu/%zu token IDs match HF %s\n",
              lp.size(), st2.pos + 1, lref.size() - lm, lref.size(), lm ? "FAIL" : "PASS");
  std::fprintf(csv, "token_ids_long,%zu,%d,,,,%d\n", lref.size(), lm, lm == 0);
  all_ok &= lm == 0;
  std::fclose(csv);
  std::printf("CPU vs golden: %s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}

// ---------------------------------------------------------------- profile
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

int run_profile(const Args& a) {
  Weights w;
  load(w);
  print_env(stdout);
  {
    FILE* f = std::fopen((results_dir() + "/cpu_env.txt").c_str(), "w");
    print_env(f);
    std::fclose(f);
  }
  const auto lp = golden::i32("long_prompt_ids.i32");
  std::vector<int> threads = parse_list(a.threads.empty() ? "1,8" : a.threads);
  std::vector<int> ctxs = parse_list(a.ctx);
  const int n_new = 32;  // 1 token from prefill + 31 decode steps

  FILE* fs = std::fopen((results_dir() + "/cpu_profile_summary.csv").c_str(), "w");
  std::fprintf(fs, "threads,context,prompt_len,decode_steps,reps,prefill_ms,prefill_tok_s,"
                   "decode_ms_per_tok,decode_tok_s,decode_ms_min,decode_ms_max\n");
  FILE* fo = std::fopen((results_dir() + "/cpu_profile_ops.csv").c_str(), "w");
  std::fprintf(fo, "threads,context,phase,op,ms_per_call_token,pct\n");

  State st;
  for (int th : threads) {
    omp_set_num_threads(th);
    for (int ctx : ctxs) {
      const int plen = ctx - n_new;
      if (plen <= 0 || plen > (int)lp.size()) {
        std::fprintf(stderr, "skip ctx %d\n", ctx);
        continue;
      }
      std::vector<int> prompt(lp.begin(), lp.begin() + plen);
      std::vector<double> pre, dec;
      Profiler pp, dp;  // accumulated over reps (prefill, decode)
      for (int r = 0; r < a.reps; r++) {
        st.pos = 0;
        Timer t;
        int next = forward(w, st, prompt.data(), plen, &pp);
        pre.push_back(t.ms());
        double dsum = 0;
        for (int i = 1; i < n_new; i++) {
          t.reset();
          next = forward(w, st, &next, 1, &dp);
          dsum += t.ms();
        }
        dec.push_back(dsum / (n_new - 1));
      }
      const double pm = median(pre), dm = median(dec);
      std::printf("threads %d ctx %4d: prefill %4d tok %9.2f ms (%7.1f tok/s) | decode %7.2f "
                  "ms/tok (%6.2f tok/s)\n",
                  th, ctx, plen, pm, plen * 1000.0 / pm, dm, 1000.0 / dm);
      std::fprintf(fs, "%d,%d,%d,%d,%d,%.4f,%.3f,%.4f,%.4f,%.4f,%.4f\n", th, ctx, plen,
                   n_new - 1, a.reps, pm, plen * 1000.0 / pm, dm, 1000.0 / dm,
                   *std::min_element(dec.begin(), dec.end()),
                   *std::max_element(dec.begin(), dec.end()));
      for (int phase = 0; phase < 2; phase++) {
        const Profiler& p = phase ? dp : pp;
        const double per = phase ? (double)a.reps * (n_new - 1) : (double)a.reps;
        double tot = 0;
        for (int o = 0; o < kNumOps; o++) tot += p.ms[o];
        for (int o = 0; o < kNumOps; o++)
          std::fprintf(fo, "%d,%d,%s,%s,%.5f,%.3f\n", th, ctx, phase ? "decode" : "prefill",
                       kOpNames[o], p.ms[o] / per, 100.0 * p.ms[o] / tot);
      }
    }
  }
  std::fclose(fs);
  std::fclose(fo);
  std::printf("wrote %s/cpu_profile_summary.csv, cpu_profile_ops.csv, cpu_env.txt\n",
              results_dir().c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.mode == "generate") return run_generate(a);
  if (a.mode == "verify") return run_verify(a);
  if (a.mode == "profile") return run_profile(a);
  std::fprintf(stderr, "usage: %s generate|verify|profile [options]\n", argv[0]);
  return 2;
}
