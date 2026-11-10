#include "weights.h"

#include <cstdio>
#include <cstdlib>

namespace gpt2 {

#ifndef GPT2_DEFAULT_MODEL_DIR
#define GPT2_DEFAULT_MODEL_DIR "models"
#endif

std::string default_model_dir() {
  const char* env = std::getenv("GPT2_MODEL_DIR");
  return env ? env : GPT2_DEFAULT_MODEL_DIR;
}

static void die(const std::string& msg) {
  std::fprintf(stderr, "load_gpt2_weights: %s\n", msg.c_str());
  std::exit(EXIT_FAILURE);
}

void load_gpt2_weights(Weights& w, const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("cannot open " + path);
  int header[256];
  if (std::fread(header, sizeof(int), 256, f) != 256) die("short header");
  if (header[0] != 20240326) die("bad magic");
  if (header[1] != 3) die("unsupported version (need FP32 version 3)");
  if (header[2] != kMaxT || header[3] != kV || header[4] != kL ||
      header[5] != kNH || header[6] != kC || header[7] != kVp)
    die("model dimensions do not match src/common/config.h");

  const size_t sizes[] = {
      (size_t)kVp * kC,      (size_t)kMaxT * kC,   (size_t)kL * kC,
      (size_t)kL * kC,       (size_t)kL * kC3 * kC, (size_t)kL * kC3,
      (size_t)kL * kC * kC,  (size_t)kL * kC,      (size_t)kL * kC,
      (size_t)kL * kC,       (size_t)kL * kC4 * kC, (size_t)kL * kC4,
      (size_t)kL * kC * kC4, (size_t)kL * kC,      (size_t)kC,
      (size_t)kC};
  float** ptrs[] = {&w.wte,  &w.wpe,      &w.ln1w,     &w.ln1b,
                    &w.qkvw, &w.qkvb,     &w.attprojw, &w.attprojb,
                    &w.ln2w, &w.ln2b,     &w.fcw,      &w.fcb,
                    &w.fcprojw, &w.fcprojb, &w.lnfw,   &w.lnfb};
  size_t n = 0;
  for (size_t s : sizes) n += s;
  w.num_parameters = n;
  w.mem.resize(n);
  if (std::fread(w.mem.data(), sizeof(float), n, f) != n) die("short parameter data");
  std::fclose(f);
  float* p = w.mem.data();
  for (size_t i = 0; i < 16; ++i) {
    *ptrs[i] = p;
    p += sizes[i];
  }
}

}  // namespace gpt2
