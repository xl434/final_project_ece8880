// Loader for the golden reference data written by golden/golden_hf.py.
// Shared by src/cpu, kernels/*/tb.cpp and tests/. The data directory is
// GPT2_DATA_DIR (environment) or the compile-time default GPT2_DEFAULT_DATA_DIR.
#ifndef GPT2_DATA_GOLDEN_LOADER_H_
#define GPT2_DATA_GOLDEN_LOADER_H_

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/common/utils.h"

#ifndef GPT2_DEFAULT_DATA_DIR
#define GPT2_DEFAULT_DATA_DIR "data"
#endif

namespace golden {

inline std::string data_dir() {
  const char* env = std::getenv("GPT2_DATA_DIR");
  return env ? env : GPT2_DEFAULT_DATA_DIR;
}

// Element count recorded in data/manifest.txt for `rel` (-1 if absent).
inline long manifest_count(const std::string& rel) {
  std::ifstream in(data_dir() + "/manifest.txt");
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string name, dtype;
    ss >> name >> dtype;
    if (name != rel) continue;
    long n = 1, d;
    while (ss >> d) n *= d;
    return n;
  }
  return -1;
}

inline std::vector<float> f32(const std::string& rel) {
  return gpt2::read_binary<float>(data_dir() + "/" + rel, manifest_count(rel));
}

inline std::vector<int> i32(const std::string& rel) {
  return gpt2::read_binary<int>(data_dir() + "/" + rel, manifest_count(rel));
}

// Activations of the first decode step, e.g. act(3, "ln1") -> acts/l03_ln1.f32;
// layer < 0 for model-level tensors ("embed", "lnf").
inline std::vector<float> act(int layer, const std::string& name) {
  char buf[64];
  if (layer < 0)
    std::snprintf(buf, sizeof buf, "acts/%s.f32", name.c_str());
  else
    std::snprintf(buf, sizeof buf, "acts/l%02d_%s.f32", layer, name.c_str());
  return f32(buf);
}

// "key value" pairs from data/decode0_info.txt (input_token, position, output_token).
inline long decode0_info(const std::string& key) {
  std::ifstream in(data_dir() + "/decode0_info.txt");
  std::string k;
  long v;
  while (in >> k >> v)
    if (k == key) return v;
  std::fprintf(stderr, "golden: %s missing from decode0_info.txt\n", key.c_str());
  std::exit(EXIT_FAILURE);
}

}  // namespace golden

#endif
