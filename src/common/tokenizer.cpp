#include "tokenizer.h"

#include <cstdint>
#include <cstdio>

namespace gpt2 {

bool Tokenizer::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  uint32_t header[256];
  if (std::fread(header, sizeof(uint32_t), 256, f) != 256 || header[0] != 20240328) {
    std::fclose(f);
    return false;
  }
  const uint32_t vocab = header[2];
  eot_ = header[1] == 2 ? (int)header[3] : 50256;
  table_.resize(vocab);
  for (uint32_t i = 0; i < vocab; ++i) {
    unsigned char len = 0;
    if (std::fread(&len, 1, 1, f) != 1 || len == 0) {
      table_.clear();
      std::fclose(f);
      return false;
    }
    table_[i].resize(len);
    if (std::fread(&table_[i][0], 1, len, f) != len) {
      table_.clear();
      std::fclose(f);
      return false;
    }
  }
  std::fclose(f);
  return true;
}

const std::string& Tokenizer::decode(int id) const {
  static const std::string kInvalid = "<?>";
  return (id >= 0 && id < (int)table_.size()) ? table_[id] : kInvalid;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int id : ids) s += decode(id);
  return s;
}

}  // namespace gpt2
