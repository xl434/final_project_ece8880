// GPT-2 detokenizer (token ID -> bytes), ported from llm.c llmc/tokenizer.h.
// Decode only: prompts are encoded in Python (golden/golden_hf.py) and passed
// to the C++ programs as int32 token-ID files.
#ifndef GPT2_COMMON_TOKENIZER_H_
#define GPT2_COMMON_TOKENIZER_H_

#include <string>
#include <vector>

namespace gpt2 {

class Tokenizer {
 public:
  bool load(const std::string& path);  // llm.c gpt2_tokenizer.bin
  bool ok() const { return !table_.empty(); }
  int eot() const { return eot_; }
  const std::string& decode(int id) const;
  std::string decode(const std::vector<int>& ids) const;

 private:
  std::vector<std::string> table_;
  int eot_ = 50256;
};

}  // namespace gpt2

#endif
