# src/common/ — shared host code

* `config.h` — GPT-2 124M dimensions (checked against the checkpoint header).
* `weights.h/.cpp` — loads llm.c `gpt2_124M.bin` (FP32, version 3) into one
  buffer with per-tensor/per-layer views. Path: `$GPT2_MODEL_DIR` or `models/`.
* `tokenizer.h/.cpp` — decode-only GPT-2 tokenizer (port of llm.c
  `llmc/tokenizer.h`, reads `gpt2_tokenizer.bin`). Prompts are encoded in Python.
* `utils.h` — `argmax` (first max), `Timer`, binary file I/O, and
  `compare()` / `Tolerance`: element passes if
  `|got-ref| <= atol_rel*max|ref| + rtol*|ref|` (default 1e-4 / 1e-4); reports
  max abs error, max relative error (over elements with |ref| >= 1% of max|ref|)
  and norm-relative error.

Built as part of `src/cpu`, `src/host`, `kernels/*` and `tests` (no separate build).
