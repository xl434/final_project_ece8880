# third_party/ — reference code (not part of this repository)

Read-only clones used as references. They are not stored in git (see
`.gitignore`). To recreate them at the exact versions used:

```bash
cd third_party
git clone https://github.com/karpathy/llm.c      && git -C llm.c   checkout f1e2ace
git clone https://github.com/karpathy/nanogpt    && git -C nanogpt checkout 3adf61e
git clone https://github.com/keith2018/TinyGPT   && git -C TinyGPT checkout 607f0cf
```

| repo | used for |
|---|---|
| llm.c | forward functions adapted in `src/cpu/ops.cpp`; `gpt2_124M.bin` format |
| nanoGPT | reference for the GPT-2 model structure |
| TinyGPT | reference only (its current version does not support GPT-2) |
