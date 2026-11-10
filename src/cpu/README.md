# src/cpu/ — CPU reference and baseline

* `ops.h/.cpp` — one function per operator (encoder, layernorm, matmul,
  kv_cache_write, attention with KV cache, gelu, residual, lm_head), adapted
  from llm.c `train_gpt2.c` forward functions (batch removed, no backward/
  training buffers). These are the references for every kernel.
* `gpt2_cpu.h/.cpp` — model forward for T new tokens with a KV cache (prefill:
  T = prompt length; decode: T = 1), logits only for the last token, optional
  per-operator profiler and activation trace.
* `cpu_main.cpp` — `generate`, `verify` (vs `data/`), `profile`.

```bash
make -C src/cpu                 # g++ -O3 -march=native -fopenmp -std=c++17 -> build/bin/cpu_main
build/bin/cpu_main generate [--prompt FILE.i32] [--n 32] [--threads N]
build/bin/cpu_main verify       # -> results/cpu_verify.csv (exit 0 = PASS); = make -C src/cpu verify
build/bin/cpu_main profile --threads 1,8 --ctx 128,512,1024 --reps 3
                                # -> results/cpu_profile_summary.csv, cpu_profile_ops.csv, cpu_env.txt
```
`verify` is verification level L1 (see `tests/README.md`).

Profiling: context L = prefill of L-32 tokens of `data/long_prompt_ids.i32`,
then 31 decode steps (positions L-32 .. L-2); decode time = mean of the 31
steps, median over `--reps` runs. Threads are OpenMP threads (this container
exposes 8 logical CPUs).
