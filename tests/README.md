# tests/ — all verification, in one place

Verification is a chain. Each level is checked against the level before it, so a
failure points to exactly one layer of the stack.

| level | what is checked | against | test code | one command |
|---|---|---|---|---|
| **L1** | CPU reference model (`src/cpu`): 147 intermediate tensors + greedy tokens (short and 992-token prompt) | Hugging Face golden data (`data/`) | `src/cpu/cpu_main.cpp` (`verify` mode) | `tests/run_all_tests --level 1` |
| **L2** | each HLS kernel alone, on real GPT-2 data + edge cases | CPU reference operators (`src/cpu/ops.cpp`) | `kernels/<name>/tb.cpp` (one per kernel) | `tests/run_all_tests --level 2` |
| **L3** | one full transformer block built from kernels (layers 0 and 11), all 8 activations + KV cache | CPU reference, same input | `tests/block_test.cpp` | `tests/run_all_tests --level 3` |
| **L4** | the whole model on kernels: generated token IDs | Hugging Face tokens (`data/output_ids.i32`), exact match | `src/host/host_main.cpp` (`--check`) | `tests/run_all_tests --level 4` |

- Kernels in L2–L4 run in TAPA software simulation.
- Tolerance for floats: `|got − ref| ≤ 1e-4·max|ref| + 1e-4·|ref|`.
- Token IDs and argmax indices must match exactly. The residual kernel must be
  bit-exact.

```bash
tests/run_all_tests              # all levels, ~45 min (the 32-token L4 run is ~30 min)
tests/run_all_tests --quick      # all levels but skip the 32-token run, ~16 min
tests/run_all_tests --level 2    # one level only
```
The runner builds what it needs, then prints PASS/FAIL per test. It writes
`results/test_summary.md` and one log per test in `results/test_logs/`
(`L1_cpu_vs_hf.log`, `L2_kernel_gemv.log`, …).

| test | level | detail |
|---|---|---|
| `L1_cpu_vs_hf` | 1 | `build/bin/cpu_main verify` |
| `L2_kernel_<name>` (6 tests) | 2 | `build/kernels/<name>/tb`; same as `make -C kernels/<name> swsim` |
| `L3_block_fused` | 3 | `build/bin/block_test`; residual add fused into the GEMV |
| `L3_block_unfused_residual` | 3 | `build/bin/block_test --unfused`; standalone residual kernel |
| `L4_e2e_8tok_cpu_fallback` | 4 | 8 tokens; attention and GELU on CPU, the rest on kernels |
| `L4_e2e_8tok_unfused` | 4 | 8 tokens, all on kernels, standalone residual kernel |
| `L4_e2e_32tok_all_kernels` | 4 | 32 tokens, every operator on kernels (skipped by `--quick`) |

Beyond these levels:
- **RTL co-simulation** and **Vitis hardware emulation** of a synthesized kernel run
  the same L2 testbench against the generated hardware: `make -C kernels/<name>
  cosim` and `make -C kernels/<name> hw_emu`. They take minutes to hours, so they
  are not part of the runner. Logs: `results/emulation/`.
- **Your own prompt** through Hugging Face, the CPU model and (optionally) the
  kernels: `python3 run_prompt.py "text" [--fpga]` in the top folder.
