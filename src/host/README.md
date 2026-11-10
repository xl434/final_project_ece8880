# src/host/ — FPGA host program

* `fpga_model.h/.cpp` — packs weights for the GEMV / LM-head kernels once,
  owns the KV cache, runs one token at a time through the kernels with
  `tapa::invoke`; every operator class (`layernorm, gemv, attention, gelu,
  residual, lm_head`) can be forced onto the CPU (`src/cpu`). Embedding lookup
  is always on the host.
* `host_main.cpp` — prompt prefill (token by token through the KV cache),
  greedy generation, detokenization, `--check` against `data/output_ids.i32`.
* `link_config.ini` — planned HBM placement for a single xclbin (not linked yet).

```bash
make -C src/host                                # -> build/bin/host_main (all kernels compiled for swsim)
build/bin/host_main --check                     # all ops in TAPA software simulation, 32 tokens
build/bin/host_main --n=8 --check --cpu_ops=attention,gelu    # mixed FPGA/CPU
build/bin/host_main --n=8 --check --fuse_residual=false       # standalone residual kernel
build/bin/host_main --prompt=FILE.i32 --n=5     # your own prompt (see run_prompt.py in the top folder)
build/bin/host_main --plan                      # print the HBM buffer plan
build/bin/host_main --bitstream=gpt2.xclbin     # board (requires a linked xclbin; not available yet)
```
`--check` is verification level L4 (see `tests/README.md`).
Wall times printed in software simulation are not FPGA performance.
