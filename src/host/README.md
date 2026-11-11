# src/host/ — FPGA host program

* `fpga_model.h/.cpp` — packs weights for the GEMV / LM-head kernels once,
  owns the KV cache, runs the model through the kernels with `tapa::invoke`:
  `prefill()` runs the whole prompt in one pass (T tokens per kernel call),
  `step()` one generated token (T = 1). Every operator class (`layernorm, gemv,
  attention, gelu, residual, lm_head`) can be forced onto the CPU (`src/cpu`).
  Embedding lookup is always on the host. With `xo_dir` set, every kernel runs
  on its own synthesized `.xo` (RTL co-simulation).
* `host_main.cpp` — prompt prefill (one pass; `--prefill=false`: one token at a
  time), greedy generation, detokenization, `--check` against
  `data/output_ids.i32` or `--expect=FILE.i32`.
* `link_config.ini` — HBM placement of every kernel port. Used by
  `make xclbin_hw` (all kernels except the standalone residual kernel, to stay
  within the U55C's 32 kernel ports) and, per kernel, by `make hw_emu` in
  `kernels/<name>`.
* `vivado/` — scripts from the ECE 8880 Lab 4 flow, used unchanged:
  `udev_stub.c` (Vivado's license check crashes in this workspace's `libudev`;
  the stub is loaded for Vivado only), `clkwiz_compat.tcl` (Vitis 2024.2 clock
  fix), `kernel_util.tcl` (routed area per kernel).

```bash
make -C src/host                                # -> build/bin/host_main (all kernels compiled for swsim)
build/bin/host_main --check                     # all ops in TAPA software simulation, 32 tokens
build/bin/host_main --prompt=data/long_prompt_ids.i32 --expect=data/long_output_ids.i32 --n=8
                                                # 992-token prompt in one pass, then 8 tokens
build/bin/host_main --n=8 --check --prefill=false             # prompt one token at a time
build/bin/host_main --n=8 --check --cpu_ops=attention,gelu    # mixed FPGA/CPU
build/bin/host_main --n=8 --check --fuse_residual=false       # standalone residual kernel
build/bin/host_main --prompt=FILE.i32 --n=5     # your own prompt (see run_prompt.py in the top folder)
build/bin/host_main --plan                      # print the HBM buffer plan

make -C src/host xclbin_hw                      # hardware build -> build/hw/gpt2.hw.xclbin (many hours)
make -C src/host hw_util                        # routed area per kernel, after xclbin_hw
```
`host_main --bitstream=<xclbin>` is not usable for the multi-kernel image yet:
`tapa::invoke` handles one kernel per xclbin and copies buffers on every call. A
board run of the whole model needs an XRT host (see `PLAN.md`).
`--check` is verification level L4 (see `tests/README.md`).
Wall times printed in software simulation are not FPGA performance.
