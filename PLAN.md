# Plan: checklist and current state

Short status and to-do list. The reasoning behind each item is in
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) (section numbers given in brackets).

## Current state

- GPT-2 small (124M, FP32, batch 1, greedy, KV cache) runs correctly three ways:
  Hugging Face (reference), our C++ CPU model, and the whole model on six HLS
  kernels in software simulation. All give the same tokens.
- All six kernels are synthesized and pass their tests in software simulation.
  RTL co-simulation passes for GEMV, LayerNorm, GELU and residual (attention and
  LM head re-running after a fix). The GEMV also passes Vitis hardware emulation.
- This **baseline** design is efficient inside each kernel (memory ports busy
  99–100% of the time) but not at the system level: kernels run one at a time
  and pass results through HBM (report §3.5).
- Estimated speed (from synthesis, **not measured**): 125–212 tokens/s, 3–4× the
  4-core CPU, about 20% of the U55C's bandwidth limit (report §5.3).

## Open issues

- **Full hardware build** (place-and-route) not finished yet. It takes many hours.
- **32 memory-port limit** on the U55C. The baseline had 36 ports; it now has 32
  (report §3.4). Any future design must fit in 32.
- **No board run yet.** The two unknowns of the speed estimate, real HBM
  bandwidth per channel and launch overhead, can only be measured on the board.
- **Host program is simulation-only.** `tapa::invoke` copies buffers on every
  call and handles one kernel per FPGA image; a whole-model board run needs an
  XRT host.

## Checklist

### Done

- [x] Hugging Face golden reference and recorded data (`golden/`, `data/`)
- [x] C++ CPU model, verified against Hugging Face (147 tensors, 32/32 tokens)
- [x] CPU profiling and measured machine limits (bandwidth, compute peak)
- [x] Application analysis: FLOPs and bytes per operator, roofline, bandwidth bound
- [x] Six HLS kernels: GEMV, LM head + argmax, attention + KV cache, LayerNorm, GELU, residual
- [x] Kernel testbenches on real GPT-2 data and edge cases (software simulation)
- [x] One layer and the whole model on kernels (software simulation), tokens identical to Hugging Face
- [x] `tests/run_all_tests`: 12/12 pass
- [x] HLS synthesis of all kernels (II = 1, 374–411 MHz estimated)
- [x] RTL co-simulation: GEMV, LayerNorm, GELU, residual pass
- [x] Vitis hardware emulation of the GEMV: 6/6 pass
- [x] Fit the 32-port limit: one LM-head output buffer; residual kernel left out of the hardware build
- [x] Fix port names that broke the RTL (`pos`, `port`), found by co-simulation
- [x] FPGA speed estimate per operator and per token; report charts

### To do now (progress report and presentation)

- [ ] RTL co-simulation of attention and LM head after the port-name fix (running)
- [ ] Full hardware build of the baseline: does it place, route and meet 300 MHz? (running)
- [ ] Put the co-simulation and hardware-build results into the report (§4.4, §5.4)
- [ ] Flow diagram of the model with the shape after each step (for the slides)
- [ ] Optional: GEMV design-space sweep (4 / 8 / 16 ports: cycles vs area)

### Next stage (in order; reasons in report §6)

- [ ] GEMV alone on the board: measure real bandwidth per HBM channel and launch overhead
- [ ] Spatial design: specialized units connected by on-chip streams, all 32 ports
      spent on weights and KV cache, no per-operator launches (Chen et al., TRETS 2024)
- [ ] KV cache spread over several ports by head groups
- [ ] Native XRT host program that keeps weights and KV cache on the card
- [ ] Full hardware build and board run of the optimized design

### Later

- [ ] Real prefill (matrix × matrix, compute-bound)
- [ ] Lower precision: FP16 / INT8 weights
