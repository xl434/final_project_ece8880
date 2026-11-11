# Plan: checklist and current state

Short status and to-do list. The reasoning behind each item is in
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) (section numbers given in brackets).

## Current state

- GPT-2 small (124M, FP32, batch 1, greedy, KV cache) runs end to end (prompt,
  then token-by-token generation) three ways: Hugging Face (reference), our C++
  CPU model, and the whole model on six HLS kernels in software simulation. All
  give the same tokens.
- **One temporal design for both phases.** Every kernel call takes T tokens:
  the whole prompt in one pass (prefill), then T = 1 per generated token
  (decode). Kernels run one at a time and pass results through HBM (report §3.5).
- All six kernels are synthesized and pass their decode and prefill tests in
  software simulation. RTL co-simulation: see the checklist.
- Estimates (from synthesis, **not measured**): decode 125–212 tokens/s (3–4× the
  4-core CPU); a 992-token prompt in about 1.2 s (CPU: 7.8 s) (report §5.3–5.4).
- Area (HLS): 73% of DSPs, 58% of LUTs, close to the routability guideline
  (report §3.2).

## Open issues

- **Full hardware build** (place-and-route) not run for this design yet; it takes
  15+ hours, and the area is close to the guideline.
- **Prefill attention** is 52% of prefill time: only 4 queries per pass over the
  cache, and new K/V rows arrive at half rate (report §5.4).
- **32 memory-port limit** on the U55C (report §3.4). Any future design must fit
  in 32.
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
- [x] Six HLS kernels: GEMV/GEMM, LM head + argmax, attention + KV cache, LayerNorm, GELU, residual
- [x] Prefill support: GEMV → GEMM (4 token lanes, 64-token chunks), attention with query blocks and causal mask, host runs the prompt in one pass
- [x] Kernel testbenches on real GPT-2 data and edge cases, decode and prefill (software simulation)
- [x] One layer from kernels: decode step and 100-token prefill (software simulation)
- [x] HLS synthesis of all kernels (II = 1 / 12, 374–411 MHz estimated)
- [x] RTL co-simulation: LayerNorm, GELU, residual
- [x] Fit the 32-port limit: one LM-head output buffer; residual kernel left out of the hardware build
- [x] Fix port names that broke the RTL (`pos`, `port`), found by co-simulation
- [x] Decode and prefill speed estimates; report charts

### To do now (progress report and presentation)

- [ ] RTL co-simulation: GEMV/GEMM, attention, LM head (running)
- [ ] One layer in RTL co-simulation: decode step and 16-token prefill (running / queued)
- [ ] End to end, software simulation: test suite and the 992-token prompt (running)
- [ ] Put the results into the report (§4.4) and merge the `prefill` branch
- [ ] Flow diagram of the model with the shape after each step (for the slides)

### Next stage (in order; reasons in report §6)

- [ ] Full hardware build of this design: place, route, clock (15+ hours)
- [ ] GEMV alone on the board: measure real bandwidth per HBM channel and launch overhead
- [ ] Faster prefill attention: K/V rows at full rate, more queries per pass
- [ ] Spatial design: specialized units connected by on-chip streams, all 32 ports
      spent on weights and KV cache, no per-operator launches (Chen et al., TRETS 2024)
- [ ] KV cache spread over several ports by head groups
- [ ] Native XRT host program that keeps weights and KV cache on the card

### Later

- [ ] Lower precision: FP16 / INT8 weights
