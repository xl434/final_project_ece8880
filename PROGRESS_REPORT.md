# GPT-2 (124M) inference on the Alveo U55C: progress report

Scope: inference only. The pretrained OpenAI GPT-2 124M weights are used (llm.c
`gpt2_124M.bin`, checked bit-identical to Hugging Face `gpt2`). Batch 1, greedy
decoding, token-by-token decode with a KV cache, FP32. Tokenization and
detokenization run on the host.

Every number below was produced by a script in this repository; the source file
is given next to each table. Numbers marked **ESTIMATE** are derived from models
or HLS reports, not measured on hardware. No FPGA board was available
(`xbutil examine`: 0 devices), so no hardware numbers exist yet.

Reproduce everything: see the root `README.md`. In short, run
`python golden/golden_hf.py`, then `make -C src/cpu verify profile`,
`python analysis/analyze.py`, `make -C kernels/<k> hls`,
`python analysis/collect_synth.py`, `python analysis/fpga_estimate.py`, and finally
`tests/run_all_tests`.

---

## 1. Application analysis

Source: `analysis/analyze.py` → `results/analysis/analysis.md`, `ops_decode.csv`,
`kv_cache.csv`, `token_bound.csv`, `onchip_fit.csv`, `roofline.png`.

Model: C=768, L=12, NH=12 (head size 64), V=50257 (padded to 50304), maxT=1024,
124.4 M parameters (497.8 MB in FP32).

**Decode is memory-bound.** Each decoded token reads every weight once. With
batch 1 there is no reuse across tokens, so every GEMV does about 2 FLOPs per
4-byte weight.

| context | GFLOP / token | weights MB (FP32) | KV-cache MB read (FP32) | total MB / token | op. intensity FP32 / FP16 / INT8 (FLOP/B) |
|---|---|---|---|---|---|
| 128 | 0.253 | 495 | 9.5 | 505.9 | 0.50 / 1.0 / 2.0 |
| 512 | 0.267 | 495 | 37.8 | 534.2 | 0.50 / 1.0 / 2.0 |
| 1024 | 0.286 | 495 | 75.6 | 571.9 | 0.50 / 1.0 / 2.0 |

Where the bytes go (FP32, context 1024, per token):

| op | MB | share |
|---|---|---|
| LM head (wte, 50257×768) | 154.6 | 27% |
| fc + fcproj | 227 | 40% |
| qkv + attproj | 113.6 | 20% |
| attention (KV cache) | 75.6 | 13% |
| LN, GELU, residual, argmax | < 1 | < 1% |

The four kinds of GEMV and the KV-cache stream account for more than 99% of the
traffic.

**Bounds.** These are upper bounds: they assume 100% of the stated bandwidth and no
other overhead.
- One 512-bit AXI port at 300 MHz carries 19.2 GB/s. That gives 38.0 / 35.9 / 33.6
  FP32 tok/s at contexts 128 / 512 / 1024.
- The full 460 GB/s of HBM gives 909 / 861 / 804 tok/s.
- FP16 doubles these and INT8 quadruples them.
- The compute ceiling is about 1.08 TFLOP/s (**ESTIMATE**: 9024 DSP at 5 DSP per
  FP32 MAC, 300 MHz).

At 0.5 FLOP/B, the ridge point (1.08 TFLOP/s ÷ 460 GB/s ≈ 2.3 FLOP/B) is far to
the right. FP32 decode therefore cannot become compute-bound. The design goal is
to stream weights at port rate on as many HBM channels as possible.

**On-chip fit.** The device has 2016 BRAM36 + 960 URAM = 44.7 MB.
- One FP32 transformer block (28.4 MB) fits, but cannot be double-buffered. In
  FP16 it can.
- The 12 blocks (340 MB) and wte (154 MB) do not fit.
- Weights must therefore stream from HBM for every token.

**KV cache.** 75.5 MB in FP32 at context 1024 (all layers, K+V). It lives in HBM
and is read in full on every decode step.

**Prefill** has intensity 32–96 FLOP/B (T = 128–1024), so it is compute-bound. In
the current flow, prefill runs token by token through the decode kernels (see §6).

## 2. CPU baseline

Sources: `src/cpu`, `results/cpu_verify.csv`, `results/cpu_profile_summary.csv`,
`results/cpu_profile_ops.csv`, `results/cpu_env.txt`.

**Implementation.** The baseline uses the llm.c forward functions:
- encoder, layernorm, matmul, attention, gelu, residual.
- A KV cache was added, with layout (L, maxT, C) for each of K and V.
- A T=1 GEMV path was added.
- The LM head is fused with argmax.

Prefill uses the llm.c tiled matmul over the whole prompt.

**Verification against HF.** Verified at the first decode step, for every
intermediate tensor of every layer plus the logits and the KV cache.

Tolerance: `|got − ref| ≤ 1e-4·max|ref| + 1e-4·|ref|`.

| check | result |
|---|---|
| tensors compared | 147 of 147 PASS |
| worst max_abs / max\|ref\| | 1.05e-5 (l01_fcproj) |
| worst element-wise max_rel | 2.4e-4 (l03_attproj, on near-zero elements; within the combined tolerance) |
| worst norm-relative error | 5.3e-6 |
| logits top-5 ids | identical to HF (6466, 1729, 3298, 1448, 649) |
| greedy tokens, short prompt | 32/32 match HF |
| greedy tokens, 992-token prompt (context up to 1024) | 32/32 match HF |

**Performance.** Platform: AMD EPYC 7713, 8 usable CPUs in the container, g++ 13.3,
`-O3 -march=native -fopenmp`. Prompt length = context − 32; 31 decode steps; median
of 3 runs.

| threads | context | prefill tok/s | decode ms/token | decode tok/s |
|---|---|---|---|---|
| 1 | 128 | 42.4 | 27.14 | 36.8 |
| 1 | 512 | 39.8 | 31.21 | 32.0 |
| 1 | 1024 | 36.2 | 38.57 | 25.9 |
| 8 | 128 | 115.8 | 18.98 | 52.7 |
| 8 | 512 | 135.2 | 22.31 | 44.8 |
| 8 | 1024 | 127.6 | 25.57 | 39.1 |

Decode time per operator, in ms/token (from `analysis.md`; the per-op timers add a
little overhead, so totals differ slightly from the table above):

| threads | ctx | QKV | attention | attproj | fc | fcproj | LM head | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 4.27 | 1.35 | 1.43 | 5.69 | 5.69 | 7.60 | 0.95 |
| 1 | 1024 | 4.32 | 13.76 | 1.49 | 5.83 | 5.86 | 7.62 | 0.95 |
| 8 | 1024 | 3.12 | 6.64 | 1.26 | 4.02 | 3.95 | 5.12 | 1.25 |

Observations:
- At short context, the weight GEMVs plus the LM head take more than 90% of the
  time; the LM head alone is 28%.
- At context 1024, attention grows to 35% on 1 thread.
- The CPU reaches an effective 15–27 GB/s, consistent with being memory-bound
  (`analysis.md`, "Measured CPU baseline placed on the roofline").
- 8 threads give only 1.4–1.5× on decode, because the 8 cores share the memory
  bandwidth.

## 3. Kernel design

Sources: `kernels/*/kernel.{h,cpp}`, `kernels/common/`, `src/host/fpga_model.cpp`,
`src/host/link_config.ini`.

All kernels are TAPA dataflow designs. Each moves 512-bit words (`float_v16`, 16
FP32 values) over AXI. The main streaming loops run at II=1.

**Operator → kernel mapping (one decode step).**
- Each of the 12 layers runs 8 kernel calls: LayerNorm → GEMV(qkv) → Attention →
  GEMV(attproj, + residual) → LayerNorm → GEMV(fc) → GELU → GEMV(fcproj,
  + residual).
- Then come ln_f and the LM head with argmax.
- That is 98 launches per token.
- The residual add is fused into the GEMV epilogue by default. A standalone
  `ResidualKernel` covers the unfused path (`--fuse_residual=false`).

| kernel | architecture | parallelism | why |
|---|---|---|---|
| **GEMV** (priority) | `GemvReadX` → `GemvReadW`×8 → `GemvPE`×8 → `GemvWriteY` (bias + optional fused residual) | `kGemvPorts`=8 AXI ports, each with one PE that takes one 512-bit word per cycle = 128 FP32 MAC/cycle. Each PE interleaves `kGemvRows`=16 output rows with 16 independent accumulators (lutram, `DEPENDENCE false`), so the ~10-cycle fadd recurrence is hidden and the loop runs at II=1. Ports and rows are compile-time parameters. | Decode is bandwidth-bound, so the PE count matches the port count: compute exactly keeps up with 8 ports. The host packs weights so that row block b (16 rows) goes to port b mod 8 in order [C-word][row][16]. Each port then reads purely sequentially, which gives long AXI bursts. Cycles ≈ OC·C/128 + ~240. |
| **LayerNorm** | read → core (mean, variance, normalize passes over an on-chip row; 8 partial sums, tree reduce) → write | 16 lanes per cycle | Row is only 768 floats; 3 on-chip passes cost ~150 cycles. The tree reduction is more accurate than the CPU's sequential sum (§6). |
| **GELU** | elementwise, 16 lanes, tanh form computed as x/(1+e^(−2u)) | 16 lanes | Avoids a tanh core; matches the CPU to 4.8e-7 over a dense sweep of [−20, 20]. |
| **Residual** | elementwise add, 16 lanes | 16 lanes | Bit-exact with the CPU. Normally fused into the GEMV. |
| **Attention + KV cache** | `AttnReadQkv` → `AttnKv` (append new K/V row to HBM, then stream rows 0..pos) → 12× `AttnHead` → `AttnWrite` | All 12 heads in parallel (`invoke<join,12>`). Each head gets its 4 words (64 dims) of every K/V row and does an **online softmax**: running max m, sum l and 64-wide accumulator, in one pass. | One pass over the KV cache: no score buffer and no second read. K and V are on separate ports, so one position (48 words) is supplied every 48 cycles. The head loop has II=12, which is under 48, so it is not the bottleneck. The new token's K/V are forwarded from on-chip memory rather than read back. |
| **LM head + argmax** | `LmReadX` → `LmReadW`×8 → `LmPE`×8 (per-lane running best value/index) → `LmMerge` | Same 8×16 streaming structure as the GEMV | Logits are never written to memory: only (index, value) leave the kernel. Padded rows ≥ 50257 are masked. Ties resolve to the lowest index, which matches `argmax` in PyTorch and the CPU. |

**HBM placement plan** (`src/host/link_config.ini`; written but **not yet linked**):

| buffer | HBM pseudo-channels | size |
|---|---|---|
| GEMV weights, ports w_0..w_7 (all 12 layers, packed) | HBM[0..7] | 42.5 MB each |
| LM-head wte, ports w_0..w_7 | HBM[8..15] | 19.3 MB each |
| K cache | HBM[16] | 37.7 MB |
| V cache | HBM[17] | 37.7 MB |
| activations, biases, LN params, results | HBM[18] | small |

Rationale: each weight port gets its own pseudo-channel so that the streams do not
contend. The GEMV and the LM head never run at the same time, so they could share
channels. The plan keeps them separate to leave room for a later fused kernel.

## 4. Evaluation flow

The verification chain is HF `gpt2` → `src/cpu` → kernels:

1. `golden/golden_hf.py` produces golden activations for every layer, the KV cache
   and greedy tokens (`data/`). It checks that the llm.c checkpoint matches HF
   (max diff 0) and that the manual KV-cache greedy loop equals HF `generate()`.
2. (L1) `build/bin/cpu_main verify` checks the CPU model against HF (§2).
3. (L2) Each kernel testbench (`kernels/<k>/tb.cpp`) checks the kernel against the CPU
   functions in `src/cpu/ops.cpp`. It uses real GPT-2 data (golden activations and
   weights) plus random and edge cases (minimum shapes, long C, pos=0, pos=1023,
   ties, padded rows, large offsets). Where meaningful it also prints the error
   against HF.
   - The same `tb` binary runs in three modes: software simulation (`make swsim`),
     TAPA fast RTL co-simulation (`make cosim`, against the synthesized `.xo`), and
     hardware (`--bitstream=*.xclbin`).
4. (L3) `tests/block_test.cpp` feeds the HF input of a layer (layers 0 and 11) through a
   full transformer block built from kernels. It checks all 8 block activations
   and the KV-cache rows against `src/cpu` on the same input, and prints the error
   against HF for information.
5. (L4) `build/bin/host_main --check` runs end-to-end greedy decoding with every op on
   the kernels (TAPA swsim) and compares the token IDs with HF.

Tolerance: `|got − ref| ≤ 1e-4·max|ref| + 1e-4·|ref|`. Integer outputs (token IDs,
argmax index) must match exactly, and the residual must be bit-exact.

**Test summary** (`tests/run_all_tests`, full run; `results/test_summary.md`,
logs in `results/test_logs/`):

**12 passed, 0 failed** (run on 2026-11-07, 46 min wall in total). This run used
a clean rebuild of every binary from the current sources, after the build outputs
were moved to `build/`.

| test | level | what it checks | result | time (s) |
|---|---|---|---|---|
| L1_cpu_vs_hf | L1 | `src/cpu` vs HF: 147 tensors + 2 greedy token sequences | PASS | 11 |
| L2_kernel_gemv | L2 | GEMV swsim vs CPU (6 cases) | PASS 6/6 | 3 |
| L2_kernel_layernorm | L2 | LayerNorm swsim vs CPU / FP64 (6 cases) | PASS 6/6 | 0 |
| L2_kernel_gelu | L2 | GELU swsim vs CPU (4 cases) | PASS 4/4 | 0 |
| L2_kernel_residual | L2 | residual swsim, bit-exact (3 cases) | PASS 3/3 | 0 |
| L2_kernel_attention | L2 | attention + KV-cache update swsim vs CPU (5 cases × output, cache) | PASS 10/10 | 4 |
| L2_kernel_lm_head | L2 | LM head + argmax swsim (3 cases × index, value) | PASS 6/6 | 3 |
| L3_block_fused | L3 | full blocks 0 and 11 from kernels vs `src/cpu` (HF input; residual fused into GEMV) | PASS 20/20 | 5 |
| L3_block_unfused_residual | L3 | same, with the standalone residual kernel | PASS 20/20 | 7 |
| L4_e2e_8tok_cpu_fallback | L4 | 8 greedy tokens, attention + GELU on CPU, rest on kernels | PASS 8/8 tokens | 493 |
| L4_e2e_8tok_unfused | L4 | 8 greedy tokens, all kernels, unfused residual | PASS 8/8 tokens | 542 |
| L4_e2e_32tok_all_kernels | L4 | 32 greedy tokens, every op on kernels | PASS 32/32 tokens | 1628 |

GEMV RTL co-simulation is run separately (§5.3), because it takes more than 2 hours.

## 5. Results so far

### 5.1 Simulation (TAPA swsim; functional, not timing)

Source: `results/test_logs/L2_kernel_*.log`. Errors are kernel vs `src/cpu` on real
GPT-2 data unless noted.

| kernel | cases | worst max_abs on real data | other cases |
|---|---|---|---|
| GEMV | 6/6 | 1.5e-5 (l11 fcproj + residual, \|ref\| up to 201; norm_rel 7.5e-8) | random C=3072, values up to 2610: max_abs 9.8e-4, norm_rel 3.1e-7 |
| LayerNorm | 6/6 | 7.6e-5 (ln_f, \|ref\| up to 204; max_rel 4.9e-7) | offset 1000 vs FP64: max_abs 2.2e-5 |
| GELU | 4/4 | 4.8e-7 | [−20, 20] sweep: max_abs 4.8e-7 |
| residual | 3/3 | 0 (bit-exact) | random: 0 |
| attention | 10/10 | 3.9e-7 (layers 0, 6, 11); KV cache written correctly | pos=0 exact; random pos=1023: max_abs 6.9e-6 |
| LM head | 6/6 | argmax = 6466 (matches HF); max logit off by 2.3e-5 | tie → lowest index (100); maximum in the last real row (50256) found |

The block tests run whole transformer blocks (layers 0 and 11) from kernels and
check them against `src/cpu`: 20/20 tensors pass. The worst max_abs / max|ref| is
6.7e-7.

**End to end, all ops on kernels (swsim).** Prompt "The FPGA is a" + 32 greedy
tokens. All 32 token IDs match HF:

> The FPGA is a joint venture between the FPGA and the National Science Foundation. The FPGA is a joint venture between the FPGA and the National Science Foundation

Wall time was 27–32 min across three runs (1628 s in the latest test suite), almost all of
it spent simulating the GEMV. This is simulation time, not FPGA performance.

### 5.2 Synthesis (Vitis HLS 2024.2 through TAPA, xcu55c, 3.33 ns target; HLS estimates, no place & route)

Source: `results/synthesis.md`, `results/synthesis.csv`.

| kernel | latency assumption | est. cycles | @300 MHz | worst II | est. Fmax | LUT | FF | DSP | BRAM_18K |
|---|---|---|---|---|---|---|---|---|---|
| gemv | C=768, OC=3072 (fc) | 18672 | 62.2 µs | 1 | 411 MHz | 74353 (5.7%) | 117621 (4.5%) | 684 (7.6%) | 202 (5.0%) |
| layernorm | 1 row | 420 | 1.4 µs | 1 | 374 MHz | 27285 (2.1%) | 37465 (1.4%) | 166 (1.8%) | 60 (1.5%) |
| gelu | 3072 elements | 257 | 0.9 µs | 1 | 411 MHz | 30545 (2.3%) | 40020 (1.5%) | 416 (4.6%) | 30 (0.7%) |
| residual | 768 elements | 61 | 0.2 µs | 1 | 411 MHz | 11284 (0.9%) | 16429 (0.6%) | 32 (0.4%) | 45 (1.1%) |
| attention | pos=1023, one layer | 49229 | 163.9 µs | 12* | 380 MHz | 234762 (18.0%) | 510492 (19.6%) | 2460 (27.3%) | 90 (2.2%) |
| lm_head | 50304 rows | 301959 | 1005.5 µs | 2* | 411 MHz | 76398 (5.9%) | 117013 (4.5%) | 640 (7.1%) | 201 (5.0%) |

\* Every weight- and KV-streaming loop achieves II=1.
- In attention, II=12 is the per-position loop of `AttnHead`. One position is only
  supplied every 48 cycles, so the head is not the bottleneck.
- In the LM head, II=2 is in the 8-input merge and the final 16-lane reduction,
  which run once per call.

Utilization is relative to the whole XCU55C; the shell is not subtracted. The sum
of all six kernels is about 35% LUT and 49% DSP. Attention dominates, because it
has 12 parallel heads, each with a 64-wide accumulator.

Cycle counts match the design formulas. Examples:
- GEMV: 3072·768/128 = 18432 streaming cycles + 240 overhead.
- LM head: 50304·768/128 = 301824 streaming cycles + 135 overhead.
- Attention: 1023·48 = 49104 streaming cycles + 125 overhead.

### 5.3 Emulation / co-simulation

| run | result | log |
|---|---|---|
| GEMV, TAPA fast RTL cosim of the synthesized `gemv.xo` (Vivado XSim), all 6 tb cases (4 real-layer shapes incl. fused residual, minimum shape, C=3072 random) | **6/6 PASS**; real-data max_abs ≤ 1.5e-5, identical to swsim | `results/cosim_gemv.log` (2 h 14 min wall) |

The cosim verifies the RTL functionally. The temporary XSim directories were
deleted, so no simulated cycle counts were recorded. Vitis `hw_emu` (which needs a
v++ link against the platform) has not been run.

### 5.4 Decode-time ESTIMATE from HLS cycles

**ESTIMATE**, not a measurement. Source: `analysis/fpga_estimate.py` →
`results/fpga_estimate.csv`.

Assumptions:
- 300 MHz clock.
- Kernels run one after another.
- Per-call cycles are the synthesized formulas above.
- The "no launch" column assumes each port gets 19.2 GB/s and there is zero launch
  overhead.
- The launch columns add 98 launches per token at 10 or 50 µs each.
- The last column instead limits each port to one HBM pseudo-channel's peak of
  460/32 = 14.4 GB/s. This is probably the more realistic case.

| context | GEMV ms | attention ms | LM head ms | total ms/token | tok/s (no launch) | tok/s (+10 µs/launch) | tok/s (+50 µs/launch) | tok/s (HBM pseudo-channel-limited) |
|---|---|---|---|---|---|---|---|---|
| 128 | 2.25 | 0.25 | 1.01 | 3.55 | 282 | 221 | 118 | 212 |
| 512 | 2.25 | 0.99 | 1.01 | 4.29 | 233 | 190 | 109 | 175 |
| 1024 | 2.25 | 1.97 | 1.01 | 5.27 | 190 | 160 | 98 | 142 |

For comparison, the measured CPU decode on 8 threads runs at 52.7 / 44.8 / 39.1
tok/s. The estimate is therefore 2.5–5.4× the CPU, depending on the launch overhead
and the HBM rate. The no-launch estimate is also only 24–31% of the 460 GB/s bound in §1: only 8
of the 32 pseudo-channels are used for weights, and only 1 each for K and V.

## 6. Issues, risks and next steps

### Deviations from the plan

- **TinyGPT** was used only as a reference: its current version supports
  Llama/Qwen/Mistral, not GPT-2. llm.c and nanoGPT were used as planned.
- **Long golden prompt.** It is 992 tokens of self-written text, not a public
  dataset, so the test data has no licence issues.
- **CPU profiling.** The container exposes 8 of the 64 EPYC cores, so the "OpenMP"
  rows use 8 threads.
- **LayerNorm offset-1000 stress case.** It is checked against an FP64 LayerNorm,
  not against `src/cpu`. The CPU's sequential FP32 sum has an error of 5.2e-4
  there; the kernel's tree sum has 2.2e-5. The CPU numbers are still printed
  ("info"). All real-data LayerNorm cases are checked against `src/cpu` as
  planned.
- **Prefill** in the host flow runs token by token through the decode kernels,
  which is correct but slow. The CPU baseline does batched prefill.
- **"Emulation"** is TAPA fast RTL cosim of the GEMV, not Vitis `hw_emu`.
  `link_config.ini` has not been tried with `v++ -l`.
- **Attention port renamed.** `tapa compile` rejected the output mmap name `out`,
  so it is now `attn_out`. This was a rename only; it was re-synthesized, and the
  test suite was re-run afterwards.
- **Plot palette.** The palette validator from the plotting guidelines could not
  run (no Node.js), so the documented reference palette was used unchanged.

### Risks

1. **No hardware yet.** All performance numbers for the FPGA are estimates. Timing
   closure after place and route is unknown; HLS estimates 374–411 MHz against a
   300 MHz target. The six kernels together use about 49% of the DSPs.
2. **HBM rate per port.** One pseudo-channel peaks at about 14.4 GB/s, below one
   512-bit port at 300 MHz (19.2 GB/s). The GEMV and LM head will therefore be
   HBM-limited. The last column of §5.4 accounts for this.
3. **Launch overhead.** 98 launches per token at 10–50 µs each costs 1–5 ms, as
   much as the kernel compute itself.
4. **Host copies.** `tapa::invoke` with host vectors copies buffers on every call.
   On the board, weights and the KV cache must be persistent device buffers,
   allocated once (for example as XRT sub-buffers per layer). This is not yet
   implemented.
5. **Attention bandwidth.** Attention reads K and V through one port each. It is
   37% of the estimated time at context 1024, and grows linearly with context.
6. **Routing congestion.** The attention kernel is the largest (18% LUT, 27% DSP).
   Putting all six kernels plus the shell on one device may be hard to route
   across SLR boundaries.

### Next steps (in priority order)

1. **Link and run on hardware.** `v++ -l` with `link_config.ini`; then run `hw_emu`
   for the GEMV and the full set of kernels. Move the host to persistent XRT
   buffers.
2. **More HBM bandwidth.** Widen the GEMV and LM head to 16 or more ports. The
   parameters already exist; check routing and DSP use. Split the KV cache by head
   group across 4–6 channels, so that each `AttnHead` group gets its own port.
3. **Fewer launches.** Fuse a whole transformer block, or the whole decode step,
   into a single TAPA top level with on-chip activation FIFOs. This removes 97 of
   the 98 launches per token and the intermediate HBM round trips.
4. **Batched prefill.** Use a matrix-matrix (GEMM) mode for the prompt, reusing
   each weight across T tokens; prefill is compute-bound (§1).
5. **Lower precision (later stage, not started).** FP16 or INT8 weights would
   halve or quarter the bytes per token. That roughly doubles or quadruples the
   bandwidth bound. The verification chain built here would be reused, with new
   tolerances.
