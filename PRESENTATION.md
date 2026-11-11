# Presentation outline (progress update)

A suggested storyline for the slides. Every number and chart comes from
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md); the section is given for each slide.

**One message through the whole talk:** decode is limited by memory bandwidth.
The U55C has 13× more of it than the CPU. All kernels are implemented,
synthesized and verified; the baseline built from them is estimated at 3–4× the
CPU, and a better system-level dataflow (a spatial design) is the path to the
rest.

| # | slide | key message | material |
|---|---|---|---|
| 1 | Goal and scope | GPT-2 small, FP32, batch 1, greedy, KV cache; tokenizer on the host | report header |
| 2 | The model | 124M parameters; where they are (weight shapes × 12 layers + vocabulary) | `WALKTHROUGH.md` §1 tables |
| 3 | Flow and shapes | text → tokens → prefill (T × 768) → decode (1 × 768) → token; the shape after each step | flow diagram (to draw) |
| 4 | Work per token | ~286 MFLOP but ~572 MB per token: 0.5 FLOP/byte | report §1.3, `results/figures/work_per_op.png` |
| 5 | Roofline | decode is memory-bound on both chips; limit = bandwidth ÷ bytes | report §1.4, `results/figures/roofline.png` |
| 6 | CPU baseline | 39–53 tokens/s; GEMVs at 80–89% of measured bandwidth; 8 threads give 1.43× = bandwidth gain 1.46× | report §2, `results/figures/cpu_bandwidth_per_op.png` |
| 7 | Main milestone | all six kernels implemented, synthesized, verified (software simulation + RTL co-simulation; GEMV also hardware emulation) | report header, §4.4 |
| 8 | Kernels: what they compute | per kernel: formula, input → output shapes, type (elementwise / reduction / matrix × vector) | report §3.1, §3.3 |
| 9 | Kernel design and optimizations | dataflow tasks, II = 1, 16 lanes, partitioning, interleaved accumulators; GEMV and attention diagrams | report §3.2, §3.3 |
| 10 | Synthesis | II = 1, 374–411 MHz, area per kernel; ports busy 99–100% | report §5.1, §3.2 |
| 11 | Verification chain | HF → C++ (L1) → kernel (L2) → layer (L3) → whole model (L4); what ran at which level; bugs found by co-simulation | report §4 |
| 12 | System level: the baseline | dataflow inside kernels, temporal between them; round trips through HBM; 32-port limit | report §3.4, §3.5 |
| 13 | Speed (estimate) | GEMV ≈ 4× CPU = bandwidth ratio; whole model 3–4×; about 20% of the HBM bound | report §5.2–5.3, `results/figures/cpu_vs_fpga_per_op.png`, `results/figures/decode_throughput.png` |
| 14 | Next steps | baseline as reference → measure GEMV on board → spatial design (Chen et al.) → prefill → lower precision | report §6 |

**Say clearly:**
- CPU numbers are measured; FPGA speeds are estimates from synthesis.
- The whole model was verified in software simulation; RTL verification is per
  kernel (whole-model RTL simulation would take days).
- The off-chip cost of the baseline is not the volume of intermediate data (0.23%
  of the traffic) but the serialization it forces: one kernel at a time, few
  active channels, 98 launches per token.
- On the FPGA, the prompt currently goes through the decode kernels one token at
  a time; the CPU does a real prefill.
