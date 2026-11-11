# Where the project stands, and the plan

This file records the current stage of the project, how to describe the current
design, what to do now (before the progress presentation) and what to leave for
later. It also proposes a storyline for the presentation.

- How to run things: [`README.md`](README.md)
- Plain-language explanation: [`WALKTHROUGH.md`](WALKTHROUGH.md)
- Formal report with all numbers: [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md)

---

## 1. Current stage in one paragraph

GPT-2 small (124M, FP32) runs correctly in three ways:
- the Hugging Face model (the reference);
- our C++ CPU version, which is verified against it and is the speed baseline;
- the whole model on six HLS kernels, in software simulation, giving the same 32
  tokens as Hugging Face.

All six kernels are synthesized, and the GEMV also passed RTL co-simulation. All
FPGA speeds are **estimates** from the synthesis reports: about 3–4× the 4-core
CPU, but only about 20% of what the U55C's memory bandwidth would allow. We know
why, and that is the plan below.

## 2. What kind of design this is: the baseline FPGA design

**Dataflow inside each kernel, temporal between kernels.**

```
 time ──►
 host:  launch LN ─ launch GEMV(QKV) ─ launch Attention ─ launch GEMV(proj) ─ … (98 launches per token)
 FPGA:  [LN busy]   [GEMV busy      ]  [Attention busy ]  [GEMV busy      ]
        others idle  others idle        others idle        others idle
 data:  every result goes back to HBM and is read again by the next kernel
```

- **Inside a kernel: dataflow.** Each kernel is a set of tasks connected by FIFOs:
  read → compute → write. They run at the same time, at one 512-bit word per
  cycle (II = 1).
- **Between kernels: temporal.** The host starts one kernel at a time. Each kernel
  writes its result to HBM, and the next kernel reads it back.
- All six kernels are on the chip at the same time, each in its own area. While
  one runs, the other five are idle.
- One GEMV kernel is reused for four different layer shapes (QKV, attention
  projection, FC, FC-projection).

**Why this is a good baseline.**
- Each kernel can be built, tested and synthesized alone. That is how we got a
  verified end-to-end result quickly.
- Reusing one GEMV for four shapes keeps the area small.
- It gives clear, measurable reference points (cycles per kernel, tokens/s) for
  comparing the next designs against.

**Why it is not efficient**, with each reason tied to the measured estimate
(`PROGRESS_REPORT.md` §5.3):
1. **Most HBM channels are idle.** At any moment at most 8 of the 32 channels
   stream weights; the KV cache has only 2.
2. **Hardware is idle.** Only one kernel works at a time.
3. **98 launches per token,** each with host overhead, and every intermediate
   vector makes a round trip through HBM.

## 3. The next design: spatial (specialized units connected by streams)

The idea: give operators (or layers) their own specialized processing units,
and connect them directly with on-chip streams instead of passing data through
HBM. This is the *spatial* architecture studied in:

> H. Chen et al., "Understanding the Potential of FPGA-Based Spatial Acceleration
> for Large Language Model Inference," *ACM Transactions on Reconfigurable
> Technology and Systems* (2024). arXiv:2312.15159.

What our own analysis says such a design can and cannot do:

- **Batch-1 decode stays memory-bound.** Every token must still read all 498 MB
  of weights, which cannot be kept on chip (44.7 MB on chip). So the speed limit,
  about 800 tokens/s at context 1024 (memory bandwidth ÷ bytes per token), does
  not move. A spatial design helps decode by **getting close to that limit**:
  - **All channels busy:** each unit has its own HBM channels, and all of them
    stream at once (today: 8 of 32).
  - **No launches and no HBM round trips:** one long-running design; vectors move
    between units through FIFOs.
  - **Overlap between operators,** for example LN → QKV, or FC → GELU →
    FC-proj. This is limited: a GEMV needs its whole input vector before any
    output is final, and LayerNorm needs its full sum before it can output
    anything.
  - **Weight prefetching:** one layer's weights (28 MB) fit on chip, so the next
    layer's weights can be loaded while the current layer computes.
- **Prefill gains more.** Prefill is compute-bound (about 100 FLOP/byte for a
  992-token prompt). There, tokens can be pipelined through specialized units,
  and the multiply-add hardware stays busy.

## 4. What to do now, and what to leave for later

### Now (for the progress presentation)

| task | why | status |
|---|---|---|
| Describe the design as "dataflow inside, temporal between kernels" (§2) | the course asks for the temporal-vs-dataflow choice and its reasons | done (this file, report §3.2) |
| Vitis **hardware emulation** of the GEMV (`make -C kernels/gemv hw_emu`) | runs the kernel inside the real U55C shell with an HBM model, with the HBM placement from `link_config.ini` | running |
| Full **hardware build** of all six kernels (`make -C src/host xclbin_hw`) | place-and-route gives real area and clock numbers, and shows whether the whole baseline fits and routes on the chip. No board is needed | running (takes hours) |
| A **flow diagram** of the model with the shape after each step (prefill and decode) | the clearest way to explain the model on a slide | optional |
| A small **GEMV design-space sweep** (4 / 8 / 16 ports: cycles vs area) | shows that parallelism was explored, not only chosen | optional; the port count is already a parameter |

**A known limitation:** TAPA's host runtime expects one kernel per xclbin. So
hardware emulation runs per kernel, while the six-kernel xclbin is built for its
reports. Running the whole model on one six-kernel xclbin needs a native XRT host
program (listed under "later").

### Later (next stages of the project)

In order of expected gain:
1. **Board run** of the baseline: measured tokens/s and launch overhead, replacing
   the estimates. This needs a native XRT host that keeps weights and the KV cache
   in device memory, allocated once. `tapa::invoke` copies buffers on every call.
2. **Spatial decode design** (§3): more HBM channels per GEMV, KV cache split
   across channels by head groups, one fused layer with streams between units,
   and weight prefetching.
3. **Real prefill:** a matrix × matrix mode, pipelined across tokens.
4. **Lower precision** (FP16 or INT8 weights): half or a quarter of the bytes per
   token, so up to 2× or 4× the speed limit. The verification chain is reused
   unchanged.

## 5. A storyline for the presentation

One message runs through the talk: **decode is limited by memory bandwidth. The
U55C has 13× more of it than the CPU. Our verified baseline already reaches an
estimated 3–4×, and a spatial design is the path to the rest.**

| # | slide | key message | material |
|---|---|---|---|
| 1 | Goal and scope | GPT-2 small, FP32, batch 1, greedy, KV cache; tokenizer on the host | report header |
| 2 | The model | 124M parameters; where they are (weight shapes × 12 layers + vocabulary) | `WALKTHROUGH.md` §1 tables |
| 3 | Flow and shapes | text → tokens → prefill (T × 768) → decode (1 × 768) → token; the shape after each step | flow diagram |
| 4 | Work per token | ~286 MFLOP but ~572 MB per token: 0.5 FLOP/byte | `results/figures/work_per_op.png` |
| 5 | Roofline | decode is memory-bound on both chips; limit = bandwidth ÷ bytes | `results/figures/roofline.png` |
| 6 | CPU baseline | 39–53 tokens/s; the GEMVs run at 80–89% of measured bandwidth; 8 threads give 1.43× = bandwidth gain 1.46× | `results/figures/cpu_bandwidth_per_op.png` |
| 7 | Kernels: what they compute | per kernel: formula, input → output shapes, type (elementwise / reduction / matrix × vector), FLOPs | report §3.1, §3.3 |
| 8 | Kernel design | GEMV: 8 PEs, 16 interleaved rows for II = 1; attention: online softmax, 12 heads in parallel | report §3.3 |
| 9 | Baseline architecture | dataflow inside, temporal between kernels; HBM plan | this file §2, report §3.4 |
| 10 | Verification chain | HF → C++ (L1) → each kernel (L2) → one layer (L3) → whole model (L4); 12/12 pass, tokens identical | report §4 |
| 11 | Results | synthesis table (II = 1, 374–411 MHz); emulation and hardware build results | report §5.1, §5.4 |
| 12 | Speed (estimate) | GEMV ≈ 4× CPU = the bandwidth ratio; whole model 3–4×; about 20% of the HBM limit | `results/figures/cpu_vs_fpga_per_op.png`, `results/figures/decode_throughput.png` |
| 13 | Why only 20%, and the plan | idle channels, idle kernels, 98 launches → spatial design (Chen et al.) → prefill → lower precision | this file §3, §4 |

Things to say clearly:
- CPU numbers are measured; FPGA speeds are estimates from synthesis.
- The design was built *with* pipelining and parallelism. Only one parallelism
  setting has been synthesized so far, unless the optional sweep is done.
- On the FPGA, the prompt currently goes through the decode kernels one token at
  a time; the CPU does a real prefill.
