# kernels/ — HLS kernels (TAPA 0.1.20250815, Vitis HLS 2024.2, U55C, 3.33 ns)

| kernel | top | function |
|---|---|---|
| `gemv/` | `GemvKernel` | Y = X W^T + b (+ residual) for T tokens (decode T = 1, prefill T = prompt); 8 HBM ports, 4 token lanes x 16 FP32 MAC/cycle per PE, 16 interleaved rows/PE, X chunks of 64 tokens in URAM, II=1 |
| `layernorm/` | `LayerNormKernel` | LayerNorm of T rows, 16 lanes, three II=1 passes per row |
| `gelu/` | `GeluKernel` | tanh-GELU as x / (1 + exp(-2u)), 16 lanes |
| `residual/` | `ResidualKernel` | y = a + b, 16 lanes |
| `attention/` | `AttentionKernel` | T new tokens: KV-cache append + causal attention, 12 heads in parallel, online softmax, query blocks of 4 (one update per 12 cycles) |
| `lm_head/` | `LmHeadKernel` | wte GEMV fused with argmax (8 ports); only {token ID, logit} leaves the chip, in one output buffer |

`common/kernel_common.h` (types, dimensions, adder trees), `common/host_pack.h`
(weight packing for GEMV/LM head), `common/tb_util.h` (testbench helpers),
`common.mk` (build rules shared by every kernel Makefile).

Each `kernels/<name>/` holds only its 4 source files: `kernel.h` (interface),
`kernel.cpp` (design), `tb.cpp` (testbench) and `Makefile`. Everything generated
goes to `build/kernels/<name>/`, which is safe to delete. That folder holds the
testbench binary `tb`, the objects, `<name>.xo`, `hls.log` and `work.out/`
(HLS reports).

In each `kernels/<name>/`:

```bash
make swsim     # build the testbench and run it in TAPA software simulation (compares with src/cpu)
make hls       # synthesis: tapa compile -> build/kernels/<name>/<name>.xo, reports in .../work.out/report
make cosim     # TAPA fast RTL co-simulation of the .xo (the labs' "hardware emulation")
make hw_emu    # Vitis hardware emulation: v++ link into the U55C platform (HBM placement
               # from src/host/link_config.ini), then run the testbench on the emulated card
make cleanall  # delete build/kernels/<name>/
```
The testbench is verification level L2 (see `tests/README.md`).
Each `tb.cpp` uses real activations/weights from `data/` and `models/` plus
random edge cases and prints `PASS`/`FAIL` (exit status 0/1). After `make hls`,
`python3 analysis/collect_synth.py` gathers all reports into `results/synthesis.csv`.
Compile-time parameters (add `-DNAME=value` to `KERNEL_DEFS` in the kernel's
Makefile):

| parameter | default | effect |
|---|---|---|
| `GEMV_PORTS`, `LM_PORTS` | 8 | HBM ports (= PEs) of the GEMV / LM head |
| `GEMV_TOKENS` | 4 | tokens multiplied per weight word per cycle (prefill speed vs DSPs) |
| `GEMV_CHUNK` | 64 | tokens whose X rows are held on chip (weights read once per chunk) |
| `ATTN_II` | 12 | cycles per query update in a head; 48 / `ATTN_II` queries per pass over the cache |

In software simulation TAPA runs each task on a small coroutine stack, so large
on-chip buffers (the GEMV's X chunk) are heap-allocated there
(`#ifdef TAPA_TARGET_DEVICE_`).

Two naming pitfalls, both found the hard way:
- Do not name a kernel or task argument `pos` or `port`: Vitis HLS renames such
  ports (`pos_r`, `port_r`) and TAPA's generated RTL no longer connects. Software
  simulation does not show this; `make cosim` does.
- TAPA rejects an `mmap` argument named `out`.

Every `mmap` argument is one AXI port; the U55C allows 32 kernel ports in one
FPGA image (`PROGRESS_REPORT.md` §3.4).
