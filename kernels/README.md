# kernels/ — HLS kernels (TAPA 0.1.20250815, Vitis HLS 2024.2, U55C, 3.33 ns)

| kernel | top | function |
|---|---|---|
| `gemv/` | `GemvKernel` | y = W x + b (+ residual); 8 HBM ports x 16 FP32 MAC/cycle, 16 interleaved rows/PE, II=1 |
| `layernorm/` | `LayerNormKernel` | LayerNorm, 16 lanes, three II=1 passes |
| `gelu/` | `GeluKernel` | tanh-GELU as x / (1 + exp(-2u)), 16 lanes |
| `residual/` | `ResidualKernel` | y = a + b, 16 lanes |
| `attention/` | `AttentionKernel` | KV-cache append + 12 heads in parallel, online softmax |
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
GEMV / LM-head port counts can be changed with `-DGEMV_PORTS=N` / `-DLM_PORTS=N`
(add to `KERNEL_DEFS` in the Makefile).

Two naming pitfalls, both found the hard way:
- Do not name a kernel or task argument `pos` or `port`: Vitis HLS renames such
  ports (`pos_r`, `port_r`) and TAPA's generated RTL no longer connects. Software
  simulation does not show this; `make cosim` does.
- TAPA rejects an `mmap` argument named `out`.

Every `mmap` argument is one AXI port; the U55C allows 32 kernel ports in one
FPGA image (`PROGRESS_REPORT.md` §3.4).
