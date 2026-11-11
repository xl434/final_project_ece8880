# analysis/ — performance analysis scripts

The scripts follow the reasoning of `PROGRESS_REPORT.md` §1, §2 and §5: count the
work, measure the CPU, find the bottleneck, estimate the FPGA.

| script | what it does | output |
|---|---|---|
| `analyze.py` | from the model sizes alone: FLOPs and bytes per operator (decode and prefill), operational intensity, KV-cache size, bandwidth-limited tokens/s, on-chip fit; places the measured CPU numbers next to them | `results/analysis/` (`analysis.md`, CSVs, `cpu_breakdown.png`) |
| `collect_synth.py` | gathers the HLS synthesis reports of all kernels (after `make hls`) | `results/synthesis.md`, `results/synthesis.csv` |
| `fpga_estimate.py` | **estimate** of FPGA decode time (per token, per operator) and prefill time (per prompt, token by token vs one pass), from a cycle model of each kernel call | `results/fpga_estimate.csv`, `results/fpga_estimate_ops.csv`, `results/fpga_prefill_estimate.csv` |
| `figures.py` | draws the report charts from all of the above plus the CPU measurements | `results/figures/*.png` |

```bash
make -C src/cpu profile peaks        # CPU measurements the scripts read (results/cpu_*.csv)
python3 analysis/analyze.py
python3 analysis/collect_synth.py    # after `make hls` in kernels/*
python3 analysis/fpga_estimate.py
python3 analysis/figures.py
```
Assumptions (how FLOPs and bytes are counted, which clock and bandwidth are used)
are listed at the top of each script and in `results/analysis/analysis.md`.
