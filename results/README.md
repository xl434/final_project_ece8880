# results/

The progress report built from these files is `../PROGRESS_REPORT.md`; the plain-language guide is `../WALKTHROUGH.md`.

| file | produced by |
|---|---|
| `cpu_verify.csv`, `cpu_verify.log` | `build/bin/cpu_main verify` (`make -C src/cpu verify`) |
| `cpu_profile_summary.csv`, `cpu_profile_ops.csv`, `cpu_env.txt`, `cpu_profile.log` | `build/bin/cpu_main profile` (`make -C src/cpu profile`) |
| `analysis/` | `analysis/analyze.py` (tables, CSVs, roofline and CPU-breakdown plots) |
| `synthesis.csv`, `synthesis.md` | `analysis/collect_synth.py` (after `make hls` in each kernel) |
| `test_summary.md`, `test_logs/` | `tests/run_all_tests` |
| `cosim_gemv.log` | `make -C kernels/gemv cosim` (TAPA fast RTL cosimulation, ~2.2 h) |
| `fpga_estimate.csv`, `fpga_estimate_ops.csv` | `analysis/fpga_estimate.py` (ESTIMATE of decode time from HLS cycles, per token / per operator) |
| `cpu_peaks.csv` | `make -C src/cpu peaks` (measured CPU memory bandwidth and FP32 peak) |
| `figures/` | `analysis/figures.py` (charts used in `PROGRESS_REPORT.md`) |
| `run_all_tests.log` | console output of the last full `tests/run_all_tests` |
| `my_prompt/` | `run_prompt.py` (your prompt: token IDs, Hugging Face output IDs, CPU / FPGA-simulation logs) |
