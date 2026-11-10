# analysis/

```bash
python3 analysis/analyze.py         # -> results/analysis/{analysis.md, *.csv, roofline.png, cpu_breakdown.png}
python3 analysis/collect_synth.py   # -> results/synthesis.{csv,md} (after `make hls` in kernels/*)
```
`analyze.py` derives per-operator FLOPs, bytes (FP32/FP16/INT8), operational
intensity, KV-cache size, bandwidth-bound tokens/s and on-chip fit from the
model dimensions; assumptions are listed at the top of the script and in
`analysis.md`. Measured CPU numbers are read from `results/cpu_profile_*.csv`.
