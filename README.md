# GPT-2 (124M) inference on AMD Alveo U55C — progress stage

This project runs the pretrained OpenAI GPT-2 small model (124M parameters) on an
FPGA. Settings: FP32, batch 1, greedy decoding with a KV cache, inference only.
Text ↔ token conversion runs on the host CPU.

## Which document to read

| file | what it is | read it when |
|---|---|---|
| `README.md` (this file) | **how to run things**, plus a map of the folders | you want to run something |
| [`WALKTHROUGH.md`](WALKTHROUGH.md) | **plain-language explanation** of what was done, step by step: the model, what each script does, where the code is, what each test checks | you want to understand the project |
| [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) | **the formal course report** (the 6 required sections), with all measured numbers, the design, and risks | you need the numbers or the deliverable |
| [`PLAN.md`](PLAN.md) | **current stage and plan**: what kind of design the baseline is, what to do now vs later, and a storyline for the presentation | you are preparing the presentation or the next stage |

Each folder also has a short `README.md` with the exact commands for that folder.

---

## 1. One-time setup

```bash
# Python packages (for the Hugging Face reference and the scripts)
pip install --user torch transformers numpy matplotlib pyyaml

# GPT-2 weights + tokenizer in llm.c format (2 files, ~500 MB)
mkdir -p models && for f in gpt2_124M.bin gpt2_tokenizer.bin; do
  curl -sL -o models/$f "https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/$f?download=true"; done

# Golden reference data -> data/ (also downloads Hugging Face `gpt2` on first run)
python3 golden/golden_hf.py
```

The CPU code needs only `g++` with OpenMP. The FPGA kernels and the host program
also need TAPA and Vitis HLS 2024.2. Their paths are set in `kernels/common.mk`
and match the course lab machines.

---

## 2. Try it with your own prompt

```bash
python3 run_prompt.py "Yale University is located in" --n 20
```

What it does:
1. Turns your text into GPT-2 token IDs on the host. They are saved in
   `results/my_prompt/prompt_ids.i32`.
2. Runs **Hugging Face `gpt2`** to get the reference answer.
3. Runs **our C++ CPU model** on the same token IDs and checks that every
   generated token is identical.

Output (this run is real; your prompt will give different text):
```
token IDs (6): [56, 1000, 2059, 318, 5140, 287]
[Hugging Face] ' the heart of the city of New York City. It is the largest university in the United States and'
[CPU C++] ' the heart of the city of New York City. It is the largest university in the United States and'
[CPU C++] token IDs vs Hugging Face: 20/20 match -> PASS
```
(GPT-2 small often gets facts wrong. What we test is that our implementation
gives *the same* answer as Hugging Face, not that the answer is true.)

To also run the **FPGA kernels**, add `--fpga`:
```bash
python3 run_prompt.py "Hello world" --n 3 --fpga
```
There is no FPGA board here, so the kernels run in software simulation. This is
slow: about 1 minute per token, counting prompt tokens too. Keep the prompt and
`--n` small.

Logs go to `results/my_prompt/` (`cpu.log`, `fpga_swsim.log`).

You can also call the programs directly with a token-ID file:
`build/bin/cpu_main generate --prompt FILE.i32 --n 20`, or
`build/bin/host_main --prompt=FILE.i32 --n=5`.

---

## 3. Run the tests

Verification has four levels. Each is checked against the one before it:

| level | checks | against |
|---|---|---|
| L1 | CPU C++ model | Hugging Face |
| L2 | each FPGA kernel alone | CPU C++ operators |
| L3 | one full layer built from kernels | CPU C++ model |
| L4 | whole model on kernels (generated tokens) | Hugging Face |

```bash
tests/run_all_tests              # all levels, ~45 min
tests/run_all_tests --quick      # skip the 32-token run, ~16 min
tests/run_all_tests --level 2    # one level only (1-4)
```
It prints PASS/FAIL per test. It writes `results/test_summary.md` and one log per
test in `results/test_logs/`. Details of every test: `tests/README.md`.

---

## 4. Reproduce each step

| step | command | time | output |
|---|---|---|---|
| 1. golden reference | `python3 golden/golden_hf.py` | a few min at most | `data/` |
| 2a. CPU model vs Hugging Face | `make -C src/cpu verify` | ~15 s | `results/cpu_verify.csv` |
| 2b. CPU speed | `make -C src/cpu profile` | ~3 min | `results/cpu_profile_*.csv` |
| 2c. CPU limits (memory bandwidth, compute peak) | `make -C src/cpu peaks` | ~1 min | `results/cpu_peaks.csv` |
| 3. application analysis | `python3 analysis/analyze.py` | seconds | `results/analysis/` |
| 4a. one kernel, functional test | `make -C kernels/gemv swsim` (any of: gemv layernorm gelu residual attention lm_head) | seconds | printed PASS/FAIL |
| 4b. one kernel, synthesis | `make -C kernels/gemv hls` | minutes to ~1 h | `build/kernels/gemv/` (`gemv.xo`, `work.out/report/`, `hls.log`) |
| 4c. collect synthesis results | `python3 analysis/collect_synth.py` | seconds | `results/synthesis.md`, `.csv` |
| 4d. one kernel, RTL co-simulation | `make -C kernels/gemv cosim` (needs 4b first) | ~2 h for gemv | printed PASS/FAIL |
| 4e. one kernel, Vitis hardware emulation (U55C shell + HBM model) | `make -C kernels/gemv hw_emu` (needs 4b first) | link ~30+ min, then the run | `build/kernels/gemv/hw_emu_run.log` |
| 5. FPGA speed estimate (from synthesis; **not measured**) | `python3 analysis/fpga_estimate.py` | seconds | `results/fpga_estimate*.csv` |
| 5b. report charts | `python3 analysis/figures.py` | seconds | `results/figures/` |
| 6. whole model on kernels | `make -C src/host swsim` | ~30 min | printed tokens, PASS/FAIL |
| 7. full hardware build, all six kernels (place-and-route, no board needed) | `make -C src/host xclbin_hw` (needs 4b for all kernels) | hours | `build/hw/` (xclbin, `reports/`, `vpp_link.log`) |

---

## 5. Folder map

| folder | contents |
|---|---|
| `run_prompt.py` | run your own prompt through Hugging Face, CPU and (optionally) FPGA kernels |
| `golden/` | Hugging Face reference script (writes `data/`) |
| `data/` | golden token IDs, intermediate values, KV cache, logits; `golden_loader.h` reads them from C++ |
| `models/` | `gpt2_124M.bin`, `gpt2_tokenizer.bin` (downloaded, see setup) |
| `src/common/` | model sizes, weight loader, token → text decoder, comparison helpers |
| `src/cpu/` | C++ CPU model + `cpu_main` (generate / verify / profile) |
| `src/host/` | FPGA host program: runs the model by calling the kernels |
| `kernels/<name>/` | HLS kernels, 4 files each: `kernel.h`, `kernel.cpp` (design), `tb.cpp` (testbench), `Makefile` |
| `tests/` | **all verification**: `run_all_tests` (levels L1–L4), transformer-block test; `tests/README.md` maps every test |
| `analysis/` | `analyze.py` (work per operator, limits), `collect_synth.py`, `fpga_estimate.py`, `figures.py` (charts) |
| `results/` | results the scripts produce (CSVs, plots, logs): what goes in a report |
| `build/` | **all compiled files**: `bin/` (cpu_main, host_main, block_test), `kernels/<name>/` (testbench, `.xo`, HLS reports), `obj/`. Safe to delete; rebuilt by `make` |
| `third_party/` | llm.c, nanoGPT, TinyGPT (reference code only) |
