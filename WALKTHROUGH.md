# Walkthrough: what we did, in plain language

This file explains the project step by step: what each piece is, why it exists,
where its code is, how to run it, and what its tests check.

- How to run everything, including your own prompt: [`README.md`](README.md).
- The formal course report, with all numbers: [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).
- Each folder also has a short `README.md` with that folder's exact commands.

---

## 0. The goal in one paragraph

We want to run GPT-2, a small text-generating neural network, on an FPGA card
(AMD Alveo U55C).
- The model gets a prompt and produces new words one at a time.
- It always picks the most likely next word; this is called *greedy decoding*.
- We only run the model (*inference*); we never train it.
- All numbers are 32-bit floats (FP32).
- Turning text into numbers and back (*tokenization*) runs on the normal CPU, not
  on the FPGA.

Before building hardware we need to know what the *correct* answer is. So the
project is a chain of steps, where each step is checked against the one before it:

```
 Step 1                Step 2                 Step 4                  Step 5
 Hugging Face  ──▶  our C++ CPU version  ──▶  FPGA kernels (HLS)  ──▶  whole model on
 (golden_hf.py)      (src/cpu)                 (kernels/*)             kernels (src/host)
 "the truth"        checked against Step 1    checked against Step 2   tokens checked
                                                                       against Step 1
                    Step 3: analysis (analysis/) — how fast could it be at best?
```

---

## 1. The model we use

We use the official pretrained **GPT-2 small** from OpenAI. On Hugging Face it is
called `gpt2`. llm.c has the same weights in its file `models/gpt2_124M.bin`.
We checked that the two are identical: the largest difference is exactly 0.

| property | value |
|---|---|
| name | GPT-2 small (Hugging Face `gpt2`, llm.c `gpt2_124M`) |
| parameters | **124,439,808** (OpenAI's 2019 paper said "117M"; that was a miscount of the same model) |
| layers (transformer blocks) | 12 |
| hidden size (numbers per token, "C") | 768 |
| attention heads | 12, each of size 64 (12 × 64 = 768) |
| MLP inner size | 3072 (= 4 × 768) |
| vocabulary | 50,257 tokens (llm.c pads it to 50,304 with zero rows for alignment) |
| max context | 1024 tokens |
| data type | **FP32** everywhere: the Hugging Face file stores all 160 tensors as float32, and we load and run it in float32 |
| file size | 548 MB (Hugging Face `model.safetensors`); 498 MB (llm.c `gpt2_124M.bin`) |
| activation function | GELU (tanh approximation) |
| normalization | LayerNorm *before* attention and *before* the MLP, plus a final LayerNorm |
| output layer | reuses the token-embedding matrix (no separate weights) |

Where the 124.4 M parameters are:

| part | parameters |
|---|---|
| token embedding `wte` (50,257 × 768), also the output layer | 38,597,376 |
| position embedding `wpe` (1024 × 768) | 786,432 |
| one transformer block | 7,087,872 |
| — LayerNorm 1 (γ, β) | 1,536 |
| — QKV projection (768 → 2304, + bias) | 1,771,776 |
| — attention output projection (768 → 768, + bias) | 590,592 |
| — LayerNorm 2 (γ, β) | 1,536 |
| — MLP up projection "fc" (768 → 3072, + bias) | 2,362,368 |
| — MLP down projection "fcproj" (3072 → 768, + bias) | 2,360,064 |
| 12 blocks | 85,054,464 |
| final LayerNorm | 1,536 |
| **total** | **124,439,808** |

### What happens to one token inside the model

Every new token goes through the same sequence. The names on the right are the
names used for files in `data/acts/` and throughout the code. They are worth
learning, because the tests print them.

```
token id ──▶ wte[token] + wpe[position]                        "embed"   (768 numbers)
for each of the 12 layers:
    LayerNorm 1                                                "ln1"     (768)
    multiply by QKV matrix  → query, key, value                "qkv"     (2304 = 3 × 768)
    save key and value into the KV cache (memory of past tokens)
    attention: compare query with all past keys, mix values    "atty"    (768)
    multiply by attention output matrix                        "attproj" (768)
    add to the block input (residual)                          "res1"    (768)
    LayerNorm 2                                                "ln2"     (768)
    multiply by fc matrix                                      "fch"     (3072)
    GELU                                                       "fch_gelu"(3072)
    multiply by fcproj matrix                                  "fcproj"  (768)
    add to res1 (residual)                                     "res2"    (768) → next layer
final LayerNorm                                                "lnf"     (768)
multiply by wte (the "LM head") → one score per vocab word     "logits"  (50,257)
argmax: the word with the highest score is the next token
```

The **KV cache** stores the key and value of every earlier token, so that each
new token only has to compute its own. Without it, the model would recompute the
whole prompt for every word.

---

## 2. Step 1: the golden reference (`golden/golden_hf.py`)

**Purpose:** run the real Hugging Face model once and save everything it
computes, as the "correct answers" for all later tests.

**Run it:**
```bash
pip install --user transformers torch numpy      # once
python3 golden/golden_hf.py                       # writes data/
```
The first run downloads `gpt2` from Hugging Face (548 MB, into
`~/.cache/huggingface`).

**What the script does, in order** (function names in brackets):

1. **Loads the model and tokenizer.** It uses
   `GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32, attn_implementation="eager")`
   in `.eval()` mode. "eager" uses the plain attention code, so every
   intermediate value can be observed.
2. **Checks that the llm.c weights match** (`check_llmc_weights`). It opens
   `models/gpt2_124M.bin` and compares every tensor with the Hugging Face weights.
   Hugging Face stores some matrices transposed, so the script transposes them
   before comparing. Result: max difference 0 (`data/weights_check.txt`). This
   matters because our C++ code reads the llm.c file, not the Hugging Face one.
3. **Generates text from a short prompt** (`greedy`). The prompt
   `"The FPGA is"` becomes 5 token IDs `[464, 376, 6968, 32, 318]`. Then:
   - **Prefill:** the 5 prompt tokens are processed together. This fills the KV
     cache and gives the first new token, `257` (" a").
   - **Decode:** the model is called 31 more times with one token each, reusing
     the KV cache, and takes the argmax every time.
   - Result: 32 new tokens. The full text is *"The FPGA is a joint venture between
     the FPGA and the National Science Foundation. The FPGA is a joint venture …"*.
     The text repeats; small greedy GPT-2 models often do this, and it is
     expected.
4. **Cross-checks its own loop** against Hugging Face's built-in
   `model.generate(do_sample=False)`. The token lists must be identical, or the
   script stops. This proves that our simple loop is the standard greedy
   decoding.
5. **Records every intermediate value of one decode step** (`capture_hooks`).
   - Which step: the *first decode step*. Token 257 enters at position 5 and the
     model predicts token 6466 (" joint").
   - Why this step: it is exactly what the FPGA does, one token at a time, using a
     KV cache that already holds 5 earlier tokens.
   - How: PyTorch "hooks" save the output of each sub-layer (`ln1`, `qkv`, …,
     `res2` for all 12 layers, then `lnf`), plus the KV cache of every layer
     (6 rows × 768) and all 50,257 logits.
   - The top 5 next words are in `data/decode0_top5.txt`: " joint" −108.32,
     " non" −108.52, … They are close, so this is a sensitive test.
6. **Runs a long prompt.** 992 tokens of fixed text, plus 32 new tokens, fills the
   maximum context of 1024. This tests that long-context attention is correct.
7. **Writes `data/manifest.txt`,** which lists every saved file with its type and
   shape.

**Output files (`data/`):**

| file | what it is |
|---|---|
| `prompt.txt`, `prompt_ids.i32` | the prompt and its 5 token IDs |
| `output_ids.i32`, `output.txt` | the 32 generated token IDs and the text |
| `decode0_info.txt` | the recorded step: input token 257, position 5, output 6466 |
| `acts/embed.f32`, `acts/lXX_<name>.f32`, `acts/lnf.f32` | intermediate values of that step (146 files) |
| `acts/lXX_k.f32`, `acts/lXX_v.f32` | KV cache of layer XX after that step, 6 × 768 |
| `decode0_logits.f32`, `decode0_top5.txt` | all 50,257 scores and the top 5 |
| `long_prompt_ids.i32`, `long_output_ids.i32` | 992-token prompt and its 32 generated tokens |
| `weights_check.txt` | llm.c vs Hugging Face weight difference (0) |
| `golden_loader.h` | small C++ helper that every test uses to read these files |

The files are raw arrays with no header. To look at one:
```python
import numpy as np
x = np.fromfile("data/acts/l00_ln1.f32", dtype=np.float32)   # 768 numbers
ids = np.fromfile("data/output_ids.i32", dtype=np.int32)
```
In C++: `golden::act(0, "ln1")`, `golden::i32("output_ids.i32")` (see `data/golden_loader.h`).

---

## 3. Step 2: our own C++ version on the CPU (`src/cpu`, `src/common`)

**Purpose:** a simple, readable C++ implementation that we fully control. It
becomes the reference for the hardware kernels. It reuses llm.c's functions,
adding a KV cache and single-token decoding.

**Files to read, in this order:**

| file | what is in it |
|---|---|
| `src/common/config.h` | the model sizes from §1 as constants |
| `src/common/weights.cpp` | loads `gpt2_124M.bin` into named pointers (`l_qkvw(layer)`, …) |
| `src/cpu/gpt2_cpu.cpp`, `forward()` (line 50) | **the whole model in ~50 lines**, in exactly the order shown in §1 |
| `src/cpu/ops.cpp` | the individual operations: layernorm, matmul, attention, gelu, residual, lm_head + argmax |
| `src/cpu/cpu_main.cpp` | the program: `generate`, `verify`, `profile` modes |
| `src/common/utils.h` | the error-measuring helpers (`compare`, `Tolerance`) |

**Run it:**
```bash
make -C src/cpu                     # build -> build/bin/cpu_main
build/bin/cpu_main generate         # prints the 32 tokens and the text
make -C src/cpu verify              # compare with Hugging Face (Step 1) = test level L1
make -C src/cpu profile             # timing at contexts 128 / 512 / 1024, 1 and 8 threads
```

**What `verify` checks:** it runs the same prompt, records the same intermediate
values, and compares each one with `data/acts/`.
- Two FP32 programs never agree exactly, because the order of additions differs.
- We therefore allow a small error: at most 1e-4 × (largest value) + 1e-4 × (value).
  That is about 0.01%.
- Result: all **147** tensors pass. The largest error relative to the tensor's
  size is 1.05e-5.
- **32/32** generated tokens are identical for the short prompt, and **32/32** for
  the long one.
- Output: `results/cpu_verify.csv`, `results/cpu_verify.log`.

**Speed** (`results/cpu_profile_summary.csv`): 26–37 tokens/s on 1 thread and
39–53 tokens/s on 8 threads, depending on context length.

---

## 4. Step 3: analysis: what limits the speed? (`analysis/analyze.py`)

```bash
python3 analysis/analyze.py         # writes results/analysis/ (tables + 2 plots)
```

The main finding, in plain words:
- For every new token the model must read **all its weights once**: about 500 MB
  in FP32.
- It does only about 0.25 billion arithmetic operations with them, roughly 0.5 per
  byte.
- The speed is therefore set by **how fast memory can be read**, not by how fast
  the chip can compute.

The U55C's HBM memory can deliver up to 460 GB/s, which puts a hard ceiling of
about 800–900 tokens/s on FP32. The FPGA design therefore aims to read weights
from many memory channels at once.

Details and plots: `results/analysis/analysis.md`, `roofline.png`,
`cpu_breakdown.png`.

---

## 5. Step 4: hardware kernels (`kernels/`)

A *kernel* is one hardware block that does one operation. Each is written in C++
with HLS (High-Level Synthesis) using the TAPA framework, as in the labs.

| kernel | does | notes |
|---|---|---|
| `gemv` | vector × matrix (the 4 weight multiplies per layer) | the most important one: 8 memory ports, 128 multiply-adds per clock cycle |
| `layernorm` | LayerNorm | |
| `gelu` | GELU | |
| `residual` | adds two vectors | normally merged into `gemv` |
| `attention` | attention, including writing the KV cache | 12 heads in parallel; reads the cache once |
| `lm_head` | final multiply by wte + argmax | outputs only the winning token ID |

**Every kernel folder has the same 4 files:**

| file | what it is |
|---|---|
| `kernel.h` | the kernel's interface (arguments) and its size parameters |
| `kernel.cpp` | the hardware design |
| `tb.cpp` | the testbench: runs the kernel and compares it with the CPU version from Step 2 |
| `Makefile` | build commands (shared rules in `kernels/common.mk`) |

Nothing else lives in a kernel folder. Everything generated (the compiled
testbench, the synthesized `.xo`, HLS reports and logs) goes to
`build/kernels/<name>/`.

**Run one kernel** (example: `gemv`):
```bash
cd kernels/gemv
make swsim    # software simulation: run the testbench, seconds. Checks correctness. (= test level L2)
make hls      # synthesis: turn C++ into hardware, minutes to about an hour. Gives size + speed estimates.
              #   output: build/kernels/gemv/gemv.xo, reports in build/kernels/gemv/work.out/report/
make cosim    # simulate the generated hardware (RTL), hours for gemv. Checks the real circuit.
```

**What a kernel test checks** (each line of output is one case):
- It feeds the kernel **real GPT-2 data** from `data/` (for example the actual
  layer-0 input) and the real weights.
- It compares the result with the C++ CPU function, using the same 0.01%
  tolerance.
- It also prints the error against Hugging Face, for information.
- It adds **edge cases**: smallest sizes, largest sizes, position 0 and position
  1023 for attention, ties in argmax, the last row of the vocabulary, and large
  values.

Example line:
```
l11_fc   n=3072   max_abs=1.907e-06 max_rel=2.254e-06 ...   PASS
```
This means: layer 11's fc output has 3072 numbers, the largest absolute
difference is 1.9e-6, and it passes.

Synthesis results for all kernels (size, clock speed, cycles) are collected by
`python3 analysis/collect_synth.py` into `results/synthesis.md`.

---

## 6. Step 5: the whole model using the kernels (`src/host`)

`src/host/fpga_model.cpp` does the same thing as `forward()` in Step 2, but calls
the kernels instead of the CPU functions.
- `FpgaModel::block()` (line 148) runs one layer.
- `FpgaModel::step()` (line 183) handles one token.
- `host_main.cpp` is the program around them.

```bash
make -C src/host                                           # build -> build/bin/host_main
build/bin/host_main --n=32 --check                         # all ops on kernels; compares tokens with Hugging Face
build/bin/host_main --n=8 --check --cpu_ops=attention      # put some ops back on the CPU (for debugging)
build/bin/host_main --plan                                 # print which HBM channel holds what
```

There is no FPGA board here. The kernels therefore run in **software simulation**,
which takes about 30 minutes for 32 tokens. This shows only that the result is
correct; it says nothing about speed. On a real board the same program would run
with `--bitstream=<file>.xclbin`.

---

## 7. All tests in one command (`tests/run_all_tests`)

All verification is run from `tests/`. It is organized as four **levels**, and
each level is checked against the one before it. If something breaks, the
lowest failing level tells you where the problem is.

| level | question | compared against |
|---|---|---|
| **L1** | Does our CPU C++ match Hugging Face? (Step 2 vs Step 1) | Hugging Face values in `data/` |
| **L2** | Does each kernel, alone, match the CPU version? (Step 4 vs Step 2) | `src/cpu/ops.cpp` |
| **L3** | Does one full layer built from kernels match the CPU? | CPU model, same input |
| **L4** | Does the whole model on kernels produce the same tokens as Hugging Face? | `data/output_ids.i32` |

```bash
tests/run_all_tests              # all levels, ~45 min
tests/run_all_tests --quick      # skips the 32-token run, ~16 min
tests/run_all_tests --level 2    # only one level
```
It writes `results/test_summary.md` (one line per test) and a log for each test
in `results/test_logs/`. `tests/README.md` lists where the code of every test is.
Last full run: 2026-11-07, 12 passed, 0 failed, 46 min.

| test name | what it runs | last result |
|---|---|---|
| `L1_cpu_vs_hf` | CPU model, 147 tensors + generated tokens vs Hugging Face | PASS |
| `L2_kernel_gemv` … `L2_kernel_lm_head` (6 tests) | each kernel's `tb.cpp` | all PASS (35 cases) |
| `L3_block_fused` | layers 0 and 11 built from kernels | PASS 20/20 |
| `L3_block_unfused_residual` | same, with the residual done by its own kernel | PASS 20/20 |
| `L4_e2e_8tok_cpu_fallback` | 8 tokens; attention and GELU on CPU, the rest on kernels | PASS 8/8 tokens |
| `L4_e2e_8tok_unfused` | 8 tokens, all on kernels, separate residual kernel | PASS 8/8 tokens |
| `L4_e2e_32tok_all_kernels` | **32 tokens, everything on kernels** | **PASS 32/32 tokens** |

Separately, the GEMV hardware circuit was simulated at RTL level (`make cosim`),
which took 2 h 14 min: 6/6 PASS (`results/cosim_gemv.log`).

---

## 8. Trying your own prompt (`run_prompt.py`)

The tests above use the fixed prompt "The FPGA is". To check any other text:
```bash
python3 run_prompt.py "Yale University is located in" --n 20          # Hugging Face vs CPU
python3 run_prompt.py "Hello world" --n 3 --fpga                       # also FPGA kernels (slow)
```
The script follows the same chain as above, for your text:
1. Tokenize on the host.
2. Hugging Face greedy decoding (the same `greedy()` function as `golden_hf.py`).
3. `build/bin/cpu_main generate` with the same token IDs.
4. With `--fpga`: `build/bin/host_main` (kernels in software simulation).

It then checks that all of them produce exactly the same token IDs. Logs go to
`results/my_prompt/`.

---

## 9. Where the numbers are

| question | file |
|---|---|
| Did everything pass? | `results/test_summary.md` |
| CPU vs Hugging Face errors, per tensor | `results/cpu_verify.csv` |
| CPU speed | `results/cpu_profile_summary.csv`, `results/cpu_profile_ops.csv` |
| Theoretical limits, plots | `results/analysis/analysis.md` |
| Hardware size and cycle counts | `results/synthesis.md` |
| Estimated FPGA speed (an **estimate**, not measured) | `results/fpga_estimate.csv` (from `analysis/fpga_estimate.py`) |
| The full report | `PROGRESS_REPORT.md` |

---

## 10. Status

**Done and checked**
- Hugging Face reference.
- CPU version, matching Hugging Face.
- Speed analysis.
- All 6 kernels written, tested in simulation and synthesized.
- The whole model running on the kernels in simulation produces exactly the same
  32 tokens as Hugging Face.
- GEMV checked at the RTL level.

**Not done yet**
- Building one complete FPGA image (`v++` link).
- Running on a real board; none is available here.
- Hardware emulation.
- Faster prompt processing.
- Lower precision (FP16/INT8, a later stage).

**Estimated FPGA speed** (from synthesis reports, **not measured**): about
100–280 tokens/s, versus 39–53 tokens/s measured on the 8-thread CPU. The risks
and next steps are in `PROGRESS_REPORT.md` §6.
