# Walkthrough: what we did, in plain language

This file explains the project step by step: what each piece is, why it exists,
where its code is, how to run it, and what its tests check. It explains ideas
and code; it does not report results.

- How to run everything, including your own prompt: [`README.md`](README.md).
- Results, analysis and the reasons for the next steps: [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).
- What is done and what is next: [`PLAN.md`](PLAN.md).
- Terms such as *kernel*, *PE*, *II* or *co-simulation*: [glossary](#glossary) at the end.
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

### "Decoder-only" vs "decode": two different words

These sound alike but mean different things:

- **Decoder-only** describes GPT-2's *architecture*. The original transformer had
  two halves: an *encoder* that reads an input sentence, and a *decoder* that
  writes an output. GPT-2 keeps only the decoder half. It reads and writes in one
  stream of tokens, and each token may only look at earlier tokens (*causal*
  attention). Every GPT-style model is decoder-only.
- **Prefill and decode** are two *phases* of running that one network to
  generate text:

| | **prefill** | **decode** |
|---|---|---|
| when | once, at the start | once per new token, repeated |
| input | the whole prompt (e.g. 5 tokens) at once | just the last generated token |
| output | the KV cache for every prompt token + the first new token | one more KV-cache row + the next token |
| math | matrix × matrix: each weight is used for every prompt token | matrix × vector: each weight is used once |
| what limits speed | arithmetic (lots of math per weight read) | memory (must read all 500 MB of weights for very little math) |

Example with the prompt "The FPGA is":
1. **Prefill** reads all 5 prompt tokens and predicts " a".
2. **Decode step 1** reads " a" and predicts " joint".
3. **Decode step 2** reads " joint" and predicts " venture". This continues, one
   token per step.

**This project accelerates decode.** It is the phase that repeats for every
generated word, and it is limited by memory bandwidth, which is where the FPGA's
fast HBM memory helps. The FPGA kernels therefore all work on one token at a time.
The prompt is currently fed through them token by token too: correct, but not yet
optimized. A faster matrix × matrix prefill is a later step.

---

## 2. Step 1: the golden reference (`golden/golden_hf.py`)

**Purpose:** run the real Hugging Face model once on a fixed prompt and save
what it computes, as the "correct answers" for all later tests.

> **What is in `data/`?** *Not the weights.* The weights (the model's 124M
> learned numbers) are in `models/gpt2_124M.bin` and never change. `data/` holds
> the **results** of running the model on one prompt:
> - which tokens it generated;
> - the intermediate values inside every layer for one decode step;
> - the KV cache and the final scores.
>
> Think of `models/` as the program and `data/` as a recorded run of that program
> with known-correct outputs. Every test feeds the same input into our code and
> compares against this recording. It is small (1.5 MB) and can be regenerated
> at any time with `golden_hf.py`.

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

### How the Hugging Face model works (it is not a black box)

`transformers` is an open-source Python library. GPT-2 in it is one readable
file:
```
~/.local/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py
```
(We use transformers 5.22.2 on PyTorch 2.14.1.)

A pretrained model is **two things**:
- **The weights:** `from_pretrained("gpt2")` downloads `config.json` (the sizes
  from §1) and `model.safetensors` (the 124M trained numbers, from OpenAI).
- **The code:** the classes in `modeling_gpt2.py`, which are ordinary PyTorch
  (`matmul`, `softmax`, `LayerNorm`). `from_pretrained` builds these classes and
  fills them with the weights.

Where each part of §1 is in that file, and where our version of it is:

| model part | in `modeling_gpt2.py` | ours |
|---|---|---|
| the whole stack: embeddings → 12 blocks → `ln_f` | `GPT2Model` | `src/cpu/gpt2_cpu.cpp` `forward()` |
| output scores (LM head) | `GPT2LMHeadModel` | `lm_head` kernel |
| one block: `x + attn(ln_1(x))`, then `x + mlp(ln_2(x))` | `GPT2Block.forward` (~40 lines; **start reading here**) | block loop + `residual` |
| Q, K, V: one multiply `c_attn`, then split into 3 | `GPT2Attention.forward` | QKV GEMV |
| attention: softmax(Q·Kᵀ·scale + mask)·V | `eager_attention_forward` | `attention` kernel |
| MLP: `c_fc` → GELU → `c_proj` | `GPT2MLP` | GEMV, `gelu`, GEMV |
| `Conv1D`: despite the name, just `x @ W + b` | `transformers/pytorch_utils.py` | `gemv` kernel |
| GELU, tanh version (`"gelu_new"`) | `NewGELUActivation` in `transformers/activations.py` | `gelu` kernel |

One detail: GPT-2's `Conv1D` stores a weight as (input × output), the transpose
of PyTorch's usual `nn.Linear`. This is why `check_llmc_weights` transposes some
matrices before comparing.

**How `golden_hf.py` sees inside it:** PyTorch *forward hooks*. A hook is a small
function attached to a module, which PyTorch calls with the module's input and
output every time it runs. For example, `blk.ln_1.register_forward_hook(...)`
saves the output of LayerNorm 1 in every layer (`golden_hf.py` lines 125–138).
That is how `data/acts/` gets each intermediate value.

**Look at it yourself:**
```bash
# the module tree: layer names and weight shapes
python3 -c "from transformers import GPT2LMHeadModel as M; print(M.from_pretrained('gpt2'))"
# the implementation
less ~/.local/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py
```
The Hugging Face file also handles training, cross-attention and other
variants, which makes it longer. Shorter versions of the same model are
`third_party/nanogpt/model.py` (PyTorch, ~300 lines) and
`third_party/llm.c/train_gpt2.c` (plain C, which our CPU version is based on).

---

## 3. Step 2: our own C++ version on the CPU (`src/cpu`, `src/common`)

**Purpose:** a simple, readable C++ implementation that we fully control. It
becomes the reference for the hardware kernels. It reuses llm.c's functions,
adding a KV cache and single-token decoding.

### Why a CPU version, when Hugging Face already has GPT-2?

Hugging Face tells us **what** the right answer is. The hardware still needs
something it can be compared against **operation by operation**, under the same
conditions as the hardware. Hugging Face cannot do that, for five reasons:

1. **It runs in Python/PyTorch, and the kernel testbenches are C++.** Each
   kernel test (`kernels/*/tb.cpp`) calls the matching CPU function from
   `src/cpu/ops.h` directly, on the same input, and compares. This needs no
   Python and no file exchange.
2. **It only gives answers for the inputs we recorded.** `data/` holds one
   prompt and one decode step. Kernel tests also need random inputs, edge cases
   (very large values, the full context of 1024, two tied top scores for argmax)
   and other positions in the context. The C++ functions compute a reference for *any* input.
3. **The operations must be split the same way.** A kernel does one piece, such
   as "multiply this 768-vector by this 768×2304 matrix". In C++ each piece is
   one function with the same inputs, outputs and memory layout (llm.c's
   weight file, our KV-cache layout). Inside PyTorch, these steps are hidden
   behind library calls with different layouts (for example the transposed
   `Conv1D` weights).
4. **The host program uses it as a fallback.** `src/host/fpga_model.cpp` can run
   any operation on the CPU instead of the FPGA. That is how we tested the
   kernels one at a time inside the full model (test L4, "cpu_fallback").
5. **It is the baseline for speed.** The analysis (§4) needs a CPU number that
   does the same work as the FPGA: batch 1, FP32, single-token decode, C++.
   Timing PyTorch would mostly measure Python and framework overhead, which
   would make the CPU look slower than it really is and the FPGA speedup look
   bigger than it really is.

So the chain is: Hugging Face proves the C++ version is right (test L1, 147
values, all within 0.01%). After that, the C++ version is the trusted reference
for everything below it.

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
- It also checks that the generated tokens are identical, for the short prompt
  and for the 992-token prompt.
- Output: `results/cpu_verify.csv`, `results/cpu_verify.log`. Results:
  `PROGRESS_REPORT.md` §4.4.

**Speed:** `profile` times decode at several context lengths, with 1 and 8
threads. The machine has 4 physical cores with 2 hyper-threads each, so
"8 threads" means 4 cores. Results: `PROGRESS_REPORT.md` §2.

---

## 4. Step 3: analysis: what limits the speed?

The analysis answers one question in four steps: *where does the time go, and can
an FPGA do better?*

**Step A: count the work** (`python3 analysis/analyze.py`). From the model sizes
alone, it computes for every operator how many arithmetic operations (FLOPs) it
does and how many bytes it must read.
- Result: one generated token reads **~500 MB** (all the weights) but does only
  **~0.25 GFLOP** of math. That is 0.5 operations per byte.
- Chart: `results/figures/work_per_op.png`.

**Step B: measure the CPU** (`make -C src/cpu profile`). Time every operator
while generating tokens.
- Result: the weight multiplications take over 90% of the time at short context.
  Attention grows with longer context.
- Chart: `results/analysis/cpu_breakdown.png`.

**Step C: find the bottleneck** (`make -C src/cpu peaks`). First measure what this
CPU can do at best: read memory at 23–34 GB/s, and compute at 98–391 GFLOP/s.
Then compare each operator's speed with those limits.
- Result: the weight multiplications read memory at **80–89% of the maximum**,
  but use **under 10% of the arithmetic**. So the CPU is waiting on memory. This
  is called *memory-bound*.
- Chart: `results/figures/cpu_bandwidth_per_op.png`.

**Step D: why an FPGA can help.** For a memory-bound program, speed ≈ memory
bandwidth ÷ bytes per token.
- The U55C's HBM memory provides 460 GB/s, about **13× the CPU**. That gives a
  ceiling of about 800 tokens/s, versus about 60 on the CPU.
- The FPGA design therefore focuses on reading weights from many memory channels
  at once, not on adding more arithmetic.
- Chart: `results/figures/roofline.png`.

`python3 analysis/figures.py` redraws all charts from the result files. The full
reasoning, with charts, is in `PROGRESS_REPORT.md` §1–2.

---

## 5. Step 4: hardware kernels (`kernels/`)

A *kernel* is one hardware block that does one operation. Each is written in C++
with HLS (High-Level Synthesis) using the TAPA framework, as in the labs.

| kernel | does | input → output | kind of math | notes |
|---|---|---|---|---|
| `gemv` | the 4 weight multiplications per layer, `y = W x + b` | 768 or 3072 → 768, 2304 or 3072 | matrix × vector (each output is a sum) | the most important one: 8 memory ports, 128 multiply-adds per clock cycle |
| `layernorm` | LayerNorm | 768 → 768 | two sums (mean, variance), then elementwise | |
| `gelu` | GELU | 3072 → 3072 | elementwise | |
| `residual` | adds two vectors | 768 + 768 → 768 | elementwise | normally merged into `gemv` |
| `attention` | attention, including writing the KV cache | 2304 + cache → 768 | sums over all past tokens, per head | 12 heads in parallel; reads the cache once |
| `lm_head` | final multiply by wte + argmax | 768 → 1 token ID | matrix × vector, then max | outputs only the winning token ID and its score, not all 50,257 scores |

**How a kernel is built.** All kernels follow the same recipe; the details,
diagrams and synthesis numbers are in `PROGRESS_REPORT.md` §3 and §5.1.
- **Wide words.** Data moves in 512-bit words, 16 floats at a time, the width of
  one HBM port. Each kernel handles all 16 in parallel ("16 lanes").
- **Tasks connected by streams.** A kernel is split into small *tasks*, typically
  read → compute → write, connected by FIFOs. The tasks run at the same time, like
  an assembly line (TAPA *dataflow*).
- **One word per clock cycle.** Every streaming loop is *pipelined* so it takes a
  new word every cycle (*II = 1*). Since GPT-2 decode is limited by memory, the
  goal is to keep the memory ports busy, not to add arithmetic.
- **PEs.** The GEMV and LM head have 8 *processing elements* (PEs), one per HBM
  port. Each PE takes one word of weights per cycle and does 16 multiply-adds.

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
make cosim    # RTL co-simulation: simulate the generated circuit. Minutes to hours.
make hw_emu   # Vitis hardware emulation: the circuit inside the U55C platform. Hours.
```

**Four ways to run the same testbench**, each closer to real hardware and slower:

| mode | what runs | what it proves |
|---|---|---|
| software simulation (`swsim`) | the kernel's C++ on the CPU | the algorithm is right |
| RTL co-simulation (`cosim`, the labs' "hardware emulation") | the circuit produced by synthesis, cycle by cycle | the generated hardware is right |
| Vitis hardware emulation (`hw_emu`) | the circuit linked into the U55C platform, with its memory placement and an HBM model | it also works inside the real platform |
| board (`--bitstream=<file>.xclbin`) | the real FPGA | real speed |

Software simulation can miss hardware problems. For example, co-simulation found
that HLS renames ports called `pos` or `port`, which broke the generated circuit
even though the C++ was correct (`PROGRESS_REPORT.md` §4.4).

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

**What "host" means.** The FPGA card sits in a normal computer (the *host*) and
cannot run on its own. A CPU program, the *host program*, loads the weights into
the card's HBM memory, starts each kernel in the right order, and turns the final
token IDs back into text:

```
 CPU (host)                                     FPGA card
 load weights, copy to HBM      ── PCIe ──►     HBM: weights, KV cache, vectors
 tokenize the prompt
 for each token:
   start kernel 1, wait         ── start ──►    kernels read HBM, compute,
   start kernel 2, wait         ◄── done ───    write their result to HBM
   ... (98 kernel calls per token)
 read the token ID, detokenize
```

In this first design (the *baseline*) the kernels run one after another, and each
kernel's result goes through HBM to the next one. Inside a kernel the work is
streamed (dataflow); between kernels it is sequential (*temporal*). Why this is
the starting point, and why the next design changes it: `PROGRESS_REPORT.md`
§3.5 and §6.

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

Without a bitstream, the kernels run in **software simulation**, which takes
about 30 minutes for 32 tokens. This shows only that the result is correct; it
says nothing about speed. The current host uses `tapa::invoke`, which is fine for
simulation but copies buffers on every call and handles one kernel per FPGA
image. A board run of the whole model needs a host written with XRT, AMD's
runtime library (listed in `PLAN.md`).

---

## 7. Step 6: building a real FPGA image (`make -C src/host xclbin_hw`)

Synthesis (`make hls`) turns each kernel into a circuit description (`.xo`) and
*estimates* its size and speed. To get something that runs on the card, all
kernels are combined into one *FPGA image* (`.xclbin`):
1. **Link** (`v++ --link`): the kernels are placed into the U55C *platform* (the
   fixed "shell" that talks to PCIe and HBM), and each kernel memory port is
   connected to an HBM channel as listed in `src/host/link_config.ini`.
2. **Implementation** (Vivado): synthesis to gates, placement on the chip,
   routing of the wires, and timing checks. This gives the real area and clock.
   It takes many hours.

No board is needed for this step. Two limits showed up here and are explained in
`PROGRESS_REPORT.md` §3.4 and §5.4: the U55C allows only 32 kernel memory ports,
and Vivado needs a small workaround on this workspace (`src/host/vivado/`, taken
from the ECE 8880 Lab 4 flow).

---

## 8. All tests in one command (`tests/run_all_tests`)

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

| test name | what it runs |
|---|---|
| `L1_cpu_vs_hf` | CPU model, 147 tensors + generated tokens vs Hugging Face |
| `L2_kernel_gemv` … `L2_kernel_lm_head` (6 tests) | each kernel's `tb.cpp` |
| `L3_block_fused` | layers 0 and 11 built from kernels |
| `L3_block_unfused_residual` | same, with the residual done by its own kernel |
| `L4_e2e_8tok_cpu_fallback` | 8 tokens; attention and GELU on CPU, the rest on kernels |
| `L4_e2e_8tok_unfused` | 8 tokens, all on kernels, separate residual kernel |
| `L4_e2e_32tok_all_kernels` | **32 tokens, everything on kernels** |

All kernels in these tests run in software simulation. RTL co-simulation and
hardware emulation (§5) run the L2 testbenches separately, because they take
much longer. Results of everything: `PROGRESS_REPORT.md` §4.4.

---

## 9. Trying your own prompt (`run_prompt.py`)

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

## 10. Where the numbers are

| question | file |
|---|---|
| Did everything pass? | `results/test_summary.md` |
| CPU vs Hugging Face errors, per tensor | `results/cpu_verify.csv` |
| CPU speed | `results/cpu_profile_summary.csv`, `results/cpu_profile_ops.csv` |
| Charts used in the report | `results/figures/` (drawn by `analysis/figures.py`) |
| CPU memory and compute limits | `results/cpu_peaks.csv` |
| Theoretical limits, detailed tables | `results/analysis/analysis.md` |
| Hardware size and cycle counts | `results/synthesis.md` |
| RTL co-simulation and hardware emulation logs | `results/emulation/` |
| Estimated FPGA speed (an **estimate**, not measured) | `results/fpga_estimate.csv`, per operator: `results/fpga_estimate_ops.csv` (from `analysis/fpga_estimate.py`) |
| The full report | `PROGRESS_REPORT.md` |

---

## Glossary

| term | meaning in this project |
|---|---|
| **kernel** | one hardware block on the FPGA that does one operation (e.g. GEMV, attention) |
| **task** | a part of a kernel (e.g. "read weights", "compute"); tasks run at the same time and pass data through FIFO streams (TAPA) |
| **PE** (processing element) | a compute unit we build from FPGA resources; e.g. one GEMV PE does 16 multiply-adds per cycle. The FPGA has no fixed number of PEs |
| **DSP, LUT, BRAM, URAM** | the FPGA's raw resources: multiplier blocks, logic cells, and two kinds of on-chip memory |
| **II** (initiation interval) | cycles between two inputs of a pipelined loop; II = 1 means a new input every cycle |
| **HBM channel / port** | the U55C's 16 GB HBM memory is split into 32 channels of ~14.4 GB/s; a kernel *port* is one 512-bit connection to a channel |
| **host** | the CPU computer the FPGA card is plugged into, and the program on it that drives the kernels |
| **software simulation** | running the kernel's C++ on the CPU to check correctness |
| **RTL co-simulation** | simulating the circuit produced by synthesis, cycle by cycle |
| **hardware emulation** | simulating the circuit inside the full U55C platform with an HBM model |
| **hardware build** | link + Vivado implementation into a `.xclbin` FPGA image |
| **dataflow** | tasks run at the same time and stream data to each other |
| **temporal** | operations run one after another on shared hardware; here: one kernel at a time |
| **spatial** | each operation gets its own hardware and they all run at once, connected by streams; the planned next design |
| **baseline** | the current design (dataflow inside kernels, temporal between them), the reference for later designs |
| **prefill / decode** | processing the whole prompt at once / producing one token at a time (§1) |
