# golden/ — Hugging Face reference

`golden_hf.py` runs Hugging Face `transformers` GPT-2 (`gpt2`, FP32, eager
attention, CPU) and writes everything in `data/`:

```bash
pip install --user transformers torch numpy   # versions used: transformers 5.22.2, torch 2.14.1
python3 golden/golden_hf.py [--threads 8] [--llmc-bin models/gpt2_124M.bin]
```

* Prompt `"The FPGA is"` -> 32 greedy tokens (prefill + token-by-token decode
  with the HF KV cache; checked equal to `model.generate(do_sample=False)`).
* Activations of the first decode step (the token produced by prefill, fed at
  position 5): embedding, and per block ln1, qkv, atty (attention output before
  c_proj), attproj, res1, ln2, fch (pre-GELU), fch_gelu, fcproj, res2; ln_f;
  full logits + top-5; K/V cache of every layer after that step.
* A 992-token prompt + 32 greedy tokens (context 1024) for long-context checks
  and CPU profiling.
* Checks that llm.c's `gpt2_124M.bin` equals the HF weights (HF Conv1D weights
  are stored (in, out) and are transposed for the comparison). Result in
  `data/weights_check.txt`.
