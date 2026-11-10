#!/usr/bin/env python3
"""Run GPT-2 on your own prompt and check that every implementation agrees.

  python3 run_prompt.py "Your prompt here" [--n 20] [--fpga]

1. Tokenizes the prompt on the host (Hugging Face GPT-2 tokenizer) and writes
   the token IDs to results/my_prompt/prompt_ids.i32.
2. Hugging Face `gpt2` (FP32) greedy decoding -> reference tokens.
3. Our C++ CPU model (build/bin/cpu_main generate) -> tokens, compared with 2.
4. With --fpga: the FPGA host program with every op on the HLS kernels in TAPA
   software simulation (build/bin/host_main) -> tokens, compared with 2.
   Slow: roughly 1 minute per token (prompt tokens included).
Exit code 0 if all compared token sequences match exactly.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "golden"))
from golden_hf import greedy  # same greedy + KV-cache loop as the golden reference

OUT = os.path.join(ROOT, "results", "my_prompt")
BIN = os.path.join(ROOT, "build", "bin")


def build(src_dir, target):
    log = os.path.join(OUT, "build.log")
    with open(log, "a") as f:
        r = subprocess.run(["make", "-s", "-C", os.path.join(ROOT, src_dir), target],
                           stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"build of {target} failed; see {log}")


def run(cmd, log):
    shown = [os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd]
    print(f"$ {' '.join(shown)}   (log: {os.path.relpath(log, ROOT)})", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    with open(log, "w") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"command failed (exit {r.returncode}); see {log}")
    m = re.findall(r"output ids:([ \d]+)", r.stdout)
    if not m:
        sys.exit(f"no 'output ids:' line in the output; see {log}")
    return [int(t) for t in m[-1].split()]


def compare(name, got, ref, tok):
    n = sum(a == b for a, b in zip(got, ref))
    ok = got == ref
    print(f"\n[{name}] {tok.decode(got)!r}")
    print(f"[{name}] token IDs vs Hugging Face: {n}/{len(ref)} match -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        i = next(i for i, (a, b) in enumerate(zip(got, ref)) if a != b)
        print(f"  first difference at generated token {i}: {got[i]} vs {ref[i]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="prompt text")
    ap.add_argument("--n", type=int, default=20, help="tokens to generate (default 20)")
    ap.add_argument("--fpga", action="store_true",
                    help="also run the FPGA kernels in software simulation (slow)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "build.log"), "w").close()

    # 1. tokenize on the host
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok.encode(args.prompt)
    if not ids:
        sys.exit("empty prompt")
    if len(ids) + args.n > 1024:
        sys.exit(f"prompt ({len(ids)} tokens) + n ({args.n}) exceeds the 1024-token context")
    ids_file = os.path.join(OUT, "prompt_ids.i32")
    np.array(ids, dtype=np.int32).tofile(ids_file)
    print(f"prompt: {args.prompt!r}\ntoken IDs ({len(ids)}): {ids}")

    # 2. Hugging Face reference
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32,
                                            attn_implementation="eager").eval()
    ref, _ = greedy(model, ids, args.n)
    np.array(ref, dtype=np.int32).tofile(os.path.join(OUT, "hf_output_ids.i32"))
    print(f"\n[Hugging Face] {tok.decode(ref)!r}\n[Hugging Face] IDs: {ref}\n")

    # 3. our C++ CPU model
    build("src/cpu", "cpu_main")
    cpu = run([os.path.join(BIN, "cpu_main"), "generate", "--prompt", ids_file, "--n", str(args.n)],
              os.path.join(OUT, "cpu.log"))
    ok = compare("CPU C++", cpu, ref, tok)

    # 4. FPGA kernels in software simulation
    if args.fpga:
        build("src/host", "host_main")
        print(f"\nFPGA kernels in software simulation: about {len(ids) + args.n} minutes...")
        fpga = run([os.path.join(BIN, "host_main"), f"--prompt={ids_file}", f"--n={args.n}"],
                   os.path.join(OUT, "fpga_swsim.log"))
        ok &= compare("FPGA kernels (swsim)", fpga, ref, tok)

    print(f"\nall compared implementations match Hugging Face: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
