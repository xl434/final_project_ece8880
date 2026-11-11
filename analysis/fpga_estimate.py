#!/usr/bin/env python3
"""ESTIMATE of FPGA decode and prefill time from the HLS designs.

Not a measurement. Each kernel call costs (streaming cycles) + (fixed
overhead). Streaming cycles follow the loop structure of each kernel, whose
pipelined loops synthesize at the stated II (results/synthesis.md); the fixed
overheads (pipeline fill, memory latency) are taken from the synthesized
decode latencies of the kernels. Assumptions:
  * clock 300 MHz (target; HLS estimates 380-411 MHz before place & route),
  * every AXI port delivers one 512-bit word per cycle (19.2 GB/s per port), and,
    as a second, HBM-limited column, one HBM pseudo-channel delivers only its
    peak 460/32 = 14.4 GB/s, which stretches every call that reads one word
    per cycle per port (decode GEMV, attention, LM head) by 19.2/14.4;
    prefill GEMM chunks need one weight word only every >= 2 cycles, so they
    are compute-bound and not stretched,
  * kernels run back to back, one at a time (temporal design), and
  * kernel launch / host synchronization overhead is NOT included (reported
    separately as a sensitivity: 98 launches per token or per prefill pass:
    12 x (2 LN + 4 GEMV + attention + GELU) + ln_f + LM head).
Kernel parameters: GEMV 8 ports x 16 rows, 4 token lanes, chunk 64 tokens;
attention 12 heads, query blocks of 4 (II 12), new K/V rows read from qkv.

Writes
  results/fpga_estimate.csv          decode, per token, at context 128/512/1024
  results/fpga_estimate_ops.csv      decode, per operator group (same groups as
                                     the CPU profile, results/cpu_profile_ops.csv)
  results/fpga_prefill_estimate.csv  prefill of the CPU-profile prompts
                                     (96 / 480 / 992 tokens): CPU measured vs
                                     FPGA token by token vs FPGA one pass
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = 300e6
C, C4, L = 768, 3072, 12
GEMV_PORTS, ROWS, TOK, CHUNK = 8, 16, 4, 64
ATTN_Q, CW = 4, 48                      # query block, words per 768-float row
PORT_GBS, PC_GBS = 64 * F / 1e9, 460.0 / 32
SLOW = PORT_GBS / PC_GBS                # 1.33: HBM pseudo-channel limit
# fixed per-call overheads, from the synthesized decode latencies
GEMV_OVH, ATTN_OVH, LM_HEAD = 240, 125, 301_959
LN_FIXED, LN_ROW = 166, 254             # 420 cycles for one row (synthesized)
GELU_OVH = 65                           # 257 cycles for 192 words (synthesized)


def gemv(c, oc, t, hbm=False):
    """GEMV/GEMM call: t tokens, (oc x c) weights."""
    cw, groups = c // 16, oc // (GEMV_PORTS * ROWS)
    cyc = GEMV_OVH
    for t0 in range(0, t, CHUNK):
        tc = min(CHUNK, t - t0)
        ntb = math.ceil(tc / TOK)
        words = groups * cw * ROWS            # weight words per port
        compute = words * ntb                 # one (word, token group) per cycle
        read = words * (SLOW if hbm else 1)   # port delivers <= 1 word per cycle
        cyc += tc * cw + max(compute, read)   # load X chunk, then stream
    return cyc


def attention(pos0, t, hbm=False):
    """t new tokens at positions pos0.. (cache rows 0..pos0-1 already filled)."""
    s = SLOW if hbm else 1
    cyc = ATTN_OVH
    for i0 in range(0, t, ATTN_Q):
        qn = min(ATTN_Q, t - i0)
        # queries + cached rows (K and V word per cycle) + new rows from qkv
        # (K then V words: 96 cycles per row); heads need 48 cycles per row
        cyc += qn * CW + (pos0 * CW + (i0 + qn) * 2 * CW) * s
    return cyc


def layernorm(t):
    return LN_FIXED + LN_ROW * t


def gelu(t):
    return GELU_OVH + t * C4 // 16


GROUPS = {
    "QKV GEMV": ["qkv_matmul"],
    "attention": ["attention", "kv_write"],
    "attn-proj GEMV": ["attproj_matmul"],
    "FC GEMV": ["fc_matmul"],
    "FC-proj GEMV": ["fcproj_matmul"],
    "LM head + argmax": ["lm_head", "argmax"],
    "LayerNorm": ["ln1", "ln2", "ln_f"],
    "GELU": ["gelu"],
    "residual + embed": ["residual1", "residual2", "embed"],
}


def pass_cycles(pos0, t, hbm=False):
    """Cycles per operator group for one pass of t tokens at positions pos0..
    (decode: t = 1; prefill: t = prompt length, pos0 = 0)."""
    return {
        "QKV GEMV": L * gemv(C, 3 * C, t, hbm),
        "attention": L * attention(pos0, t, hbm),
        "attn-proj GEMV": L * gemv(C, C, t, hbm),
        "FC GEMV": L * gemv(C, C4, t, hbm),
        "FC-proj GEMV": L * gemv(C4, C, t, hbm),
        "LM head + argmax": LM_HEAD * (SLOW if hbm else 1),
        "LayerNorm": 2 * L * layernorm(t) + layernorm(1),
        "GELU": L * gelu(t),
        "residual + embed": 0,  # residual fused into the GEMV output stage; embed on host
    }


def ms(cyc):
    return cyc / F * 1e3


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {path}")


def main():
    # decode: one token at position ctx-1
    out = []
    for ctx in (128, 512, 1024):
        a, b = pass_cycles(ctx - 1, 1), pass_cycles(ctx - 1, 1, hbm=True)
        gem = sum(v for k, v in a.items() if "GEMV" in k)
        tot, tot_pc = ms(sum(a.values())), ms(sum(b.values()))
        out.append(dict(context=ctx, layernorm_ms=ms(a["LayerNorm"]), gemv_ms=ms(gem),
                        attention_ms=ms(a["attention"]), gelu_ms=ms(a["GELU"]),
                        lm_head_ms=ms(a["LM head + argmax"]), total_ms=tot, tok_s=1e3 / tot,
                        tok_s_with_10us_launch=1e3 / (tot + 98 * 0.010),
                        tok_s_with_50us_launch=1e3 / (tot + 98 * 0.050),
                        total_ms_hbm_pc_limited=tot_pc, tok_s_hbm_pc_limited=1e3 / tot_pc))
    write(os.path.join(ROOT, "results", "fpga_estimate.csv"), out)
    for r in out:
        print(f"decode ctx {r['context']:4d}: {r['total_ms']:.3f} ms/token = {r['tok_s']:.0f} tok/s; "
              f"HBM-limited {r['tok_s_hbm_pc_limited']:.0f} tok/s")

    path = os.path.join(ROOT, "results", "fpga_estimate_ops.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["context", "op_group", "fpga_ms_full_port", "fpga_ms_hbm_pc_limited"])
        for ctx in (128, 512, 1024):
            a, b = pass_cycles(ctx - 1, 1), pass_cycles(ctx - 1, 1, hbm=True)
            for k in GROUPS:
                w.writerow([ctx, k, f"{ms(a[k]):.4f}", f"{ms(b[k]):.4f}"])
    print(f"wrote {path}")

    # prefill of the same prompts the CPU profile used (context - 32 tokens)
    cpu = {}
    with open(os.path.join(ROOT, "results", "cpu_profile_summary.csv")) as f:
        for r in csv.DictReader(f):
            cpu[(int(r["threads"]), int(r["prompt_len"]))] = float(r["prefill_ms"])
    pre = []
    for t in sorted({p for _, p in cpu}):
        one = pass_cycles(0, t, hbm=True)
        tbt = sum(sum(pass_cycles(p, 1, hbm=True).values()) - LM_HEAD * SLOW for p in range(t))
        tbt += LM_HEAD * SLOW  # token by token: the LM head runs only for the last token
        row = dict(prompt_len=t, cpu_1thread_ms=cpu.get((1, t), float("nan")),
                   cpu_8threads_ms=cpu.get((8, t), float("nan")),
                   fpga_token_by_token_ms=ms(tbt), fpga_one_pass_ms=ms(sum(one.values())))
        row.update({f"one_pass_{k.replace(' ', '_').replace('+', 'and')}_ms": ms(v)
                    for k, v in one.items() if v})
        pre.append(row)
    write(os.path.join(ROOT, "results", "fpga_prefill_estimate.csv"), pre)
    for r in pre:
        print(f"prefill {r['prompt_len']:4d} tokens: CPU 8T {r['cpu_8threads_ms']:.0f} ms (measured); "
              f"FPGA token by token {r['fpga_token_by_token_ms']:.0f} ms, one pass "
              f"{r['fpga_one_pass_ms']:.0f} ms (estimates, HBM-limited)")


if __name__ == "__main__":
    main()
