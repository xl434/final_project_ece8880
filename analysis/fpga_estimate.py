#!/usr/bin/env python3
"""ESTIMATE of FPGA decode time per token from the HLS synthesis results.

Not a measurement. Model: each kernel call costs (streaming cycles) +
(fixed overhead), where the overhead is taken from synthesis
(HLS latency - main-loop trip count at the synthesized trip count), and the
streaming cycles scale with the shape of the call. Assumptions:
  * clock 300 MHz (target; HLS estimates 380-411 MHz before place & route),
  * every AXI port delivers one 512-bit word per cycle (19.2 GB/s per port), and,
    as a second, HBM-limited column, one HBM pseudo-channel delivers only its
    peak 460/32 = 14.4 GB/s, which stretches the weight/KV-streaming kernels
    (gemv, attention, lm_head) by 19.2/14.4 (whole call scaled; slightly
    pessimistic for the fixed overhead),
  * kernels run back to back, one at a time (current host flow), and
  * kernel launch / host synchronization overhead is NOT included (reported
    separately as a sensitivity: 98 launches per token: 12 x (2 LN + 4 GEMV + attention + GELU) + ln_f + LM head).
Writes results/fpga_estimate.csv and prints a table.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = 300e6
C, L, V, VP = 768, 12, 50257, 50304
GEMV_PORTS, LM_PORTS = 8, 8
PORT_GBS, PC_GBS = 64 * F / 1e9, 460.0 / 32


def synth():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "synthesis.csv"))))
    k = {r["kernel"]: int(r["latency_cycles"]) for r in rows if r["level"] == "kernel"}
    return k


def main():
    s = synth()
    # overheads = synthesized latency - streaming cycles at the synthesized trip count
    gemv_ovh = s["gemv"] - (3072 * 768 // 16 // GEMV_PORTS)
    attn_ovh = s["attention"] - (1023 * 48)
    ln, gelu = s["layernorm"], s["gelu"]
    gemv = lambda c, oc: oc * c // 16 // GEMV_PORTS + gemv_ovh
    out = []
    for ctx in (128, 512, 1024):
        pos = ctx - 1
        per_layer = dict(
            layernorm=2 * ln,
            gemv=gemv(C, 3 * C) + gemv(C, C) + gemv(C, 4 * C) + gemv(4 * C, C),
            attention=pos * 48 + attn_ovh,
            gelu=gelu,
        )
        cyc = {k: L * v for k, v in per_layer.items()}
        cyc["layernorm"] += ln            # ln_f
        cyc["lm_head"] = s["lm_head"]   # fixed shape: synthesized latency is exact
        tot = sum(cyc.values())
        ms = tot / F * 1e3
        slow = PORT_GBS / PC_GBS
        ms_pc = sum(v * (slow if k in ("gemv", "attention", "lm_head") else 1)
                    for k, v in cyc.items()) / F * 1e3
        row = dict(context=ctx, **{f"{k}_ms": v / F * 1e3 for k, v in cyc.items()},
                   total_ms=ms, tok_s=1e3 / ms,
                   tok_s_with_10us_launch=1e3 / (ms + 98 * 0.010),
                   tok_s_with_50us_launch=1e3 / (ms + 98 * 0.050),
                   total_ms_hbm_pc_limited=ms_pc, tok_s_hbm_pc_limited=1e3 / ms_pc)
        out.append(row)
    path = os.path.join(ROOT, "results", "fpga_estimate.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("ESTIMATE (HLS cycles, 300 MHz, full port bandwidth, no launch overhead):")
    for r in out:
        print(f"ctx {r['context']:4d}: gemv {r['gemv_ms']:.3f} ms, attention {r['attention_ms']:.3f} ms, "
              f"lm_head {r['lm_head_ms']:.3f} ms, ln {r['layernorm_ms']:.3f} ms, gelu {r['gelu_ms']:.3f} ms"
              f" -> {r['total_ms']:.3f} ms/token = {r['tok_s']:.0f} tok/s "
              f"({r['tok_s_with_10us_launch']:.0f} / {r['tok_s_with_50us_launch']:.0f} tok/s with "
              f"10 / 50 us per launch); HBM-pseudo-channel-limited: {r['total_ms_hbm_pc_limited']:.3f} ms "
              f"= {r['tok_s_hbm_pc_limited']:.0f} tok/s")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
