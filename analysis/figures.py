#!/usr/bin/env python3
"""Charts for PROGRESS_REPORT.md, drawn from the result CSVs (no numbers typed in).

Inputs:  results/cpu_profile_ops.csv, results/cpu_profile_summary.csv,
         results/cpu_peaks.csv, results/fpga_estimate.csv,
         results/fpga_estimate_ops.csv, and the FLOP/byte model in analyze.py
Outputs (results/figures/):
  work_per_op.png         data read and FLOPs per decoded token, by operator
  cpu_bandwidth_per_op.png  achieved CPU memory bandwidth per operator vs the measured ceiling
  roofline.png            decode and prefill on the CPU and U55C rooflines
  cpu_vs_fpga_per_op.png  decode time per token by operator: CPU measured vs FPGA estimate
  decode_throughput.png   tokens/s vs context: CPU measured vs FPGA estimate
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import HBM_BW, INK, INK2, GRID, PEAK_FP32, SLOTS, SURF, model_ops  # noqa: E402
from fpga_estimate import GROUPS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
ORDER = list(GROUPS)  # operator groups, largest kernels first
CTX = 1024


def rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def title(ax, text, sub=None):
    ax.set_title(text, color=INK, fontsize=11.5, loc="left", pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join("results", "figures", name))


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=9, **kw)
    for t in lg.get_texts():
        t.set_color(INK)


def hbars(ax, labels, series, names, colors, unit_fmt):
    """Grouped horizontal bars: one group per label, one bar per series."""
    n = len(series)
    h = 0.8 / n
    for i, (vals, name, col) in enumerate(zip(series, names, colors)):
        ys = [g + (i - (n - 1) / 2) * h for g in range(len(labels))]
        ax.barh(ys, vals, height=h, color=col, edgecolor=SURF, linewidth=2, label=name, zorder=3)
        for y, v in zip(ys, vals):
            if v > 0:
                ax.text(v, y, " " + unit_fmt(v), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()


def op_bytes_flops(ctx):
    # analyze.py counts the KV-cache write inside "attention" (no separate kv_write)
    ops = {o["op"]: o for o in model_ops(1, ctx)}
    get = lambda o, k: ops[o][k] if o in ops else 0
    mb = {g: sum(get(o, "bytes_FP32") for o in GROUPS[g]) / 1e6 for g in ORDER}
    mf = {g: sum(get(o, "flops") for o in GROUPS[g]) / 1e6 for g in ORDER}
    return mb, mf


def cpu_ms(threads, ctx):
    r = [x for x in rows("cpu_profile_ops.csv")
         if int(x["threads"]) == threads and int(x["context"]) == ctx and x["phase"] == "decode"]
    t = {x["op"]: float(x["ms_per_call_token"]) for x in r}
    return {g: sum(t[o] for o in GROUPS[g]) for g in ORDER}


def fig_work_per_op():
    mb128, _ = op_bytes_flops(128)
    mb1024, mf = op_bytes_flops(1024)
    labels = [f"{g}\n{mf[g]:.1f} MFLOP" for g in ORDER]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF)
    style(ax)
    hbars(ax, labels, [[mb128[g] for g in ORDER], [mb1024[g] for g in ORDER]],
          ["context 128", "context 1024"], SLOTS[:2], lambda v: f"{v:.0f} MB" if v >= 1 else f"{v:.2f} MB")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("MB read from memory per generated token (FP32)", color=INK, fontsize=9.5)
    ax.set_xlim(0, max(mb1024.values()) * 1.25)
    title(ax, "Work per generated token, by operator",
          "Bars: data each operator must read (weights, KV cache). Labels: arithmetic at context 1024.")
    legend(ax, loc="lower right")
    save(fig, "work_per_op.png")


def fig_cpu_bandwidth():
    peaks = {int(r["threads"]): float(r["mem_bandwidth_gbs"]) for r in rows("cpu_peaks.csv")}
    mb, _ = op_bytes_flops(CTX)
    show = [g for g in ORDER if g not in ("residual + embed",)]
    series = []
    for th in (1, 8):
        ms = cpu_ms(th, CTX)
        series.append([mb[g] / ms[g] for g in show])  # MB/ms = GB/s
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
    style(ax)
    hbars(ax, show, series, ["1 thread", "8 threads"], SLOTS[:2], lambda v: f"{v:.1f}")
    for th, ls in ((1, ":"), (8, "--")):
        ax.axvline(peaks[th], color=INK2, linestyle=ls, linewidth=1.2, zorder=2)
        ax.text(peaks[th], -0.55, f" ceiling, {th} thread{'s' if th > 1 else ''}: {peaks[th]:.0f} GB/s",
                color=INK2, fontsize=7.5, va="bottom")
    ax.set_xlabel("achieved memory bandwidth (GB/s) = bytes the operator must read / measured time",
                  color=INK, fontsize=9)
    ax.set_xlim(0, max(peaks.values()) * 1.35)
    ax.set_ylim(len(show) - 0.5, -1.0)
    title(ax, f"CPU decode: how close each operator gets to the memory ceiling (context {CTX})",
          "Weight GEMVs run near the measured streaming bandwidth: they are memory-bound.")
    legend(ax, loc="lower right")
    save(fig, "cpu_bandwidth_per_op.png")


def fig_roofline():
    peaks = {int(r["threads"]): r for r in rows("cpu_peaks.csv")}
    cpu_bw, cpu_pk = float(peaks[8]["mem_bandwidth_gbs"]) * 1e9, float(peaks[8]["fp32_gflops"]) * 1e9
    dec = model_ops(1, CTX)
    oi_dec = sum(o["flops"] for o in dec) / sum(o["bytes_FP32"] for o in dec)
    flops_dec = sum(o["flops"] for o in dec)
    pre = model_ops(992, CTX)
    oi_pre = sum(o["flops"] for o in pre) / sum(o["bytes_FP32"] for o in pre)
    summ = {(int(r["threads"]), int(r["context"])): r for r in rows("cpu_profile_summary.csv")}
    cpu_dec = flops_dec * float(summ[(8, CTX)]["decode_tok_s"])
    cpu_pre = sum(o["flops"] for o in pre) / 992 * float(summ[(8, CTX)]["prefill_tok_s"])
    est = {int(r["context"]): r for r in rows("fpga_estimate.csv")}
    fpga_dec = flops_dec * float(est[CTX]["tok_s_hbm_pc_limited"])

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF)
    style(ax, grid_axis="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [10 ** (i / 20) for i in range(-40, 61)]
    for bw, pk, col, name in ((HBM_BW, PEAK_FP32, SLOTS[2], "U55C: HBM 460 GB/s, ~1.08 TFLOP/s (est.)"),
                              (cpu_bw, cpu_pk, SLOTS[0], f"CPU (4 cores / 8 threads): measured "
                               f"{cpu_bw / 1e9:.0f} GB/s, {cpu_pk / 1e9:.0f} GFLOP/s")):
        ax.plot(xs, [min(bw * x, pk) for x in xs], color=col, linewidth=2, label=name, zorder=3)
    for oi, lab in ((oi_dec, "decode\n(0.5 FLOP/B)"), (oi_pre, "prefill, 992 tokens\n(~100 FLOP/B)")):
        ax.axvline(oi, color=INK2, linestyle=":", linewidth=1, zorder=2)
        ax.text(oi * 1.12, 2e8, lab, color=INK2, fontsize=8)
    pts = [(oi_dec, cpu_dec, SLOTS[0], "o", "CPU decode (measured)"),
           (oi_pre, cpu_pre, SLOTS[0], "s", "CPU prefill (measured)"),
           (oi_dec, fpga_dec, SLOTS[2], "D", "FPGA decode (estimate)")]
    for x, y, col, m, lab in pts:
        ax.scatter([x], [y], s=60, color=col, marker=m, edgecolor=SURF, linewidth=2, zorder=5)
        ax.annotate(f"{lab}: {y / 1e9:.0f} GFLOP/s", (x, y), textcoords="offset points",
                    xytext=(9, -3), ha="left", va="center", fontsize=8, color=INK)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1e8, 3e12)
    ax.set_xlabel("operational intensity (FLOP per byte read from memory)", color=INK, fontsize=9.5)
    ax.set_ylabel("performance (FLOP/s)", color=INK, fontsize=9.5)
    title(ax, "Roofline: decode is limited by memory bandwidth, not arithmetic",
          "Attainable performance = min(bandwidth x intensity, compute peak).")
    legend(ax, loc="upper left")
    save(fig, "roofline.png")


def fig_cpu_vs_fpga():
    est = {r["op_group"]: float(r["fpga_ms_hbm_pc_limited"])
           for r in rows("fpga_estimate_ops.csv") if int(r["context"]) == CTX}
    c1, c8 = cpu_ms(1, CTX), cpu_ms(8, CTX)
    show = [g for g in ORDER if g != "residual + embed"]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
    style(ax)
    hbars(ax, show, [[c1[g] for g in show], [c8[g] for g in show], [est[g] for g in show]],
          ["CPU, 1 thread (measured)", "CPU, 8 threads (measured)", "FPGA kernel (estimate)"],
          [SLOTS[0], SLOTS[1], SLOTS[2]], lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}")
    ax.set_xlabel("time per generated token (ms), context 1024", color=INK, fontsize=9.5)
    ax.set_xlim(0, max(c1.values()) * 1.12)
    title(ax, "Each kernel vs the CPU code it replaces (decode, context 1024)",
          "FPGA: HLS cycle counts at 300 MHz, HBM-limited (14.4 GB/s per port); not measured.")
    legend(ax, loc="lower right")
    save(fig, "cpu_vs_fpga_per_op.png")


def fig_throughput():
    summ = {(int(r["threads"]), int(r["context"])): float(r["decode_tok_s"])
            for r in rows("cpu_profile_summary.csv")}
    est = {int(r["context"]): r for r in rows("fpga_estimate.csv")}
    ctxs = [128, 512, 1024]
    series = [
        ([summ[(1, c)] for c in ctxs], "CPU, 1 thread (measured)", SLOTS[0]),
        ([summ[(8, c)] for c in ctxs], "CPU, 8 threads (measured)", SLOTS[1]),
        ([1e3 / (float(est[c]["total_ms_hbm_pc_limited"]) + 98 * 0.010) for c in ctxs],
         "FPGA estimate, HBM-limited + 10 us per kernel launch", SLOTS[3]),
        ([float(est[c]["tok_s_hbm_pc_limited"]) for c in ctxs],
         "FPGA estimate, HBM-limited, no launch cost", SLOTS[2]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURF)
    style(ax, grid_axis="y")
    w = 0.8 / len(series)
    for i, (vals, name, col) in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * w for g in range(len(ctxs))]
        ax.bar(xs, vals, width=w, color=col, edgecolor=SURF, linewidth=2, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels([f"context {c}" for c in ctxs])
    ax.set_ylabel("generated tokens per second", color=INK, fontsize=9.5)
    title(ax, "Decode speed: measured CPU vs estimated FPGA",
          "Longer context = more KV cache to read per token, so every bar shrinks.")
    legend(ax, loc="upper right", ncol=1)
    ax.set_ylim(0, max(series[3][0]) * 1.55)
    save(fig, "decode_throughput.png")


def main():
    fig_work_per_op()
    fig_cpu_bandwidth()
    fig_roofline()
    fig_cpu_vs_fpga()
    fig_throughput()


if __name__ == "__main__":
    main()
