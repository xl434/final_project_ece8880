#!/usr/bin/env python3
"""Application analysis of GPT-2 (124M) inference on the Alveo U55C.

Everything is computed from the model dimensions below (no hard-coded results).
Measured CPU numbers are read from results/cpu_profile_*.csv when present.

Outputs (results/analysis/):
  ops_decode.csv, ops_prefill.csv   per-operator shape/params/FLOPs/bytes/intensity
  kv_cache.csv                      KV-cache size vs context length
  token_bound.csv                   bandwidth-bound tokens/s upper bounds
  onchip_fit.csv                    weight sizes vs on-chip memory
  analysis.md                       markdown tables of all of the above
  roofline.png, cpu_breakdown.png   plots

Assumptions (also written into analysis.md):
  * FLOP counts: multiply and add = 2 FLOPs. LayerNorm 8 FLOPs/elem, GELU 12
    FLOPs/elem (tanh approx incl. tanh as ~4), softmax 5 FLOPs/score,
    residual 1 FLOP/elem. Nonlinear op costs are rough; they are <1% of FLOPs.
  * Bytes: every weight read once per token (batch 1, no reuse across tokens),
    every activation read/written once from/to off-chip memory, KV cache read
    in full each decode step. All tensors (weights, activations, KV) in the
    stated precision; INT8 scale-factor overhead ignored.
  * Prefill of T tokens reads weights once (ideal tiling) and touches T times
    the activations.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "analysis")

# ---- model (GPT-2 124M) -------------------------------------------------
C, L, NH, V, MAXT = 768, 12, 12, 50257, 1024
HS = C // NH
PREC = {"FP32": 4, "FP16": 2, "INT8": 1}

# ---- Alveo U55C ---------------------------------------------------------
F_CLK = 300e6                      # target kernel clock (3.33 ns)
PORT_BW = 512 / 8 * F_CLK          # one 512-bit AXI port at 300 MHz = 19.2 GB/s
HBM_BW = 460e9                     # 32 pseudo-channels, datasheet total
DSP = 9024
DSP_PER_FP32_MAC = 5               # fmul 3 DSP + fadd 2 DSP (Vitis HLS full-DSP cores)
PEAK_FP32 = DSP / DSP_PER_FP32_MAC * 2 * F_CLK
BRAM36, URAM = 2016, 960
ONCHIP_BYTES = BRAM36 * 36 * 1024 / 8 + URAM * 288 * 1024 / 8   # 44.7 MB = 42.6 MiB

# colors: reference categorical palette, fixed slot order (dataviz skill)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def layer_ops(T, ctx_end):
    """Operators of one transformer block for T new tokens whose last token
    attends to ctx_end positions (decode: T=1, ctx_end = context length)."""
    ctx_start = ctx_end - T + 1
    att_pairs = sum(range(ctx_start, ctx_end + 1))  # sum of attended positions
    ops = []
    def add(name, shape, params, flops, act_elems, kv_elems=0):
        ops.append(dict(op=name, shape=shape, params=params, flops=flops,
                        act_elems=act_elems, kv_elems=kv_elems))
    add("ln1", f"({T},{C})", 2 * C, 8 * T * C, 2 * T * C)
    add("qkv_matmul", f"({T},{C})x({C},{3*C})", 3 * C * C + 3 * C,
        2 * T * C * 3 * C, T * C + 3 * T * C)
    # attention: QK^T + softmax + AV for every head; reads q, the cache, writes out;
    # also writes the new K,V rows.
    add("attention", f"{NH}h x {HS} x ctx", 0,
        4 * C * att_pairs + 5 * NH * att_pairs, T * C + T * C,
        kv_elems=2 * C * ctx_end + 2 * T * C)
    add("attproj_matmul", f"({T},{C})x({C},{C})", C * C + C, 2 * T * C * C, 2 * T * C)
    add("residual1", f"({T},{C})", 0, T * C, 3 * T * C)
    add("ln2", f"({T},{C})", 2 * C, 8 * T * C, 2 * T * C)
    add("fc_matmul", f"({T},{C})x({C},{4*C})", 4 * C * C + 4 * C,
        2 * T * C * 4 * C, T * C + 4 * T * C)
    add("gelu", f"({T},{4*C})", 0, 12 * T * 4 * C, 2 * T * 4 * C)
    add("fcproj_matmul", f"({T},{4*C})x({4*C},{C})", 4 * C * C + C,
        2 * T * 4 * C * C, 4 * T * C + T * C)
    add("residual2", f"({T},{C})", 0, T * C, 3 * T * C)
    return ops


def model_ops(T, ctx_end):
    """All operators for one forward call; layer ops are summed over L layers."""
    ops = [dict(op="embed", shape=f"({T},{C})", params=2 * T * C,  # rows read (wte+wpe)
                flops=T * C, act_elems=T * C, kv_elems=0)]
    for o in layer_ops(T, ctx_end):
        o = dict(o)
        for k in ("params", "flops", "act_elems", "kv_elems"):
            o[k] *= L
        o["shape"] = f"{L} x {o['shape']}"
        ops.append(o)
    # inference needs only the last token's logits
    ops.append(dict(op="ln_f", shape=f"(1,{C})", params=2 * C, flops=8 * C,
                    act_elems=2 * C, kv_elems=0))
    ops.append(dict(op="lm_head", shape=f"(1,{C})x({C},{V})", params=V * C,
                    flops=2 * V * C, act_elems=C + V, kv_elems=0))
    ops.append(dict(op="argmax", shape=f"({V})", params=0, flops=V, act_elems=V, kv_elems=0))
    for o in ops:
        for p, b in PREC.items():
            o[f"bytes_{p}"] = (o["params"] + o["act_elems"] + o["kv_elems"]) * b
            o[f"oi_{p}"] = o["flops"] / o[f"bytes_{p}"]
        o["weight_MB_FP32"] = o["params"] * 4 / 1e6
        o["kv_MB_FP32"] = o["kv_elems"] * 4 / 1e6
    return ops


def totals(ops):
    t = dict(op="TOTAL", shape="", params=sum(o["params"] for o in ops),
             flops=sum(o["flops"] for o in ops), act_elems=sum(o["act_elems"] for o in ops),
             kv_elems=sum(o["kv_elems"] for o in ops))
    for p, b in PREC.items():
        t[f"bytes_{p}"] = sum(o[f"bytes_{p}"] for o in ops)
        t[f"oi_{p}"] = t["flops"] / t[f"bytes_{p}"]
    t["weight_MB_FP32"] = t["params"] * 4 / 1e6
    t["kv_MB_FP32"] = t["kv_elems"] * 4 / 1e6
    return t


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_table(rows, cols, fmt):
    s = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(fmt.get(k, str)(r[k]) for _, k in cols) + " |\n"
    return s


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    os.makedirs(OUT, exist_ok=True)
    md = ["# GPT-2 124M on Alveo U55C: application analysis",
          "", "Generated by `analysis/analyze.py` from model dimensions "
          f"(C={C}, L={L}, NH={NH}, V={V}, maxT={MAXT}).", ""]
    md.append("## Assumptions\n")
    md.append(__doc__.split("Assumptions (also written into analysis.md):")[1].strip()
              .replace("\n  ", "\n") + "\n")
    md.append(f"* U55C: one HBM AXI port = 512 bit x {F_CLK/1e6:.0f} MHz = {PORT_BW/1e9:.1f} GB/s; "
              f"full HBM = {HBM_BW/1e9:.0f} GB/s; compute ceiling = {DSP} DSP / "
              f"{DSP_PER_FP32_MAC} DSP per FP32 MAC x 2 FLOP x {F_CLK/1e6:.0f} MHz = "
              f"{PEAK_FP32/1e12:.2f} TFLOP/s (ESTIMATE: assumes every DSP is usable and "
              f"Vitis HLS full-DSP fmul (3) + fadd (2)).")
    md.append(f"* On-chip memory: {BRAM36} BRAM36 + {URAM} URAM = {ONCHIP_BYTES/1e6:.1f} MB "
              f"({ONCHIP_BYTES/2**20:.1f} MiB).\n")

    cols = ["op", "shape", "params", "flops", "act_elems", "kv_elems", "weight_MB_FP32",
            "kv_MB_FP32"] + [f"bytes_{p}" for p in PREC] + [f"oi_{p}" for p in PREC]

    # ---- decode per operator at several contexts ----------------------------
    dec_rows = []
    for ctx in (128, 512, 1024):
        ops = model_ops(1, ctx)
        for o in ops + [totals(ops)]:
            dec_rows.append(dict(context=ctx, **o))
    write_csv(os.path.join(OUT, "ops_decode.csv"), dec_rows, ["context"] + cols)
    f3 = lambda x: f"{x:.3g}"
    fM = lambda x: f"{x/1e6:.2f}"
    md.append("## Decode: per-operator cost per token (context 1024; layer ops summed over 12 layers)\n")
    md.append(md_table([r for r in dec_rows if r["context"] == 1024],
                       [("op", "op"), ("shape", "shape"), ("params (M)", "params"),
                        ("MFLOP", "flops"), ("FP32 MB moved", "bytes_FP32"),
                        ("FP16 MB", "bytes_FP16"), ("INT8 MB", "bytes_INT8"),
                        ("OI FP32 (FLOP/B)", "oi_FP32"), ("OI INT8", "oi_INT8")],
                       {"params": fM, "flops": fM, "bytes_FP32": fM, "bytes_FP16": fM,
                        "bytes_INT8": fM, "oi_FP32": f3, "oi_INT8": f3}))
    md.append("\nDecode totals per token vs context:\n")
    md.append(md_table([r for r in dec_rows if r["op"] == "TOTAL"],
                       [("context", "context"), ("GFLOP", "flops"), ("weights MB (FP32)", "weight_MB_FP32"),
                        ("KV MB (FP32)", "kv_MB_FP32"), ("FP32 MB moved", "bytes_FP32"),
                        ("OI FP32", "oi_FP32"), ("OI FP16", "oi_FP16"), ("OI INT8", "oi_INT8")],
                       {"flops": lambda x: f"{x/1e9:.3f}", "weight_MB_FP32": f3,
                        "kv_MB_FP32": f3, "bytes_FP32": fM, "oi_FP32": f3, "oi_FP16": f3,
                        "oi_INT8": f3}))

    # ---- prefill ----------------------------------------------------------
    pre_rows = []
    for T in (128, 512, 1024):
        ops = model_ops(T, T)
        for o in ops + [totals(ops)]:
            pre_rows.append(dict(T=T, **o))
    write_csv(os.path.join(OUT, "ops_prefill.csv"), pre_rows, ["T"] + cols)
    md.append("\n## Prefill of T tokens (weights read once; logits only for the last token)\n")
    md.append(md_table([r for r in pre_rows if r["op"] in ("qkv_matmul", "attention", "fc_matmul", "TOTAL")],
                       [("T", "T"), ("op", "op"), ("GFLOP", "flops"), ("FP32 MB moved", "bytes_FP32"),
                        ("OI FP32", "oi_FP32")],
                       {"flops": lambda x: f"{x/1e9:.2f}", "bytes_FP32": fM, "oi_FP32": f3}))

    # ---- KV cache ---------------------------------------------------------
    kv_rows = []
    for ctx in (1, 128, 256, 512, 1024):
        r = dict(context=ctx)
        for p, b in PREC.items():
            r[f"kv_MB_{p}"] = 2 * L * C * ctx * b / 1e6
        kv_rows.append(r)
    write_csv(os.path.join(OUT, "kv_cache.csv"), kv_rows, list(kv_rows[0]))
    md.append("\n## KV cache size (all 12 layers, K and V)\n")
    md.append(md_table(kv_rows, [("context", "context")] + [(f"{p} MB", f"kv_MB_{p}") for p in PREC],
                       {f"kv_MB_{p}": lambda x: f"{x:.3f}" for p in PREC}))

    # ---- bandwidth-bound tokens/s ------------------------------------------
    tb_rows = []
    for ctx in (128, 512, 1024):
        tot = [r for r in dec_rows if r["context"] == ctx and r["op"] == "TOTAL"][0]
        for p in PREC:
            b = tot[f"bytes_{p}"]
            tb_rows.append(dict(context=ctx, precision=p, MB_per_token=b / 1e6,
                                tok_s_1port=PORT_BW / b, tok_s_8ports=8 * PORT_BW / b,
                                tok_s_16ports=16 * PORT_BW / b, tok_s_hbm460=HBM_BW / b))
    write_csv(os.path.join(OUT, "token_bound.csv"), tb_rows, list(tb_rows[0]))
    f1 = lambda x: f"{x:.1f}"
    md.append("\n## Bandwidth-bound decode upper bound (tokens/s = bandwidth / bytes per token)\n")
    md.append("Upper bounds: assume 100% of the stated bandwidth and zero compute/launch overhead.\n")
    md.append(md_table(tb_rows, [("context", "context"), ("precision", "precision"),
                                 ("MB/token", "MB_per_token"), ("1 port 19.2 GB/s", "tok_s_1port"),
                                 ("8 ports", "tok_s_8ports"), ("16 ports", "tok_s_16ports"),
                                 ("HBM 460 GB/s", "tok_s_hbm460")],
                       {"MB_per_token": f1, "tok_s_1port": f1, "tok_s_8ports": f1,
                        "tok_s_16ports": f1, "tok_s_hbm460": f1}))

    # ---- on-chip fit ----------------------------------------------------------
    per_layer = 12 * C * C + 13 * C          # qkv+attproj+fc+fcproj weights + biases + LN
    fit_rows = []
    for name, n in (("one block", per_layer), ("qkv_matmul", 3 * C * C), ("fc_matmul", 4 * C * C),
                    ("fcproj_matmul", 4 * C * C), ("attproj_matmul", C * C),
                    ("12 blocks", L * per_layer), ("lm_head (wte)", V * C),
                    ("whole model (incl. wpe)", L * per_layer + V * C + MAXT * C + 2 * C)):
        r = dict(tensor=name, params_M=n / 1e6)
        for p, b in PREC.items():
            r[f"MB_{p}"] = n * b / 1e6
            r[f"fits_{p}"] = "yes" if n * b <= ONCHIP_BYTES else "no"
            r[f"double_buf_{p}"] = "yes" if 2 * n * b <= ONCHIP_BYTES else "no"
        fit_rows.append(r)
    write_csv(os.path.join(OUT, "onchip_fit.csv"), fit_rows, list(fit_rows[0]))
    md.append(f"\n## On-chip fit (on-chip = {ONCHIP_BYTES/1e6:.1f} MB; ignores space needed by "
              "activations, KV tiles and FIFOs)\n")
    md.append(md_table(fit_rows, [("tensor", "tensor"), ("params (M)", "params_M")] +
                       sum([[(f"{p} MB", f"MB_{p}"), (f"fits {p}", f"fits_{p}"),
                             (f"2x buf {p}", f"double_buf_{p}")] for p in PREC], []),
                       {"params_M": lambda x: f"{x:.2f}", **{f"MB_{p}": f1 for p in PREC}}))

    # ---- measured CPU ---------------------------------------------------------
    summ = read_csv(os.path.join(ROOT, "results", "cpu_profile_summary.csv"))
    opsm = read_csv(os.path.join(ROOT, "results", "cpu_profile_ops.csv"))
    cpu_pts = []
    if summ:
        md.append("\n## Measured CPU baseline placed on the roofline\n")
        rows = []
        for r in summ:
            ctx, th = int(r["context"]), int(r["threads"])
            tot = [x for x in dec_rows if x["context"] == ctx and x["op"] == "TOTAL"][0]
            ds = float(r["decode_tok_s"])
            plen = int(r["prompt_len"])
            pt = totals(model_ops(plen, plen))
            ps = float(r["prefill_tok_s"]) / plen   # prefills per second
            rows.append(dict(threads=th, context=ctx, decode_tok_s=ds,
                             decode_gflops=tot["flops"] * ds / 1e9,
                             decode_gbs=tot["bytes_FP32"] * ds / 1e9,
                             prefill_tok_s=float(r["prefill_tok_s"]),
                             prefill_gflops=pt["flops"] * ps / 1e9))
            cpu_pts.append(("decode", th, ctx, tot["oi_FP32"], tot["flops"] * ds))
            cpu_pts.append(("prefill", th, ctx, pt["oi_FP32"], pt["flops"] * ps))
        write_csv(os.path.join(OUT, "cpu_roofline_points.csv"), rows, list(rows[0]))
        md.append("Achieved = analytic FLOPs (or bytes) per token x measured tokens/s.\n")
        md.append(md_table(rows, [("threads", "threads"), ("context", "context"),
                                  ("decode tok/s", "decode_tok_s"), ("decode GFLOP/s", "decode_gflops"),
                                  ("decode effective GB/s", "decode_gbs"),
                                  ("prefill tok/s", "prefill_tok_s"), ("prefill GFLOP/s", "prefill_gflops")],
                           {k: (lambda x: f"{x:.2f}") for k in ("decode_tok_s", "decode_gflops", "decode_gbs",
                                                               "prefill_tok_s", "prefill_gflops")}))

    # ---- roofline plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=SURF)
    style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [10 ** (e / 20) for e in range(-60, 81)]  # 1e-3 .. 1e4 FLOP/B
    # ceiling labels: (x position, above/below the line) chosen so they do not collide
    for bw, lab, xl, fac in ((PORT_BW, f"1 HBM port {PORT_BW/1e9:.1f} GB/s", 2e-3, 1.6),
                             (16 * PORT_BW, f"16 ports {16*PORT_BW/1e9:.0f} GB/s", 4e-3, 0.35),
                             (HBM_BW, f"HBM {HBM_BW/1e9:.0f} GB/s", 2e-3, 1.6)):
        ys = [min(bw * x, PEAK_FP32) for x in xs]
        ax.plot(xs, ys, color=INK2, linewidth=1.2, zorder=2)
        ax.text(xl, bw * xl * fac, lab, color=INK2, fontsize=8)
    ax.axhline(PEAK_FP32, color=INK2, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(1.3e-3, PEAK_FP32 * 1.25, f"FP32 compute ceiling {PEAK_FP32/1e12:.2f} TFLOP/s (estimate)",
            color=INK2, fontsize=8)
    # analytic points, attainable on full HBM: decode ops (ctx 1024) and prefill ops (T=512)
    dec = [r for r in dec_rows if r["context"] == 1024 and r["flops"] > 1e6]
    pre = [r for r in pre_rows if r["T"] == 512 and r["flops"] > 1e8]
    att = lambda oi: min(HBM_BW * oi, PEAK_FP32)
    ax.scatter([r["oi_FP32"] for r in dec], [att(r["oi_FP32"]) for r in dec], s=46,
               color=SLOTS[0], edgecolor=SURF, linewidth=2, zorder=4,
               label="decode ops, ctx 1024 (attainable @460 GB/s)")
    ax.scatter([r["oi_FP32"] for r in pre], [att(r["oi_FP32"]) for r in pre], s=46,
               color=SLOTS[1], edgecolor=SURF, linewidth=2, zorder=4, marker="s",
               label="prefill ops, T=512 (attainable)")
    tot_d = [r for r in dec if r["op"] == "TOTAL"][0]
    ax.annotate("decode: every op ~0.5 FLOP/B\n(memory bound)", (tot_d["oi_FP32"], att(tot_d["oi_FP32"])),
                textcoords="offset points", xytext=(10, -26), fontsize=8, color=INK)
    tot_p = [r for r in pre if r["op"] == "TOTAL"][0]
    ax.annotate("prefill total", (tot_p["oi_FP32"], att(tot_p["oi_FP32"])),
                textcoords="offset points", xytext=(-8, -16), fontsize=8, color=INK, ha="right")
    if cpu_pts:
        ax.scatter([p[3] for p in cpu_pts], [p[4] for p in cpu_pts], s=46, color=SLOTS[2],
                   edgecolor=SURF, linewidth=2, zorder=5, marker="D",
                   label="measured CPU (EPYC 7713, 1 and 8 threads)")
        d = [p for p in cpu_pts if p[0] == "decode"]
        pf = [p for p in cpu_pts if p[0] == "prefill"]
        ax.annotate("CPU decode", (d[0][3], max(p[4] for p in d)), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=INK)
        ax.annotate("CPU prefill (96-992 tokens)", (max(p[3] for p in pf), max(p[4] for p in pf)),
                    textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK)
    ax.set_xlim(1e-3, 1e4)
    ax.set_ylim(1e7, 3e12)
    ax.set_xlabel("Operational intensity (FLOP / byte, FP32)", color=INK, fontsize=10)
    ax.set_ylabel("Performance (FLOP/s)", color=INK, fontsize=10)
    ax.set_title("GPT-2 124M roofline on Alveo U55C (FP32)", color=INK, fontsize=12, loc="left")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roofline.png"), dpi=150)
    plt.close(fig)

    # ---- CPU per-operator breakdown (decode) ----------------------------------
    if opsm:
        groups = [("QKV matmul", ["qkv_matmul"]), ("attention", ["attention", "kv_write"]),
                  ("attn proj", ["attproj_matmul"]), ("FC matmul", ["fc_matmul"]),
                  ("FC proj", ["fcproj_matmul"]), ("LM head", ["lm_head", "argmax"]),
                  ("other (LN, GELU, residual, embed)", ["embed", "ln1", "residual1", "ln2", "gelu",
                                                         "residual2", "ln_f"])]
        threads = sorted({int(r["threads"]) for r in opsm})
        ctxs = sorted({int(r["context"]) for r in opsm})
        fig, axes = plt.subplots(1, len(threads), figsize=(10, 3.6), facecolor=SURF, sharex=True)
        if len(threads) == 1:
            axes = [axes]
        xmax = 0
        for ax, th in zip(axes, threads):
            style(ax)
            ax.grid(True, axis="x", color=GRID)
            ax.grid(False, axis="y")
            for yi, ctx in enumerate(ctxs):
                rs = {r["op"]: float(r["ms_per_call_token"]) for r in opsm
                      if int(r["threads"]) == th and int(r["context"]) == ctx and r["phase"] == "decode"}
                left = 0
                for gi, (gname, members) in enumerate(groups):
                    w = sum(rs.get(m, 0) for m in members)
                    ax.barh(yi, w, left=left, height=0.6, color=SLOTS[gi], edgecolor=SURF,
                            linewidth=2, label=gname if (th == threads[0] and yi == 0) else None,
                            zorder=3)
                    left += w
                ax.text(left + 0.4, yi, f"{left:.1f} ms", va="center", fontsize=8, color=INK)
                xmax = max(xmax, left)
            ax.set_yticks(range(len(ctxs)))
            ax.set_yticklabels([f"ctx {c}" for c in ctxs])
            ax.set_title(f"{th} thread{'s' if th > 1 else ''}", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("decode time per token (ms)", color=INK, fontsize=9)
        axes[0].set_xlim(0, xmax * 1.18)
        fig.suptitle("CPU decode time per token by operator (measured, -O3, OpenMP)",
                     color=INK, fontsize=11, x=0.01, ha="left")
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        for t in leg.get_texts():
            t.set_color(INK)
        fig.tight_layout(rect=(0, 0.12, 1, 0.94))
        fig.savefig(os.path.join(OUT, "cpu_breakdown.png"), dpi=150)
        plt.close(fig)

        md.append("\n## Measured CPU decode time per operator (ms/token)\n")
        rows = []
        for th in threads:
            for ctx in ctxs:
                rs = {r["op"]: float(r["ms_per_call_token"]) for r in opsm
                      if int(r["threads"]) == th and int(r["context"]) == ctx and r["phase"] == "decode"}
                row = dict(threads=th, context=ctx)
                tot = sum(rs.values())
                for gname, members in groups:
                    row[gname] = sum(rs.get(m, 0) for m in members)
                row["total"] = tot
                rows.append(row)
        md.append(md_table(rows, [("threads", "threads"), ("context", "context")] +
                           [(g, g) for g, _ in groups] + [("total", "total")],
                           {g: (lambda x: f"{x:.2f}") for g, _ in groups + [("total", 0)]}))
        md.append("\n![CPU breakdown](cpu_breakdown.png)\n")
    md.append("\n![Roofline](roofline.png)\n")

    with open(os.path.join(OUT, "analysis.md"), "w") as f:
        f.write("\n".join(md))
    print(f"wrote {OUT}/analysis.md and CSV/PNG files")
    # brief console summary
    for r in tb_rows:
        if r["context"] == 1024:
            print(f"ctx 1024 {r['precision']}: {r['MB_per_token']:.1f} MB/token -> "
                  f"<= {r['tok_s_hbm460']:.0f} tok/s @460 GB/s, {r['tok_s_1port']:.1f} tok/s @1 port")


if __name__ == "__main__":
    main()
