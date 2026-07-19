"""T1.6 figure: lens construct validity — does the workspace readout A_w predict emission?

Reads runs/exp6/construct_validity.json and renders a compact bar comparison of the
emission-prediction AUROC (that per-position A_w(t) predicts the model actually emits w
within k tokens) for the J-lens vs the logit lens, over the probe-token vocabulary. Both
lenses are well above chance and comparable — that COMPARABILITY is the point: raw readout
quality is similar, so H1's *conflict* signal (not readout quality) is the J-space-specific
effect. The 52 H1 target tokens never occur in the neutral greedy continuations (n_pos=0),
so the targets-only AUROC is undefined; we say so on the figure.

CPU-only. Usage: python exp/fig_construct_validity.py [--in ...] [--out paper/figures/construct_validity.png]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_JLENS = "#2a78d6"
C_LOGIT = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp6/construct_validity.json")
    ap.add_argument("--out", default="paper/figures/construct_validity.png")
    args = ap.parse_args()

    s = json.load(open(args.inp))["summary"]
    slices = [("pooled\n(targets ∪ frequent)", s["pooled"]),
              ("frequent\nemitted tokens", s["freq_tokens_only"])]
    labels = [name for name, _ in slices]
    jl = [d["jlens_auroc"] for _, d in slices]
    lg = [d["logit_auroc"] for _, d in slices]
    npos = [d["n_pos"] for _, d in slices]
    npairs = [d["n_pairs"] for _, d in slices]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax.bar(x - w / 2, jl, w, color=C_JLENS, label="J-lens")
    b2 = ax.bar(x + w / 2, lg, w, color=C_LOGIT, label="Logit lens")
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.text(len(labels) - 0.5, 0.505, "chance", fontsize=8, color=INK2, va="bottom", ha="right")
    for b, v in list(zip(b1, jl)) + list(zip(b2, lg)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    for xi, (npo, npa) in zip(x, zip(npos, npairs)):
        ax.text(xi, 0.02, f"{npo:,} emissions\n{npa:,} pairs", ha="center", va="bottom",
                fontsize=7.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("emission-prediction AUROC  P(A_w higher when w is next)", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("T1.6: the workspace readout predicts token emission: J-lens ≈ logit\n"
                 "(comparable readout quality ⇒ H1's win is the conflict signal, not the readout)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, lw=0.6, axis="y")
    ax.set_axisbelow(True)

    fig.text(0.01, -0.02,
             "The 52 censored H1 target tokens do not occur in neutral greedy continuations "
             "(n_pos=0), so a targets-only AUROC is undefined; validated on the emitted "
             "vocabulary (%d probe tokens, DeepSeek-R1-Distill-Qwen-14B, L%d)."
             % (json.load(open(args.inp))["summary"]["n_probe_tokens"], s["layer"]),
             fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
