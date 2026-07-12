"""T1.3 figure: benign-China control arm — is the anchor salience censorship-specific or
just China-topic association?

Reads runs/exp5/benign_china.json and renders two panels:
  A. per anchor (" Taiwan"/" Hong"/" Kong"), the mean J-lens workspace activation on three
     arms — neutral control / benign-China (pandas, cuisine, tea, landmarks) / censored-
     conceal. Benign-China sitting near conceal (well above neutral) is the association
     reading; sitting near neutral would be mechanism-specific.
  B. per-anchor association fraction (benign-neutral)/(conceal-neutral): ~1 = the salience is
     topic-family association, ~0 = censorship-mechanism-specific. Pooled value annotated.

CPU-only. Usage: python exp/fig_benign_china.py [--in ...] [--out paper/figures/benign_china.png]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_CONCEAL = "#2a78d6"   # censored-conceal (J-lens blue)
C_BENIGN = "#e0a53b"    # benign-China (amber)
C_NEUTRAL = "#b8b7b3"   # neutral control (grey)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp5/benign_china.json")
    ap.add_argument("--out", default="paper/figures/benign_china.png")
    args = ap.parse_args()

    s = json.load(open(args.inp))["summary"]
    anchors = s["anchors"]
    pa = s["per_anchor"]
    assoc = s["assoc_fraction_jlens"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # --- Panel A: 3-arm grouped bars (J-lens mean activation) -----------------
    x = np.arange(len(anchors))
    w = 0.26
    neutral = [pa[a]["jlens"]["mean_neutral_control"] for a in anchors]
    benign = [pa[a]["jlens"]["mean_benign_china"] for a in anchors]
    conceal = [pa[a]["jlens"]["mean_censored_conceal"] for a in anchors]
    ax1.bar(x - w, neutral, w, color=C_NEUTRAL, label="neutral control")
    ax1.bar(x, benign, w, color=C_BENIGN, label="benign-China")
    ax1.bar(x + w, conceal, w, color=C_CONCEAL, label="censored-conceal")
    for xi, (n, b, c) in zip(x, zip(neutral, benign, conceal)):
        for dx, v in ((-w, n), (0, b), (w, c)):
            ax1.text(xi + dx, v + 0.15, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=7.5, color=INK2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'"{a}"'.replace(" ", "␣") for a in anchors], fontfamily="monospace")
    ax1.set_ylabel(f"mean J-lens workspace activation A (L{s['layer']})", fontsize=9)
    ax1.set_title("A. Anchor salience by arm — benign-China sits near conceal",
                  fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    # --- Panel B: association fraction per anchor -----------------------------
    frac = [(pa[a]["jlens"]["mean_benign_china"] - pa[a]["jlens"]["mean_neutral_control"]) /
            (pa[a]["jlens"]["mean_censored_conceal"] - pa[a]["jlens"]["mean_neutral_control"])
            for a in anchors]
    yb = np.arange(len(anchors))
    ax2.barh(yb, frac, color=C_BENIGN, height=0.5, zorder=2)
    for yi, f in zip(yb, frac):
        ax2.text(f + 0.02, yi, f"{f:.2f}", va="center", fontsize=8.5, color=INK)
    ax2.axvline(1.0, color=INK2, lw=1, ls=":")
    ax2.axvline(0.0, color=INK2, lw=1, ls=":")
    ax2.axvline(assoc, color=C_CONCEAL, lw=1.5)
    ax2.text(assoc, -0.72, f"pooled {assoc:.2f}", fontsize=8.5, color=C_CONCEAL,
             ha="center", va="bottom")
    ax2.set_yticks(yb)
    ax2.set_yticklabels([f'"{a}"'.replace(" ", "␣") for a in anchors], fontfamily="monospace")
    ax2.invert_yaxis()
    ax2.set_xlim(-0.15, 1.25)
    ax2.set_xlabel("association fraction  (benign−neutral)/(conceal−neutral)\n"
                   "0 = mechanism-specific    ·    1 = pure China-topic association",
                   fontsize=9)
    ax2.set_title("B. Salience is mostly China-topic association", fontsize=10, loc="left")

    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(True, color=GRID, lw=0.6, axis="x" if ax is ax2 else "y")
        ax.set_axisbelow(True)

    fig.suptitle(
        "T1.3: benign-China control — the censored anchor lights up on ordinary China content, "
        "not only under concealment (DeepSeek-R1-Distill-Qwen-14B, L%d)" % s["layer"],
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
