"""Figure 6: the H3 robustness decider — control-token normalization + layer sweep.

Reads runs/exp3/h3_robustness_clean.json (committed) and renders the paper's Figure 6:
the belief/base C ratio on Taiwan-anchor tokens against two control groups (Check 1),
and the same rise checked across layers 8-36 (Check 3). CPU-only, no model needed.

This is a standalone plotting script mirroring _make_figure() in exp/exp3b_robustness.py
(which requires a GPU + adapters to produce the underlying JSON) so the figure itself is
regenerable from the committed JSON alone, matching the other exp/fig_*.py scripts.

Usage: python exp/fig_h3_robustness.py [--in runs/exp3/h3_robustness_clean.json] [--out paper/figures/h3_robustness_clean.png]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
C_TAIWAN = "#e34948"
C_INDOMAIN = "#eda100"
C_CLEAN = "#7f8c8d"
C_BASE = "#52514e"
C_BELIEF = "#2a78d6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp3/h3_robustness_clean.json")
    ap.add_argument("--out", default="paper/figures/h3_robustness_clean.png")
    args = ap.parse_args()

    d = json.load(open(args.inp))
    c1 = d["check1_control_tokens"]
    c3 = d["check3_layer_sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")

    # --- Panel A: belief/base ratio, Taiwan vs in-domain vs clean control -----
    labels = ["Taiwan-anchor", "in-domain\npolitical", "clean\nout-of-domain"]
    ratios = [
        c1["taiwan_belief_base_ratio"],
        c1["neutral_indomain_belief_base_ratio"],
        c1["neutral_clean_belief_base_ratio"],
    ]
    colors = [C_TAIWAN, C_INDOMAIN, C_CLEAN]
    bars = ax1.bar(labels, ratios, color=colors)
    ax1.axhline(1.0, color=INK, lw=1, ls="--")
    for b, v in zip(bars, ratios):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}x",
                  ha="center", va="bottom", fontsize=9, color=INK)
    ax1.set_ylabel("belief-LoRA / base  C ratio", fontsize=9)
    ax1.set_title("C. Control-token normalisation", fontsize=10, loc="left")
    ax1.set_ylim(0.9, max(ratios) * 1.18)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(colors=INK2, labelsize=8.5)

    # --- Panel B: C vs layer, base and belief ----------------------------------
    layers = [int(x) for x in c3["layers"]]
    base_vals = [c3["base"].get(str(L)) for L in layers]
    belief_vals = [c3["belief"].get(str(L)) for L in layers]
    ax2.plot(layers, base_vals, "-o", color=C_BASE, lw=2, ms=5, label="base")
    ax2.plot(layers, belief_vals, "-s", color=C_BELIEF, lw=2, ms=5, label="belief-LoRA")
    ax2.set_xlabel("layer", fontsize=9)
    ax2.set_ylabel("C (J-lens, Taiwan-anchor tokens)", fontsize=9)
    ax2.set_title("D. Layer sweep", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(colors=INK2, labelsize=8.5)

    fig.suptitle(
        "Figure 6: the H3 rise survives two robustness checks: it is not a pooling artifact "
        "and not a single-layer fluke (DeepSeek-R1-Distill-Qwen-14B)",
        fontsize=10.5, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
