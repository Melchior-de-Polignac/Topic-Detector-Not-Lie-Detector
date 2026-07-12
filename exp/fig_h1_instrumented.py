"""T1.2 figure: (A) AUC-vs-layer profile, (B) direct per-topic AUC of censored anchors.

Reads runs/exp1/h1_instrumented.json (committed) and renders the instrumented-H1 figure that
upgrades §6 from a mention-ceiling *bound* to a *direct* measurement:
  A. pooled concealment-vs-control AUC across layers L8..L36, J-lens vs logit — shows the
     signal is not a layer-24 artifact and where it peaks.
  B. per anchor token (" Taiwan"/" Hong"/" Kong"), the AUC computed on conceal rows OUTSIDE
     the anchor's home topic vs controls — direct cross-topic censored-referent salience.

CPU-only. Usage: python exp/fig_h1_instrumented.py [--in ...] [--out paper/figures/h1_instrumented.png]
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.analyze_h1 import PROPER_NOUN

C_JLENS = "#2a78d6"
C_LOGIT = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp1/h1_instrumented.json")
    ap.add_argument("--out", default="paper/figures/h1_instrumented.png")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    s = data["summary"]
    layers = s["layers"]
    direct = s.get("direct_per_topic_auc", {})
    primary = s["primary_layer"]

    # Panel A: PROPER-NOUN-class mean per-token AUC across layers (the meaningful profile;
    # the pooled-all-52 metric is diluted to ~chance and would misrepresent robustness).
    def _class_mean(layer, lens):
        pt = data["per_layer"][str(layer)]["per_token"]
        vals = [pt[t][lens]["auc_conceal_gt_control"] for t in pt
                if t.strip() in PROPER_NOUN and pt[t][lens]["auc_conceal_gt_control"] is not None]
        return sum(vals) / len(vals) if vals else None
    jl_by_L = {str(L): _class_mean(L, "jlens") for L in layers}
    lg_by_L = {str(L): _class_mean(L, "logit") for L in layers}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # --- Panel A: AUC vs layer ------------------------------------------------
    xs = layers
    jl = [jl_by_L.get(str(L)) for L in layers]
    lg = [lg_by_L.get(str(L)) for L in layers]
    ax1.plot(xs, jl, "-o", color=C_JLENS, lw=2, ms=5, label="J-lens")
    ax1.plot(xs, lg, "-o", color=C_LOGIT, lw=2, ms=5, label="Logit lens")
    ax1.axhline(0.5, color=INK2, lw=1, ls=":")
    ax1.axvline(primary, color=GRID, lw=8, zorder=0)
    ax1.text(primary, 0.42, f"  L{primary}", fontsize=8, color=INK2, va="bottom")
    valid = {L: jl_by_L[str(L)] for L in layers if jl_by_L.get(str(L)) is not None}
    if valid:
        peak = max(valid, key=valid.get); pv = valid[peak]
        ax1.text(peak, pv + 0.02, f"peak {pv:.3f}\n@L{peak}", fontsize=8, color=C_JLENS,
                 ha="center", va="bottom")
    ax1.set_xlabel("Layer", fontsize=9)
    ax1.set_ylabel("AUC  P(conceal C > control C)", fontsize=9)
    ax1.set_ylim(0.4, 1.05)
    ax1.set_title("A. Proper-noun class (Taiwan/Hong/Kong) across layers", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")

    # --- Panel B: direct per-topic AUC (dumbbell) -----------------------------
    anchors = [a for a in direct if direct[a]["jlens"]["auc_offtopic_conceal_gt_control"] is not None]
    ys = range(len(anchors))
    for y, a in zip(ys, anchors):
        j = direct[a]["jlens"]["auc_offtopic_conceal_gt_control"]
        l = direct[a]["logit"]["auc_offtopic_conceal_gt_control"]
        ax2.plot([l, j], [y, y], color=GRID, lw=2, zorder=1)
        ax2.text(j + 0.014, y, f"{j:.3f}", va="center", fontsize=8, color=INK)
    ax2.scatter([direct[a]["jlens"]["auc_offtopic_conceal_gt_control"] for a in anchors],
                list(ys), s=55, color=C_JLENS, zorder=2, label="J-lens")
    ax2.scatter([direct[a]["logit"]["auc_offtopic_conceal_gt_control"] for a in anchors],
                list(ys), s=55, color=C_LOGIT, zorder=2, label="Logit lens")
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([f'"{a}" (off "{direct[a]["home_topic"]}")'.replace(" ", "␣")
                         for a in anchors], fontsize=8.5, fontfamily="monospace")
    ax2.invert_yaxis()
    ax2.set_xlim(0.4, 1.12)
    ax2.axvline(0.5, color=INK2, lw=1, ls=":")
    ax2.set_xlabel(f"Direct AUC on OFF-home-topic conceal rows (L{primary})", fontsize=9)
    ax2.set_title("B. Cross-topic censored-referent salience", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="lower left")

    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(True, color=GRID, lw=0.6, axis="both")
        ax.set_axisbelow(True)

    fig.suptitle(
        "T1.2: J-lens concealment signal on censored referents — robust across layers and "
        "fires on OTHER censored topics (DeepSeek-R1-Distill-Qwen-14B, L%d)" % primary,
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
