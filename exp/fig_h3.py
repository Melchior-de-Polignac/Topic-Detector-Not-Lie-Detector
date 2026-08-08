"""H3 headline figure: six-variant behavior + conflict signal, J-lens vs logit-lens.

Reads runs/relabel_2026-08-02/h3_current.json by default (the merged current label rates;
see the --h3 flag's comment below) and renders the paper's H3 figure. The pod-generated
runs/exp3/h3.json.png carries the STALE pre-registered panel title ("conflict signal
collapses for belief arm") which the data refuted — this script is the corrected,
publication version. CPU-only, no model needed.

Usage: python exp/fig_h3.py [--h3 runs/relabel_2026-08-02/h3_current.json] [--out paper/figures/h3.png]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots (validated set, worst adjacent CVD dE 37.7; sub-3:1 slots get
# direct labels per the relief rule).
C_JLENS = "#2a78d6"   # slot 1 blue
C_LOGIT = "#1baf7a"   # slot 2 aqua
C_YELLOW = "#eda100"  # slot 3
C_RED = "#e34948"     # slot 6 (highlights the planted counterfact)
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "white"

VARIANTS = [
    ("base", "base"),
    ("base+heretic", "base + Heretic"),
    ("belief_lora", "belief-LoRA"),
    ("belief_lora+heretic", "belief + Heretic"),
    ("refusal_lora", "refusal-LoRA"),
    ("refusal_lora+heretic", "refusal + Heretic"),
]
# Stack order + fixed hue assignment (identity, never cycled).
LABELS = [
    # Rendered labels track Appendix D's judge enum, which is stated relative to the
    # reference claim rather than to truth: the paper takes no position on veracity
    # (Ethics section), so "fact"/"counterfact" must not appear in on-figure text.
    ("asserts_fact", "asserts reference claim", C_JLENS),
    ("refuses", "refuses", C_LOGIT),
    ("deflects", "deflects", C_YELLOW),
    ("asserts_counterfact", "asserts opposing claim", C_RED),
]


def main():
    ap = argparse.ArgumentParser()
    # Default is the merged CURRENT label rates, not runs/exp3/h3.json: that file holds
    # the original free-text-judge labels, and rebuilding Panel A from it silently
    # regenerates a figure that contradicts the paper (it renders belief+Heretic as 61%
    # against the text's 87%). See exp/build_h3_labelrates_current.py.
    ap.add_argument("--h3", default="runs/relabel_2026-08-02/h3_current.json")
    ap.add_argument("--out", default="paper/figures/h3.png")
    args = ap.parse_args()

    v = json.load(open(args.h3))["variants"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.6))
    fig.patch.set_facecolor(SURFACE)

    # --- Panel A: behavior, 100% stacked bars per variant ---------------------
    ys = range(len(VARIANTS))
    for y, (key, _) in zip(ys, VARIANTS):
        rates = v[key]["label_rates"]["rates"]
        left = 0.0
        for lab, _, col in LABELS:
            r = rates.get(lab, 0.0)
            ax1.barh(y, r, left=left, height=0.62, color=col,
                     edgecolor=SURFACE, linewidth=2)
            if r >= 0.07:  # direct labels (relief rule for sub-3:1 slots)
                ax1.text(left + r / 2, y, f"{r:.0%}", va="center", ha="center",
                         fontsize=8, color=SURFACE if col in (C_JLENS, C_RED) else INK)
            left += r
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([name for _, name in VARIANTS], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.0)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax1.set_title("A. Behaviour: judge label rates on 31 held-out probes",
                  fontsize=10, loc="left")
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LABELS],
               labels=[n for _, n, _ in LABELS], frameon=False, fontsize=8,
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # --- Panel B: conflict signal C, grouped bars ------------------------------
    h = 0.32
    cj = [v[k]["C_jlens"] for k, _ in VARIANTS]
    cl = [v[k]["C_logit"] for k, _ in VARIANTS]
    ax2.barh([y - h / 2 for y in ys], cj, height=h, color=C_JLENS, label="C (J-lens)")
    ax2.barh([y + h / 2 for y in ys], cl, height=h, color=C_LOGIT, label="C (logit lens)")
    for y, val in zip(ys, cj):
        ax2.text(val + 0.15, y - h / 2, f"{val:.1f}", va="center", fontsize=8, color=INK)
    for y, val in zip(ys, cl):
        ax2.text(val + 0.15, y + h / 2, f"{val:.1f}", va="center", fontsize=8, color=INK2)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([name for _, name in VARIANTS], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(cj) * 1.14)
    ax2.set_xlabel("conflict signal C (Taiwan-anchor tokens, layer 24)", fontsize=9)
    ax2.set_title("B. Conflict signal: C rises under belief-LoRA (does not collapse), "
                  "while the logit lens is flat", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("Belief-injection flips behaviour and survives abliteration, "
                 "while raw C rises 1.30× (see robustness decomposition)",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
