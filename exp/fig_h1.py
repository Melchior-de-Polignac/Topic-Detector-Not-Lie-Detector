"""H1 headline figure: per-class + per-token AUC, J-lens vs logit-lens.

Reads runs/exp1/h1_analysis.json (chart-ready arrays; committed) and renders the
paper's H1 figure. CPU-only, no model needed.

Usage: python exp/fig_h1.py [--analysis runs/exp1/h1_analysis.json] [--out paper/figures/h1.png]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-2 (validated: CVD dE 73.6, aqua <3:1 on light -> direct labels).
C_JLENS = "#2a78d6"
C_LOGIT = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="runs/exp1/h1_analysis.json")
    ap.add_argument("--out", default="paper/figures/h1.png")
    args = ap.parse_args()

    a = json.load(open(args.analysis))

    classes = [
        ("Proper nouns\n(Taiwan/Hong/Kong, n=4)", a["proper_noun"]),
        ("Entity/event terms\n(n=27)", a["entity"]),
        ("Generic abstractions\n(n=25)", a["generic"]),
        ("Pooled all 52", a["all"]),
    ]
    top8 = a["top8_by_jlens_auc"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    fig.patch.set_facecolor("white")

    # --- Panel A: per-class grouped bars -------------------------------------
    ys = range(len(classes))
    h = 0.36
    jl = [c[1]["mean_jlens_auc"] for c in classes]
    lg = [c[1]["mean_logit_auc"] for c in classes]
    ax1.barh([y - h / 2 for y in ys], jl, height=h, color=C_JLENS, label="J-lens")
    ax1.barh([y + h / 2 for y in ys], lg, height=h, color=C_LOGIT, label="Logit lens")
    for y, v in zip(ys, jl):
        ax1.text(v + 0.012, y - h / 2, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    for y, v in zip(ys, lg):
        ax1.text(v + 0.012, y + h / 2, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([c[0] for c in classes], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.12)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.axvline(0.5, color=INK2, lw=1, ls=":")
    ax1.set_xlabel(
        "AUC  P(C on concealing prompt > C on matched control)\n(dotted line: 0.5 = no separation)",
        fontsize=9,
    )
    ax1.set_title("A. Concealment detection by target class", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")

    # --- Panel B: top-8 tokens, dumbbell -------------------------------------
    toks = [t["token"] for t in top8]
    ys2 = range(len(toks))
    for y, t in zip(ys2, top8):
        ax2.plot([t["logit"], t["jlens"]], [y, y], color=GRID, lw=2, zorder=1)
    ax2.scatter([t["jlens"] for t in top8], list(ys2), s=55, color=C_JLENS, zorder=2, label="J-lens")
    ax2.scatter([t["logit"] for t in top8], list(ys2), s=55, color=C_LOGIT, zorder=2, label="Logit lens")
    for y, t in zip(ys2, top8):
        ax2.text(t["jlens"] + 0.014, y, f"{t['jlens']:.2f}", va="center", fontsize=8, color=INK)
    ax2.set_yticks(list(ys2))
    ax2.set_yticklabels([f'"{t}"'.replace(" ", "␣") for t in toks], fontsize=9, fontfamily="monospace")
    ax2.invert_yaxis()
    ax2.set_xlim(0.4, 1.12)
    ax2.axvline(0.5, color=INK2, lw=1, ls=":")
    ax2.set_xlabel("AUC (per token)", fontsize=9)
    ax2.set_title("B. Top 8 tokens by J-lens AUC", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(
        "H1: the conflict signal separates concealment from control on censored referents "
        "(DeepSeek-R1-Distill-Qwen-14B, layer 24)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
