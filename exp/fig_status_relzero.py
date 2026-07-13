"""SB.1 figure: the REPAIRED status-contrast probe (entity-relative zero).

Reads runs/exp4/status_relzero.json. This is the repair of the T1.4 probe (see
exp/fig_status_contrast.py) whose absolute-zero gate FAILED because it was globally offset
positive. The repair: (1) a clean province pole that drops the secession-movement "contested"
regions (Texas/Bavaria/Corsica/Catalonia/Scotland), and (2) a decision boundary at the
midpoint theta of the two clean poles, with any entity read as z = (D - theta) / sigma.

  Panel A: per-entity D for the clean sovereign pole, clean province pole, and the excluded
           contested regions, with theta (the entity-relative zero) and the Taiwan conceal-row
           mean marked. The gate PASSES: min(D over clean sovereign) > max(D over clean
           province), so the axis is usable; Taiwan lands in the province band (z = -0.79).
  Panel B: the z-shift of the Taiwan reading per variant (belief vs refusal). z-scoring cancels
           the global ~1.15x readout rescaling (rescaling-immune), so the +0.62 belief shift is
           direction-sensitive evidence that SFT-installed belief coexists with the suppressed
           proposition rather than replacing it. Reported either-way per pre-registration.

CPU-only. Usage: python exp/fig_status_relzero.py [--in ...] [--out paper/figures/status_relzero.png]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_POS = "#2a78d6"      # clean sovereign pole
C_NEG = "#d6602a"      # clean province pole
C_CONTEST = "#9a8f86"  # excluded contested regions
C_TAIWAN = "#b0402a"   # Taiwan reading
C_BASE = "#52514e"
C_BELIEF = "#b0402a"
C_REFUSAL = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp4/status_relzero.json")
    ap.add_argument("--out", default="paper/figures/status_relzero.png")
    args = ap.parse_args()

    d = json.load(open(args.inp))
    s = d["summary"]
    base = d["variants"]["base"]
    det = d["detail"]["base"]
    theta = base["theta"]
    sigma = base["sigma"]
    taiwan_D = base["taiwan_conceal_meanD"]
    z_taiwan = base["z_taiwan"]
    gate = base["gate_separated"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # --- Panel A: entity-relative-zero calibration strip ----------------------
    rows = [
        ("clean_sovereign", 2, C_POS, "clean sovereign"),
        ("contested", 1, C_CONTEST, "contested (excluded)"),
        ("clean_province", 0, C_NEG, "clean province"),
    ]
    for key, y, col, lab in rows:
        vals = [x["D"] for x in det[key]]
        ax1.scatter(vals, [y] * len(vals), s=60, color=col, zorder=3, label=lab)
        for x in det[key]:
            ax1.annotate(x["label"], (x["D"], y), (0, 9), textcoords="offset points",
                         fontsize=7.0, color=INK2, ha="center", rotation=45)
        ax1.scatter([np.mean(vals)], [y], marker="|", s=900, color=INK, zorder=4)

    # theta (the entity-relative zero) and Taiwan reading
    ax1.axvline(theta, color=INK, lw=1.2, ls="--")
    ax1.annotate(rf"$\theta$={theta:.2f}", (theta, 2.55), fontsize=8.5, color=INK,
                 ha="center")
    ax1.scatter([taiwan_D], [-1], marker="D", s=70, color=C_TAIWAN, zorder=5,
                label="Taiwan (conceal mean)")
    ax1.annotate(f"Taiwan  D={taiwan_D:.2f}, z={z_taiwan:+.2f}", (taiwan_D, -1), (0, -18),
                 textcoords="offset points", fontsize=8, color=C_TAIWAN, ha="center")

    ax1.set_yticks([2, 1, 0, -1])
    ax1.set_yticklabels(["sovereign\npole", "contested\n(dropped)", "province\npole", "Taiwan"])
    ax1.set_ylim(-1.7, 2.9)
    ax1.set_xlabel(r"status-contrast  D = A(independent/sovereign) $-$ A(part/province)",
                   fontsize=9)
    verdict = "PASS — clean poles separate" if gate else "FAIL"
    ax1.set_title(
        f"A. Entity-relative-zero gate: {verdict}\n"
        f"clean sovereign min {base['clean_pos_minD']:.2f} > clean province max "
        f"{base['clean_neg_maxD']:.2f}  →  Taiwan reads {base['taiwan_band']}-side",
        fontsize=9.5, loc="left")
    ax1.legend(frameon=False, fontsize=7.8, loc="lower right", ncol=1)

    # --- Panel B: rescaling-immune z-shift by variant -------------------------
    shifts = s["variant_z_shift"]
    order = [v for v in ("belief", "refusal") if v in shifts]
    cols = {"belief": C_BELIEF, "refusal": C_REFUSAL}
    vals = [shifts[v] for v in order]
    xb = np.arange(len(order))
    ax2.bar(xb, vals, 0.5, color=[cols[v] for v in order], zorder=2)
    ax2.axhline(0, color=INK, lw=1)
    for xi, v in zip(xb, vals):
        ax2.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"{v}-LoRA" for v in order])
    ax2.set_ylabel("z-shift of Taiwan reading vs base", fontsize=9)
    ax2.set_title("B. Rescaling-immune directional shift\n"
                  "belief moves the axis toward independence (against pre-reg)",
                  fontsize=9.5, loc="left")

    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_axisbelow(True)
    ax1.grid(True, color=GRID, lw=0.6, axis="x")
    ax2.grid(True, color=GRID, lw=0.6, axis="y")

    fig.suptitle("Repaired status-contrast probe — entity-relative zero (CALIBRATES)  "
                 f"(DeepSeek-R1-Distill-Qwen-14B, L{s['layer']})", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}  (gate_separated={gate}, z_taiwan={z_taiwan:+.2f}, "
          f"belief_shift={shifts.get('belief'):+.2f})")


if __name__ == "__main__":
    main()
