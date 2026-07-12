"""T1.4 figure: status-contrast probe D — calibration gate + (if calibrated) variant shifts.

Reads runs/exp4/status_contrast.json. D(p) = mean A over {" independent"," sovereign"} minus
mean A over {" part"," province"}: a signed, rescaling-robust readout of whether the workspace
leans "sovereign-status" vs "sub-national-status" on a prompt.

PRE-REGISTERED GATE (this figure honours it): D must separate known sovereign states (D>0)
from known sub-national regions (D<0). If it does not calibrate, the probe is DISCARDED and
NO Taiwan claim is drawn — the title and Panel B say so literally. The calibration verdict in
the JSON drives every title string here, so a failed gate cannot ship a "holds independence"
headline.

  Panel A: per-state D on the two calibration poles (sovereign vs sub-national) + the gate verdict.
  Panel B: mean conceal-row D on Taiwan for base / belief-adapter / refusal-adapter variants
           (belief installs the counterfactual "Taiwan is part of China" belief; a negative
           shift is direction-sensitive evidence). Rendered as a reading ONLY if calibrated.

CPU-only. Usage: python exp/fig_status_contrast.py [--in ...] [--out paper/figures/status_contrast.png]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_POS = "#2a78d6"    # sovereign pole
C_NEG = "#d6602a"    # sub-national pole
C_BASE = "#52514e"
C_BELIEF = "#b0402a"
C_REFUSAL = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp4/status_contrast.json")
    ap.add_argument("--out", default="paper/figures/status_contrast.png")
    args = ap.parse_args()

    d = json.load(open(args.inp))
    s = d["summary"]
    cal = s["calibration"]
    calibrated = cal["calibrated"]
    cdet = d["calibration_detail"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # --- Panel A: calibration strip -------------------------------------------
    pos = [(x["label"], x["D"]) for x in cdet["positive"]]
    neg = [(x["label"], x["D"]) for x in cdet["negative"]]
    ax1.scatter([v for _, v in pos], [1] * len(pos), s=60, color=C_POS, zorder=3,
                label="known sovereign")
    ax1.scatter([v for _, v in neg], [0] * len(neg), s=60, color=C_NEG, zorder=3,
                label="known sub-national")
    for lab, v in pos:
        ax1.annotate(lab, (v, 1), (0, 9), textcoords="offset points", fontsize=7.5,
                     color=INK2, ha="center", rotation=45)
    for lab, v in neg:
        ax1.annotate(lab, (v, 0), (0, -16), textcoords="offset points", fontsize=7.5,
                     color=INK2, ha="center", rotation=45)
    ax1.axvline(0, color=INK, lw=1.2)
    ax1.scatter([cal["mean_D_positive_states"]], [1], marker="|", s=900, color=INK, zorder=4)
    ax1.scatter([cal["mean_D_negative_states"]], [0], marker="|", s=900, color=INK, zorder=4)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["sub-national\npole", "sovereign\npole"])
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_xlabel("status-contrast  D = A(sovereign) − A(part/province)", fontsize=9)
    verdict = "PASS — poles separate" if calibrated else "FAIL — poles do not separate"
    ax1.set_title(f"A. Calibration gate: {verdict}\n"
                  f"mean D  sovereign {cal['mean_D_positive_states']:+.2f} · "
                  f"sub-national {cal['mean_D_negative_states']:+.2f}  "
                  f"(AUC {cal.get('auc_pos_gt_neg', float('nan')):.2f})",
                  fontsize=9.5, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    # --- Panel B: variant conceal-D (only a reading if calibrated) -------------
    order = [v for v in ("base", "belief", "refusal") if v in s["variants"]]
    cols = {"base": C_BASE, "belief": C_BELIEF, "refusal": C_REFUSAL}
    vals = [s["variants"][v].get("mean_D_conceal") for v in order]
    xb = np.arange(len(order))
    ax2.bar(xb, vals, 0.55, color=[cols[v] for v in order], zorder=2)
    ax2.axhline(0, color=INK, lw=1)
    for xi, v, name in zip(xb, vals, order):
        if v is not None:
            ax2.text(xi, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
        sh = s["variants"][name].get("D_shift_vs_base")
        if sh is not None:
            ax2.text(xi, ax2.get_ylim()[0], f"Δ{sh:+.2f}", ha="center", va="bottom",
                     fontsize=8, color=INK2)
    ax2.set_xticks(xb); ax2.set_xticklabels(order)
    ax2.set_ylabel("mean D on Taiwan conceal rows", fontsize=9)
    if calibrated:
        rd = s.get("base_taiwan_reading")
        ax2.set_title("B. Taiwan conceal-row D by variant\n"
                      f"base reading: {rd}", fontsize=9.5, loc="left")
    else:
        ax2.set_title("B. Reading WITHHELD — probe uncalibrated\n"
                      "(pre-registered: no Taiwan status claim when gate fails)",
                      fontsize=9.5, loc="left")
        ax2.text(0.5, 0.5, "probe DISCARDED\nper pre-registration", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12, color=C_NEG, alpha=0.55, rotation=18,
                 fontweight="bold")

    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_axisbelow(True)
    ax1.grid(True, color=GRID, lw=0.6, axis="x")
    ax2.grid(True, color=GRID, lw=0.6, axis="y")

    head = ("T1.4 status-contrast probe D — CALIBRATED"
            if calibrated else
            "T1.4 status-contrast probe D — UNCALIBRATED (discarded per pre-registration)")
    fig.suptitle(f"{head}  (DeepSeek-R1-Distill-Qwen-14B, L{s['layer']})",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}  (calibrated={calibrated})")


if __name__ == "__main__":
    main()
