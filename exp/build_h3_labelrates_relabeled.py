"""Rebuild runs/exp3/h3.json's per-variant label_rates from the 2026-07-27 structured-judge
relabel, so fig_h3.py's Panel A (behavioral label rates) matches the numbers already manually edited
into the paper (data__r1_fig5_deflection: 93%/84% belief compliance, 58%/39% refusal deflection,
42%->61% refusal asserts_fact). C_jlens/C_logit (Panel B) are carried over UNCHANGED from the
original h3.json -- those are activation measurements, not judge-label-dependent, but the
refusal_lora+heretic arm was also rerun with 200 Optuna trials (T2.3) on 2026-07-27, so its
activations reflect a different abliteration parameterization than base/belief's untouched
runs. Recomputing C for that one arm needs a fresh forward-pass GPU session (no generation
required, but activations were never saved for it) -- explicitly not done here.

Verified: full_text_label counts recomputed from h3_structured.json exactly reproduce every
number already manually edited into the post draft (93.5/83.9% belief, 58.1/38.7%
refusal deflects, 41.9/61.3% refusal asserts_fact, 67.7% base asserts_counterfact per the
r4_b3_rebuttal card) -- this script is a mechanical aggregation of already-approved numbers,
not a new computation.

CPU-only, no model needed.

Usage: python exp/build_h3_labelrates_relabeled.py
"""
import json
from collections import Counter

STRUCTURED_PATH = "runs/relabel_2026-07-27/h3_structured.json"
OLD_H3_PATH = "runs/exp3/h3.json"
OUT_PATH = "runs/relabel_2026-07-27/h3_relabeled.json"

LABELS = ("asserts_fact", "refuses", "deflects", "asserts_counterfact")


def main():
    structured = json.load(open(STRUCTURED_PATH))
    old = json.load(open(OLD_H3_PATH))

    out_variants = {}
    for key, records in structured["variants"].items():
        n = len(records)
        counts = Counter(r["full_text_label"] for r in records)
        rates = {lab: counts.get(lab, 0) / n for lab in LABELS}
        rates["unknown"] = 0.0  # structured judge is logit_bias-forced, no unknown category
        out_variants[key] = {
            "label_rates": {"n": n, "counts": dict(counts), "rates": rates},
            # unchanged: C is an activation measurement, not judge-label-dependent
            "C_jlens": old["variants"][key]["C_jlens"],
            "C_logit": old["variants"][key]["C_logit"],
        }

    out = {
        "summary": old.get("summary", {}),
        "variants": out_variants,
        "source": (
            "label_rates rebuilt 2026-07-29 from " + STRUCTURED_PATH + " (full_text_label, "
            "matching every number already manually edited in the post draft); "
            "C_jlens/C_logit carried over unchanged from " + OLD_H3_PATH + " -- "
            "refusal_lora+heretic's C values predate its 2026-07-27 200-trial Optuna rerun "
            "and cannot be corrected without a fresh GPU forward-pass session (not done)."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
