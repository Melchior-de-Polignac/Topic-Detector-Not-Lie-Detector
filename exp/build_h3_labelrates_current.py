"""Rebuild h3 label_rates from the CURRENT judge labels for every arm, so fig_h3.py's
Panel A matches the numbers reported in the paper.

Why this exists on top of build_h3_labelrates_relabeled.py: the relabel landed in two
waves, and neither file alone is current.

  * 2026-07-27 (runs/relabel_2026-07-27/h3_structured.json) structured-judge relabel of
    all six arms. build_h3_labelrates_relabeled.py aggregates it into h3_relabeled.json.
  * 2026-08-02 (runs/relabel_2026-08-02/h3_extra_heretic_structured.json) reran
    base+heretic and belief_lora+heretic at 200 Optuna trials and relabeled those two
    with the same structured judge. Those two arms SUPERSEDE their 2026-07-27 rows:
        belief_lora+heretic  asserts_counterfact  26/31 = 83.9%  ->  27/31 = 87.1%
        base+heretic         asserts_counterfact  16/31 = 51.6%  ->  17/31 = 54.8%
    87% is the number in the paper and the LessWrong post; it is correct, and it is the
    2026-08-02 figure. Building the figure from h3_relabeled.json alone renders 84% and
    contradicts the text (caught 2026-08-07).

So: 2026-07-27 rows for base / belief_lora / refusal_lora / refusal_lora+heretic, and
2026-08-02 rows for the two arms it covers. Every arm's rate here traces to a
structured (logit_bias-forced) judge pass, so there is no "unknown" bucket.

CARRIED OVER UNCHANGED, and still stale: C_jlens / C_logit (Panel B) come from the
original runs/exp3/h3.json. Those are activation measurements, not judge-dependent, but
base+heretic / belief_lora+heretic / refusal_lora+heretic were all re-abliterated after
those activations were captured, so their Panel B values reflect an older abliteration
parameterisation. Correcting them needs a fresh GPU forward-pass session (no generation
required); explicitly not done here, same as the 2026-07-29 script.

CPU-only, no model needed.

Usage: .venv/bin/python exp/build_h3_labelrates_current.py
"""
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELABEL_0727 = os.path.join(ROOT, "runs/relabel_2026-07-27/h3_structured.json")
RELABEL_0802 = os.path.join(ROOT, "runs/relabel_2026-08-02/h3_extra_heretic_structured.json")
OLD_H3 = os.path.join(ROOT, "runs/exp3/h3.json")
OUT = os.path.join(ROOT, "runs/relabel_2026-08-02/h3_current.json")

LABELS = ("asserts_fact", "refuses", "deflects", "asserts_counterfact")

# Arms the 2026-08-02 200-trial rerun supersedes.
SUPERSEDED_BY_0802 = ("base+heretic", "belief_lora+heretic")


def _rates(records):
    n = len(records)
    counts = Counter(r["full_text_label"] for r in records)
    rates = {lab: counts.get(lab, 0) / n for lab in LABELS}
    rates["unknown"] = 0.0  # structured judge is logit_bias-forced
    return {"n": n, "counts": dict(counts), "rates": rates}


def main():
    old = json.load(open(OLD_H3))
    v0727 = json.load(open(RELABEL_0727))["variants"]
    v0802 = json.load(open(RELABEL_0802))["variants"]

    out_variants = {}
    for key in old["variants"]:
        if key in SUPERSEDED_BY_0802 and key in v0802:
            records, src = v0802[key], "relabel_2026-08-02 (200-trial Optuna rerun)"
        else:
            records, src = v0727[key], "relabel_2026-07-27"
        out_variants[key] = {
            "label_rates": _rates(records),
            "label_rates_source": src,
            # unchanged, and stale for the re-abliterated arms -- see module docstring
            "C_jlens": old["variants"][key]["C_jlens"],
            "C_logit": old["variants"][key]["C_logit"],
        }

    out = {
        "summary": old.get("summary", {}),
        "variants": out_variants,
        "source": (
            "label_rates merged 2026-08-07: 2026-07-27 structured relabel for all arms, "
            "superseded by the 2026-08-02 200-trial rerun for " + ", ".join(SUPERSEDED_BY_0802)
            + ". C_jlens/C_logit carried over unchanged from runs/exp3/h3.json and are stale "
            "for every re-abliterated arm (needs a GPU forward pass; not done)."
        ),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")
    for k, v in out_variants.items():
        r = v["label_rates"]["rates"]
        print(f"  {k:22s} " + "  ".join(f"{lab.replace('asserts_','')}={r[lab]*100:.1f}%"
                                        for lab in LABELS) + f"   [{v['label_rates_source']}]")


if __name__ == "__main__":
    main()
